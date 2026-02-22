"""Experiment: Sequential Graphical Testing over a (threshold × alpha) grid.

Uses the graphical testing procedure (Bretz et al. 2009) to discover
**all** (threshold, alpha) pairs for which a risk guarantee holds,
rather than testing a single fixed alpha like ``guaranteed_risk_cascade``.

This solves the problem where an ambitious alpha target causes FST to
fail entirely, even though valid guarantees exist at less strict levels.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cascade_utils import (
    BatchCascadeStatistics,
    ThresholdCascadeResult,
    compute_batch_statistics,
    compute_overall_metrics,
)
from clearml_serialization import (
    artifact_field,
    derived_field,
    scalar_field,
)
from config import load_config
from dotenv import load_dotenv

from reliable_monitoring.cascade import offline_batch_cascade, run_llm_baseline
from reliable_monitoring.dataset import ActivationConfig, load_dataset, sample_from_dataset, split_dataset
from reliable_monitoring.graphical_test_graphs import lattice_graph, row_chain_graph, uniform_lattice_graph
from reliable_monitoring.learn_then_test import (
    GraphicalTestResult,
    Hypothesis,
    compute_p_values,
    graphical_testing,
    is_pareto,
)
from reliable_monitoring.probes import DegradedProbe, SequenceProbe
from reliable_monitoring.risks import (
    RISK_RGISTRY,
    BudgetCostRisk,
    ThresholdEvaluationResult,
    evaluate_threshold_risks,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_SAMPLE_SIZE = 256

GRAPH_FACTORIES = {
    "lattice": lattice_graph,
    "uniform_lattice": uniform_lattice_graph,
    "row_chain": row_chain_graph,
}


@dataclass
class SGTCascadeResults:
    """Results for the SGT cascade experiment."""

    # Experiment metadata
    config: dict = artifact_field()
    seed: int = scalar_field()
    debug_mode: bool = scalar_field()

    # Dataset information
    test_size: int = scalar_field()
    cascade_batch_size: int = scalar_field()
    num_batches: int = scalar_field()

    # Risk configuration
    guaranteed_risk_name: str = scalar_field()
    guarantee_probability: float = scalar_field()

    # SGT configuration
    sgt_graph_type: str = scalar_field()
    n_thresholds: int = scalar_field()
    n_alphas: int = scalar_field()
    n_hypotheses: int = scalar_field()

    # SGT results
    n_rejected: int = scalar_field()
    rejected_pairs: list[tuple[int, int]] = artifact_field()
    ordered_thresholds: np.ndarray = artifact_field()
    ordered_alphas: np.ndarray = artifact_field()

    # All per-threshold cascade results (one per unique valid threshold)
    threshold_results: list[ThresholdCascadeResult] = artifact_field()
    selection_mode: str = scalar_field()  # "best_alpha" or "best_threshold"

    # Selected best (threshold, alpha) pair
    reliable_threshold: float = scalar_field()
    achieved_alpha: float = scalar_field()

    # Overall cascade batch statistics
    mean_budget_cost: float = scalar_field()
    std_budget_cost: float = scalar_field()
    min_budget_cost: float = scalar_field()
    max_budget_cost: float = scalar_field()

    # Overall cascade performance
    cascade_accuracy: float = scalar_field()
    cascade_f1_score: float = scalar_field()
    cascade_roc_auc: float = scalar_field()

    # Probe-only baseline
    probe_only_accuracy: float = scalar_field()
    probe_only_f1_score: float = scalar_field()
    probe_only_roc_auc: float = scalar_field()

    # LLM-baseline-only
    baseline_only_accuracy: float = scalar_field()
    baseline_only_f1_score: float = scalar_field()
    baseline_only_roc_auc: float = scalar_field()

    # Per-batch detailed results
    batches: list[BatchCascadeStatistics] = artifact_field()

    # Score distributions
    train_probe_scores: np.ndarray = artifact_field()
    calib_probe_scores: np.ndarray = artifact_field()
    test_probe_scores: np.ndarray = artifact_field()
    train_labels: np.ndarray = artifact_field()
    calib_labels: np.ndarray = artifact_field()
    test_labels: np.ndarray = artifact_field()

    # Baseline and cascade scores
    test_baseline_scores: np.ndarray = artifact_field()
    cascade_final_scores: np.ndarray = artifact_field()

    # Calibration evaluation (for diagnostics)
    calib_evaluation_risks: ThresholdEvaluationResult = artifact_field()

    # Pareto testing results (None when pareto_testing is disabled)
    opt_evaluation_risks: ThresholdEvaluationResult | None = artifact_field()
    pareto_mask: np.ndarray | None = artifact_field()
    n_original_thresholds: int = scalar_field()
    n_pareto_thresholds: int | None = scalar_field()

    # Derived
    budget_costs: np.ndarray = derived_field(derive_fn=lambda r: np.array([b.budget_cost for b in r.batches]))


def parse_args():
    default_config_path = Path(__file__).parent / "configs" / "sgt_cascade.yaml"
    parser = argparse.ArgumentParser(description="SGT Cascade Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--use-clearml",
        action="store_true",
        help="Enable ClearML experiment tracking.",
    )
    return parser.parse_args()


def run_sgt_cascade_experiment(config) -> SGTCascadeResults | None:
    """Run the SGT cascade experiment.

    Returns ``None`` when no reliable (threshold, alpha) pair can be found.
    """
    seed = config.seed
    np.random.seed(seed)

    cascade_batch_size = config.cascade_batch_size

    # --- Risk setup ---
    guaranteed_risk_name = getattr(config, "guaranteed_risk", "accuracy_error")
    GuaranteedRisk = RISK_RGISTRY.get(guaranteed_risk_name)
    if GuaranteedRisk is None:
        raise ValueError(f"Invalid guaranteed_risk: '{guaranteed_risk_name}'. Available: {list(RISK_RGISTRY.keys())}")

    degrade_enabled = getattr(config, "probe_degradation_enabled", False)
    if degrade_enabled:
        logger.warning("Probe degradation enabled (fixed settings).")

    activation_config = ActivationConfig(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
    )

    # --- Load data ---
    logger.info("Loading datasets...")
    train_dataset = load_dataset(Path(config.train_dataset_path), activation_config=activation_config)
    calib_dataset = load_dataset(Path(config.calib_dataset_path), activation_config=activation_config)
    test_dataset = load_dataset(Path(config.test_dataset_path), activation_config=activation_config)

    debug_mode = getattr(config, "debug", False)
    if debug_mode:
        logger.warning("Running in debug mode with smaller datasets.")
        train_dataset = sample_from_dataset(train_dataset, DEBUG_SAMPLE_SIZE, seed=seed)
        calib_dataset = sample_from_dataset(calib_dataset, DEBUG_SAMPLE_SIZE, seed=seed)
        test_dataset = sample_from_dataset(test_dataset, DEBUG_SAMPLE_SIZE, seed=seed)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Calibration dataset size: {len(calib_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # --- Data splitting (probe calibration and/or Pareto optimisation) ---
    calibration_method = getattr(config, "calibration_method", None)
    needs_auxiliary_data = calibration_method is not None

    pareto_testing = getattr(config, "pareto_testing", False)
    if pareto_testing:
        pareto_proportion = getattr(config, "pareto_split_proportion", 0.2)
        logger.info(
            f"Splitting calibration data for Pareto testing "
            f"(opt={pareto_proportion:.0%}, calib={1 - pareto_proportion:.0%})"
        )
        calib_dataset, opt_dataset = split_dataset(
            calib_dataset,
            proportions=[1 - pareto_proportion, pareto_proportion],
            shuffle=True,
            seed=seed,
        )
        if needs_auxiliary_data:
            logger.info("Auxiliary operations (probe calibration) will use opt split")
        auxiliary_dataset = opt_dataset if needs_auxiliary_data else None
    elif needs_auxiliary_data:
        auxiliary_proportion = getattr(config, "auxiliary_split_proportion", 0.15)
        logger.info(
            f"Splitting calibration data for probe calibration "
            f"(auxiliary={auxiliary_proportion:.0%}, calib={1 - auxiliary_proportion:.0%})"
        )
        calib_dataset, auxiliary_dataset = split_dataset(
            calib_dataset,
            proportions=[1 - auxiliary_proportion, auxiliary_proportion],
            shuffle=True,
            seed=seed,
        )
    else:
        auxiliary_dataset = None

    logger.info(f"Effective calibration set size for hypothesis testing: {len(calib_dataset)}")

    # --- Probe ---
    logger.info("Fitting probe...")
    base_probe = SequenceProbe(reduction_strategy=config.reduction_strategy)
    probe = DegradedProbe(base_probe, enabled=degrade_enabled, seed=seed)
    probe.fit(train_dataset)

    if calibration_method is not None:
        assert auxiliary_dataset is not None
        logger.info(f"Calibrating probe scores ({calibration_method}) on auxiliary dataset...")
        probe.calibrate(auxiliary_dataset, method=calibration_method)

    logger.info("Computing probe scores on training dataset...")
    train_probe_scores = probe.predict(train_dataset)
    train_labels = train_dataset.labels_numpy()

    # --- Calibration scores ---
    logger.info("Computing probe scores on calibration dataset...")
    calib_probe_scores = probe.predict(calib_dataset)
    calib_labels = calib_dataset.labels_numpy()

    logger.info("Computing baseline scores on calibration dataset...")
    calib_baseline_scores = run_llm_baseline(
        baseline_model_name=config.baseline_model_name,
        dataset=calib_dataset,
        baseline_batch_size=config.baseline_batch_size,
        local=not getattr(config, "use_modal", False),
        gpu=getattr(config, "modal_gpu", None),
    )

    if pareto_testing:
        logger.info("Computing probe scores on optimisation dataset...")
        opt_probe_scores = probe.predict(opt_dataset)
        logger.info("Computing baseline scores on optimisation dataset...")
        opt_baseline_scores = run_llm_baseline(
            baseline_model_name=config.baseline_model_name,
            dataset=opt_dataset,
            baseline_batch_size=config.baseline_batch_size,
            local=not getattr(config, "use_modal", False),
            gpu=getattr(config, "modal_gpu", None),
        )

    # --- Candidate thresholds ---
    thresholds = np.linspace(
        getattr(config, "threshold_start", 0.5),
        getattr(config, "threshold_end", 1.0),
        getattr(config, "threshold_steps", 10),
    )

    # --- Evaluate guaranteed risk on calibration data ---
    guaranteed_needs_dataset = GuaranteedRisk is not BudgetCostRisk

    calib_eval_result = evaluate_threshold_risks(
        calib_probe_scores,
        calib_baseline_scores,
        thresholds,
        risks=GuaranteedRisk,
        dataset=calib_dataset if guaranteed_needs_dataset else None,
        merge_strategy=config.cascade_merge_strategy,
    )

    logger.info(f"Empirical {GuaranteedRisk.description} risks computed.")
    for thr, risk in zip(calib_eval_result.thresholds, calib_eval_result[guaranteed_risk_name], strict=True):
        logger.info(f"  Threshold: {thr:.4f}, Empirical risk: {risk:.4f}")

    n_original_thresholds = len(thresholds)

    # --- Pareto pre-filtering (optional) ---
    if pareto_testing:
        logger.info("Performing Pareto pre-filtering with multiple risks...")
        assert hasattr(config, "opt_risk"), "opt_risk must be specified in config when pareto_testing is enabled"
        opt_risk_name = config.opt_risk
        OptRisk = RISK_RGISTRY.get(opt_risk_name)
        if OptRisk is None:
            raise ValueError(f"Invalid opt_risk: '{opt_risk_name}'. Available: {list(RISK_RGISTRY.keys())}")
        if OptRisk.name == GuaranteedRisk.name:
            raise ValueError(f"opt_risk and guaranteed_risk must differ, both are '{OptRisk.name}'")

        opt_eval_result = evaluate_threshold_risks(
            opt_probe_scores,
            opt_baseline_scores,
            thresholds,
            risks=[GuaranteedRisk, OptRisk],
            dataset=opt_dataset,
            merge_strategy=config.cascade_merge_strategy,
        )

        empirical_risks_2d = opt_eval_result.get_empirical_risks_array()
        pareto_mask = is_pareto(empirical_risks_2d, maximize=False)
        n_pareto = int(pareto_mask.sum())
        logger.info(f"Found {n_pareto}/{len(thresholds)} Pareto-efficient thresholds")

        if n_pareto == 0:
            logger.warning("No Pareto-efficient points found! Falling back to all thresholds.")
            pareto_mask = np.ones(len(thresholds), dtype=bool)
            n_pareto = len(thresholds)

        thresholds = thresholds[pareto_mask]
        calib_empirical = calib_eval_result[guaranteed_risk_name][pareto_mask]
    else:
        opt_eval_result = None
        pareto_mask = None
        n_pareto = None
        calib_empirical = calib_eval_result[guaranteed_risk_name]

    # --- Deduplicate thresholds with identical calib empirical risk (optional) ---
    deduplicate_thresholds = getattr(config, "deduplicate_thresholds", False)
    if deduplicate_thresholds:
        _, unique_indices = np.unique(calib_empirical, return_index=True)
        unique_indices = np.sort(unique_indices)  # preserve order
        n_before_dedup = len(thresholds)
        thresholds = thresholds[unique_indices]
        calib_empirical = calib_empirical[unique_indices]
        logger.info(
            f"Deduplicated thresholds: {n_before_dedup} → {len(thresholds)} "
            f"(removed {n_before_dedup - len(thresholds)} duplicates)"
        )

    # =================================================================
    # SGT: Sequential Graphical Testing over (threshold × alpha) grid
    # =================================================================
    alpha_grid = np.linspace(config.alpha_start, config.alpha_end, getattr(config, "alpha_steps", 10))
    n_t, n_a = len(thresholds), len(alpha_grid)
    delta = 1 - config.guarantee_probability

    logger.info(f"SGT grid: {n_t} thresholds × {n_a} alphas = {n_t * n_a} hypotheses")
    if pareto_testing or deduplicate_thresholds:
        logger.info(f"  (reduced from {n_original_thresholds} original thresholds)")
    logger.info(f"Alpha range: [{alpha_grid.min():.3f}, {alpha_grid.max():.3f}], FWER delta={delta:.3f}")

    # Order thresholds from easiest to hardest empirical risk
    if guaranteed_risk_name == "budget":
        threshold_order = np.argsort(thresholds)
    else:
        threshold_order = np.argsort(-thresholds)
    # Order alphas from most permissive (largest) to strictest (smallest)
    alpha_order = np.argsort(-alpha_grid)

    ordered_thresholds = thresholds[threshold_order]
    ordered_alphas = alpha_grid[alpha_order]
    ordered_empirical = calib_empirical[threshold_order]

    # Row dimension: controls which parameter varies across rows of the
    # graph and which varies within each row (columns).
    #   "threshold" (default): rows=thresholds, cols=alphas
    #       → "for each budget level, what is the best reliability?"
    #   "alpha": rows=alphas, cols=thresholds
    #       → "for each reliability target, what is the best budget?"
    row_dim = getattr(config, "sgt_row_dimension", "threshold")

    n_samples = calib_eval_result.n_samples
    bound_fn = GuaranteedRisk.p_value_bound_fn

    if row_dim == "threshold":
        n_rows, n_cols = n_t, n_a
        hypotheses = [
            Hypothesis(
                p_value_fn=lambda r=risk, a=alpha: float(bound_fn(r, n_samples, a)),
                params={"threshold": float(ordered_thresholds[t_idx]), "alpha": float(alpha)},
            )
            for t_idx, risk in enumerate(ordered_empirical)
            for alpha in ordered_alphas
        ]
    elif row_dim == "alpha":
        n_rows, n_cols = n_a, n_t
        hypotheses = [
            Hypothesis(
                p_value_fn=lambda r=risk, a=alpha: float(bound_fn(r, n_samples, a)),
                params={"threshold": float(ordered_thresholds[t_idx]), "alpha": float(alpha)},
            )
            for alpha in ordered_alphas
            for t_idx, risk in enumerate(ordered_empirical)
        ]
    else:
        raise ValueError(f"Invalid sgt_row_dimension: '{row_dim}'. Use 'threshold' or 'alpha'.")

    logger.info(f"SGT row dimension: {row_dim} ({n_rows} rows × {n_cols} cols)")

    flat_p_values = compute_p_values(hypotheses)

    # Build graph
    graph_type = getattr(config, "sgt_graph_type", "lattice")
    if graph_type not in GRAPH_FACTORIES:
        raise ValueError(f"Unknown sgt_graph_type: '{graph_type}'. Available: {list(GRAPH_FACTORIES.keys())}")
    weights, transitions = GRAPH_FACTORIES[graph_type](n_rows, n_cols)

    # Run graphical testing
    logger.info(f"Running graphical testing (graph={graph_type})...")
    sgt_result: GraphicalTestResult = graphical_testing(flat_p_values, weights, transitions, delta=delta)

    if not sgt_result.rejected:
        logger.warning("SGT: No hypotheses rejected! Cannot run cascade.")
        return None

    # Map flat indices back to (threshold_idx, alpha_idx) pairs
    if row_dim == "threshold":
        rejected_pairs = [(idx // n_cols, idx % n_cols) for idx in sgt_result.rejected]
    else:
        rejected_pairs = [(idx % n_cols, idx // n_cols) for idx in sgt_result.rejected]
    logger.info(f"SGT rejected {len(rejected_pairs)}/{n_t * n_a} hypotheses")

    # Log achievable guarantees per alpha level
    for a_idx, alpha_val in enumerate(ordered_alphas):
        valid_t = [t for (t, a) in rejected_pairs if a == a_idx]
        if valid_t:
            t_range = f"{ordered_thresholds[min(valid_t)]:.4f}–{ordered_thresholds[max(valid_t)]:.4f}"
            logger.info(f"  alpha={alpha_val:.3f}: {len(valid_t)} valid thresholds ({t_range})")

    # --- Determine selection mode from graph structure ---
    # row_dim="threshold" → graph chains within each threshold row (across alphas) → best_alpha
    # row_dim="alpha"     → graph chains within each alpha row (across thresholds) → best_threshold
    selection_mode = "best_alpha" if row_dim == "threshold" else "best_threshold"
    logger.info(f"Selection mode: {selection_mode} (row_dim={row_dim})")

    # --- Test-set scores (expensive -- LLM inference) ---
    logger.info("Computing probe scores on test dataset...")
    test_probe_scores = probe.predict(test_dataset)
    test_labels = test_dataset.labels_numpy()

    logger.info("Computing baseline scores on test dataset...")
    test_baseline_scores = run_llm_baseline(
        baseline_model_name=config.baseline_model_name,
        dataset=test_dataset,
        baseline_batch_size=config.baseline_batch_size,
        local=not getattr(config, "use_modal", False),
        gpu=getattr(config, "modal_gpu", None),
    )

    # --- Run cascade for every unique valid threshold ---
    from collections import defaultdict

    valid_by_threshold: dict[int, list[int]] = defaultdict(list)
    for t_idx, a_idx in rejected_pairs:
        valid_by_threshold[t_idx].append(a_idx)

    num_batches = (len(test_probe_scores) + cascade_batch_size - 1) // cascade_batch_size
    merge_strategy = config.cascade_merge_strategy

    logger.info(f"Running cascade for {len(valid_by_threshold)} unique valid thresholds...")
    threshold_results: list[ThresholdCascadeResult] = []

    for t_idx in sorted(valid_by_threshold):
        thr = float(ordered_thresholds[t_idx])
        alpha_indices = sorted(valid_by_threshold[t_idx])
        # ordered_alphas is sorted descending, so largest index = tightest alpha
        best_alpha = float(ordered_alphas[max(alpha_indices)])

        cascade_result = offline_batch_cascade(
            probe_scores=test_probe_scores,
            baseline_scores=test_baseline_scores,
            batch_size=cascade_batch_size,
            selection_strategy="fixed_threshold",
            merge_strategy=merge_strategy,
            threshold=thr,
        )

        batches: list[BatchCascadeStatistics] = []
        for batch_idx in range(num_batches):
            s = batch_idx * cascade_batch_size
            e = min(s + cascade_batch_size, len(test_probe_scores))
            batches.append(
                compute_batch_statistics(
                    batch_index=batch_idx,
                    probe_scores=cascade_result.probe_scores[s:e],
                    baseline_scores=cascade_result.baseline_scores[s:e],
                    used_baseline=cascade_result.used_baseline[s:e],
                    final_scores=cascade_result.final_scores[s:e],
                    labels=test_labels[s:e],
                )
            )

        cascade_m = compute_overall_metrics(cascade_result.final_scores, test_labels)
        budget_costs = np.array([b.budget_cost for b in batches])

        threshold_results.append(
            ThresholdCascadeResult(
                threshold=thr,
                best_alpha=best_alpha,
                valid_alpha_indices=alpha_indices,
                cascade_accuracy=cascade_m["accuracy"],
                cascade_f1_score=cascade_m["f1_score"],
                cascade_roc_auc=cascade_m["roc_auc"],
                mean_budget_cost=float(budget_costs.mean()),
                batches=batches,
                cascade_final_scores=cascade_result.final_scores.copy(),
            )
        )
        logger.info(
            f"  threshold={thr:.4f}: alpha<={best_alpha:.4f}, "
            f"budget={budget_costs.mean():.4f}, acc={cascade_m['accuracy']:.4f}"
        )

    # --- Select headline threshold ---
    if selection_mode == "best_alpha":
        # Tightest alpha (largest alpha_idx since ordered_alphas is descending)
        # Break ties: lowest budget
        selected = min(threshold_results, key=lambda r: (-max(r.valid_alpha_indices), r.mean_budget_cost))
    else:  # best_threshold
        # Lowest budget (most aggressive threshold)
        # Break ties: tightest alpha
        selected = min(threshold_results, key=lambda r: (r.mean_budget_cost, -max(r.valid_alpha_indices)))

    reliable_threshold = selected.threshold
    achieved_alpha = selected.best_alpha
    budget_costs = np.array([b.budget_cost for b in selected.batches])

    logger.info(
        f"Selected ({selection_mode}): threshold={reliable_threshold:.4f}, "
        f"alpha={achieved_alpha:.4f}, budget={selected.mean_budget_cost:.4f}"
    )

    # --- Overall metrics ---
    probe_m = compute_overall_metrics(test_probe_scores, test_labels)
    baseline_m = compute_overall_metrics(test_baseline_scores, test_labels)

    logger.info("\n=== OVERALL PERFORMANCE METRICS ===")
    logger.info(
        f"Probe Only:    Acc={probe_m['accuracy']:.4f}, F1={probe_m['f1_score']:.4f}, ROC-AUC={probe_m['roc_auc']:.4f}"
    )
    logger.info(
        f"Baseline Only: Acc={baseline_m['accuracy']:.4f}, F1={baseline_m['f1_score']:.4f}, "
        f"ROC-AUC={baseline_m['roc_auc']:.4f}"
    )
    logger.info(
        f"Cascade:       Acc={selected.cascade_accuracy:.4f}, F1={selected.cascade_f1_score:.4f}, "
        f"ROC-AUC={selected.cascade_roc_auc:.4f}"
    )
    logger.info(f"Guaranteed:    risk({guaranteed_risk_name}) <= {achieved_alpha:.4f}")
    logger.info("===================================\n")

    return SGTCascadeResults(
        config=vars(config),
        seed=seed,
        debug_mode=debug_mode,
        test_size=len(test_dataset),
        cascade_batch_size=cascade_batch_size,
        num_batches=num_batches,
        guaranteed_risk_name=guaranteed_risk_name,
        guarantee_probability=config.guarantee_probability,
        sgt_graph_type=graph_type,
        n_thresholds=n_t,
        n_alphas=n_a,
        n_hypotheses=n_t * n_a,
        n_rejected=len(rejected_pairs),
        rejected_pairs=rejected_pairs,
        ordered_thresholds=ordered_thresholds,
        ordered_alphas=ordered_alphas,
        threshold_results=threshold_results,
        selection_mode=selection_mode,
        reliable_threshold=reliable_threshold,
        achieved_alpha=achieved_alpha,
        mean_budget_cost=float(budget_costs.mean()),
        std_budget_cost=float(budget_costs.std()),
        min_budget_cost=float(budget_costs.min()),
        max_budget_cost=float(budget_costs.max()),
        cascade_accuracy=selected.cascade_accuracy,
        cascade_f1_score=selected.cascade_f1_score,
        cascade_roc_auc=selected.cascade_roc_auc,
        probe_only_accuracy=probe_m["accuracy"],
        probe_only_f1_score=probe_m["f1_score"],
        probe_only_roc_auc=probe_m["roc_auc"],
        baseline_only_accuracy=baseline_m["accuracy"],
        baseline_only_f1_score=baseline_m["f1_score"],
        baseline_only_roc_auc=baseline_m["roc_auc"],
        batches=selected.batches,
        train_probe_scores=train_probe_scores,
        calib_probe_scores=calib_probe_scores,
        test_probe_scores=test_probe_scores,
        train_labels=train_labels,
        calib_labels=calib_labels,
        test_labels=test_labels,
        test_baseline_scores=test_baseline_scores,
        cascade_final_scores=selected.cascade_final_scores,
        calib_evaluation_risks=calib_eval_result,
        opt_evaluation_risks=opt_eval_result,
        pareto_mask=pareto_mask,
        n_original_thresholds=n_original_thresholds,
        n_pareto_thresholds=n_pareto,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    args = parse_args()
    config = load_config(args.config)

    clearml_logger = None
    if args.use_clearml:
        from clearml_logger import ClearMLLogger
        from clearml_serialization import ClearMLSerializer

        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="sgt_cascade_experiment",
            enabled=True,
        )

    results = run_sgt_cascade_experiment(config)

    if results is None:
        logger.warning("Experiment failed: no reliable (threshold, alpha) pair found.")
    else:
        from sgt_cascade_plotting import log_sgt_figures_to_clearml, make_sgt_figures

        logger.info("Generating plots...")
        figures = make_sgt_figures(results)

        if clearml_logger is not None:
            clearml_logger.connect_configuration(results.config)

            tags = [
                f"sgt-{results.sgt_graph_type}",
                f"guaranteed_risk-{results.guaranteed_risk_name}",
                f"achieved_alpha-{results.achieved_alpha:.3f}",
                f"rejected-{results.n_rejected}/{results.n_hypotheses}",
                f"probe-{results.config['reduction_strategy']}",
                f"merge-{results.config['cascade_merge_strategy']}",
                f"probe-degraded-{results.config.get('probe_degradation_enabled', False)}",
            ]
            calibration_method = results.config.get("calibration_method")
            tags.append(f"calibration-{calibration_method}" if calibration_method else "not-calibrated")
            tags.append(f"pareto_testing-{results.config.get('pareto_testing', False)}")
            if results.config.get("pareto_testing", False):
                tags.append(f"opt_risk-{results.config.get('opt_risk', 'budget')}")
                if results.n_pareto_thresholds is not None:
                    tags.append(f"pareto-{results.n_pareto_thresholds}/{results.n_original_thresholds}")
            if results.debug_mode:
                tags.append("debug")
            clearml_logger.add_tags(tags)

            serializer = ClearMLSerializer()
            scalars = serializer.to_clearml_scalars(results)
            # Ensure all scalar values are plain Python types (not numpy arrays)
            scalars = {
                k: float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v for k, v in scalars.items()
            }
            clearml_logger.log_scalars(scalars)
            clearml_logger.log_artifacts(serializer.to_clearml_artifacts(results))

            from cascade_utils import save_results_to_clearml

            save_results_to_clearml(clearml_logger, results)

            log_sgt_figures_to_clearml(clearml_logger, figures)
            logger.info("All plots generated and logged to ClearML.")
            clearml_logger.finalize()

    logger.info("Experiment complete!")
