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
from cascade_comparison import BatchCascadeStatistics, compute_batch_statistics
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

    # --- Optional probe calibration ---
    calibration_method = getattr(config, "calibration_method", None)
    auxiliary_dataset = None
    if calibration_method is not None:
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

    # =================================================================
    # SGT: Sequential Graphical Testing over (threshold × alpha) grid
    # =================================================================
    alpha_grid = np.linspace(config.alpha_start, config.alpha_end, getattr(config, "alpha_steps", 10))
    n_t, n_a = len(thresholds), len(alpha_grid)
    delta = 1 - config.guarantee_probability

    logger.info(f"SGT grid: {n_t} thresholds × {n_a} alphas = {n_t * n_a} hypotheses")
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
    ordered_empirical = calib_eval_result[guaranteed_risk_name][threshold_order]

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

    # Select best: tightest achievable alpha, breaking ties by most
    # aggressive threshold (highest index in ordered sequence).
    best_pair = min(rejected_pairs, key=lambda p: (p[1], -p[0]))
    reliable_threshold = float(ordered_thresholds[best_pair[0]])
    achieved_alpha = float(ordered_alphas[best_pair[1]])

    logger.info(f"SGT best: threshold={reliable_threshold:.4f}, alpha={achieved_alpha:.4f}")

    # --- Test-set cascade ---
    logger.info("Computing probe scores on test dataset...")
    test_probe_scores = probe.predict(test_dataset)
    test_labels = test_dataset.labels_numpy()

    logger.info("Computing baseline scores on test dataset...")
    test_baseline_scores = run_llm_baseline(
        baseline_model_name=config.baseline_model_name,
        dataset=test_dataset,
        baseline_batch_size=config.baseline_batch_size,
    )

    logger.info(f"Running adaptive cascade (threshold={reliable_threshold})...")
    cascade_result = offline_batch_cascade(
        probe_scores=test_probe_scores,
        baseline_scores=test_baseline_scores,
        batch_size=cascade_batch_size,
        selection_strategy="fixed_threshold",
        merge_strategy=config.cascade_merge_strategy,
        threshold=reliable_threshold,
    )

    # --- Per-batch statistics ---
    logger.info("Computing per-batch statistics...")
    batches: list[BatchCascadeStatistics] = []
    num_batches = (len(test_probe_scores) + cascade_batch_size - 1) // cascade_batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * cascade_batch_size
        end_idx = min(start_idx + cascade_batch_size, len(test_probe_scores))
        batch_labels = test_labels[start_idx:end_idx]

        batch_stats = compute_batch_statistics(
            batch_index=batch_idx,
            probe_scores=cascade_result.probe_scores[start_idx:end_idx],
            baseline_scores=cascade_result.baseline_scores[start_idx:end_idx],
            used_baseline=cascade_result.used_baseline[start_idx:end_idx],
            final_scores=cascade_result.final_scores[start_idx:end_idx],
            labels=batch_labels,
        )
        batches.append(batch_stats)

    budget_costs = np.array([b.budget_cost for b in batches])
    logger.info(f"Mean budget cost: {budget_costs.mean():.3f}")

    # --- Overall metrics ---
    logger.info("Computing overall performance metrics...")
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # Cascade
    cascade_preds = (cascade_result.final_scores >= 0.5).astype(int)
    cascade_accuracy = float(accuracy_score(test_labels, cascade_preds))
    cascade_f1 = float(f1_score(test_labels, cascade_preds))
    cascade_roc_auc = float(roc_auc_score(test_labels, cascade_result.final_scores))

    # Probe only
    probe_preds = (test_probe_scores >= 0.5).astype(int)
    probe_only_accuracy = float(accuracy_score(test_labels, probe_preds))
    probe_only_f1 = float(f1_score(test_labels, probe_preds))
    probe_only_roc_auc = float(roc_auc_score(test_labels, test_probe_scores))

    # Baseline only
    baseline_preds = (test_baseline_scores >= 0.5).astype(int)
    baseline_only_accuracy = float(accuracy_score(test_labels, baseline_preds))
    baseline_only_f1 = float(f1_score(test_labels, baseline_preds))
    baseline_only_roc_auc = float(roc_auc_score(test_labels, test_baseline_scores))

    logger.info("\n=== OVERALL PERFORMANCE METRICS ===")
    logger.info(
        f"Probe Only:    Acc={probe_only_accuracy:.4f}, F1={probe_only_f1:.4f}, ROC-AUC={probe_only_roc_auc:.4f}"
    )
    logger.info(
        f"Baseline Only: Acc={baseline_only_accuracy:.4f}, F1={baseline_only_f1:.4f}, ROC-AUC={baseline_only_roc_auc:.4f}"
    )
    logger.info(f"Cascade:       Acc={cascade_accuracy:.4f}, F1={cascade_f1:.4f}, ROC-AUC={cascade_roc_auc:.4f}")
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
        reliable_threshold=reliable_threshold,
        achieved_alpha=achieved_alpha,
        mean_budget_cost=float(budget_costs.mean()),
        std_budget_cost=float(budget_costs.std()),
        min_budget_cost=float(budget_costs.min()),
        max_budget_cost=float(budget_costs.max()),
        cascade_accuracy=cascade_accuracy,
        cascade_f1_score=cascade_f1,
        cascade_roc_auc=cascade_roc_auc,
        probe_only_accuracy=probe_only_accuracy,
        probe_only_f1_score=probe_only_f1,
        probe_only_roc_auc=probe_only_roc_auc,
        baseline_only_accuracy=baseline_only_accuracy,
        baseline_only_f1_score=baseline_only_f1,
        baseline_only_roc_auc=baseline_only_roc_auc,
        batches=batches,
        train_probe_scores=train_probe_scores,
        calib_probe_scores=calib_probe_scores,
        test_probe_scores=test_probe_scores,
        train_labels=train_labels,
        calib_labels=calib_labels,
        test_labels=test_labels,
        test_baseline_scores=test_baseline_scores,
        cascade_final_scores=cascade_result.final_scores,
        calib_evaluation_risks=calib_eval_result,
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

            log_sgt_figures_to_clearml(clearml_logger, figures)
            logger.info("All plots generated and logged to ClearML.")
            clearml_logger.finalize()

    logger.info("Experiment complete!")
