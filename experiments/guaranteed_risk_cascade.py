"""Experiment studying an adaptive cascade with a configurable guaranteed risk.

Unlike ``cascade_comparison`` (which compares adaptive vs fixed-budget cascading
and always guarantees *budget*), this experiment:

* Runs **only** the adaptive cascade.
* Lets the user choose **which risk to guarantee** (e.g. accuracy error) and
  at what level (``guarantee_alpha``).
* Optionally optimises a second risk via Pareto testing (``opt_risk``).

The design is intended to be extended to multiple guaranteed / optimised risks.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cascade_utils import BatchCascadeStatistics, compute_batch_statistics, compute_overall_metrics
from clearml_serialization import (
    artifact_field,
    derived_field,
    scalar_field,
)
from config import load_config
from dotenv import load_dotenv
from matplotlib.figure import Figure

from reliable_monitoring.cascade import offline_batch_cascade, run_llm_baseline
from reliable_monitoring.dataset import ActivationConfig, load_dataset, sample_from_dataset, split_dataset
from reliable_monitoring.learn_then_test import fixed_sequence_testing, is_pareto
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


@dataclass
class GuaranteedRiskCascadeResults:
    """Results for the guaranteed-risk adaptive cascade experiment."""

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
    guarantee_alpha: float = scalar_field()
    guarantee_probability: float = scalar_field()

    # Reliable threshold found via hypothesis testing
    reliable_threshold: float = scalar_field()

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

    # LLM-baseline-only (all examples use the baseline model)
    baseline_only_accuracy: float = scalar_field()
    baseline_only_f1_score: float = scalar_field()
    baseline_only_roc_auc: float = scalar_field()

    # Per-batch detailed results
    batches: list[BatchCascadeStatistics] = artifact_field()

    # Probe score distributions
    train_probe_scores: np.ndarray = artifact_field()
    calib_probe_scores: np.ndarray = artifact_field()
    test_probe_scores: np.ndarray = artifact_field()
    train_labels: np.ndarray = artifact_field()
    calib_labels: np.ndarray = artifact_field()
    test_labels: np.ndarray = artifact_field()

    # Baseline and cascade scores
    test_baseline_scores: np.ndarray = artifact_field()
    cascade_final_scores: np.ndarray = artifact_field()

    # Evaluation results (for Pareto / diagnostics plots)
    calib_evaluation_risks: ThresholdEvaluationResult = artifact_field()
    opt_evaluation_risks: ThresholdEvaluationResult | None = artifact_field()
    pareto_mask: np.ndarray | None = artifact_field()

    # Derived
    budget_costs: np.ndarray = derived_field(derive_fn=lambda r: np.array([b.budget_cost for b in r.batches]))


def parse_args():
    default_config_path = Path(__file__).parent / "configs" / "guaranteed_risk_cascade.yaml"
    parser = argparse.ArgumentParser(description="Guaranteed-Risk Cascade Experiment")
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


def run_guaranteed_risk_cascade_experiment(config) -> GuaranteedRiskCascadeResults | None:
    """Run the guaranteed-risk cascade experiment.

    Returns ``None`` when no reliable threshold can be found.
    """
    seed = config.seed
    np.random.seed(seed)

    cascade_batch_size = config.cascade_batch_size

    # --- Guaranteed / optimisation risk setup ---
    guaranteed_risk_name = getattr(config, "guaranteed_risk", "budget")
    GuaranteedRisk = RISK_RGISTRY.get(guaranteed_risk_name)
    if GuaranteedRisk is None:
        raise ValueError(f"Invalid guaranteed_risk: '{guaranteed_risk_name}'. Available: {list(RISK_RGISTRY.keys())}")
    guarantee_alpha = config.guarantee_alpha

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

    calibration_method = getattr(config, "calibration_method", None)
    needs_auxiliary_data = calibration_method is not None

    # --- Data splitting ---
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

    # --- Candidate thresholds (linear in score space) ---
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
        logger.info(f"Threshold: {thr:.4f}, Empirical {GuaranteedRisk.description} Risk: {risk:.4f}")

    all_p_values = calib_eval_result.compute_p_values(alpha=guarantee_alpha)[guaranteed_risk_name]

    # --- Pareto testing (optional) ---
    if pareto_testing:
        logger.info("Performing Pareto testing with multiple risks...")
        OptRisk = RISK_RGISTRY.get(getattr(config, "opt_risk", "accuracy_error"))
        if OptRisk is None:
            raise ValueError(f"Invalid opt_risk: '{config.opt_risk}'. Available: {list(RISK_RGISTRY.keys())}")
        if OptRisk.name == GuaranteedRisk.name:
            raise ValueError(f"opt_risk and guaranteed_risk must be different, both are '{OptRisk.name}'")

        opt_eval_result = evaluate_threshold_risks(
            opt_probe_scores,
            opt_baseline_scores,
            thresholds,
            risks=[GuaranteedRisk, OptRisk],
            dataset=opt_dataset,
            merge_strategy=config.cascade_merge_strategy,
        )

        empirical_risks_2d = opt_eval_result.get_empirical_risks_array()
        logger.info(f"Empirical risks shape: {empirical_risks_2d.shape}")

        pareto_mask = is_pareto(empirical_risks_2d, maximize=False)
        n_pareto = pareto_mask.sum()
        logger.info(f"Found {n_pareto}/{len(thresholds)} Pareto-efficient thresholds")

        if n_pareto == 0:
            logger.warning("No Pareto-efficient points found! Falling back to all thresholds.")
            pareto_mask = np.ones(len(thresholds), dtype=bool)

        p_values = all_p_values[pareto_mask]
        pareto_thresholds = thresholds[pareto_mask]
        logger.info(f"P-values array length: {len(p_values)} (reduced from {len(thresholds)})")
    else:
        p_values = all_p_values
        pareto_thresholds = thresholds
        opt_eval_result = None
        pareto_mask = None

    # --- Order thresholds from lowest to highest expected guaranteed risk ---
    # fixed_sequence_testing rejects sequentially and stops at the first
    # failure, so we must traverse from the easiest (lowest risk) to the
    # hardest (highest risk) hypothesis.
    #
    # When a Pareto opt set is available we use its empirical risks (which
    # are independent of the calib set used for hypothesis testing).
    # Otherwise we rely on domain knowledge: higher thresholds delegate
    # more to the baseline, so budget risk increases with threshold while
    # performance-error risks (accuracy, ROC-AUC) decrease.
    opt_risks_sorted = None
    if pareto_testing and opt_eval_result is not None:
        guaranteed_risks_for_ordering = opt_eval_result[guaranteed_risk_name][pareto_mask]
        sort_order = np.argsort(guaranteed_risks_for_ordering)
        opt_risks_sorted = opt_eval_result[OptRisk.name][pareto_mask][sort_order]
    else:
        if guaranteed_risk_name == "budget":
            sort_order = np.argsort(pareto_thresholds)  # ascending threshold
        else:
            sort_order = np.argsort(-pareto_thresholds)  # descending threshold

    p_values = p_values[sort_order]
    pareto_thresholds = pareto_thresholds[sort_order]

    # --- Hypothesis testing ---
    delta = 1 - config.guarantee_probability
    reliable_hyperparams = fixed_sequence_testing(p_values=p_values, delta=delta)

    if len(reliable_hyperparams) == 0:
        logger.warning("No reliable thresholds found! Cannot run cascade.")
        return None

    # Among reliable thresholds, pick the one that minimises the opt risk.
    # When no opt risk is available, fall back to the most aggressive
    # threshold for the guaranteed risk (last rejected hypothesis).
    if opt_risks_sorted is not None:
        reliable_opt_risks = opt_risks_sorted[reliable_hyperparams]
        best_among_reliable = int(np.argmin(reliable_opt_risks))
        best_idx = reliable_hyperparams[best_among_reliable]
        logger.info(
            f"Selected threshold minimising {OptRisk.name} "
            f"(opt risk = {opt_risks_sorted[best_idx]:.4f}) "
            f"among {len(reliable_hyperparams)} reliable thresholds"
        )
    else:
        best_idx = reliable_hyperparams[-1]

    reliable_threshold = float(pareto_thresholds[best_idx])
    logger.info(f"Found reliable threshold: {reliable_threshold}")

    # --- Test-set cascade ---
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
    cascade_m = compute_overall_metrics(cascade_result.final_scores, test_labels)
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
        f"Cascade:       Acc={cascade_m['accuracy']:.4f}, F1={cascade_m['f1_score']:.4f}, "
        f"ROC-AUC={cascade_m['roc_auc']:.4f}"
    )
    logger.info("===================================\n")

    return GuaranteedRiskCascadeResults(
        config=vars(config),
        seed=seed,
        debug_mode=debug_mode,
        test_size=len(test_dataset),
        cascade_batch_size=cascade_batch_size,
        num_batches=num_batches,
        guaranteed_risk_name=guaranteed_risk_name,
        guarantee_alpha=guarantee_alpha,
        guarantee_probability=config.guarantee_probability,
        reliable_threshold=reliable_threshold,
        mean_budget_cost=float(budget_costs.mean()),
        std_budget_cost=float(budget_costs.std()),
        min_budget_cost=float(budget_costs.min()),
        max_budget_cost=float(budget_costs.max()),
        cascade_accuracy=cascade_m["accuracy"],
        cascade_f1_score=cascade_m["f1_score"],
        cascade_roc_auc=cascade_m["roc_auc"],
        probe_only_accuracy=probe_m["accuracy"],
        probe_only_f1_score=probe_m["f1_score"],
        probe_only_roc_auc=probe_m["roc_auc"],
        baseline_only_accuracy=baseline_m["accuracy"],
        baseline_only_f1_score=baseline_m["f1_score"],
        baseline_only_roc_auc=baseline_m["roc_auc"],
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
        opt_evaluation_risks=opt_eval_result,
        pareto_mask=pareto_mask,
    )


def make_figures(results: GuaranteedRiskCascadeResults) -> dict[str, Figure | None | dict[str, Figure]]:
    """Generate all figures for the experiment."""
    from guaranteed_risk_cascade_plotting import (
        plot_batch_distributions,
        plot_batch_metric_boxplots,
        plot_batch_uncertainty_vs_metrics,
        plot_cascade_vs_probe,
        plot_overall_performance,
    )
    from plot_utils import (
        plot_pareto_frontier,
        plot_reliability_diagrams_by_split,
        plot_roc_curves_by_score_set,
        plot_score_histograms_by_split,
    )

    splits = {
        "train": (results.train_probe_scores, results.train_labels),
        "calibration": (results.calib_probe_scores, results.calib_labels),
        "test": (results.test_probe_scores, results.test_labels),
    }

    figures: dict[str, Figure | None | dict[str, Figure]] = {}

    figures["overall"] = plot_overall_performance(results)
    figures["distributions"] = plot_batch_distributions(results)
    figures["probe_uncertainty"] = plot_batch_uncertainty_vs_metrics(results)
    figures["boxes"] = plot_batch_metric_boxplots(results)
    figures["cascade_vs_probe"] = plot_cascade_vs_probe(results)

    # Shared utilities
    figures["probe_score_hists"] = plot_score_histograms_by_split(splits)
    figures["reliability_diagrams"] = plot_reliability_diagrams_by_split(splits)
    figures["roc_curves"] = plot_roc_curves_by_score_set(
        results.test_labels,
        {
            "probe": results.test_probe_scores,
            "baseline": results.test_baseline_scores,
            "cascade": results.cascade_final_scores,
        },
    )

    # Pareto frontier (only when Pareto testing was used)
    if results.opt_evaluation_risks is not None and results.pareto_mask is not None:
        guaranteed_risk_name = results.guaranteed_risk_name
        opt_risk_name = results.config.get("opt_risk", "accuracy_error")
        figures["pareto"] = plot_pareto_frontier(
            results.opt_evaluation_risks,
            results.pareto_mask,
            guaranteed_risk_name,
            opt_risk_name,
        )
    else:
        figures["pareto"] = None

    return figures


def log_to_clearml(
    clearml_logger: "ClearMLLogger",
    results: GuaranteedRiskCascadeResults,
    figures: dict[str, Figure | None | dict[str, Figure]],
):
    """Log the experiment results to ClearML."""
    clearml_logger.connect_configuration(results.config)

    # Tags
    tags = []
    if results.debug_mode:
        tags.append("debug")
    tags.append(f"guaranteed_risk-{results.guaranteed_risk_name}")
    tags.append(f"guarantee_alpha-{results.guarantee_alpha:.2f}")
    tags.append(f"probe-{results.config['reduction_strategy']}")
    tags.append(f"merge-{results.config['cascade_merge_strategy']}")
    tags.append(f"pareto_testing-{results.config.get('pareto_testing', False)}")
    calibration_method = results.config.get("calibration_method")
    tags.append(f"calibration-{calibration_method}" if calibration_method else "not-calibrated")
    if results.config.get("pareto_testing", False):
        tags.append(f"opt_risk-{results.config.get('opt_risk', 'accuracy_error')}")
    tags.append(f"probe-degraded-{results.config.get('probe_degradation_enabled', False)}")
    clearml_logger.add_tags(tags)

    # Scalars & artifacts
    serializer = ClearMLSerializer()
    clearml_logger.log_scalars(serializer.to_clearml_scalars(results))
    clearml_logger.log_artifacts(serializer.to_clearml_artifacts(results))

    from cascade_utils import save_results_to_clearml

    save_results_to_clearml(clearml_logger, results)

    # Figures
    logger.info("Generating comparison plots...")

    clearml_logger.log_figure(title="Performance", series="Overall Performance", figure=figures["overall"])

    if figures["distributions"] is not None:
        for metric_name, fig in figures["distributions"].items():  # type: ignore
            clearml_logger.log_figure(title="Distributions", series=metric_name, figure=fig)

    clearml_logger.log_figure(
        title="Analysis", series="Probe Uncertainty vs Performance", figure=figures["probe_uncertainty"]
    )
    clearml_logger.log_figure(title="Analysis", series="Metric Ranges", figure=figures["boxes"])
    clearml_logger.log_figure(
        title="Analysis", series="Cascade vs Probe Performance", figure=figures["cascade_vs_probe"]
    )

    if figures.get("probe_score_hists") is not None:
        for dataset_name, fig in figures["probe_score_hists"].items():  # type: ignore
            clearml_logger.log_figure(
                title="Probe Score Histograms", series=f"Probe Scores ({dataset_name})", figure=fig
            )

    if figures.get("reliability_diagrams") is not None:
        for dataset_name, fig in figures["reliability_diagrams"].items():  # type: ignore
            clearml_logger.log_figure(
                title="Reliability Diagrams", series=f"Probe Reliability ({dataset_name})", figure=fig
            )

    if figures.get("roc_curves") is not None:
        for score_type, fig in figures["roc_curves"].items():  # type: ignore
            clearml_logger.log_figure(title="ROC Curves", series=f"ROC ({score_type.title()})", figure=fig)

    if figures["pareto"] is not None:
        clearml_logger.log_figure(title="Pareto Frontier", series="Pareto Frontier", figure=figures["pareto"])

    import matplotlib.pyplot as plt

    plt.close("all")

    logger.info("All plots generated and logged to ClearML")
    clearml_logger.finalize()


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
            task_name="guaranteed_risk_cascade_experiment",
            enabled=True,
        )

    results = run_guaranteed_risk_cascade_experiment(config)

    if results is None:
        logger.warning("Experiment failed: no reliable threshold found.")
    else:
        figures = make_figures(results)

        if clearml_logger is not None:
            log_to_clearml(clearml_logger, results=results, figures=figures)

    logger.info("Experiment complete!")
