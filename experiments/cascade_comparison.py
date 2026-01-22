"""
Experiment comparing adaptive reliable threshold cascading with fixed budget cascading.

This experiment demonstrates that the reliable threshold approach (from guaranteed_budget)
behaves adaptively in batch settings: on easy instances, fewer examples call the baseline
(lower budget), while on difficult instances, more examples call the baseline (higher budget).

In contrast, fixed budget approaches always use the same budget, which wastes resources
on easy instances and may fail to surge enough on difficult instances.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
from reliable_monitoring.probes import SequenceProbe
from reliable_monitoring.risks import AccuracyRisk, BudgetCostRisk, ThresholdEvaluationResult, evaluate_threshold_risks

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_SAMPLE_SIZE = 64


@dataclass
class BatchCascadeStatistics:
    """Statistics for a single batch cascade run."""

    # Batch identifier
    batch_index: int = scalar_field()

    # Cascade statistics
    budget_cost: float = scalar_field()  # Fraction of examples using baseline
    num_examples: int = scalar_field()

    # Probe uncertainty distribution in batch
    probe_uncertainty_mean: float = scalar_field()  # Mean(min(p, 1-p))
    probe_uncertainty_std: float = scalar_field()
    probe_uncertainty_min: float = scalar_field()
    probe_uncertainty_max: float = scalar_field()

    # Baseline score distribution in batch (for examples that used baseline)
    baseline_score_mean: float = scalar_field()
    baseline_score_std: float = scalar_field()

    # Performance metrics (cascade)
    accuracy: float = scalar_field()
    f1_score: float = scalar_field()
    roc_auc: float = scalar_field()

    # Performance metrics (probe only - baseline for comparison)
    probe_accuracy: float = scalar_field()
    probe_f1_score: float = scalar_field()
    probe_roc_auc: float = scalar_field()

    # Raw data for detailed analysis
    probe_scores: np.ndarray = artifact_field()
    baseline_scores: np.ndarray = artifact_field()  # Contains NaN where not used
    used_baseline: np.ndarray = artifact_field()  # Boolean mask
    final_scores: np.ndarray = artifact_field()


@dataclass
class CascadeComparisonResults:
    """Results comparing adaptive reliable threshold with fixed budget cascading.

    Pure data structure comparing two cascade strategies on the same data:
    1. Adaptive: Uses reliable threshold (from guaranteed budget approach)
    2. Fixed: Uses fixed budget rate (middle percentiles)

    Shows that adaptive approach varies budget per batch based on difficulty,
    while fixed approach always uses same budget.
    """

    # Experiment metadata
    config: dict = artifact_field()
    seed: int = scalar_field()
    debug_mode: bool = scalar_field()

    # Dataset information
    test_size: int = scalar_field()
    cascade_batch_size: int = scalar_field()
    num_batches: int = scalar_field()

    # Reliable threshold information
    reliable_threshold: float = scalar_field()
    guarantee_probability: float = scalar_field()
    budget: float = scalar_field()

    # Overall statistics - adaptive threshold approach
    adaptive_mean_budget_cost: float = scalar_field()
    adaptive_std_budget_cost: float = scalar_field()
    adaptive_min_budget_cost: float = scalar_field()
    adaptive_max_budget_cost: float = scalar_field()

    # Overall statistics - fixed budget approach
    fixed_mean_budget_cost: float = scalar_field()
    fixed_std_budget_cost: float = scalar_field()
    fixed_min_budget_cost: float = scalar_field()
    fixed_max_budget_cost: float = scalar_field()

    # Best batch statistics - adaptive (lowest budget cost)
    adaptive_best_batch_index: int = scalar_field()
    adaptive_best_batch_budget_cost: float = scalar_field()
    adaptive_best_batch_accuracy: float = scalar_field()
    adaptive_best_batch_f1_score: float = scalar_field()
    adaptive_best_batch_roc_auc: float = scalar_field()
    adaptive_best_batch_probe_uncertainty: float = scalar_field()

    # Worst batch statistics - adaptive (highest budget cost)
    adaptive_worst_batch_index: int = scalar_field()
    adaptive_worst_batch_budget_cost: float = scalar_field()
    adaptive_worst_batch_accuracy: float = scalar_field()
    adaptive_worst_batch_f1_score: float = scalar_field()
    adaptive_worst_batch_roc_auc: float = scalar_field()
    adaptive_worst_batch_probe_uncertainty: float = scalar_field()

    # Best batch statistics - fixed (lowest budget cost, usually all same)
    fixed_best_batch_index: int = scalar_field()
    fixed_best_batch_budget_cost: float = scalar_field()
    fixed_best_batch_accuracy: float = scalar_field()
    fixed_best_batch_f1_score: float = scalar_field()
    fixed_best_batch_roc_auc: float = scalar_field()
    fixed_best_batch_probe_uncertainty: float = scalar_field()

    # Worst batch statistics - fixed (highest budget cost, usually all same)
    fixed_worst_batch_index: int = scalar_field()
    fixed_worst_batch_budget_cost: float = scalar_field()
    fixed_worst_batch_accuracy: float = scalar_field()
    fixed_worst_batch_f1_score: float = scalar_field()
    fixed_worst_batch_roc_auc: float = scalar_field()
    fixed_worst_batch_probe_uncertainty: float = scalar_field()

    # Per-batch detailed results (raw data for analysis)
    adaptive_batches: list[BatchCascadeStatistics] = artifact_field()
    fixed_batches: list[BatchCascadeStatistics] = artifact_field()

    # Paired t-test results for accuracy
    accuracy_t_stat: float = scalar_field()
    accuracy_p_value: float = scalar_field()
    accuracy_mean_diff: float = scalar_field()
    accuracy_std_diff: float = scalar_field()

    # Paired t-test results for F1 score
    f1_score_t_stat: float = scalar_field()
    f1_score_p_value: float = scalar_field()
    f1_score_mean_diff: float = scalar_field()
    f1_score_std_diff: float = scalar_field()

    # Paired t-test results for ROC-AUC
    roc_auc_t_stat: float = scalar_field()
    roc_auc_p_value: float = scalar_field()
    roc_auc_mean_diff: float = scalar_field()
    roc_auc_std_diff: float = scalar_field()

    # Overall performance metrics - probe only (baseline for comparison)
    probe_only_accuracy: float = scalar_field()
    probe_only_f1_score: float = scalar_field()
    probe_only_roc_auc: float = scalar_field()

    # Overall performance metrics - baseline only (all examples use baseline)
    baseline_only_accuracy: float = scalar_field()
    baseline_only_f1_score: float = scalar_field()
    baseline_only_roc_auc: float = scalar_field()

    # Overall performance metrics - adaptive cascade (on full dataset)
    adaptive_overall_accuracy: float = scalar_field()
    adaptive_overall_f1_score: float = scalar_field()
    adaptive_overall_roc_auc: float = scalar_field()

    # Overall performance metrics - fixed cascade (on full dataset)
    fixed_overall_accuracy: float = scalar_field()
    fixed_overall_f1_score: float = scalar_field()
    fixed_overall_roc_auc: float = scalar_field()

    calib_evaluation_risks: ThresholdEvaluationResult = artifact_field()
    opt_evaluation_risks: ThresholdEvaluationResult | None = artifact_field()

    pareto_mask: np.ndarray | None = artifact_field()  # Boolean mask

    # Derived statistics
    adaptive_budget_costs: np.ndarray = derived_field(
        derive_fn=lambda r: np.array([b.budget_cost for b in r.adaptive_batches])
    )
    fixed_budget_costs: np.ndarray = derived_field(
        derive_fn=lambda r: np.array([b.budget_cost for b in r.fixed_batches])
    )


def compute_paired_t_tests(metrics: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Compute paired t-tests for multiple metrics.

    Since both methods evaluate the same batches, we can use paired t-tests
    to determine if performance differences are statistically significant.

    Args:
        metrics: Dictionary mapping metric name to (adaptive_data, fixed_data) arrays

    Returns:
        Dictionary mapping metric names to dicts with 't_stat', 'p_value', 'mean_diff', 'std_diff'
    """
    from scipy import stats

    results_dict = {}
    for metric_name, (adaptive_data, fixed_data) in metrics.items():
        diff = adaptive_data - fixed_data
        t_stat, p_value = stats.ttest_rel(adaptive_data, fixed_data)
        results_dict[metric_name] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std()),
        }
        logger.info(
            f"Paired t-test for {metric_name}: t={t_stat:.4f}, p={p_value:.4f}, "
            f"mean_diff={diff.mean():.4f}±{diff.std():.4f}"
        )

    return results_dict


def parse_args():
    """Parse command line arguments to get config file path."""
    default_config_path = Path(__file__).parent / "configs" / "cascade_comparison.yaml"

    parser = argparse.ArgumentParser(description="Cascade Comparison Experiment")
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


def compute_batch_statistics(
    batch_index: int,
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    used_baseline: np.ndarray,
    final_scores: np.ndarray,
    labels: np.ndarray,
) -> BatchCascadeStatistics:
    """Compute statistics for a batch cascade run."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # Probe uncertainty (closeness to decision boundary)
    probe_uncertainty = np.minimum(probe_scores, 1 - probe_scores)

    # Baseline scores for examples that used baseline
    baseline_subset = baseline_scores[used_baseline]
    baseline_score_mean = float(np.nanmean(baseline_subset)) if len(baseline_subset) > 0 else 0.0
    baseline_score_std = float(np.nanstd(baseline_subset)) if len(baseline_subset) > 0 else 0.0

    # Cascade performance metrics
    predictions = (final_scores >= 0.5).astype(int)
    accuracy = float(accuracy_score(labels, predictions))
    f1 = float(f1_score(labels, predictions))
    roc_auc = float(roc_auc_score(labels, final_scores))

    # Probe-only performance metrics
    probe_predictions = (probe_scores >= 0.5).astype(int)
    probe_accuracy = float(accuracy_score(labels, probe_predictions))
    probe_f1 = float(f1_score(labels, probe_predictions))
    probe_roc_auc = float(roc_auc_score(labels, probe_scores))

    return BatchCascadeStatistics(
        batch_index=batch_index,
        budget_cost=float(used_baseline.mean()),
        num_examples=len(probe_scores),
        probe_uncertainty_mean=float(probe_uncertainty.mean()),
        probe_uncertainty_std=float(probe_uncertainty.std()),
        probe_uncertainty_min=float(probe_uncertainty.min()),
        probe_uncertainty_max=float(probe_uncertainty.max()),
        baseline_score_mean=baseline_score_mean,
        baseline_score_std=baseline_score_std,
        accuracy=accuracy,
        f1_score=f1,
        roc_auc=roc_auc,
        probe_accuracy=probe_accuracy,
        probe_f1_score=probe_f1,
        probe_roc_auc=probe_roc_auc,
        probe_scores=probe_scores.copy(),
        baseline_scores=baseline_scores.copy(),
        used_baseline=used_baseline.copy(),
        final_scores=final_scores.copy(),
    )


def run_cascade_comparison_experiment(config) -> CascadeComparisonResults | None:
    """Run the cascade comparison experiment.

    Args:
        config: Configuration object loaded from YAML file.
    """
    seed = config.seed
    np.random.seed(seed)

    cascade_batch_size = config.cascade_batch_size
    budget = config.budget

    activation_config = ActivationConfig(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
    )
    # Load data
    logger.info("Loading datasets...")
    train_dataset = load_dataset(
        Path(config.train_dataset_path),
        activation_config=activation_config,
    )
    calib_dataset = load_dataset(
        Path(config.calib_dataset_path),
        activation_config=activation_config,
    )
    test_dataset = load_dataset(
        Path(config.test_dataset_path),
        activation_config=activation_config,
    )

    debug_mode = getattr(config, "debug", False)
    if debug_mode:
        logger.warning("Running in debug mode with smaller datasets.")
        train_dataset = sample_from_dataset(train_dataset, DEBUG_SAMPLE_SIZE, seed=seed)
        calib_dataset = sample_from_dataset(calib_dataset, DEBUG_SAMPLE_SIZE, seed=seed)
        test_dataset = sample_from_dataset(test_dataset, DEBUG_SAMPLE_SIZE, seed=seed)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Calibration dataset size: {len(calib_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    if config.pareto_testing:
        logger.info("Using Pareto testing")
        # split the calibration dataset into two halves: one for optimisation the other for calibration
        calib_dataset, opt_dataset = split_dataset(
            calib_dataset,
            proportions=[0.5, 0.5],
            shuffle=True,
            seed=seed,
        )

    # Load or train probe
    logger.info("Fitting probe...")
    probe = SequenceProbe(reduction_strategy=config.reduction_strategy)
    probe.fit(train_dataset)

    # Determine reliable threshold
    logger.info("Learning reliable threshold from calibration data...")

    # Compute calibration scores
    logger.info("Computing probe scores on calibration dataset...")
    calib_probe_scores = probe.predict(calib_dataset)

    logger.info("Computing baseline scores on calibration dataset...")
    calib_baseline_scores = run_llm_baseline(
        baseline_model_name=config.baseline_model_name,
        dataset=calib_dataset,
        baseline_batch_size=config.baseline_batch_size,
    )

    if config.pareto_testing:
        logger.info("Computing probe scores on optimisation dataset...")
        opt_probe_scores = probe.predict(opt_dataset)

        logger.info("Computing baseline scores on optimisation dataset...")
        opt_baseline_scores = run_llm_baseline(
            baseline_model_name=config.baseline_model_name,
            dataset=opt_dataset,
            baseline_batch_size=config.baseline_batch_size,
        )

    # Evaluate empirical risks
    thresholds = np.linspace(
        getattr(config, "threshold_start", 0.5),
        getattr(config, "threshold_end", 1.0),
        getattr(config, "threshold_steps", 10),
    )

    calib_eval_result = evaluate_threshold_risks(
        calib_probe_scores,
        calib_baseline_scores,
        thresholds,
        risks=BudgetCostRisk,
        merge_strategy=config.cascade_merge_strategy,
    )

    # Compute p-values using the risk's appropriate bound (binomial for budget cost)
    all_p_values = calib_eval_result.compute_p_values(alpha=config.budget)["Budget Cost"]

    if config.pareto_testing:
        logger.info("Performing Pareto testing with multiple risks...")

        # Step 1: Evaluate both risks on optimization set only
        opt_eval_result = evaluate_threshold_risks(
            opt_probe_scores,
            opt_baseline_scores,
            thresholds,
            risks=[BudgetCostRisk, AccuracyRisk],
            dataset=opt_dataset,  # Required for AccuracyRisk
            merge_strategy=config.cascade_merge_strategy,
        )

        # Step 2: Extract empirical risks into 2D array for Pareto computation
        empirical_risks_2d = opt_eval_result.get_empirical_risks_array()
        logger.info(f"Empirical risks shape: {empirical_risks_2d.shape}")

        # Step 3: Find Pareto-efficient thresholds (minimize both risks)
        pareto_mask = is_pareto(empirical_risks_2d, maximize=False)
        n_pareto = pareto_mask.sum()
        logger.info(f"Found {n_pareto}/{len(thresholds)} Pareto-efficient thresholds")

        if n_pareto == 0:
            logger.warning("No Pareto-efficient points found! Falling back to all thresholds.")
            pareto_mask = np.ones(len(thresholds), dtype=bool)

        # Step 4: Extract p-values and thresholds for only Pareto-efficient points
        # Use the already-computed p-values from calib_eval_result (BudgetCostRisk only)
        p_values = all_p_values[pareto_mask]
        pareto_thresholds = thresholds[pareto_mask]

        logger.info(f"P-values array length: {len(p_values)} (reduced from {len(thresholds)})")

    else:
        p_values = all_p_values
        pareto_thresholds = thresholds  # No filtering
        opt_eval_result = None
        pareto_mask = None

    # hypothesis testing to find reliable hyperparameters
    delta = 1 - config.guarantee_probability

    # Apply fixed-sequence testing
    reliable_hyperparams = fixed_sequence_testing(
        p_values=p_values,
        delta=delta,
    )

    # Check if a reliable threshold was found
    if len(reliable_hyperparams) == 0:
        logger.warning("No reliable thresholds found! Cannot run cascade comparison.")
        return None

    # Use most aggressive reliable threshold
    # For Pareto: reliable_hyperparams indexes into pareto_thresholds
    # For non-Pareto: reliable_hyperparams indexes into thresholds (same thing)
    best_idx = reliable_hyperparams[-1]  # TODO: generalise and allow choice of selection strategy
    reliable_threshold = float(pareto_thresholds[best_idx])

    logger.info(f"Found reliable threshold: {reliable_threshold}")

    # Compute test scores
    logger.info("Computing probe scores on test dataset...")
    test_probe_scores = probe.predict(test_dataset)

    logger.info("Computing baseline scores on test dataset...")
    test_baseline_scores = run_llm_baseline(
        baseline_model_name=config.baseline_model_name,
        dataset=test_dataset,
        baseline_batch_size=config.baseline_batch_size,
    )

    # Extract test labels for performance metrics
    test_labels = test_dataset.labels_numpy()
    logger.info(f"Extracted test labels: {len(test_labels)} labels")

    # Run batch cascades
    logger.info(f"Running batch cascades (batch_size={cascade_batch_size})...")

    # Adaptive threshold approach
    logger.info(f"Running adaptive threshold cascade (threshold={reliable_threshold})...")
    adaptive_result = offline_batch_cascade(
        probe_scores=test_probe_scores,
        baseline_scores=test_baseline_scores,
        batch_size=cascade_batch_size,
        selection_strategy="fixed_threshold",
        merge_strategy=config.cascade_merge_strategy,
        threshold=reliable_threshold,
    )

    # Fixed budget approach
    logger.info(f"Running fixed budget cascade (rate={budget})...")
    fixed_result = offline_batch_cascade(
        probe_scores=test_probe_scores,
        baseline_scores=test_baseline_scores,
        batch_size=cascade_batch_size,
        selection_strategy="fixed_budget_rate",
        merge_strategy=config.cascade_merge_strategy,
        rate=budget,
    )

    # Compute per-batch statistics
    logger.info("Computing per-batch statistics...")
    adaptive_batches = []
    fixed_batches = []

    num_batches = (len(test_probe_scores) + cascade_batch_size - 1) // cascade_batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * cascade_batch_size
        end_idx = min(start_idx + cascade_batch_size, len(test_probe_scores))
        batch_labels = test_labels[start_idx:end_idx]

        # Extract batch data for adaptive approach
        adaptive_batch_stats = compute_batch_statistics(
            batch_index=batch_idx,
            probe_scores=adaptive_result.probe_scores[start_idx:end_idx],
            baseline_scores=adaptive_result.baseline_scores[start_idx:end_idx],
            used_baseline=adaptive_result.used_baseline[start_idx:end_idx],
            final_scores=adaptive_result.final_scores[start_idx:end_idx],
            labels=batch_labels,
        )
        adaptive_batches.append(adaptive_batch_stats)

        # Extract batch data for fixed approach
        fixed_batch_stats = compute_batch_statistics(
            batch_index=batch_idx,
            probe_scores=fixed_result.probe_scores[start_idx:end_idx],
            baseline_scores=fixed_result.baseline_scores[start_idx:end_idx],
            used_baseline=fixed_result.used_baseline[start_idx:end_idx],
            final_scores=fixed_result.final_scores[start_idx:end_idx],
            labels=batch_labels,
        )
        fixed_batches.append(fixed_batch_stats)

    # Compute overall statistics
    adaptive_costs = np.array([b.budget_cost for b in adaptive_batches])
    fixed_costs = np.array([b.budget_cost for b in fixed_batches])

    logger.info(f"Adaptive approach: mean budget cost = {adaptive_costs.mean():.3f}")
    logger.info(f"Fixed approach: mean budget cost = {fixed_costs.mean():.3f}")

    # Compute best/worst batch statistics for ClearML logging
    def get_best_batch(batches):
        """Get batch with lowest budget cost."""
        best_idx = np.argmin([b.budget_cost for b in batches])
        return best_idx, batches[best_idx]

    def get_worst_batch(batches):
        """Get batch with highest budget cost."""
        worst_idx = np.argmax([b.budget_cost for b in batches])
        return worst_idx, batches[worst_idx]

    adaptive_best_idx, adaptive_best = get_best_batch(adaptive_batches)
    adaptive_worst_idx, adaptive_worst = get_worst_batch(adaptive_batches)
    fixed_best_idx, fixed_best = get_best_batch(fixed_batches)
    fixed_worst_idx, fixed_worst = get_worst_batch(fixed_batches)

    # Compute paired t-tests
    logger.info("Computing paired t-tests for performance metrics...")
    adaptive_accuracy = np.array([b.accuracy for b in adaptive_batches])
    fixed_accuracy = np.array([b.accuracy for b in fixed_batches])
    adaptive_f1 = np.array([b.f1_score for b in adaptive_batches])
    fixed_f1 = np.array([b.f1_score for b in fixed_batches])
    adaptive_roc_auc = np.array([b.roc_auc for b in adaptive_batches])
    fixed_roc_auc = np.array([b.roc_auc for b in fixed_batches])

    t_test_results = compute_paired_t_tests(
        {
            "accuracy": (adaptive_accuracy, fixed_accuracy),
            "f1_score": (adaptive_f1, fixed_f1),
            "roc_auc": (adaptive_roc_auc, fixed_roc_auc),
        }
    )

    # Compute overall performance metrics across all examples
    logger.info("Computing overall performance metrics...")
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # Probe only metrics
    probe_predictions = (test_probe_scores >= 0.5).astype(int)
    probe_only_accuracy = float(accuracy_score(test_labels, probe_predictions))
    probe_only_f1 = float(f1_score(test_labels, probe_predictions))
    probe_only_roc_auc = float(roc_auc_score(test_labels, test_probe_scores))

    # Baseline only metrics (all examples use baseline)
    baseline_predictions = (test_baseline_scores >= 0.5).astype(int)
    baseline_only_accuracy = float(accuracy_score(test_labels, baseline_predictions))
    baseline_only_f1 = float(f1_score(test_labels, baseline_predictions))
    baseline_only_roc_auc = float(roc_auc_score(test_labels, test_baseline_scores))

    # Adaptive cascade overall metrics
    adaptive_predictions = (adaptive_result.final_scores >= 0.5).astype(int)
    adaptive_overall_accuracy = float(accuracy_score(test_labels, adaptive_predictions))
    adaptive_overall_f1 = float(f1_score(test_labels, adaptive_predictions))
    adaptive_overall_roc_auc = float(roc_auc_score(test_labels, adaptive_result.final_scores))

    # Fixed cascade overall metrics
    fixed_predictions = (fixed_result.final_scores >= 0.5).astype(int)
    fixed_overall_accuracy = float(accuracy_score(test_labels, fixed_predictions))
    fixed_overall_f1 = float(f1_score(test_labels, fixed_predictions))
    fixed_overall_roc_auc = float(roc_auc_score(test_labels, fixed_result.final_scores))

    # Log overall metrics
    logger.info("\n=== OVERALL PERFORMANCE METRICS ===")
    logger.info(
        f"Probe Only:        Acc={probe_only_accuracy:.4f}, F1={probe_only_f1:.4f}, ROC-AUC={probe_only_roc_auc:.4f}"
    )
    logger.info(
        f"Baseline Only:     Acc={baseline_only_accuracy:.4f}, F1={baseline_only_f1:.4f}, ROC-AUC={baseline_only_roc_auc:.4f}"
    )
    logger.info(
        f"Adaptive Cascade:  Acc={adaptive_overall_accuracy:.4f}, F1={adaptive_overall_f1:.4f}, ROC-AUC={adaptive_overall_roc_auc:.4f}"
    )
    logger.info(
        f"Fixed Cascade:     Acc={fixed_overall_accuracy:.4f}, F1={fixed_overall_f1:.4f}, ROC-AUC={fixed_overall_roc_auc:.4f}"
    )
    logger.info("===================================\n")

    return CascadeComparisonResults(
        config=vars(config),
        seed=seed,
        debug_mode=debug_mode,
        test_size=len(test_dataset),
        cascade_batch_size=cascade_batch_size,
        num_batches=num_batches,
        reliable_threshold=reliable_threshold,
        guarantee_probability=config.guarantee_probability,
        budget=budget,
        adaptive_mean_budget_cost=float(adaptive_costs.mean()),
        adaptive_std_budget_cost=float(adaptive_costs.std()),
        adaptive_min_budget_cost=float(adaptive_costs.min()),
        adaptive_max_budget_cost=float(adaptive_costs.max()),
        fixed_mean_budget_cost=float(fixed_costs.mean()),
        fixed_std_budget_cost=float(fixed_costs.std()),
        fixed_min_budget_cost=float(fixed_costs.min()),
        fixed_max_budget_cost=float(fixed_costs.max()),
        # Best batch statistics - adaptive
        adaptive_best_batch_index=int(adaptive_best_idx),
        adaptive_best_batch_budget_cost=float(adaptive_best.budget_cost),
        adaptive_best_batch_accuracy=float(adaptive_best.accuracy),
        adaptive_best_batch_f1_score=float(adaptive_best.f1_score),
        adaptive_best_batch_roc_auc=float(adaptive_best.roc_auc),
        adaptive_best_batch_probe_uncertainty=float(adaptive_best.probe_uncertainty_mean),
        # Worst batch statistics - adaptive
        adaptive_worst_batch_index=int(adaptive_worst_idx),
        adaptive_worst_batch_budget_cost=float(adaptive_worst.budget_cost),
        adaptive_worst_batch_accuracy=float(adaptive_worst.accuracy),
        adaptive_worst_batch_f1_score=float(adaptive_worst.f1_score),
        adaptive_worst_batch_roc_auc=float(adaptive_worst.roc_auc),
        adaptive_worst_batch_probe_uncertainty=float(adaptive_worst.probe_uncertainty_mean),
        # Best batch statistics - fixed
        fixed_best_batch_index=int(fixed_best_idx),
        fixed_best_batch_budget_cost=float(fixed_best.budget_cost),
        fixed_best_batch_accuracy=float(fixed_best.accuracy),
        fixed_best_batch_f1_score=float(fixed_best.f1_score),
        fixed_best_batch_roc_auc=float(fixed_best.roc_auc),
        fixed_best_batch_probe_uncertainty=float(fixed_best.probe_uncertainty_mean),
        # Worst batch statistics - fixed
        fixed_worst_batch_index=int(fixed_worst_idx),
        fixed_worst_batch_budget_cost=float(fixed_worst.budget_cost),
        fixed_worst_batch_accuracy=float(fixed_worst.accuracy),
        fixed_worst_batch_f1_score=float(fixed_worst.f1_score),
        fixed_worst_batch_roc_auc=float(fixed_worst.roc_auc),
        fixed_worst_batch_probe_uncertainty=float(fixed_worst.probe_uncertainty_mean),
        adaptive_batches=adaptive_batches,
        fixed_batches=fixed_batches,
        # Paired t-test results
        accuracy_t_stat=float(t_test_results["accuracy"]["t_stat"]),
        accuracy_p_value=float(t_test_results["accuracy"]["p_value"]),
        accuracy_mean_diff=float(t_test_results["accuracy"]["mean_diff"]),
        accuracy_std_diff=float(t_test_results["accuracy"]["std_diff"]),
        f1_score_t_stat=float(t_test_results["f1_score"]["t_stat"]),
        f1_score_p_value=float(t_test_results["f1_score"]["p_value"]),
        f1_score_mean_diff=float(t_test_results["f1_score"]["mean_diff"]),
        f1_score_std_diff=float(t_test_results["f1_score"]["std_diff"]),
        roc_auc_t_stat=float(t_test_results["roc_auc"]["t_stat"]),
        roc_auc_p_value=float(t_test_results["roc_auc"]["p_value"]),
        roc_auc_mean_diff=float(t_test_results["roc_auc"]["mean_diff"]),
        roc_auc_std_diff=float(t_test_results["roc_auc"]["std_diff"]),
        # Overall performance metrics
        probe_only_accuracy=probe_only_accuracy,
        probe_only_f1_score=probe_only_f1,
        probe_only_roc_auc=probe_only_roc_auc,
        baseline_only_accuracy=baseline_only_accuracy,
        baseline_only_f1_score=baseline_only_f1,
        baseline_only_roc_auc=baseline_only_roc_auc,
        adaptive_overall_accuracy=adaptive_overall_accuracy,
        adaptive_overall_f1_score=adaptive_overall_f1,
        adaptive_overall_roc_auc=adaptive_overall_roc_auc,
        fixed_overall_accuracy=fixed_overall_accuracy,
        fixed_overall_f1_score=fixed_overall_f1,
        fixed_overall_roc_auc=fixed_overall_roc_auc,
        # Evaluation results
        calib_evaluation_risks=calib_eval_result,
        opt_evaluation_risks=opt_eval_result,
        pareto_mask=pareto_mask,
    )


def make_figures(results: CascadeComparisonResults) -> dict[str, Figure | None | dict[str, Figure]]:
    """Generate all figures for the cascade comparison results."""
    # Compute all the figures
    from cascade_plotting import (
        plot_batch_distributions,
        plot_cascade_vs_probe_performance,
        plot_metric_boxplots,
        plot_overall_performance_comparison,
        plot_paired_method_comparison,
        plot_pareto_frontier,
        plot_probe_uncertainty_vs_metrics,
        plot_summary_comparison,
    )

    figures: dict[str, Figure | None | dict[str, Figure]] = {}
    # Overall performance comparison
    figures["overall"] = plot_overall_performance_comparison(results)
    # Summary bar chart
    figures["summary"] = plot_summary_comparison(results)
    # Distribution histograms
    figures["distributions"] = plot_batch_distributions(results)
    # Probe uncertainty correlation
    figures["probe_uncertainty"] = plot_probe_uncertainty_vs_metrics(results)
    # Paired method comparison
    figures["paired"] = plot_paired_method_comparison(results)
    # Box plots
    figures["boxes"] = plot_metric_boxplots(results)
    # Cascade vs Probe performance
    figures["cascade_vs_probe"] = plot_cascade_vs_probe_performance(results)

    # Pareto frontier (only if Pareto testing was used)
    if results.opt_evaluation_risks is not None and results.pareto_mask is not None:
        figures["pareto"] = plot_pareto_frontier(results)
    else:
        figures["pareto"] = None
    return figures


def log_to_clearml(
    clearml_logger: "ClearMLLogger",
    results: CascadeComparisonResults,
    figures: dict[str, Figure | None | dict[str, Figure]],
):
    """log the experiment results to ClearML"""
    # Log configuration
    clearml_logger.connect_configuration(results.config)

    # Add tags
    tags = []
    if results.debug_mode:
        tags.append("debug")
    tags.append(f"budget-{results.budget:.2f}")
    tags.append(f"probe-{results.config['reduction_strategy']}")
    tags.append(f"merge-{results.config['cascade_merge_strategy']}")
    tags.append(f"pareto_testing-{results.config['pareto_testing']}")
    clearml_logger.add_tags(tags)

    # Use serializer for clean data extraction
    serializer = ClearMLSerializer()
    clearml_logger.log_scalars(serializer.to_clearml_scalars(results))
    clearml_logger.log_artifacts(serializer.to_clearml_artifacts(results))

    logger.info("Generating comparison plots...")

    clearml_logger.log_figure(
        title="Comparison",
        series="Overall Performance",
        figure=figures["overall"],
    )

    clearml_logger.log_figure(
        title="Comparison",
        series="Summary Statistics",
        figure=figures["summary"],
    )
    if figures["distributions"] is not None:
        for metric_name, fig in figures["distributions"].items():  # type: ignore
            clearml_logger.log_figure(
                title="Distributions",
                series=metric_name,
                figure=fig,
            )

    clearml_logger.log_figure(
        title="Analysis",
        series="Probe Uncertainty vs Performance",
        figure=figures["probe_uncertainty"],
    )

    clearml_logger.log_figure(
        title="Analysis",
        series="Paired Method Comparison",
        figure=figures["paired"],
    )

    clearml_logger.log_figure(
        title="Comparison",
        series="Metric Ranges",
        figure=figures["boxes"],
    )

    clearml_logger.log_figure(
        title="Analysis",
        series="Cascade vs Probe Performance",
        figure=figures["cascade_vs_probe"],
    )

    # Log Pareto frontier only if it exists
    if figures["pareto"] is not None:
        clearml_logger.log_figure(
            title="Pareto Frontier",
            series="Pareto Frontier",
            figure=figures["pareto"],
        )

    # Close figures to free memory
    import matplotlib.pyplot as plt

    plt.close("all")

    logger.info("All plots generated and logged to ClearML")
    clearml_logger.finalize()


if __name__ == "__main__":
    import os

    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize ClearML logger if enabled
    clearml_logger = None
    if args.use_clearml:
        from clearml_logger import ClearMLLogger
        from clearml_serialization import ClearMLSerializer

        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="cascade_comparison_experiment",
            enabled=True,
        )

    # Run experiment
    results = run_cascade_comparison_experiment(config)

    # Skip logging if experiment failed (no reliable threshold found)
    if results is None:
        logger.warning("Experiment failed: no reliable threshold found.")
    else:
        figures = make_figures(results)

        if clearml_logger is not None:
            log_to_clearml(clearml_logger, results=results, figures=figures)

    logger.info("Experiment complete!")
