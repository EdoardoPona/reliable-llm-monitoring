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

from reliable_monitoring.bounds import hb_p_value
from reliable_monitoring.cascade import offline_batch_cascade, run_llm_baseline
from reliable_monitoring.dataset import ActivationConfig, load_dataset, sample_from_dataset
from reliable_monitoring.learn_then_test import fixed_sequence_testing
from reliable_monitoring.probes import SequenceProbe
from reliable_monitoring.risks import evaluate_threshold_risks

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_SAMPLE_SIZE = 32


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

    # Performance metrics
    accuracy: float = scalar_field()
    f1_score: float = scalar_field()
    roc_auc: float = scalar_field()

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

    # Derived statistics
    adaptive_budget_costs: np.ndarray = derived_field(
        derive_fn=lambda r: np.array([b.budget_cost for b in r.adaptive_batches])
    )
    fixed_budget_costs: np.ndarray = derived_field(
        derive_fn=lambda r: np.array([b.budget_cost for b in r.fixed_batches])
    )


def parse_args():
    """Parse command line arguments to get config file path."""
    default_config_path = Path(__file__).parent / "configs" / "cascade_comparison.yaml"

    parser = argparse.ArgumentParser(description="Cascade Comparison Experiment")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with smaller datasets.",
    )
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

    # Probe uncertainty (distance from decision boundary)
    probe_uncertainty = np.minimum(probe_scores, 1 - probe_scores)

    # Baseline scores for examples that used baseline
    baseline_subset = baseline_scores[used_baseline]
    baseline_score_mean = float(np.nanmean(baseline_subset)) if len(baseline_subset) > 0 else 0.0
    baseline_score_std = float(np.nanstd(baseline_subset)) if len(baseline_subset) > 0 else 0.0

    # Performance metrics
    predictions = (final_scores >= 0.5).astype(int)
    accuracy = float(accuracy_score(labels, predictions))
    f1 = float(f1_score(labels, predictions))
    roc_auc = float(roc_auc_score(labels, final_scores))

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
        probe_scores=probe_scores.copy(),
        baseline_scores=baseline_scores.copy(),
        used_baseline=used_baseline.copy(),
        final_scores=final_scores.copy(),
    )


def run_cascade_comparison_experiment(args: argparse.Namespace) -> CascadeComparisonResults | None:
    """Run the cascade comparison experiment."""
    config = load_config(args.config)

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

    if args.debug:
        logger.warning("Running in debug mode with smaller datasets.")
        train_dataset = sample_from_dataset(train_dataset, DEBUG_SAMPLE_SIZE, seed=seed)
        calib_dataset = sample_from_dataset(calib_dataset, DEBUG_SAMPLE_SIZE, seed=seed)
        test_dataset = sample_from_dataset(test_dataset, DEBUG_SAMPLE_SIZE, seed=seed)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Calibration dataset size: {len(calib_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

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

    # Evaluate empirical risks
    thresholds = np.linspace(
        getattr(config, "threshold_start", 0.5),
        getattr(config, "threshold_end", 1.0),
        getattr(config, "threshold_steps", 10),
    )

    eval_result = evaluate_threshold_risks(
        calib_probe_scores,
        calib_baseline_scores,
        thresholds,
        merge_strategy=config.cascade_merge_strategy,
    )

    # Compute p-values using Hoeffding-Bentkus bound
    p_values = hb_p_value(
        r_hat=eval_result.empirical_risks,
        n=eval_result.n_samples,
        alpha=config.budget,
    )

    # Apply fixed-sequence testing
    delta = 1 - config.guarantee_probability
    reliable_hyperparams = fixed_sequence_testing(
        p_values=p_values,
        delta=delta,
    )

    # Check if a reliable threshold was found
    if len(reliable_hyperparams) == 0:
        logger.warning("No reliable thresholds found! Cannot run cascade comparison.")
        return None

    # Use most aggressive reliable threshold
    best_idx = reliable_hyperparams[-1]
    reliable_threshold = float(eval_result.thresholds[best_idx])
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

    return CascadeComparisonResults(
        config=vars(config),
        seed=seed,
        debug_mode=args.debug,
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
    )


if __name__ == "__main__":
    import os

    from clearml_logger import ClearMLLogger
    from clearml_serialization import ClearMLSerializer

    args = parse_args()

    # Initialize ClearML logger if enabled
    clearml_logger = None
    if args.use_clearml:
        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="cascade_comparison_experiment",
            enabled=True,
        )

    # Run experiment
    results = run_cascade_comparison_experiment(args)

    # Skip logging if experiment failed (no reliable threshold found)
    if results is None:
        logger.warning("Experiment failed: no reliable threshold found.")
    else:
        # Log to ClearML if enabled
        if clearml_logger:
            # Log configuration
            clearml_logger.connect_configuration(results.config)

            # Add tags
            tags = []
            if results.debug_mode:
                tags.append("debug")
            tags.append(f"threshold_{results.reliable_threshold:.2f}")
            tags.append(f"budget_{results.budget:.2f}")
            clearml_logger.add_tags(tags)

            # Use serializer for clean data extraction
            serializer = ClearMLSerializer()
            clearml_logger.log_scalars(serializer.to_clearml_scalars(results))
            clearml_logger.log_artifacts(serializer.to_clearml_artifacts(results))

            # Generate and log plots
            from cascade_plotting import (
                plot_batch_distributions,
                plot_difficulty_vs_metrics,
                plot_metric_boxplots,
                plot_summary_comparison,
            )

            logger.info("Generating comparison plots...")

            # Summary bar chart
            fig_summary = plot_summary_comparison(results)
            clearml_logger.log_figure(
                title="Comparison",
                series="Summary Statistics",
                figure=fig_summary,
            )

            # Distribution histograms
            fig_dists = plot_batch_distributions(results)
            for metric_name, fig in fig_dists.items():
                clearml_logger.log_figure(
                    title="Distributions",
                    series=metric_name,
                    figure=fig,
                )

            # Difficulty correlation
            fig_difficulty = plot_difficulty_vs_metrics(results)
            clearml_logger.log_figure(
                title="Analysis",
                series="Difficulty vs Performance",
                figure=fig_difficulty,
            )

            # Box plots
            fig_boxes = plot_metric_boxplots(results)
            clearml_logger.log_figure(
                title="Comparison",
                series="Metric Ranges",
                figure=fig_boxes,
            )

            # Close figures to free memory
            import matplotlib.pyplot as plt

            plt.close("all")

            logger.info("All plots generated and logged to ClearML")
            clearml_logger.finalize()

    logger.info("Experiment complete!")
