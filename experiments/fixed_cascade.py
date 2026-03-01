"""Fixed-budget cascade experiment using test data from a previous adaptive experiment.

Loads saved results from an adaptive cascade experiment (SGT or guaranteed-risk)
via ClearML, runs a fixed-budget cascade on the **same** test data, and produces
comparable results.  Fast to run (no LLM inference, no probe training).
"""

import argparse
import logging
import os
from dataclasses import dataclass

import numpy as np
from cascade_utils import (
    BatchCascadeStatistics,
    CascadeExperimentResults,
    compute_batch_statistics,
    compute_overall_metrics,
    load_results_from_clearml,
    save_results_to_clearml,
)
from clearml_logger import ClearMLLogger
from clearml_serialization import ClearMLSerializer, artifact_field, derived_field, scalar_field

from reliable_monitoring.cascade import offline_batch_cascade

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FixedCascadeResults:
    """Results for a fixed-budget cascade experiment.

    Field names intentionally match ``SGTCascadeResults`` /
    ``GuaranteedRiskCascadeResults`` so that this type satisfies the
    :class:`CascadeExperimentResults` protocol.
    """

    # Experiment metadata
    config: dict = artifact_field()
    seed: int = scalar_field()
    source_task_id: str = scalar_field()

    # Dataset information
    test_size: int = scalar_field()
    cascade_batch_size: int = scalar_field()
    num_batches: int = scalar_field()

    # Fixed cascade configuration
    fixed_budget_rate: float = scalar_field()

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

    # Baseline-only
    baseline_only_accuracy: float = scalar_field()
    baseline_only_f1_score: float = scalar_field()
    baseline_only_roc_auc: float = scalar_field()

    # Per-batch detailed results
    batches: list[BatchCascadeStatistics] = artifact_field()

    # Score distributions
    test_probe_scores: np.ndarray = artifact_field()
    test_baseline_scores: np.ndarray = artifact_field()
    test_labels: np.ndarray = artifact_field()
    cascade_final_scores: np.ndarray = artifact_field()

    # Mixed dataset group labels (propagated from source experiment, None if single-source)
    test_groups: np.ndarray | None = artifact_field()

    # Derived
    budget_costs: np.ndarray = derived_field(derive_fn=lambda r: np.array([b.budget_cost for b in r.batches]))


def run_fixed_cascade(
    source_results: CascadeExperimentResults,
    budget_rate: float | None = None,
    cascade_batch_size: int | None = None,
    merge_strategy: str | None = None,
) -> FixedCascadeResults:
    """Run a fixed-budget cascade using test data from a previous experiment.

    Args:
        source_results: Results from an adaptive cascade experiment.
        budget_rate: Budget rate for ``fixed_budget_rate`` strategy.
            Defaults to the source experiment's ``mean_budget_cost``.
        cascade_batch_size: Override batch size (defaults to source's).
        merge_strategy: Override merge strategy (defaults to source config's).
    """
    test_probe_scores = source_results.test_probe_scores
    test_baseline_scores = source_results.test_baseline_scores
    test_labels = source_results.test_labels

    if budget_rate is None:
        budget_rate = source_results.mean_budget_cost
        logger.info(f"Using empirical mean budget from source: {budget_rate:.4f}")

    if cascade_batch_size is None:
        cascade_batch_size = getattr(source_results, "cascade_batch_size", 128)

    if merge_strategy is None:
        src_config = getattr(source_results, "config", {})
        merge_strategy = (
            src_config.get("cascade_merge_strategy", "replace") if isinstance(src_config, dict) else "replace"
        )

    logger.info(f"Running fixed cascade (rate={budget_rate:.4f}, batch_size={cascade_batch_size})...")
    cascade_result = offline_batch_cascade(
        probe_scores=test_probe_scores,
        baseline_scores=test_baseline_scores,
        batch_size=cascade_batch_size,
        selection_strategy="fixed_budget_rate",
        merge_strategy=merge_strategy,
        rate=budget_rate,
    )

    # Per-batch statistics
    num_batches = (len(test_probe_scores) + cascade_batch_size - 1) // cascade_batch_size
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

    budget_costs = np.array([b.budget_cost for b in batches])

    cascade_m = compute_overall_metrics(cascade_result.final_scores, test_labels)
    probe_m = compute_overall_metrics(test_probe_scores, test_labels)
    baseline_m = compute_overall_metrics(test_baseline_scores, test_labels)

    logger.info("\n=== FIXED CASCADE RESULTS ===")
    logger.info(f"Budget rate:   {budget_rate:.4f}")
    logger.info(f"Mean budget:   {budget_costs.mean():.4f} (std={budget_costs.std():.4f})")
    logger.info(
        f"Cascade:       Acc={cascade_m['accuracy']:.4f}, F1={cascade_m['f1_score']:.4f}, "
        f"ROC-AUC={cascade_m['roc_auc']:.4f}"
    )
    logger.info("=============================\n")

    seed = getattr(source_results, "seed", 42)

    return FixedCascadeResults(
        config={
            "fixed_budget_rate": budget_rate,
            "cascade_batch_size": cascade_batch_size,
            "cascade_merge_strategy": merge_strategy,
        },
        seed=seed,
        source_task_id="",
        test_size=len(test_labels),
        cascade_batch_size=cascade_batch_size,
        num_batches=num_batches,
        fixed_budget_rate=budget_rate,
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
        test_probe_scores=test_probe_scores,
        test_baseline_scores=test_baseline_scores,
        test_labels=test_labels,
        cascade_final_scores=cascade_result.final_scores,
        test_groups=getattr(source_results, "test_groups", None),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-Budget Cascade Experiment")
    parser.add_argument(
        "--source-task",
        type=str,
        required=True,
        help="ClearML task ID of the adaptive experiment to load test data from.",
    )
    parser.add_argument(
        "--budget-rate",
        type=float,
        default=None,
        help="Budget rate for fixed_budget_rate strategy. Defaults to the source experiment's mean_budget_cost.",
    )
    parser.add_argument(
        "--use-clearml",
        action="store_true",
        help="Enable ClearML experiment tracking for this experiment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Loading results from ClearML task: {args.source_task}")
    source_results = load_results_from_clearml(args.source_task)

    results = run_fixed_cascade(source_results, budget_rate=args.budget_rate)
    results.source_task_id = args.source_task

    if args.use_clearml:
        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="fixed_cascade_experiment",
            enabled=True,
        )
        clearml_logger.connect_configuration(results.config)
        clearml_logger.add_tags(
            [
                f"fixed_budget_rate-{results.fixed_budget_rate:.3f}",
                f"source_task-{args.source_task}",
            ]
        )

        serializer = ClearMLSerializer()
        clearml_logger.log_scalars(serializer.to_clearml_scalars(results))
        clearml_logger.log_artifacts(serializer.to_clearml_artifacts(results))
        save_results_to_clearml(clearml_logger, results)

        clearml_logger.finalize()

    logger.info("Fixed cascade experiment complete!")
