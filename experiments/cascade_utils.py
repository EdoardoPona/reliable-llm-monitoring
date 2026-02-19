"""Shared data structures and utilities for cascade experiments.

Contains:
- ``BatchCascadeStatistics``: Per-batch statistics dataclass
- ``compute_batch_statistics``: Compute stats for a single batch
- ``compute_overall_metrics``: Compute accuracy/f1/roc_auc for any score array
- ``CascadeExperimentResults``: Protocol formalising shared fields across result types
- ``save_results_to_clearml`` / ``load_results_from_clearml``: Full-object ClearML serialization
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from clearml_logger import ClearMLLogger
from clearml_serialization import artifact_field, scalar_field

logger = logging.getLogger(__name__)


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


def compute_overall_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute accuracy, f1, and roc_auc for a score array against labels.

    Returns:
        Dict with keys ``'accuracy'``, ``'f1_score'``, ``'roc_auc'``.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    predictions = (scores >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_score": float(f1_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


@runtime_checkable
class CascadeExperimentResults(Protocol):
    """Protocol formalising the shared fields across all cascade result types.

    Any cascade result dataclass (``SGTCascadeResults``,
    ``GuaranteedRiskCascadeResults``, ``FixedCascadeResults``) satisfies this
    protocol, enabling generic analysis and comparison code.
    """

    batches: list[BatchCascadeStatistics]

    cascade_accuracy: float
    cascade_f1_score: float
    cascade_roc_auc: float

    mean_budget_cost: float

    probe_only_accuracy: float
    probe_only_f1_score: float
    probe_only_roc_auc: float

    baseline_only_accuracy: float
    baseline_only_f1_score: float
    baseline_only_roc_auc: float

    test_probe_scores: np.ndarray
    test_baseline_scores: np.ndarray
    test_labels: np.ndarray
    cascade_final_scores: np.ndarray


def save_results_to_clearml(clearml_logger: ClearMLLogger, results: Any) -> None:
    """Upload the full results object as a pickle artifact to ClearML.

    This supplements the existing per-field scalar/artifact logging and enables
    complete result reconstruction via :func:`load_results_from_clearml`.
    """
    clearml_logger.log_pickle_artifact("results_object", results)
    logger.info("Saved full results object to ClearML artifact 'results_object'")


def load_results_from_clearml(task_id: str) -> Any:
    """Load the full results object from a ClearML task by ID.

    Args:
        task_id: ClearML task ID (e.g. ``'abc123'``).

    Returns:
        The deserialized results dataclass.
    """
    from clearml import Task

    task = Task.get_task(task_id=task_id)
    artifact = task.artifacts.get("results_object")
    if artifact is None:
        raise ValueError(
            f"ClearML task '{task_id}' has no 'results_object' artifact. "
            "Was it run with the updated code that saves full results?"
        )
    local_path = artifact.get_local_copy()
    with open(local_path, "rb") as f:
        return pickle.load(f)  # noqa: S301
