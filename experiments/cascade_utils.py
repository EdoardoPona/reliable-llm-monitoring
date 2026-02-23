"""Shared data structures and utilities for cascade experiments.

Contains:
- ``BatchCascadeStatistics``: Per-batch statistics dataclass
- ``compute_batch_statistics``: Compute stats for a single batch
- ``compute_overall_metrics``: Compute accuracy/f1/roc_auc for any score array
- ``ThresholdCascadeResult``: Per-threshold cascade results (for SGT)
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


@dataclass
class ThresholdCascadeResult:
    """Cascade results for a single threshold value.

    Stores the cascade output and metrics for one threshold from the SGT
    rejected set.  Lightweight — raw probe/baseline scores live on the
    parent ``SGTCascadeResults``.
    """

    threshold: float = scalar_field()
    best_alpha: float = scalar_field()  # tightest valid alpha for this threshold
    valid_alpha_indices: list[int] = artifact_field()  # indices into ordered_alphas

    # Overall metrics at this threshold
    cascade_accuracy: float = scalar_field()
    cascade_f1_score: float = scalar_field()
    cascade_roc_auc: float = scalar_field()
    mean_budget_cost: float = scalar_field()

    # Per-batch stats and scores
    batches: list[BatchCascadeStatistics] = artifact_field()
    cascade_final_scores: np.ndarray = artifact_field()


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


def extract_batch_arrays(results: CascadeExperimentResults) -> dict[str, np.ndarray]:
    """Extract per-batch data into flat numpy arrays for analysis.

    Returns a dict with:
    - Per-batch scalars: ``budget_cost``, ``accuracy``, ``f1_score``, ``roc_auc``,
      ``probe_accuracy``, ``probe_f1_score``, ``probe_roc_auc``,
      ``probe_uncertainty_mean``, ``probe_uncertainty_std``.
    - Per-example arrays (test set): ``probe_scores``, ``baseline_scores``,
      ``labels``, ``cascade_final_scores``.
    - Per-batch lists of arrays: ``batch_probe_scores``, ``batch_baseline_scores``,
      ``batch_labels``, ``batch_used_baseline``, ``batch_final_scores``,
      ``batch_uncertainty`` — one array per batch for within-batch analysis.
    - Scalars: ``threshold`` (if available), ``n_batches``, ``batch_size``.
    """
    batches = results.batches
    n_batches = len(batches)
    batch_size = batches[0].num_examples if batches else 0

    # Per-batch scalar arrays
    scalar_fields = [
        "budget_cost",
        "accuracy",
        "f1_score",
        "roc_auc",
        "probe_accuracy",
        "probe_f1_score",
        "probe_roc_auc",
        "probe_uncertainty_mean",
        "probe_uncertainty_std",
    ]
    out: dict[str, Any] = {}
    for field in scalar_fields:
        out[field] = np.array([getattr(b, field) for b in batches])

    # Full test-set arrays
    out["probe_scores"] = results.test_probe_scores
    out["baseline_scores"] = results.test_baseline_scores
    out["labels"] = results.test_labels
    out["cascade_final_scores"] = results.cascade_final_scores

    # Per-batch example-level arrays
    out["batch_probe_scores"] = [b.probe_scores for b in batches]
    out["batch_baseline_scores"] = [b.baseline_scores for b in batches]
    out["batch_used_baseline"] = [b.used_baseline for b in batches]
    out["batch_final_scores"] = [b.final_scores for b in batches]
    out["batch_uncertainty"] = [np.minimum(b.probe_scores, 1 - b.probe_scores) for b in batches]

    # Reconstruct per-batch labels from test_labels and batch_size
    labels = results.test_labels
    out["batch_labels"] = [labels[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]

    # Metadata
    out["n_batches"] = n_batches
    out["batch_size"] = batch_size
    threshold = getattr(results, "reliable_threshold", None)
    out["threshold"] = threshold

    return out


def save_results_to_clearml(clearml_logger: ClearMLLogger, results: Any) -> None:
    """Upload the full results object as a pickle artifact to ClearML.

    This supplements the existing per-field scalar/artifact logging and enables
    complete result reconstruction via :func:`load_results_from_clearml`.
    """
    clearml_logger.log_pickle_artifact("results_object", results)
    logger.info("Saved full results object to ClearML artifact 'results_object'")


class _ExperimentUnpickler(pickle.Unpickler):
    """Unpickler that resolves ``__main__`` classes from experiment modules.

    When an experiment script (e.g. ``sgt_cascade.py``) is run directly,
    pickle stores its classes under ``__main__``.  Loading from a different
    script fails because the class doesn't exist on *that* ``__main__``.
    This unpickler tries known experiment modules as fallbacks.
    """

    _MODULES = (
        "sgt_cascade",
        "guaranteed_risk_cascade",
        "fixed_cascade",
        "cascade_utils",
        "reliable_monitoring.risks",
    )

    def find_class(self, module: str, name: str):  # noqa: ANN001
        if module == "__main__":
            import importlib

            for mod_name in self._MODULES:
                try:
                    mod = importlib.import_module(mod_name)
                    if hasattr(mod, name):
                        return getattr(mod, name)
                except ImportError:
                    continue
        return super().find_class(module, name)


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
        return _ExperimentUnpickler(f).load()  # noqa: S301
