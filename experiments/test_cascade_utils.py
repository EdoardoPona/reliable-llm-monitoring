"""Integration tests for cascade_utils ClearML save/load round-trip.

Creates a real ClearML task, uploads fake results, fetches them back,
and checks they match.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from experiments.cascade_utils import (
    BatchCascadeStatistics,
    CascadeExperimentResults,
    load_results_from_clearml,
    save_results_to_clearml,
)
from experiments.clearml_logger import ClearMLLogger
from experiments.clearml_serialization import artifact_field, scalar_field


@dataclass
class _FakeResults:
    """Minimal dataclass satisfying CascadeExperimentResults."""

    batches: list = scalar_field()
    cascade_accuracy: float = scalar_field()
    cascade_f1_score: float = scalar_field()
    cascade_roc_auc: float = scalar_field()
    mean_budget_cost: float = scalar_field()
    probe_only_accuracy: float = scalar_field()
    probe_only_f1_score: float = scalar_field()
    probe_only_roc_auc: float = scalar_field()
    baseline_only_accuracy: float = scalar_field()
    baseline_only_f1_score: float = scalar_field()
    baseline_only_roc_auc: float = scalar_field()
    test_probe_scores: np.ndarray = artifact_field()
    test_baseline_scores: np.ndarray = artifact_field()
    test_labels: np.ndarray = artifact_field()
    cascade_final_scores: np.ndarray = artifact_field()


def _make_fake_results() -> _FakeResults:
    rng = np.random.default_rng(42)
    n = 20
    return _FakeResults(
        batches=[
            BatchCascadeStatistics(
                batch_index=0,
                budget_cost=0.3,
                num_examples=n,
                probe_uncertainty_mean=0.15,
                probe_uncertainty_std=0.05,
                probe_uncertainty_min=0.02,
                probe_uncertainty_max=0.45,
                baseline_score_mean=0.85,
                baseline_score_std=0.1,
                accuracy=0.85,
                f1_score=0.82,
                roc_auc=0.90,
                probe_accuracy=0.80,
                probe_f1_score=0.78,
                probe_roc_auc=0.88,
                probe_scores=rng.random(n),
                baseline_scores=rng.random(n),
                used_baseline=rng.random(n) > 0.5,
                final_scores=rng.random(n),
            )
        ],
        cascade_accuracy=0.85,
        cascade_f1_score=0.82,
        cascade_roc_auc=0.90,
        mean_budget_cost=0.35,
        probe_only_accuracy=0.80,
        probe_only_f1_score=0.78,
        probe_only_roc_auc=0.88,
        baseline_only_accuracy=0.90,
        baseline_only_f1_score=0.89,
        baseline_only_roc_auc=0.95,
        test_probe_scores=rng.random(n),
        test_baseline_scores=rng.random(n),
        test_labels=rng.integers(0, 2, size=n).astype(float),
        cascade_final_scores=rng.random(n),
    )


@pytest.fixture(scope="module")
def clearml_round_trip():
    """Push fake results to a real ClearML task, then load them back."""
    original = _make_fake_results()

    clearml_logger = ClearMLLogger(
        project_name="reliable-llm-monitoring/tests",
        task_name="test_cascade_utils",
        enabled=True,
    )
    assert clearml_logger.task is not None, "ClearML not available"
    task_id = clearml_logger.task.id

    save_results_to_clearml(clearml_logger, original)
    clearml_logger.finalize()

    loaded = load_results_from_clearml(task_id)
    return original, loaded


class TestClearMLRoundTrip:
    def test_scalar_fields(self, clearml_round_trip):
        original, loaded = clearml_round_trip
        for field in (
            "cascade_accuracy",
            "cascade_f1_score",
            "cascade_roc_auc",
            "mean_budget_cost",
            "probe_only_accuracy",
            "probe_only_f1_score",
            "probe_only_roc_auc",
            "baseline_only_accuracy",
            "baseline_only_f1_score",
            "baseline_only_roc_auc",
        ):
            assert getattr(loaded, field) == getattr(original, field), f"{field} mismatch"

    def test_numpy_arrays(self, clearml_round_trip):
        original, loaded = clearml_round_trip
        for field in ("test_probe_scores", "test_baseline_scores", "test_labels", "cascade_final_scores"):
            np.testing.assert_array_equal(getattr(loaded, field), getattr(original, field), err_msg=f"{field} mismatch")

    def test_batches_preserved(self, clearml_round_trip):
        original, loaded = clearml_round_trip
        assert len(loaded.batches) == len(original.batches)
        ob, lb = original.batches[0], loaded.batches[0]
        assert lb.batch_index == ob.batch_index
        assert lb.accuracy == ob.accuracy
        assert lb.budget_cost == ob.budget_cost
        np.testing.assert_array_equal(lb.probe_scores, ob.probe_scores)

    def test_loaded_satisfies_protocol(self, clearml_round_trip):
        _, loaded = clearml_round_trip
        assert isinstance(loaded, CascadeExperimentResults)
