"""Tests for GuaranteedBudgetResults ClearML round-tripping.

These are unit tests that do not require a live ClearML server.
An optional integration test is included but skipped unless explicitly enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _import_guaranteed_budget_results():
    """Import GuaranteedBudgetResults from experiments/guaranteed_budget.py.

    The experiments scripts use local (non-package) imports like `from config import ...`,
    so we add the experiments folder to sys.path for test-time imports.
    """

    repo_root = Path(__file__).resolve().parents[1]
    experiments_dir = repo_root / "experiments"
    sys.path.insert(0, str(experiments_dir))
    try:
        from guaranteed_budget import GuaranteedBudgetResults  # type: ignore

        return GuaranteedBudgetResults
    finally:
        if sys.path and sys.path[0] == str(experiments_dir):
            sys.path.pop(0)


class _Artifact:
    def __init__(self, path: Path):
        self._path = path

    def get_local_copy(self) -> str:
        return str(self._path)


class _FakeTask:
    def __init__(self, *, config_dict: dict, artifacts: dict[str, _Artifact], reported_scalars: dict):
        self._config_dict = config_dict
        self.artifacts = artifacts
        self._reported_scalars = reported_scalars

    def get_parameter(self, name: str):
        assert name == "Configuration"
        return None

    def get_configuration_object_as_dict(self, name: str):
        assert name == "Configuration"
        return self._config_dict

    def get_reported_scalars(self):
        return self._reported_scalars


def _write_artifacts(tmp_path: Path) -> dict[str, _Artifact]:
    thresholds = np.linspace(0.5, 1.0, 5)
    p_values = np.array([0.01, 0.02, 0.1, 0.2, 0.3], dtype=float)
    empirical_budget_risks = np.array([0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)
    calib_probe_scores = np.array([0.1, 0.2, 0.3], dtype=float)
    calib_baseline_scores = np.array([0.3, 0.2, 0.1], dtype=float)

    # Reliable indices = [0, 1]
    reliable_mask = np.array([1, 1, 0, 0, 0], dtype=np.uint8)

    def save(name: str, arr: np.ndarray) -> _Artifact:
        path = tmp_path / f"{name}.npy"
        np.save(path, arr)
        return _Artifact(path)

    artifacts: dict[str, _Artifact] = {
        "thresholds": save("thresholds", thresholds),
        "empirical_budget_risks": save("empirical_budget_risks", empirical_budget_risks),
        "p_values": save("p_values", p_values),
        "calib_probe_scores": save("calib_probe_scores", calib_probe_scores),
        "calib_baseline_scores": save("calib_baseline_scores", calib_baseline_scores),
    }

    artifacts["reliable_mask"] = save("reliable_mask", reliable_mask)

    return artifacts


def test_from_clearml_reconstructs_results(tmp_path: Path):
    GuaranteedBudgetResults = _import_guaranteed_budget_results()

    artifacts = _write_artifacts(tmp_path)

    config_dict = {
        "seed": 123,
        "reduction_strategy": "pca",
        "activations_model_name": "modelA",
        "activations_layer": 12,
        "baseline_model_name": "base/model",
        "baseline_batch_size": 8,
        "cascade_merge_strategy": "max",
        "budget": 0.1,
        "guarantee_probability": 0.9,
        "debug": False,
    }

    reported_scalars = {
        "Results": {
            "success": {"name": "success", "x": [0], "y": [1]},
            "delta": {"name": "delta", "x": [0], "y": [0.1]},
            "train_size": {"name": "train_size", "x": [0], "y": [10]},
            "calib_size": {"name": "calib_size", "x": [0], "y": [3]},
            "test_size": {"name": "test_size", "x": [0], "y": [4]},
            "best_threshold": {"name": "best_threshold", "x": [0], "y": [0.75]},
            "best_index": {"name": "best_index", "x": [0], "y": [1]},
            "test_budget_cost": {"name": "test_budget_cost", "x": [0], "y": [0.09]},
        }
    }

    task = _FakeTask(config_dict=config_dict, artifacts=artifacts, reported_scalars=reported_scalars)
    results = GuaranteedBudgetResults.from_clearml(task)

    assert results.success is True
    assert results.seed == 123
    assert results.probe_reduction_strategy == "pca"

    assert isinstance(results.thresholds, np.ndarray)
    assert results.best_index == 1
    assert results.best_threshold == pytest.approx(0.75)
    assert results.test_budget_cost == pytest.approx(0.09)

    assert results.reliable_hyperparameters == [0, 1]


def test_to_clearml_artifacts_uses_reliable_mask():
    GuaranteedBudgetResults = _import_guaranteed_budget_results()

    results = GuaranteedBudgetResults(
        config={"debug": False},
        seed=0,
        debug_mode=False,
        train_size=1,
        calib_size=1,
        test_size=1,
        probe_reduction_strategy="pca",
        thresholds=np.array([0.5, 0.6]),
        empirical_budget_risks=np.array([0.1, 0.2]),
        p_values=np.array([0.01, 0.2]),
        delta=0.1,
        reliable_hyperparameters=[0],
        calib_probe_scores=np.array([0.1]),
        calib_baseline_scores=np.array([0.2]),
        success=False,
        best_threshold=None,
        best_index=None,
        test_budget_cost=None,
        test_probe_scores=None,
        test_baseline_scores=None,
        test_cascade_scores=None,
    )

    artifacts = results.to_clearml_artifacts()
    assert "reliable_mask" in artifacts
    assert "rejected_mask" not in artifacts


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::DeprecationWarning:clearml\\.utilities\\.pyhocon\\.config_parser")
@pytest.mark.filterwarnings("ignore::DeprecationWarning:pyparsing\\..*")
def test_from_clearml_integration_real_task():
    """Integration test against a real ClearML task."""
    try:
        from clearml import Task
    except ImportError:
        pytest.skip("clearml package not installed")

    GuaranteedBudgetResults = _import_guaranteed_budget_results()

    task = Task.get_task(task_id="cf2e4edd49a5497c87cbee6c24ff5f18")
    results = GuaranteedBudgetResults.from_clearml(task)

    assert isinstance(results.thresholds, np.ndarray)
    assert isinstance(results.p_values, np.ndarray)
