"""Load post-hoc analysis inputs from saved probe-ablation score artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dv_ltt_cascade import split_calib_eval
from sklearn.metrics import roc_auc_score

from reliable_monitoring.cascade import probe_uncertainty


def artifact_path(root: Path, expert: str, cell: str, seed: int) -> Path:
    """Return the standard path for one probe-ablation score artifact."""
    return Path(root) / expert / cell / f"seed_{seed}" / "scores.npz"


def load_saved_run(path: Path, *, require_ltt: bool = False) -> dict:
    """Reconstruct the calibration and evaluation views without loading models.

    The split is reproduced from the seed and calibration fraction saved beside
    ``scores.npz``. When requested, the budget-guarantee table is loaded from
    the ``results.json`` produced by ``probe_ablation_analysis.py``.
    """
    path = Path(path).resolve()
    metadata_path = path.parent / "metadata.json"
    results_path = path.parent / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing score artifact: {path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing artifact metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    config = SimpleNamespace(**metadata["config"])
    required = {
        "test_probe",
        "test_expert",
        "test_labels",
        "test_value",
        "test_dv",
        "test_groups",
    }
    with np.load(path) as saved:
        missing = required.difference(saved.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Score artifact {path} is missing: {names}")
        arrays = {name: saved[name] for name in required}

    calibration, evaluation = split_calib_eval(
        arrays["test_probe"],
        arrays["test_expert"],
        arrays["test_labels"],
        arrays["test_dv"],
        arrays["test_value"],
        arrays["test_groups"],
        calib_fraction=float(config.calib_fraction),
        seed=int(config.seed),
    )
    calib_ps, calib_bs, calib_labels, calib_dv, calib_v, calib_groups = calibration
    eval_ps, eval_bs, eval_labels, eval_dv, eval_v, eval_groups = evaluation
    assert calib_ps is not None and calib_bs is not None
    assert calib_labels is not None and calib_dv is not None
    assert eval_ps is not None and eval_bs is not None
    assert eval_labels is not None and eval_dv is not None
    assert eval_v is not None and eval_groups is not None

    saved_results = None
    if results_path.exists():
        saved_results = json.loads(results_path.read_text())
        expected_auc = float(saved_results["probe_auc"])
        actual_auc = float(roc_auc_score(eval_labels, eval_ps))
        if not np.isclose(actual_auc, expected_auc, atol=1e-10):
            raise ValueError(
                f"Probe AUC mismatch for {path}: reconstructed={actual_auc:.10f}, saved={expected_auc:.10f}"
            )
    elif require_ltt:
        raise FileNotFoundError(
            f"Missing calibrated results: {results_path}. "
            f"Run `uv run experiments/probe_ablation_analysis.py {path.parents[3]}` first."
        )

    return {
        "config": config,
        "calib_ps": calib_ps,
        "calib_bs": calib_bs,
        "calib_labels": calib_labels,
        "calib_dv": calib_dv,
        "calib_v": calib_v,
        "calib_groups": calib_groups,
        "eval_ps": eval_ps,
        "eval_bs": eval_bs,
        "eval_labels": eval_labels,
        "eval_dv": eval_dv,
        "eval_v": eval_v,
        "eval_groups": eval_groups,
        "unc": probe_uncertainty(eval_ps, reference=calib_ps),
        "probe_auc": float(roc_auc_score(eval_labels, eval_ps)),
        "baseline_auc": float(roc_auc_score(eval_labels, eval_bs)),
        "ltt": saved_results["ltt"] if saved_results is not None else None,
        "run_dir": path.parent,
        "artifact_path": path,
        "cell": config.cell_name,
        "expert": config.expert_name,
        "seed": int(config.seed),
    }


def run_label(run: dict) -> str:
    """Return a concise expert label using saved model metadata."""
    model = run["config"].baseline_model_name.split("/")[-1]
    return f"{run['expert'].title()} expert ({model})"
