import json
from types import SimpleNamespace

import numpy as np
import pytest
from batch_analysis import get_tau_for_alpha
from dv_cascade_comparison import run_ltt_calibration
from ltt_coverage_validation import run_accuracy_trial
from probe_ablation_analysis import evaluate_accuracy_guarantee_artifact, evaluate_artifact
from saved_ablation_runs import load_saved_run
from sklearn.metrics import roc_auc_score


def test_evaluate_saved_score_artifact(tmp_path):
    rng = np.random.default_rng(12)
    n = 512
    labels = rng.integers(0, 2, n)
    signal = labels + rng.normal(0, 0.9, n)
    probe = 1 / (1 + np.exp(-signal))
    expert = 1 / (1 + np.exp(-(labels + rng.normal(0, 0.55, n))))
    true_probe = np.where(labels == 1, probe, 1 - probe)
    true_expert = np.where(labels == 1, expert, 1 - expert)
    value = true_expert - true_probe
    dv = value + rng.normal(0, 0.15, n)

    scores_path = tmp_path / "scores.npz"
    np.savez_compressed(
        scores_path,
        test_probe=probe,
        test_expert=expert,
        test_labels=labels,
        test_value=value,
        test_dv=dv,
        test_groups=np.array(["a", "b"] * (n // 2)),
    )
    metadata = {
        "config": {
            "cell_name": "attention_attention",
            "expert_name": "strong",
            "seed": 42,
            "probe": {"type": "attention"},
            "dv_probe": {"type": "attention"},
            "calib_fraction": 0.5,
            "guarantee_probability": 0.9,
            "tau_steps": 12,
            "n_alpha_steps": 6,
            "pareto_testing": True,
            "pareto_split_proportion": 0.3,
            "opt_risk": "accuracy_error",
            "merge_strategy": "replace",
            "batch_size": 32,
            "n_k_steps": 6,
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    result = evaluate_artifact(scores_path)

    assert result["cell"] == "attention_attention"
    assert 0 <= result["probe_auc"] <= 1
    assert -1 <= result["dv_spearman"] <= 1
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "ablation_ranking_comparison_B32.pdf").exists()


def test_evaluate_accuracy_guarantee_from_saved_scores(tmp_path):
    rng = np.random.default_rng(7)
    n = 800
    labels = rng.integers(0, 2, n)
    probe = np.clip(0.25 + 0.5 * labels + rng.normal(0, 0.2, n), 0.001, 0.999)
    expert = np.clip(0.1 + 0.8 * labels + rng.normal(0, 0.1, n), 0.001, 0.999)
    dv = -np.abs(probe - 0.5) + rng.normal(0, 0.03, n)

    scores_path = tmp_path / "scores.npz"
    np.savez_compressed(
        scores_path,
        test_probe=probe,
        test_expert=expert,
        test_labels=labels,
        test_dv=dv,
    )
    metadata = {
        "config": {
            "cell_name": "attention_attention",
            "expert_name": "strong",
            "seed": 42,
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    result = evaluate_accuracy_guarantee_artifact(
        scores_path,
        guarantee_performance_levels=[0.70, 0.75],
        shared_config={
            "calib_fraction": 0.5,
            "guarantee_probability": 0.9,
            "tau_steps": 20,
            "pareto_testing": True,
            "pareto_split_proportion": 0.3,
            "merge_strategy": "replace",
        },
    )

    assert result["guaranteed_risk"] == "accuracy_error"
    assert result["opt_risk"] == "budget"
    assert result["ltt"]["CTD"]
    assert all("target_performance" in row for row in result["ltt"]["CTD"])
    assert not (tmp_path / "results.json").exists()


def test_accuracy_validation_trial_reports_risk_and_budget():
    rng = np.random.default_rng(22)
    n = 1000
    labels = rng.integers(0, 2, n)
    probe = np.clip(0.25 + 0.5 * labels + rng.normal(0, 0.2, n), 0.001, 0.999)
    expert = np.clip(0.1 + 0.8 * labels + rng.normal(0, 0.1, n), 0.001, 0.999)
    dv = -np.abs(probe - 0.5) + rng.normal(0, 0.03, n)

    result = run_accuracy_trial(
        probe,
        expert,
        labels,
        dv,
        target_accuracy=0.70,
        delta=0.1,
        dv_tau_grid=np.linspace(dv.min() - 0.01, dv.max() + 0.01, 30),
        seed=42,
        calib_fraction=0.5,
        pareto_split_proportion=0.3,
        merge_strategy="replace",
    )

    assert result["valid"]
    assert 0 <= result["realized_accuracy"] <= 1
    assert 0 <= result["realized_budget"] <= 1
    assert result["violation"] == (result["realized_accuracy"] < result["target_accuracy"])
    assert result["empirical_valid"]
    assert 0 <= result["empirical_realized_accuracy"] <= 1
    assert 0 <= result["empirical_realized_budget"] <= 1
    assert result["empirical_violation"] == (result["empirical_realized_accuracy"] < result["target_accuracy"])


def test_ltt_calibration_requires_explicit_guaranteed_risk():
    scores = np.linspace(0.1, 0.9, 20)
    labels = np.tile([0, 1], 10)
    config = SimpleNamespace(guarantee_probability=0.9, tau_steps=10)

    with pytest.raises(ValueError, match="missing required 'guaranteed_risk'"):
        run_ltt_calibration(
            scores,
            scores,
            labels,
            scores,
            scores,
            scores,
            labels,
            scores,
            scores,
            scores,
            "continuous",
            config,
            "replace",
        )


def test_saved_analysis_loader_reconstructs_split_without_models(tmp_path):
    rng = np.random.default_rng(4)
    n = 120
    labels = np.tile([0, 1], n // 2)
    probe = np.clip(0.2 + 0.6 * labels + rng.normal(0, 0.1, n), 0.001, 0.999)
    expert = np.clip(0.1 + 0.8 * labels + rng.normal(0, 0.1, n), 0.001, 0.999)
    value = np.where(labels == 1, expert - probe, probe - expert)
    dv = value + rng.normal(0, 0.05, n)
    groups = np.asarray(["a", "b", "c", "d"] * (n // 4))
    score_path = tmp_path / "scores.npz"
    np.savez_compressed(
        score_path,
        test_probe=probe,
        test_expert=expert,
        test_labels=labels,
        test_value=value,
        test_dv=dv,
        test_groups=groups,
    )
    config = {
        "cell_name": "attention_attention",
        "expert_name": "strong",
        "seed": 42,
        "calib_fraction": 0.5,
        "baseline_model_name": "expert/model",
        "merge_strategy": "replace",
    }
    (tmp_path / "metadata.json").write_text(json.dumps({"config": config}))

    permutation = np.random.default_rng(42).permutation(n)
    eval_indices = permutation[n // 2 :]
    ltt = {
        "CTD": [{"alpha": 0.2, "tau": 0.1, "realized_budget": 0.15}],
        "Unc. calibrated": [{"alpha": 0.2, "tau": -0.1, "realized_budget": 0.18}],
    }
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "probe_auc": roc_auc_score(labels[eval_indices], probe[eval_indices]),
                "ltt": ltt,
            }
        )
    )

    run = load_saved_run(score_path, require_ltt=True)

    np.testing.assert_array_equal(run["eval_ps"], probe[eval_indices])
    np.testing.assert_array_equal(run["eval_groups"], groups[eval_indices])
    assert run["cell"] == "attention_attention"
    assert run["ltt"] == ltt
    assert get_tau_for_alpha(run["ltt"], 0.2) == (0.1, 0.15)
