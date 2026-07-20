import json

import numpy as np
from probe_ablation_analysis import evaluate_artifact


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
