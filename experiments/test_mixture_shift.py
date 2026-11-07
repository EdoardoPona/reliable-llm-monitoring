import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mixture_shift
import numpy as np
import pytest
from mixture_shift import json_safe, run_experiment, run_trial
from mixture_shift_inventory import add_row_inventory, load_artifact, reconstruct_rows

MAPPING = {"anthropic": "anthropic", "mt": "medical", "mts": "medical", "toolace": "toolace"}
SETTINGS = {
    "calib_fraction": 0.5,
    "guarantee_probability": 0.9,
    "tau_steps": 12,
    "pareto_split_proportion": 0.3,
    "merge_strategy": "replace",
}


@pytest.fixture
def artifact(tmp_path):
    data = tmp_path / "data"
    sources = []
    for source in MAPPING:
        paths = {}
        for split in ("dev", "test"):
            path = data / "evals" / split / f"{source}_{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for i in range(20):
                text = f"{source} {split} case {i} " * 20
                rows.append(
                    {
                        "ids": f"{split}_{i}_chosen",
                        "inputs": [{"role": "user", "content": text}],
                        "labels": "high-stakes" if i % 2 else "low-stakes",
                        "transcription": text,
                        "section_text": text,
                    }
                )
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths[split] = f"/original/machine/data/evals/{split}/{path.name}"
        sources.append({"group": source, **paths})
    metadata = {
        "config": {
            "seed": 42,
            "expert_name": "strong",
            "cell_name": "attention_attention",
            "mixed_datasets": {"balance_strategy": 15, "sources": sources},
        }
    }
    selected, _ = reconstruct_rows(metadata, data)
    arrays = {}
    for split in ("dev", "test"):
        labels = np.array([int(r["record"]["labels"] == "high-stakes") for r in selected[split]])
        arrays.update(
            {
                f"{split}_labels": labels,
                f"{split}_probe": 0.1 + 0.8 * labels,
                f"{split}_expert": 0.2 + 0.6 * labels,
                f"{split}_dv": np.linspace(-0.5, 0.5, len(labels)),
            }
        )
    arrays["test_groups"] = np.array([r["source"] for r in selected["test"]])
    path = tmp_path / "artifacts/strong/attention_attention/seed_42/scores.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, **arrays)
    path.with_name("metadata.json").write_text(json.dumps(metadata))
    return path, data


def configuration(artifact_path):
    return SimpleNamespace(
        artifact_root=str(artifact_path.parents[3]),
        experts=["strong"],
        cell="attention_attention",
        seeds=[42],
        source_to_domain=MAPPING,
        reference_weights={"anthropic": 15, "medical": 30, "toolace": 15},
        mixture_resolution=5,
        shared=SETTINGS,
        guaranteed_risk="accuracy_error",
        targets={"strong": [0.5]},
        trials={"n": 1, "base_seed": None},
    )


def test_inventory_preserves_every_inherited_row_and_split(artifact):
    path, data = artifact
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    arrays, metadata, inventory = load_artifact(path, MAPPING)
    add_row_inventory(arrays, metadata, inventory, data)
    assert inventory["test_sources"] == dict.fromkeys(MAPPING, 15)
    assert inventory["test_domains"] == {"anthropic": 15, "medical": 30, "toolace": 15}
    assert len(inventory["row_manifest"]["test"]) == len(inventory["row_manifest"]["dev"]) == 60
    assert inventory["inherited_overlap"]["test_rows_with_exact_input_in_dev"] == 0
    assert all(r["source_id"].startswith("dev_") for r in inventory["row_manifest"]["dev"])
    assert all(r["source_id"].startswith("test_") for r in inventory["row_manifest"]["test"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_inventory_only_never_calls_calibration(artifact, tmp_path, monkeypatch):
    path, data = artifact

    def forbidden(*args, **kwargs):
        raise AssertionError("Inventory must not perform risk control")

    monkeypatch.setattr(mixture_shift, "run_trial", forbidden)
    output = tmp_path / "inventory"
    run_experiment(configuration(path), output, inventory_only=True, data_root=data)
    assert not (output / "trials.jsonl").exists()
    assert json.loads((output / "inventory.json").read_text())["status"] == "complete"


def test_end_to_end_saved_scores_and_matched_comparison(artifact, tmp_path):
    path, data = artifact
    config = configuration(path)
    output = tmp_path / "run"
    run_experiment(config, output, data_root=data)
    trial = json.loads((output / "trials.jsonl").read_text())
    results = {r["method"]: r for r in trial["results"]}
    assert set(results) == {"pooled", "pooled_matched", "mixture"}
    assert (
        results["mixture"]["selection"]["candidate_thresholds"]
        == results["pooled_matched"]["selection"]["candidate_thresholds"]
    )
    indices = trial["split_indices"]
    sets = [set(v) for v in indices.values()]
    assert len(set.union(*sets)) == 60
    assert sum(map(len, sets)) == 60
    assert trial["results"][0]["alpha"] == 0.5
    assert results["mixture"]["selection"]["domain_p_values"].keys() == {"anthropic", "medical", "toolace"}
    assert "NaN" not in (output / "trials.jsonl").read_text()
    with pytest.raises(FileExistsError):
        run_experiment(config, output)


def test_eval_outcomes_cannot_change_thresholds(artifact):
    path, _ = artifact
    arrays, _, _ = load_artifact(path, MAPPING)
    kwargs: dict = dict(
        settings=SETTINGS,
        risk_name="accuracy_error",
        targets=[0.5],
        split_seed=42,
        domains=("anthropic", "medical", "toolace"),
        reference_weights={"anthropic": 1, "medical": 2, "toolace": 1},
        mixtures=np.eye(3),
    )
    first = run_trial(arrays, **kwargs)
    evaluation = first["split_indices"]["evaluation"]
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["test_labels"][evaluation] = 1 - changed["test_labels"][evaluation]
    changed["test_dv"][evaluation] = 1000
    second = run_trial(changed, **kwargs)
    for a, b in zip(first["results"], second["results"], strict=True):
        assert json_safe(a["selection"]) == json_safe(b["selection"])


def test_wrong_source_order_is_rejected(artifact):
    path, data = artifact
    arrays, metadata, inventory = load_artifact(path, MAPPING)
    arrays["test_groups"] = arrays["test_groups"][::-1]
    with pytest.raises(ValueError, match="source order"):
        add_row_inventory(arrays, metadata, inventory, data)


def test_unmapped_group_or_missing_development_grid_fails(artifact):
    path, _ = artifact
    with pytest.raises(ValueError, match="Unmapped"):
        load_artifact(path, {"anthropic": "a"})
    with np.load(path) as source:
        arrays = {k: source[k] for k in source.files if k != "dev_dv"}
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="dev_dv"):
        load_artifact(path, MAPPING)


def test_cannot_write_into_original_artifact_directory(artifact):
    path, _ = artifact
    with pytest.raises(ValueError, match="outside the original"):
        run_experiment(configuration(path), path.parent / "new_results")


def test_shipped_configs_are_valid():
    from config import load_config

    base = Path(__file__).parent / "configs/mixture_shift"
    for path in base.glob("*.yaml"):
        config = load_config(path)
        assert config.source_to_domain == MAPPING
        assert config.cell == "attention_attention"
        assert sum(config.reference_weights.values()) == 1586
        assert config.shared["tau_steps"] == 100


def test_certified_policy_has_linear_mixture_metrics():
    n = 900
    sources = np.array(list(MAPPING) * (n // 4))
    arrays = {
        "dev_dv": np.array([-1.0, 1.0]),
        "test_probe": np.zeros(n),
        "test_expert": np.zeros(n),
        "test_labels": np.zeros(n),
        "test_dv": np.linspace(-0.9, 0.9, n),
        "test_groups": sources,
        "test_domains": np.array([MAPPING[g] for g in sources]),
    }
    result = run_trial(
        arrays,
        settings=SETTINGS,
        risk_name="accuracy_error",
        targets=[0.2],
        split_seed=42,
        domains=("anthropic", "medical", "toolace"),
        reference_weights={"anthropic": 1, "medical": 2, "toolace": 1},
        mixtures=np.eye(3),
    )
    for row in result["results"]:
        assert row["selection"]["threshold"] is not None
        evaluation = row["evaluation"]
        np.testing.assert_array_equal(evaluation["accuracy_error"]["mixtures"], np.zeros(3))
        assert evaluation["accuracy_error"]["worst_domain"] == 0
        assert evaluation["budget"]["reference"] == 0
