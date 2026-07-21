from pathlib import Path
from types import SimpleNamespace

import numpy as np
import probe_ablation
from models_under_pressure.interfaces.dataset import LabelledDataset
from probe_ablation_sweep import build_runs


def test_mixed_split_balances_with_aligned_python_indices(monkeypatch):
    datasets = {}
    for name, size in (("a", 7), ("b", 5)):
        marker = np.arange(size) + (100 if name == "b" else 0)
        datasets[name] = LabelledDataset(
            inputs=[f"{name}-{i}" for i in range(size)],
            ids=[f"{name}-{i}" for i in range(size)],
            other_fields={"labels": (marker % 2).tolist(), "marker": marker.tolist()},
        )

    monkeypatch.setattr(
        probe_ablation.LabelledDataset,
        "load_from",
        lambda path: datasets[str(path)].model_copy(deep=True),
    )
    monkeypatch.setattr(
        probe_ablation,
        "_fetch_activations",
        lambda dataset, path, config, kinds: {"mean": np.asarray(dataset.other_fields["marker"], dtype=float)[:, None]},
    )
    monkeypatch.setattr(
        probe_ablation,
        "compute_or_fetch_baseline",
        lambda model_name, dataset, dataset_path, **kwargs: np.asarray(dataset.other_fields["marker"], dtype=float),
    )
    config = SimpleNamespace(
        seed=42,
        baseline_model_name="expert",
        use_modal=False,
        mixed_datasets={
            "balance_strategy": 5,
            "sources": [
                {"group": "a", "dev": "a"},
                {"group": "b", "dev": "b"},
            ],
        },
    )

    combined, baselines = probe_ablation._load_mixed_split(config, "dev", {"mean"})

    activations = np.asarray(combined.other_fields["activations_mean"]).squeeze(1)
    assert len(combined) == 10
    np.testing.assert_array_equal(activations, baselines)


def test_sweep_uses_mckenzie_safety_hyperparameters():
    runs = build_runs(Path("experiments/configs/probe_ablation.yaml"))
    attention = next(run for run in runs if run.cell_name == "attention_attention")
    softmax = next(run for run in runs if run.cell_name == "softmax_softmax")
    mlp = next(run for run in runs if run.cell_name == "mlp_mlp")

    assert attention.probe["hyperparams"] == {
        "batch_size": 256,
        "validation_batch_size": 128,
        "epochs": 200,
        "learning_rate": 5e-3,
        "final_learning_rate": 5e-4,
        "patience": 50,
        "weight_decay": 1e-3,
    }
    assert softmax.probe["hyperparams"]["gradient_accumulation_steps"] == 1
    assert softmax.probe["hyperparams"]["patience"] == 10
    assert mlp.probe["hyperparams"]["epochs"] == 50
    assert mlp.dv_probe["hyperparams"]["patience"] == 10


def test_sweep_runs_all_seeds_for_deterministic_baseline():
    runs = build_runs(Path("experiments/configs/probe_ablation.yaml"))
    mean_runs = [run for run in runs if run.cell_name == "mean_ridge"]

    assert len(runs) == 42
    assert {(run.expert_name, run.seed) for run in mean_runs} == {
        (expert, seed) for expert in ("strong", "weak") for seed in (42, 43, 44)
    }
