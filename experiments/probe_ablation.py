"""Run one probe-architecture ablation cell and persist all score arrays.

The expensive stage (activation loading and probe training) writes ``scores.npz``.
The cascade metrics and figures can be rebuilt from that file with
``probe_ablation_analysis.py`` without retraining either probe.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from activation_registry import compute_or_fetch_activations
from baseline_registry import compute_or_fetch_baseline
from config import load_config
from dotenv import load_dotenv
from models_under_pressure.interfaces.dataset import LabelledDataset

from reliable_monitoring.dv_probes import build_dv_probe
from reliable_monitoring.probes import Probe, build_probe, probe_requires_raw_activations

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SAFETY_PROBE_CACHE: dict[str, Probe] = {}


def continuous_delegation_value(probe_scores: np.ndarray, expert_scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return P_expert(y|x) - P_probe(y|x)."""
    probe_true = np.where(labels == 1, probe_scores, 1.0 - probe_scores)
    expert_true = np.where(labels == 1, expert_scores, 1.0 - expert_scores)
    return expert_true - probe_true


def _pad_raw(arrays: list[np.ndarray]) -> list[np.ndarray]:
    max_length = max(array.shape[1] for array in arrays)
    return [
        np.pad(array, ((0, 0), (0, max_length - array.shape[1]), (0, 0))) if array.shape[1] < max_length else array
        for array in arrays
    ]


def _activation_kinds(safety_spec: dict, dv_spec: dict) -> set[str]:
    kinds = {"mean"}
    if probe_requires_raw_activations(safety_spec) or probe_requires_raw_activations(dv_spec):
        kinds.add("raw")
    return kinds


def _fetch_activations(dataset, path: str, config, kinds: set[str]) -> dict[str, np.ndarray]:
    return {
        kind: compute_or_fetch_activations(
            model_name=config.activations_model_name,
            layer=config.activations_layer,
            reduction=kind,
            dataset=dataset,
            dataset_path=path,
            local=not config.use_modal,
            gpu=getattr(config, "modal_gpu", None),
            batch_size=getattr(config, "activation_batch_size", 8),
            local_only=getattr(config, "local_only", False),
            sync_clearml_on_local_hit=False,
        )
        for kind in kinds
    }


def _attach_activations(dataset, activations: dict[str, np.ndarray]):
    fields = {f"activations_{kind}": value for kind, value in activations.items() if kind != "raw"}
    if "raw" in activations:
        raw = activations["raw"]
        fields["activations"] = raw
        fields["attention_mask"] = np.any(raw != 0, axis=-1)
    return dataset.assign(**fields)


def _load_train(config, kinds: set[str]):
    path = config.train_dataset_path
    dataset = LabelledDataset.load_from(Path(path))
    return _attach_activations(dataset, _fetch_activations(dataset, path, config, kinds))


def _load_mixed_split(config, split: str, kinds: set[str]):
    sources = config.mixed_datasets["sources"]
    balance = config.mixed_datasets.get("balance_strategy", "min_size")
    datasets, baselines, activation_sets = [], [], []
    for source in sources:
        path = source[split]
        dataset = LabelledDataset.load_from(Path(path))
        activations = _fetch_activations(dataset, path, config, kinds)
        baseline = compute_or_fetch_baseline(
            model_name=config.baseline_model_name,
            dataset=dataset,
            dataset_path=path,
            local=not config.use_modal,
            gpu=getattr(config, "modal_gpu", None),
            local_only=getattr(config, "local_only", False),
            sync_clearml_on_local_hit=False,
            baseline_batch_size=getattr(config, "baseline_batch_size", 8),
        )
        datasets.append(dataset)
        baselines.append(baseline)
        activation_sets.append(activations)

    if balance == "min_size":
        target = min(map(len, datasets))
    elif isinstance(balance, int):
        target = balance
    elif balance == "none":
        target = None
    else:
        raise ValueError(f"Unknown balance strategy: {balance}")

    for i, dataset in enumerate(datasets):
        indices = np.arange(len(dataset))
        if target is not None and len(dataset) > target:
            indices = np.array(random.Random(config.seed).sample(range(len(dataset)), target))
        datasets[i] = dataset[indices.tolist()]
        baselines[i] = baselines[i][indices]
        activation_sets[i] = {name: values[indices] for name, values in activation_sets[i].items()}

    if "raw" in kinds:
        padded = _pad_raw([values["raw"] for values in activation_sets])
        for values, raw in zip(activation_sets, padded, strict=True):
            values["raw"] = raw

    enriched = []
    for source, dataset, activations in zip(sources, datasets, activation_sets, strict=True):
        enriched.append(_attach_activations(dataset, activations).assign(group=[source["group"]] * len(dataset)))
    combined = LabelledDataset.concatenate(enriched, col_conflict="intersection")
    return combined, np.concatenate(baselines)


def _spec_with_seed(spec: dict, seed: int) -> dict:
    result = {**spec, "hyperparams": dict(spec.get("hyperparams", {}))}
    result["hyperparams"].setdefault("seed", seed)
    return result


def compute_cell(config) -> dict[str, np.ndarray]:
    """Train the configured safety and DV probes and return reusable arrays."""
    safety_spec = _spec_with_seed(config.probe, config.seed)
    dv_spec = _spec_with_seed(config.dv_probe, config.seed)
    kinds = _activation_kinds(safety_spec, dv_spec)

    safety_key = json.dumps(
        {
            "model": config.activations_model_name,
            "layer": config.activations_layer,
            "probe": safety_spec,
            "seed": config.seed,
        },
        sort_keys=True,
    )
    safety_probe = _SAFETY_PROBE_CACHE.get(safety_key)
    if safety_probe is None:
        logger.info("Loading training activations: %s", sorted(kinds))
        train = _load_train(config, kinds)
        safety_probe = build_probe(safety_spec)
        safety_probe.fit(train)
        _SAFETY_PROBE_CACHE[safety_key] = safety_probe
    else:
        logger.info("Reusing fitted safety probe for this architecture and seed")

    logger.info("Loading mixed development split")
    dev, dev_expert = _load_mixed_split(config, "dev", kinds)
    dev_probe = safety_probe.predict(dev)
    dev_labels = dev.labels_numpy()
    dev_value = continuous_delegation_value(dev_probe, dev_expert, dev_labels)

    dv_probe = build_dv_probe(dv_spec, safety_probe=safety_probe)
    dv_probe.fit(dev, dev_value)

    logger.info("Loading mixed test split")
    test, test_expert = _load_mixed_split(config, "test", kinds)
    test_probe = safety_probe.predict(test)
    test_labels = test.labels_numpy()
    test_value = continuous_delegation_value(test_probe, test_expert, test_labels)
    test_dv = dv_probe.predict(test)
    test_groups = np.asarray(test.other_fields["group"])

    return {
        "dev_probe": dev_probe,
        "dev_expert": dev_expert,
        "dev_labels": dev_labels,
        "dev_value": dev_value,
        "dev_dv": dv_probe.predict(dev),
        "test_probe": test_probe,
        "test_expert": test_expert,
        "test_labels": test_labels,
        "test_value": test_value,
        "test_dv": test_dv,
        "test_groups": test_groups,
    }


def save_cell(config, arrays: dict[str, np.ndarray], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scores.npz"
    np.savez_compressed(path, **arrays)
    metadata = {
        "config": vars(config),
        "created_at": datetime.now().isoformat(),
        "artifact": path.name,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Saved reusable score artifact to %s", path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one probe-ablation cell")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        destination = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = Path("results/probe_ablation") / timestamp
    save_cell(cfg, compute_cell(cfg), destination)
