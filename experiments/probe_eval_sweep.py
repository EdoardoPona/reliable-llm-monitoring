"""Run a sequence of probe evaluation experiments with a sweep of configurations.

Edit BASE_CONFIG and SWEEP_CONFIG to define fixed values and swept values.
The sweep is executed in a deterministic order using the insertion order of SWEEP_CONFIG.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

from config import expand_env_vars
from probe_eval import log_to_clearml, make_figures, run_probe_eval

from reliable_monitoring.reductions import _REDUCTION_REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_CONFIG = {
    # Fixed values
    # "reduction_strategy": "mean",
    "activations_model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "activations_layer": 11,
    "train_dataset_path": "${DATA_DIR}/training/prompts_4x/train.jsonl",
    "seed": 42,
    "debug": True,
    "auto_compute_activations": True,
    "cleanup_activations_after_load": True,
    # Batch sizes
    "activation_batch_size": 32,  # Batch size for computing activations
    "reduction_batch_size": 512,  # Batch size for reduction operations
}

# Values to sweep over. Order matters (insertion order defines iteration order).
SWEEP_CONFIG = {
    # Sweep over different evaluation datasets
    "dev_dataset_path": [
        "${DATA_DIR}/evals/dev/anthropic_balanced_apr_23.jsonl",
        "${DATA_DIR}/evals/dev/anthropic_raw_apr_23.jsonl",
        "${DATA_DIR}/evals/dev/mt_balanced_apr_30.jsonl",
        "${DATA_DIR}/evals/dev/mt_raw_apr_30.jsonl",
        "${DATA_DIR}/evals/dev/mts_balanced_apr_22.jsonl",
        "${DATA_DIR}/evals/dev/mts_raw_apr_22.jsonl",
        "${DATA_DIR}/evals/dev/toolace_balanced_apr_22.jsonl",
        "${DATA_DIR}/evals/dev/toolace_raw_apr_22.jsonl",
    ],
    "test_dataset_path": [
        "${DATA_DIR}/evals/test/anthropic_test_balanced_apr_23.jsonl",
        "${DATA_DIR}/evals/test/anthropic_test_raw_apr_23.jsonl",
        "${DATA_DIR}/evals/test/mt_test_balanced_apr_30.jsonl",
        "${DATA_DIR}/evals/test/mt_test_raw_apr_30.jsonl",
        "${DATA_DIR}/evals/test/mts_test_balanced_apr_22.jsonl",
        "${DATA_DIR}/evals/test/mts_test_raw_apr_22.jsonl",
        "${DATA_DIR}/evals/test/toolace_test_balanced_apr_22.jsonl",
        "${DATA_DIR}/evals/test/toolace_test_raw_apr_22.jsonl",
    ],
    "reduction_strategy": list(_REDUCTION_REGISTRY.keys()),
}


@dataclass(frozen=True)
class SweepRun:
    index: int
    config: SimpleNamespace
    label: str


def _normalize_sweep_values(values: object) -> list[object]:
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def build_sweep_configs(base_config: dict, sweep_config: dict) -> list[SweepRun]:
    """Build configurations for sweep, handling paired datasets specially."""
    overlap = set(base_config).intersection(sweep_config)
    if overlap:
        overlap_list = ", ".join(sorted(overlap))
        raise ValueError(
            "Overlapping keys in BASE_CONFIG and SWEEP_CONFIG are not allowed. "
            f"Remove from BASE_CONFIG or SWEEP_CONFIG: {overlap_list}"
        )

    # Special handling: if both dev_dataset_path and test_dataset_path are present,
    # pair them instead of taking cartesian product
    if "dev_dataset_path" in sweep_config and "test_dataset_path" in sweep_config:
        dev_paths = _normalize_sweep_values(sweep_config["dev_dataset_path"])
        test_paths = _normalize_sweep_values(sweep_config["test_dataset_path"])

        if len(dev_paths) != len(test_paths):
            raise ValueError(
                f"dev_dataset_path and test_dataset_path must have same length for pairing. "
                f"Got {len(dev_paths)} dev paths and {len(test_paths)} test paths."
            )

        # Remove dataset paths from sweep_config for normal processing
        other_sweep_config = {k: v for k, v in sweep_config.items() if k not in ("dev_dataset_path", "test_dataset_path")}

        # Build configs with paired datasets
        sweep_runs: list[SweepRun] = []
        run_index = 1

        # Get other sweep combinations
        other_keys = list(other_sweep_config.keys())
        other_values_list = [_normalize_sweep_values(other_sweep_config[key]) for key in other_keys]

        if not other_keys:
            other_combinations = [{}]
        else:
            other_combinations = []
            for values in product(*other_values_list):
                combo = {k: v for k, v in zip(other_keys, values, strict=True)}
                other_combinations.append(combo)

        # Combine paired datasets with other sweep values
        for dev_path, test_path in zip(dev_paths, test_paths, strict=True):
            for other_combo in other_combinations:
                cfg = deepcopy(base_config)
                cfg["dev_dataset_path"] = dev_path
                cfg["test_dataset_path"] = test_path
                cfg.update(other_combo)

                # Create label from dataset name
                dataset_name = dev_path.split("/")[-1].replace(".jsonl", "").replace("_apr_", "_")
                label_parts = [f"dataset={dataset_name}"]
                for k, v in other_combo.items():
                    label_parts.append(f"{k}={v}")
                label = ",".join(label_parts)

                expanded = expand_env_vars(cfg)
                sweep_runs.append(SweepRun(index=run_index, config=SimpleNamespace(**expanded), label=label))
                run_index += 1

        return sweep_runs

    # Standard cartesian product sweep
    keys = list(sweep_config.keys())
    values_list = [_normalize_sweep_values(sweep_config[key]) for key in keys]

    if not keys:
        configs = [deepcopy(base_config)]
        labels = ["base"]
    else:
        configs = []
        labels = []
        for values in product(*values_list):
            cfg = deepcopy(base_config)
            label_parts = []
            for key, value in zip(keys, values, strict=True):
                cfg[key] = value
                label_parts.append(f"{key}={value}")
            configs.append(cfg)
            labels.append(",".join(label_parts))

    sweep_runs = []
    for i, (cfg, label) in enumerate(zip(configs, labels, strict=True), start=1):
        expanded = expand_env_vars(cfg)
        sweep_runs.append(SweepRun(index=i, config=SimpleNamespace(**expanded), label=label))

    return sweep_runs


def _make_task_name(label: str) -> str:
    if not label or label == "base":
        return "probe_eval"
    sanitized = label.replace("/", "-").replace(" ", "").replace("${DATA_DIR}", "")
    # Truncate if too long
    if len(sanitized) > 80:
        sanitized = sanitized[:80]
    return f"probe_eval_{sanitized}"


def run_sweep(sweep_runs: Iterable[SweepRun], use_clearml: bool) -> None:
    import gc

    import matplotlib.pyplot as plt
    import torch

    for run in sweep_runs:
        logger.info("\n=== Running sweep %d: %s ===", run.index, run.label)

        clearml_logger = None
        if use_clearml:
            from clearml_logger import ClearMLLogger

            clearml_logger = ClearMLLogger(
                project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
                task_name=_make_task_name(run.label),
                enabled=True,
            )

        results = run_probe_eval(run.config)
        figures = make_figures(results)

        if clearml_logger is not None:
            log_to_clearml(clearml_logger, results, figures)

        # Clean up between runs to prevent memory accumulation
        plt.close("all")
        del results, figures
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("All sweep runs complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probe evaluation sweep")
    parser.add_argument(
        "--use-clearml",
        action="store_true",
        help="Enable ClearML experiment tracking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runs = build_sweep_configs(BASE_CONFIG, SWEEP_CONFIG)
    logger.info(f"Built {len(runs)} sweep configurations")
    run_sweep(runs, use_clearml=args.use_clearml)
