"""Run a sequence of cascade comparison experiments with a sweep of configurations.

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

import cascade_comparison as cascade_module
from cascade_comparison import make_figures, run_cascade_comparison_experiment
from config import expand_env_vars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_CONFIG = {
    # Fixed values (do not change across experiments)
    "budget": 0.4,
    "cascade_merge_strategy": "avg",
    "guarantee_probability": 0.95,
    "reduction_strategy": "mean",
    "cascade_batch_size": 128,
    "baseline_batch_size": 16,
    "activations_model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "activations_layer": 11,
    "baseline_model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "train_dataset_path": "${DATA_DIR}/training/prompts_4x/train.jsonl",
    "calib_dataset_path": "${DATA_DIR}/evals/dev/anthropic_balanced_apr_23.jsonl",
    "test_dataset_path": "${DATA_DIR}/evals/test/anthropic_test_balanced_apr_23.jsonl",
    "seed": 42,
    "debug": True,
}

# Values to sweep over. Order matters (insertion order defines iteration order).
SWEEP_CONFIG = {
    # Examples:
    # "budget": [0.25, 0.4, 0.6],
    # "cascade_merge_strategy": ["avg", "replace"],
    "pareto_testing": [True, False],
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
    overlap = set(base_config).intersection(sweep_config)
    if overlap:
        overlap_list = ", ".join(sorted(overlap))
        raise ValueError(
            "Overlapping keys in BASE_CONFIG and SWEEP_CONFIG are not allowed. "
            f"Remove from BASE_CONFIG or SWEEP_CONFIG: {overlap_list}"
        )

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

    sweep_runs: list[SweepRun] = []
    for i, (cfg, label) in enumerate(zip(configs, labels, strict=True), start=1):
        expanded = expand_env_vars(cfg)
        sweep_runs.append(SweepRun(index=i, config=SimpleNamespace(**expanded), label=label))

    return sweep_runs


def _make_task_name(label: str) -> str:
    if not label or label == "base":
        return "cascade_comparison_experiment"
    sanitized = label.replace("/", "-").replace(" ", "")
    return f"cascade_comparison_experiment_{sanitized}"


def _ensure_cascade_args(config: SimpleNamespace) -> None:
    cascade_module.args = SimpleNamespace(
        debug_mode=bool(getattr(config, "debug", False)),
        pareto_testing=bool(getattr(config, "pareto_testing", False)),
    )


def run_sweep(sweep_runs: Iterable[SweepRun], use_clearml: bool) -> None:
    for run in sweep_runs:
        logger.info("\n=== Running sweep %d: %s ===", run.index, run.label)
        _ensure_cascade_args(run.config)

        clearml_logger = None
        if use_clearml:
            from clearml_logger import ClearMLLogger
            from clearml_serialization import ClearMLSerializer

            clearml_logger = ClearMLLogger(
                project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
                task_name=_make_task_name(run.label),
                enabled=True,
            )
            cascade_module.ClearMLSerializer = ClearMLSerializer

        results = run_cascade_comparison_experiment(run.config)
        if results is None:
            logger.warning("Sweep %d skipped: no reliable threshold found.", run.index)
            continue

        figures = make_figures(results)
        if clearml_logger is not None:
            cascade_module.log_to_clearml(clearml_logger, results, figures)

    logger.info("All sweep runs complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cascade comparison sweep")
    parser.add_argument(
        "--use-clearml",
        action="store_true",
        help="Enable ClearML experiment tracking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runs = build_sweep_configs(BASE_CONFIG, SWEEP_CONFIG)
    run_sweep(runs, use_clearml=args.use_clearml)
