"""Configuration utilities for experiments."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

import yaml


def expand_env_vars(value):
    """Expand environment variables in a value (string or recursively in dict/list)."""
    if isinstance(value, str):
        # Replace ${VAR_NAME} with environment variable values
        def replace_env_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(r"\$\{([^}]+)\}", replace_env_var, value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    else:
        return value


def load_config(config_path):
    """Load configuration from a YAML file and expand environment variables.

    Returns a SimpleNamespace object allowing attribute access (e.g., config.budget)
    instead of dict-style access (config['budget']).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Expand environment variables in all config values
    config = expand_env_vars(config)

    # Convert to SimpleNamespace for attribute-style access
    return SimpleNamespace(**config)


# ---------------------------------------------------------------------------
# Sweep utilities
# ---------------------------------------------------------------------------


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
    """Build a list of sweep runs from base config and sweep dimensions.

    Sweep keys can be strings (single config key) or tuples of strings
    (paired keys).  Paired keys are swept together (zipped), not crossed::

        SWEEP_CONFIG = {
            "reduction_strategy": ["mean", "max"],
            ("activations_model_name", "activations_layer"): [
                ("meta-llama/Llama-3.2-1B-Instruct", 11),
                ("meta-llama/Llama-3.2-3B-Instruct", 25),
            ],
        }

    All top-level sweep keys are crossed via cartesian product; paired
    keys expand into multiple config entries per combination.
    """
    # Flatten tuple keys for overlap check
    all_sweep_keys: set[str] = set()
    for key in sweep_config:
        if isinstance(key, tuple):
            all_sweep_keys.update(key)
        else:
            all_sweep_keys.add(key)

    overlap = set(base_config).intersection(all_sweep_keys)
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
                if isinstance(key, tuple):
                    for k, v in zip(key, value, strict=True):
                        cfg[k] = v
                    label_parts.append(",".join(f"{k}={v}" for k, v in zip(key, value, strict=True)))
                else:
                    cfg[key] = value
                    label_parts.append(f"{key}={value}")
            configs.append(cfg)
            labels.append(",".join(label_parts))

    sweep_runs: list[SweepRun] = []
    for i, (cfg, label) in enumerate(zip(configs, labels, strict=True), start=1):
        expanded = expand_env_vars(cfg)
        sweep_runs.append(SweepRun(index=i, config=SimpleNamespace(**expanded), label=label))

    return sweep_runs
