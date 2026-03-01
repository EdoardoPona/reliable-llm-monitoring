"""Utilities for building mixed multi-source datasets with group labels.

Loads multiple evaluation datasets, tags each with a 'group' field indicating
its source, and concatenates them into a single LabelledDataset. This enables
group-stratified batching where batches are homogeneous by source, creating
natural performance variability across groups.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from models_under_pressure.interfaces.dataset import LabelledDataset

from reliable_monitoring.dataset import ActivationConfig, load_dataset, sample_from_dataset

logger = logging.getLogger(__name__)


def load_mixed_dataset(
    sources: list[dict[str, str]],
    split: str,
    activation_config: ActivationConfig | None,
    balance_strategy: str | int = "min_size",
    seed: int = 42,
    **load_kwargs: Any,
) -> LabelledDataset:
    """Load and combine multiple source datasets with a 'group' field.

    Each source dataset is loaded, tagged with its group name, optionally
    balanced, then all are concatenated into a single dataset.

    Args:
        sources: List of source dicts, each with keys 'group', 'dev', 'test'.
        split: Which split to load — 'dev' or 'test'.
        activation_config: Configuration for loading activations (passed to load_dataset).
        balance_strategy: How to handle different-sized sources:
            - "min_size": Subsample all groups to the size of the smallest.
            - "none": Keep original sizes.
            - int: Cap each group at this many examples.
        seed: Random seed for reproducibility.
        **load_kwargs: Additional keyword arguments passed to load_dataset.

    Returns:
        A single LabelledDataset with a "group" field in other_fields.
    """
    datasets: list[LabelledDataset] = []

    for source in sources:
        group_name = source["group"]
        path = Path(source[split])

        logger.info(f"Loading {split} dataset for group '{group_name}' from {path}")
        ds = load_dataset(path, activation_config, **load_kwargs)
        ds = ds.assign(group=[group_name] * len(ds))
        logger.info(f"  Group '{group_name}': {len(ds)} examples")
        datasets.append(ds)

    # Balance groups if requested
    if balance_strategy == "min_size":
        min_n = min(len(ds) for ds in datasets)
        logger.info(f"Balancing to min group size: {min_n}")
        datasets = [sample_from_dataset(ds, min_n, seed=seed) if len(ds) > min_n else ds for ds in datasets]
    elif isinstance(balance_strategy, int):
        cap = balance_strategy
        logger.info(f"Capping each group at {cap} examples")
        datasets = [sample_from_dataset(ds, cap, seed=seed) if len(ds) > cap else ds for ds in datasets]
    elif balance_strategy != "none":
        raise ValueError(f"Unknown balance_strategy: {balance_strategy!r}. Use 'min_size', 'none', or an integer.")

    combined = LabelledDataset.concatenate(datasets, col_conflict="intersection")
    logger.info(f"Combined mixed dataset: {len(combined)} total examples across {len(sources)} groups")
    return combined


def has_mixed_config(config: SimpleNamespace) -> bool:
    """Check whether the config uses the mixed_datasets section."""
    return hasattr(config, "mixed_datasets") and config.mixed_datasets is not None


def get_mixed_splits(config: SimpleNamespace) -> set[str]:
    """Return which splits should be mixed (e.g. {'test'} or {'test', 'calib'})."""
    if not has_mixed_config(config):
        return set()
    return set(config.mixed_datasets.get("splits", ["test"]))
