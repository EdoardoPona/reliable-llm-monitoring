import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from models_under_pressure.activation_store import ActivationStore
from models_under_pressure.interfaces.dataset import LabelledDataset


@dataclass
class ActivationConfig:
    """Configuration for loading and processing activations.

    Attributes:
        model_name: Name of the model used to extract activations
        layer: Layer number to extract activations from
        aggregation_strategy: Strategy or dict of strategies for reducing sequence dimension.
            - Single strategy: "mean" | "max" | "last" | "first"
            - Multiple strategies: {"mean": None, "max": None} will compute both
            - Custom function: {"custom": my_reduction_fn}
            - None: No reduction, keep raw activations (default)
    """

    model_name: str
    layer: int
    aggregation_strategy: None | str | dict[str, Callable | None] = None


def reduce_activations(
    dataset: LabelledDataset,
    strategies: str | dict[str, str | Callable],
    *,
    activation_field: str = "activations",
    mask_field: str = "attention_mask",
    drop_raw: bool = False,
    batch_size: int = 256,
    device: torch.device | None = None,
    inplace: bool = True,
) -> LabelledDataset:
    """Add reduced activations to dataset using specified reduction strategies.

    This function computes reduced (aggregated) versions of sequence-level activations
    and adds them as new fields to the dataset. This is useful for:
    - Precomputing reductions once instead of repeated computation
    - Experimenting with multiple reduction strategies
    - Saving memory by dropping raw activations after reduction

    Args:
        dataset: Dataset with activation fields to reduce
        strategies: Reduction strategy specification:
            - Single string: "mean" → adds "activations_mean" field
            - Dict mapping names to strategies:
              {"mean": "mean", "max": "max"} → adds "activations_mean" and "activations_max"
              {"custom": my_fn} → adds "activations_custom" field
        activation_field: Name of field containing raw activations (default: "activations")
        mask_field: Name of field containing attention mask (default: "attention_mask")
        drop_raw: If True, remove raw activation field after reduction to save memory
        batch_size: Batch size for reduction computation
        device: Device to use for computation (auto-detected if None)
        inplace: If True, modify dataset in place; if False, return a copy

    Returns:
        Dataset with reduced activation fields added (and optionally raw removed)

    Examples:
        # Single reduction
        dataset = reduce_activations(dataset, "mean")
        # Now dataset has "activations_mean" field

        # Multiple reductions
        dataset = reduce_activations(dataset, {"mean": "mean", "max": "max"})
        # Now dataset has "activations_mean" and "activations_max" fields

        # Save memory by dropping raw
        dataset = reduce_activations(dataset, "mean", drop_raw=True)
        # Raw "activations" field is removed, only "activations_mean" remains
    """
    from reliable_monitoring.reductions import apply_reduction_batched, get_reduction_function

    # Normalize strategies to dict format
    if isinstance(strategies, str):
        strategies = {strategies: strategies}

    # Validate that dataset has required fields
    if activation_field not in dataset.other_fields:
        raise ValueError(f"Dataset missing field '{activation_field}'")
    if mask_field not in dataset.other_fields:
        raise ValueError(f"Dataset missing field '{mask_field}'")

    activations = dataset.other_fields[activation_field]
    attention_mask = dataset.other_fields[mask_field]

    # Validate shapes
    if not isinstance(activations, torch.Tensor):
        activations = torch.tensor(activations)
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask)

    if len(activations.shape) != 3:
        raise ValueError(
            f"Expected activations with shape (n_samples, seq_len, hidden_dim), got shape {activations.shape}"
        )

    # Copy dataset if not inplace
    if not inplace:
        # LabelledDataset slicing creates a copy
        dataset = dataset[:]

    # Compute each reduction
    new_fields = {}
    for name, strategy in strategies.items():
        # Get reduction function
        if isinstance(strategy, str):
            reduction_fn = get_reduction_function(strategy)
        elif callable(strategy):
            reduction_fn = strategy
        else:
            raise TypeError(f"Strategy must be string or callable, got {type(strategy)}")

        # Apply reduction
        reduced = apply_reduction_batched(
            activations=activations,
            attention_mask=attention_mask,
            reduction_fn=reduction_fn,
            batch_size=batch_size,
            device_override=device,
            show_progress=True,
        )

        # Store with naming convention: {activation_field}_{name}
        field_name = f"{activation_field}_{name}"
        new_fields[field_name] = reduced
        print(f"Added field '{field_name}' with shape {reduced.shape}")

    # Add all new fields to dataset
    dataset = dataset.assign(**new_fields)

    # Drop raw activations if requested
    if drop_raw:
        dataset = dataset.drop_cols(activation_field)
        print(f"Removed raw field '{activation_field}' to save memory")

    return dataset


def load_dataset(
    dataset_path: Path,
    activation_config: ActivationConfig | None,
    *,
    compute_reductions: bool = False,
    drop_raw_after_reduction: bool = False,
    reduction_batch_size: int = 256,
) -> LabelledDataset:
    """Load dataset with optional activation enrichment and reduction.

    Args:
        dataset_path: Path to dataset file
        activation_config: Configuration for loading activations.
            If None, no activations loaded.
            If aggregation_strategy is set and compute_reductions=True,
            reductions will be computed at load time.
        compute_reductions: If True and activation_config has aggregation_strategy,
            compute reductions immediately after loading
        drop_raw_after_reduction: If True, remove raw activations after computing
            reductions to save memory (only applies if compute_reductions=True)
        reduction_batch_size: Batch size for reduction computation

    Returns:
        Loaded dataset with optional activation fields

    Examples:
        # Load without activations
        dataset = load_dataset(path, activation_config=None)

        # Load with raw activations only
        config = ActivationConfig(model="llama", layer=11)
        dataset = load_dataset(path, config)

        # Load and immediately compute reduction, keep raw
        config = ActivationConfig(model="llama", layer=11, aggregation_strategy="mean")
        dataset = load_dataset(path, config, compute_reductions=True)
        # Has both "activations" and "activations_mean"

        # Load, compute reduction, drop raw to save memory
        config = ActivationConfig(model="llama", layer=11, aggregation_strategy="mean")
        dataset = load_dataset(
            path, config,
            compute_reductions=True,
            drop_raw_after_reduction=True
        )
        # Only has "activations_mean", raw "activations" dropped
    """
    dataset = LabelledDataset.load_from(dataset_path)
    if not activation_config:
        return dataset

    # Load and attach precomputed activations
    store = ActivationStore()  # uses DATA_DIR/activations via config
    dataset = store.enrich(
        dataset=dataset,
        path=dataset_path,
        model_name=activation_config.model_name,
        layer=activation_config.layer,
        mmap=True,  # Use memory-mapped files for large datasets
    )

    # Compute reductions if requested and configured
    if compute_reductions and activation_config.aggregation_strategy is not None:
        # Normalize strategy to dict format for reduce_activations
        strategy = activation_config.aggregation_strategy
        if isinstance(strategy, str):
            strategy_dict = {strategy: strategy}
        elif isinstance(strategy, dict):
            # Filter out None values - reduce_activations expects str | Callable only
            strategy_dict = {k: v for k, v in strategy.items() if v is not None}
        else:
            raise TypeError(f"aggregation_strategy must be str or dict, got {type(strategy)}")

        dataset = reduce_activations(
            dataset,
            strategies=strategy_dict,
            drop_raw=drop_raw_after_reduction,
            batch_size=reduction_batch_size,
            inplace=True,
        )

    return dataset


def sample_from_dataset(dataset: LabelledDataset, num_samples: int, seed: int | None = None) -> LabelledDataset:
    if seed is not None:
        random.seed(seed)
    indices = list(range(len(dataset)))
    sample = random.sample(indices, num_samples)
    return dataset[sample]


def split_dataset(
    dataset: LabelledDataset,
    proportions: Sequence[float],
    shuffle: bool = True,
    seed: int | None = None,
) -> list[LabelledDataset]:
    """
    Split a LabelledDataset into multiple subsets according to specified proportions.

    Args:
        dataset: The LabelledDataset to split
        proportions: A sequence of floats (e.g. [0.6, 0.2, 0.2]) that sum to 1.0.
                    Determines the relative sizes of the returned subsets.
        shuffle: Whether to shuffle the dataset before splitting. Default: True
        seed: Seed for reproducibility. If None, uses random state.

    Returns:
        A list of LabelledDataset objects split according to proportions.

    Example:
        >>> train, val, test = split_dataset(dataset, [0.6, 0.2, 0.2])
    """
    # Validate proportions
    proportions = list(proportions)
    total = sum(proportions)
    if not np.isclose(total, 1.0):
        raise ValueError(f"Proportions must sum to 1.0, got {total}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create indices
    indices = list(range(len(dataset)))

    # Shuffle if requested
    if shuffle:
        random.shuffle(indices)

    # Split indices according to proportions
    splits = []
    start = 0
    for prop in proportions:
        end = start + int(np.round(prop * len(dataset)))
        splits.append(indices[start:end])
        start = end

    # Return split datasets
    return [dataset[split_indices] for split_indices in splits]
