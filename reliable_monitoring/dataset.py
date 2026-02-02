from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from models_under_pressure.activation_store import ActivationStore
from models_under_pressure.interfaces.dataset import LabelledDataset

if TYPE_CHECKING:
    from models_under_pressure.model import LLMModel


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


@dataclass
class EvalDataset:
    """
    A paired dev/test dataset for evaluation.
    These should always be statistically exchangeable (we use them as calibration sets).
    """
    name: str
    dev: Path
    test: Path


# Paths relative to BASE_DATA_DIR
TRAIN_DATASET = Path("training/prompts_4x/train.jsonl")

EVAL_DATASETS: list[EvalDataset] = [
    EvalDataset("anthropic_balanced", Path("evals/dev/anthropic_balanced_apr_23.jsonl"), Path("evals/test/anthropic_test_balanced_apr_23.jsonl")),
    EvalDataset("anthropic_raw", Path("evals/dev/anthropic_raw_apr_23.jsonl"), Path("evals/test/anthropic_test_raw_apr_23.jsonl")),
    EvalDataset("mt_balanced", Path("evals/dev/mt_balanced_apr_30.jsonl"), Path("evals/test/mt_test_balanced_apr_30.jsonl")),
    EvalDataset("mt_raw", Path("evals/dev/mt_raw_apr_30.jsonl"), Path("evals/test/mt_test_raw_apr_30.jsonl")),
    EvalDataset("mts_balanced", Path("evals/dev/mts_balanced_apr_22.jsonl"), Path("evals/test/mts_test_balanced_apr_22.jsonl")),
    EvalDataset("mts_raw", Path("evals/dev/mts_raw_apr_22.jsonl"), Path("evals/test/mts_test_raw_apr_22.jsonl")),
    EvalDataset("toolace_balanced", Path("evals/dev/toolace_balanced_apr_22.jsonl"), Path("evals/test/toolace_test_balanced_apr_22.jsonl")),
    EvalDataset("toolace_raw", Path("evals/dev/toolace_raw_apr_22.jsonl"), Path("evals/test/toolace_test_raw_apr_22.jsonl")),
]


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
        activation_field: Name of field containing raw activations
        mask_field: Name of field containing attention mask
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


def _load_model(model: str | LLMModel, batch_size: int) -> LLMModel:
    """Load model if given as string, otherwise return as-is."""
    if isinstance(model, str):
        from models_under_pressure.model import LLMModel as LLMModelClass

        return LLMModelClass.load(model, batch_size=batch_size)
    return model


def _get_model_name(model: str | LLMModel) -> str:
    """Get model name from string or LLMModel instance."""
    if isinstance(model, str):
        return model
    return model.name


def _compute_raw_activations(
    dataset: LabelledDataset,
    model: LLMModel,
    layer: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute activations using a loaded model.

    Returns:
        Tuple of (activations, inputs) where activations is a tensor of shape
        (n_layers, n_samples, seq_len, hidden_dim) and inputs contains
        input_ids and attention_mask tensors.
    """
    # Returns activations with shape (n_layers, n_samples, seq_len, hidden_dim)
    return model.get_batched_activations_for_layers(
        dataset=dataset,
        layers=[layer],
    )


def enrich_with_activations(
    dataset: LabelledDataset,
    activations: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> LabelledDataset:
    """Attach activation tensors directly to a dataset.

    Args:
        dataset: Dataset to enrich
        activations: Activation tensor of shape (n_samples, seq_len, hidden_dim)
        input_ids: Input IDs tensor
        attention_mask: Attention mask tensor

    Returns:
        Dataset with activation fields attached
    """
    return dataset.assign(
        activations=activations,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )


def _apply_attention_mask_to_activations(dataset: LabelledDataset) -> LabelledDataset:
    """Zero out activations at padding positions.

    This matches the behavior of models-under-pressure Activation.__post_init__,
    which multiplies activations by the attention mask to zero out padding.

    The ActivationStore.enrich() path bypasses the Activation class, so we need
    to apply this masking explicitly after loading.

    Args:
        dataset: Dataset with 'activations' and 'attention_mask' fields

    Returns:
        Dataset with masked activations
    """
    activations = dataset.other_fields["activations"]
    attention_mask = dataset.other_fields["attention_mask"]

    # Apply mask: activations *= attention_mask[:, :, None]
    if isinstance(activations, torch.Tensor):
        mask = attention_mask
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        masked_activations = activations * mask.unsqueeze(-1)
    else:
        # Handle numpy arrays
        masked_activations = activations * np.expand_dims(attention_mask, -1)

    return dataset.assign(activations=masked_activations)


def compute_activations(
    dataset_path: Path,
    model: str | LLMModel,
    layer: int,
    *,
    batch_size: int = 4,
) -> None:
    """Compute and store activations for a dataset.

    Args:
        dataset_path: Path to the dataset file
        model: Model name (str) or pre-loaded LLMModel instance
        layer: Layer number to extract activations from
        batch_size: Batch size for processing (only used if model is a str)
    """
    from models_under_pressure.activation_store import ActivationsSpec

    model_name = _get_model_name(model)

    store = ActivationStore()
    spec = ActivationsSpec(model_name=model_name, dataset_path=dataset_path, layer=layer)

    if store.exists(spec):
        return  # Already computed

    loaded_model = _load_model(model, batch_size)
    dataset = LabelledDataset.load_from(dataset_path)
    activations, inputs = _compute_raw_activations(dataset, loaded_model, layer)

    store.save(model_name, dataset_path, [layer], activations, inputs)


def cleanup_activations(
    dataset_path: Path,
    model_name: str,
    layer: int,
) -> None:
    """Delete stored activations for a dataset/model/layer combination.

    Args:
        dataset_path: Path to the dataset file
        model_name: Name of the model
        layer: Layer number
    """
    from models_under_pressure.activation_store import ActivationsSpec

    store = ActivationStore()
    spec = ActivationsSpec(model_name=model_name, dataset_path=dataset_path, layer=layer)
    store.sync(add_specs=[], remove_specs=[spec])


def load_dataset(
    dataset_path: Path,
    activation_config: ActivationConfig | None,
    *,
    compute_reductions: bool = False,
    drop_raw_after_reduction: bool = False,
    reduction_batch_size: int = 512,
    auto_compute: bool = False,
    cleanup_after_load: bool = False,
    model: str | LLMModel | None = None,
    compute_batch_size: int = 32,
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
        auto_compute: If True, compute activations if not present
        cleanup_after_load: If True, delete activation files after loading
        model: Model for activation computation - either model name (str) or
            pre-loaded LLMModel instance. Uses activation_config.model_name if None.
        compute_batch_size: Batch size for activation computation

    Returns:
        Loaded dataset with optional activation fields

    Raises:
        FileNotFoundError: If activations not found and auto_compute=False

    Examples:
        # Load without activations
        dataset = load_dataset(path, activation_config=None)

        # Load with raw activations only
        config = ActivationConfig(model="llama", layer=11)
        dataset = load_dataset(path, config)

        # Auto-compute activations if missing
        config = ActivationConfig(model="llama", layer=11)
        dataset = load_dataset(path, config, auto_compute=True)

        # Load and cleanup activation files after loading
        config = ActivationConfig(model="llama", layer=11)
        dataset = load_dataset(path, config, cleanup_after_load=True)

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

        # Compute in-memory (no disk I/O) when auto-computing ephemeral activations
        config = ActivationConfig(model="llama", layer=11)
        dataset = load_dataset(path, config, auto_compute=True, cleanup_after_load=True)
        # Activations computed in memory, never written to disk
    """
    dataset = LabelledDataset.load_from(dataset_path)
    if not activation_config:
        return dataset

    from models_under_pressure.activation_store import ActivationsSpec

    store = ActivationStore()
    spec = ActivationsSpec(
        model_name=activation_config.model_name,
        dataset_path=dataset_path,
        layer=activation_config.layer,
    )

    activations_exist = store.exists(spec)

    # Use in-memory computation when we'd compute and immediately delete (no point saving)
    use_in_memory = auto_compute and cleanup_after_load and not activations_exist

    if use_in_memory:
        # Compute activations in memory without disk I/O
        loaded_model = _load_model(
            model if model is not None else activation_config.model_name,
            compute_batch_size,
        )
        activations, inputs = _compute_raw_activations(
            dataset, loaded_model, activation_config.layer
        )
        # activations has shape (n_layers, n_samples, seq_len, hidden_dim)
        # Extract the single layer with [0]
        dataset = enrich_with_activations(
            dataset,
            activations[0],
            inputs["input_ids"],
            inputs["attention_mask"],
        )
    else:
        # Standard path: load from disk, computing first if needed
        if not activations_exist:
            if not auto_compute:
                raise FileNotFoundError(
                    f"Activations not found for {activation_config.model_name} "
                    f"layer {activation_config.layer} on {dataset_path}. "
                    f"Set auto_compute=True to compute them automatically."
                )
            compute_activations(
                dataset_path=dataset_path,
                model=model if model is not None else activation_config.model_name,
                layer=activation_config.layer,
                batch_size=compute_batch_size,
            )
            # Recreate store to pick up newly saved activations
            store = ActivationStore()

        # Load and attach activations from disk
        dataset = store.enrich(
            dataset=dataset,
            path=dataset_path,
            model_name=activation_config.model_name,
            layer=activation_config.layer,
            mmap=True,
        )

        # Cleanup if requested
        if cleanup_after_load:
            cleanup_activations(
                dataset_path=dataset_path,
                model_name=activation_config.model_name,
                layer=activation_config.layer,
            )

    # Apply attention mask to zero out padding positions
    # This matches models-under-pressure Activation.__post_init__ behavior
    dataset = _apply_attention_mask_to_activations(dataset)

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
