"""Activation reduction strategies for converting sequence-level to sample-level representations.

This module provides a registry of reduction functions that aggregate activations
across the sequence dimension while respecting attention masks. Reductions are
applied in batches for memory efficiency.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
import tqdm

# Device detection
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


@runtime_checkable
class ReductionFunction(Protocol):
    """Protocol for reduction functions that aggregate sequence dimension.

    A reduction function takes activations and attention mask and returns
    aggregated activations with sequence dimension removed.
    """

    def __call__(
        self,
        activations: torch.Tensor,  # (batch, seq_len, hidden_dim)
        attention_mask: torch.Tensor,  # (batch, seq_len)
    ) -> torch.Tensor:  # (batch, hidden_dim)
        """Reduce sequence dimension to produce sample-level representation."""
        ...


# Registry of built-in reduction strategies
_REDUCTION_REGISTRY: dict[str, ReductionFunction] = {}


def register_reduction(name: str) -> Callable[[ReductionFunction], ReductionFunction]:
    """Decorator to register a reduction function.

    Args:
        name: Name to register the reduction function under

    Returns:
        Decorator function

    Example:
        @register_reduction("custom_mean")
        def my_custom_mean(activations, attention_mask):
            return activations.mean(dim=1)
    """

    def decorator(fn: ReductionFunction) -> ReductionFunction:
        _REDUCTION_REGISTRY[name] = fn
        return fn

    return decorator


def get_reduction_function(name: str) -> ReductionFunction:
    """Get a registered reduction function by name.

    Args:
        name: Name of the reduction strategy

    Returns:
        Reduction function

    Raises:
        ValueError: If reduction strategy is not registered
    """
    if name not in _REDUCTION_REGISTRY:
        raise ValueError(f"Unknown reduction strategy: {name}. Available: {list(_REDUCTION_REGISTRY.keys())}")
    return _REDUCTION_REGISTRY[name]


# Built-in reduction strategies


@register_reduction("mean")
def reduce_mean(
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average over sequence dimension, respecting attention mask.

    Computes the mean of activations for each sample, but only over
    the non-masked (valid) tokens.

    Args:
        activations: Input activations, shape (batch, seq_len, hidden_dim)
        attention_mask: Binary mask for valid tokens, shape (batch, seq_len)

    Returns:
        Mean-reduced activations, shape (batch, hidden_dim)
    """
    masked_acts = activations * attention_mask.unsqueeze(-1)
    sum_acts = masked_acts.sum(dim=1)  # (batch, hidden_dim)
    lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid div by zero
    return sum_acts / lengths


@register_reduction("max")
def reduce_max(
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Max pooling over sequence dimension, respecting attention mask.

    Takes the maximum value across the sequence dimension for each feature,
    ignoring masked positions.

    Args:
        activations: Input activations, shape (batch, seq_len, hidden_dim)
        attention_mask: Binary mask for valid tokens, shape (batch, seq_len)

    Returns:
        Max-pooled activations, shape (batch, hidden_dim)
    """
    # Set masked positions to -inf so they don't affect max
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(activations)
    masked_acts = torch.where(
        mask_expanded.bool(),
        activations,
        torch.tensor(float("-inf"), dtype=activations.dtype, device=activations.device),
    )
    return masked_acts.max(dim=1)[0]  # [0] gets values, [1] would be indices


@register_reduction("last")
def reduce_last(
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Take the last token with non-zero activations for each sequence.

    Finds the last position with non-zero activations rather than relying
    on the attention mask, which may not accurately reflect activation boundaries
    due to batch processing in the upstream activation computation.

    Args:
        activations: Input activations, shape (batch, seq_len, hidden_dim)
        attention_mask: Binary mask for valid tokens, shape (batch, seq_len)
            Note: This parameter is kept for API compatibility but not used.

    Returns:
        Last-token activations, shape (batch, hidden_dim)
    """
    # Find positions with non-zero activations (sum of absolute values > 0)
    nonzero_mask = activations.abs().sum(dim=-1) > 0  # (batch, seq_len)
    seq_len = activations.size(1)

    # Find last non-zero position by flipping and using argmax
    # argmax returns first occurrence, so on flipped tensor it finds last non-zero
    flipped = nonzero_mask.flip(dims=[1])
    last_nonzero_idx = seq_len - 1 - flipped.long().argmax(dim=1)

    batch_indices = torch.arange(activations.size(0), device=activations.device)
    return activations[batch_indices, last_nonzero_idx]  # (batch, hidden_dim)


@register_reduction("first")
def reduce_first(
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Take the first token for each sequence.

    Extracts the representation of the first token in each sequence.
    Commonly used for classification tasks.

    Args:
        activations: Input activations, shape (batch, seq_len, hidden_dim)
        attention_mask: Binary mask for valid tokens, shape (batch, seq_len)

    Returns:
        First-token activations, shape (batch, hidden_dim)
    """
    return activations[:, 0, :]  # (batch, hidden_dim)


def apply_reduction_batched(
    activations: torch.Tensor,  # (n_samples, seq_len, hidden_dim)
    attention_mask: torch.Tensor,  # (n_samples, seq_len)
    reduction_fn: ReductionFunction,
    batch_size: int = 256,
    device_override: torch.device | None = None,
    show_progress: bool = True,
) -> torch.Tensor:  # (n_samples, hidden_dim)
    """Apply reduction function in batches for memory efficiency.

    Processes a large dataset of sequence activations in batches, applying
    a reduction function to each batch. This prevents OOM errors when working
    with large datasets or GPUs with limited memory.

    Args:
        activations: Raw activations with sequence dimension
        attention_mask: Mask indicating valid tokens
        reduction_fn: Function to apply for reduction
        batch_size: Number of samples to process at once
        device_override: Device to use (defaults to auto-detected device)
        show_progress: Whether to show tqdm progress bar

    Returns:
        Reduced activations without sequence dimension
    """
    target_device = device_override or device
    n_samples = activations.shape[0]
    reduced_acts = []

    iterator = range(0, n_samples, batch_size)
    if show_progress:
        iterator = tqdm.tqdm(iterator, desc="Reducing activations")

    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, n_samples)

        # Move batch to device
        batch_acts = activations[start_idx:end_idx].to(target_device)
        batch_mask = attention_mask[start_idx:end_idx].to(target_device)

        # Apply reduction
        reduced_batch = reduction_fn(batch_acts, batch_mask)

        # Move back to CPU to save GPU memory
        reduced_acts.append(reduced_batch.cpu())

    return torch.cat(reduced_acts, dim=0)
