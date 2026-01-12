from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from models_under_pressure.baselines.continuation import LikelihoodContinuationBaseline, likelihood_continuation_prompts
from models_under_pressure.config import LOCAL_MODELS
from models_under_pressure.experiments.monitoring_cascade import get_abbreviated_model_name, get_model_baseline_prompt
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.model import LLMModel

from reliable_monitoring.probes import Probe


@dataclass
class CascadePredictionResults:
    """Organises the results of running a cascade of a probe and a baseline model."""

    probe_scores: np.ndarray
    baseline_scores: np.ndarray
    used_baseline: np.ndarray  # Boolean array indicating where the baseline was used
    final_scores: np.ndarray


@runtime_checkable
class SelectionStrategy(Protocol):
    """Protocol for selection strategies that determine which examples go to baseline.

    A selection strategy takes probe scores and returns a boolean mask indicating
    which examples should be sent to the baseline model.
    """

    def __call__(self, probe_scores: np.ndarray, **kwargs) -> np.ndarray:
        """Select examples to send to baseline.

        Args:
            probe_scores: Array of probe scores for all examples
            **kwargs: Strategy-specific parameters (threshold, rate, amount, etc.)

        Returns:
            Boolean array indicating which examples to send to baseline
        """
        ...


# Registry of selection strategies
_SELECTION_REGISTRY: dict[str, SelectionStrategy] = {}


def register_selection_strategy(name: str):
    """Decorator to register a selection strategy.

    Args:
        name: Name to register the selection strategy under

    Returns:
        Decorator function
    """

    def decorator(fn: SelectionStrategy) -> SelectionStrategy:
        _SELECTION_REGISTRY[name] = fn
        return fn

    return decorator


def get_selection_strategy(name: str) -> SelectionStrategy:
    """Get a registered selection strategy by name.

    Args:
        name: Name of the selection strategy

    Returns:
        Selection strategy function

    Raises:
        ValueError: If strategy is not registered
    """
    if name not in _SELECTION_REGISTRY:
        raise ValueError(f"Unknown selection strategy: {name}. Available: {list(_SELECTION_REGISTRY.keys())}")
    return _SELECTION_REGISTRY[name]


# Built-in selection strategies


@register_selection_strategy("fixed_threshold")
def select_fixed_threshold(probe_scores: np.ndarray, threshold: float, **kwargs) -> np.ndarray:
    """Send examples where probe is uncertain (score between threshold and 1-threshold).

    Args:
        probe_scores: Array of probe scores for all examples
        threshold: Threshold value; sends examples where probe score is between
                   threshold and 1-threshold (i.e., uncertain examples)

    Returns:
        Boolean array indicating which examples to send to baseline
    """
    if threshold is None:
        raise ValueError("fixed_threshold strategy requires 'threshold' parameter")
    if not (0 < threshold <= 0.5):
        raise ValueError(f"threshold must be in (0, 0.5], got {threshold}")
    return np.logical_and(probe_scores < threshold, 1 - probe_scores < threshold)


@register_selection_strategy("fixed_budget_rate")
def select_fixed_budget_rate(probe_scores: np.ndarray, rate: float, **kwargs) -> np.ndarray:
    """Send middle portion of examples based on percentiles.

    Args:
        probe_scores: Array of probe scores for all examples
        rate: Fraction of examples to send (e.g., 0.5 sends middle 50%)

    Returns:
        Boolean array indicating which examples to send to baseline
    """
    if rate is None:
        raise ValueError("fixed_budget_rate strategy requires 'rate' parameter")
    if not (0 < rate <= 1):
        raise ValueError(f"rate must be in (0, 1], got {rate}")

    lower_percentile = (0.5 - rate / 2) * 100
    upper_percentile = (0.5 + rate / 2) * 100
    lower_bound = np.percentile(probe_scores, lower_percentile)
    upper_bound = np.percentile(probe_scores, upper_percentile)
    return (probe_scores >= lower_bound) & (probe_scores <= upper_bound)


@register_selection_strategy("fixed_budget_amount")
def select_fixed_budget_amount(probe_scores: np.ndarray, amount: int, **kwargs) -> np.ndarray:
    """Send fixed number of examples centered around median.

    Args:
        probe_scores: Array of probe scores for all examples
        amount: Number of examples to send to baseline

    Returns:
        Boolean array indicating which examples to send to baseline
    """
    if amount is None:
        raise ValueError("fixed_budget_amount strategy requires 'amount' parameter")
    if not isinstance(amount, int) or amount <= 0:
        raise ValueError(f"amount must be a positive integer, got {amount}")

    n_samples = len(probe_scores)
    sorted_indices = np.argsort(probe_scores)
    median_rank = n_samples // 2

    # Compute range centered on median
    half_amount = amount // 2
    start_rank = max(0, median_rank - half_amount)
    end_rank = min(n_samples, median_rank + half_amount)

    # Create boolean mask
    mask = np.zeros(n_samples, dtype=bool)
    selected_indices = sorted_indices[start_rank:end_rank]
    mask[selected_indices] = True

    return mask


def select_examples_for_baseline(
    probe_scores: np.ndarray,
    strategy: str | SelectionStrategy = "fixed_threshold",
    **kwargs,
) -> np.ndarray:
    """Apply a selection strategy to determine which examples go to baseline.

    Args:
        probe_scores: Probe scores for all examples
        strategy: Strategy name (str) or custom SelectionStrategy callable
        **kwargs: Parameters for the strategy (threshold, rate, amount, etc.)

    Returns:
        Boolean array indicating which examples to send to baseline

    Raises:
        TypeError: If strategy is not a string or callable
    """
    if isinstance(strategy, str):
        strategy_fn = get_selection_strategy(strategy)
    elif callable(strategy):
        strategy_fn = strategy
    else:
        raise TypeError(f"Strategy must be string or callable, got {type(strategy)}")

    return strategy_fn(probe_scores, **kwargs)


def run_llm_baseline(
    baseline_model_name: str,
    dataset: LabelledDataset,
    baseline_batch_size: int = 16,
) -> np.ndarray:
    """
    Run the baseline LLM model on the given dataset and return the high-stakes probabilities.
    Args:
        baseline_model_name: The name of the baseline model to use (from the McKenzie et al. codebase).
        dataset: The dataset to run the baseline model on.
        baseline_batch_size: The batch size to use when calling the baseline model.
    Returns:
        A numpy array of high-stakes probabilities from the baseline model.
    """
    prompt_key = get_model_baseline_prompt(get_abbreviated_model_name(baseline_model_name))
    prompt_config = likelihood_continuation_prompts[prompt_key]
    model = LLMModel.load(LOCAL_MODELS[get_abbreviated_model_name(baseline_model_name)])
    baseline_model = LikelihoodContinuationBaseline(model, prompt_config=prompt_config)
    baseline_results = baseline_model.likelihood_classify_dataset(dataset, batch_size=baseline_batch_size)
    baseline_high_stakes_prob = np.array(baseline_results.other_fields["high_stakes_score"])
    return baseline_high_stakes_prob


def run_online_cascade(
    probe: Probe,
    baseline_model_name: str,
    dataset: LabelledDataset,
    selection_strategy: str | SelectionStrategy = "fixed_threshold",
    baseline_batch_size: int = 16,
    merge_strategy: str = "avg",
    **selection_kwargs,
) -> CascadePredictionResults:
    """Run a cascade online: call the baseline model only for examples that need it.

    Args:
        probe: A Probe instance that implements the Probe protocol (has .predict() method).
        baseline_model_name: The name of the baseline model to use (from the McKenzie et al. codebase).
        dataset: The dataset to run the cascade on.
        selection_strategy: Strategy for selecting which examples to send to baseline.
                           Can be a strategy name ("fixed_threshold", "fixed_budget_rate", "fixed_budget_amount")
                           or a custom SelectionStrategy callable.
        baseline_batch_size: The batch size to use when calling the baseline model.
        merge_strategy: The strategy to merge probe and baseline model scores ("avg" or "replace").
        **selection_kwargs: Strategy-specific parameters (e.g., threshold=0.5, rate=0.5, amount=500)

    Returns:
        CascadePredictionResults with probe scores, baseline scores, and final merged scores.
    """
    probe_scores = probe.predict(dataset)
    to_call_baseline = select_examples_for_baseline(probe_scores, selection_strategy, **selection_kwargs)

    # Use boolean array to select examples where we need to call the baseline
    baseline_indices = np.where(to_call_baseline)[0].tolist()
    dataset_to_call_baseline = dataset[baseline_indices]
    baseline_high_stakes_prob = run_llm_baseline(
        baseline_model_name=baseline_model_name,
        dataset=dataset_to_call_baseline,
        baseline_batch_size=baseline_batch_size,
    )

    # re-index baseline scores to match the original dataset
    # baseline_scores is nan where baseline was not called
    baseline_scores = np.full_like(probe_scores, np.nan)
    baseline_scores[to_call_baseline] = baseline_high_stakes_prob

    if merge_strategy == "avg":
        final_scores = np.where(to_call_baseline, (probe_scores + baseline_scores) / 2, probe_scores)
    elif merge_strategy == "replace":
        final_scores = np.where(to_call_baseline, baseline_scores, probe_scores)
    else:
        raise ValueError(f"Unknown merge strategy: {merge_strategy}")
    return CascadePredictionResults(
        probe_scores=probe_scores,
        baseline_scores=baseline_scores,
        used_baseline=to_call_baseline,
        final_scores=final_scores,
    )


def run_offline_cascade(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    selection_strategy: str | SelectionStrategy = "fixed_threshold",
    merge_strategy: str = "avg",
    **selection_kwargs,
) -> CascadePredictionResults:
    """Merge precomputed probe and baseline scores using the cascade logic.

    Args:
        probe_scores: Scores from the probe for all examples.
        baseline_scores: Scores from the baseline for all examples.
        selection_strategy: Strategy for selecting which examples to use baseline.
                           Can be a strategy name ("fixed_threshold", "fixed_budget_rate", "fixed_budget_amount")
                           or a custom SelectionStrategy callable.
        merge_strategy: "avg" to average probe and baseline when baseline is used,
                        "replace" to replace probe with baseline when baseline is used.
        **selection_kwargs: Strategy-specific parameters (e.g., threshold=0.5, rate=0.5, amount=500)

    Returns:
        CascadePredictionResults with merged scores.
    """
    to_call_baseline = select_examples_for_baseline(probe_scores, selection_strategy, **selection_kwargs)

    # Mask baseline_scores to only where we would have called the baseline
    masked_baseline_scores = np.full_like(probe_scores, np.nan)
    masked_baseline_scores[to_call_baseline] = baseline_scores[to_call_baseline]

    if merge_strategy == "avg":
        final_scores = np.where(to_call_baseline, (probe_scores + masked_baseline_scores) / 2, probe_scores)
    elif merge_strategy == "replace":
        final_scores = np.where(to_call_baseline, masked_baseline_scores, probe_scores)
    else:
        raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    return CascadePredictionResults(
        probe_scores=probe_scores,
        baseline_scores=masked_baseline_scores,
        used_baseline=to_call_baseline,
        final_scores=final_scores,
    )


def offline_batch_cascade(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    batch_size: int,
    selection_strategy: str | SelectionStrategy = "fixed_threshold",
    merge_strategy: str = "avg",
    **selection_kwargs,
) -> CascadePredictionResults:
    """Run offline cascade in batches.
    Depending on the selection strategy, this might give different results than the non-batched cascades
    because the selection strategy is applied independently to each batch.

    Args:
        probe_scores: Scores from the probe for all examples.
        baseline_scores: Scores from the baseline for all examples.
        batch_size: Number of examples to process in each batch.
        selection_strategy: Strategy for selecting which examples to use baseline.
                           Can be a strategy name ("fixed_threshold", "fixed_budget_rate", "fixed_budget_amount")
                           or a custom SelectionStrategy callable.
        merge_strategy: "avg" to average probe and baseline when baseline is used,
                        "replace" to replace probe with baseline when baseline is used.
        **selection_kwargs: Strategy-specific parameters (e.g., threshold=0.5, rate=0.5, amount=500)

    Returns:
        CascadePredictionResults with merged scores.
    """
    n_samples = len(probe_scores)
    all_used_baseline = np.zeros(n_samples, dtype=bool)
    all_final_scores = np.zeros(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_probe_scores = probe_scores[start_idx:end_idx]
        batch_baseline_scores = baseline_scores[start_idx:end_idx]

        batch_results = run_offline_cascade(
            probe_scores=batch_probe_scores,
            baseline_scores=batch_baseline_scores,
            selection_strategy=selection_strategy,
            merge_strategy=merge_strategy,
            **selection_kwargs,
        )

        all_used_baseline[start_idx:end_idx] = batch_results.used_baseline
        all_final_scores[start_idx:end_idx] = batch_results.final_scores

    return CascadePredictionResults(
        probe_scores=probe_scores,
        baseline_scores=baseline_scores,
        used_baseline=all_used_baseline,
        final_scores=all_final_scores,
    )
