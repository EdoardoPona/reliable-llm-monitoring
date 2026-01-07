from dataclasses import dataclass

import numpy as np
from models_under_pressure.baselines.continuation import LikelihoodContinuationBaseline, likelihood_continuation_prompts
from models_under_pressure.config import LOCAL_MODELS
from models_under_pressure.experiments.monitoring_cascade import get_abbreviated_model_name, get_model_baseline_prompt
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.model import LLMModel


@dataclass
class CascadePredictionResults:
    """Organises the results of running a cascade of a probe and a baseline model."""

    probe_scores: np.ndarray
    baseline_scores: np.ndarray
    used_baseline: np.ndarray  # Boolean array indicating where the baseline was used
    final_scores: np.ndarray


def run_cascade(
    probe: callable,  # TODO refine the type here, really we are happy with anything that returns the logits
    baseline_model_name: str,
    threshold: float,
    dataset: LabelledDataset,
    baseline_batch_size: int = 16,
    merge_strategy: str = "avg",
) -> CascadePredictionResults:
    """
    Run a cascade of a probe and a baseline model on the given dataset.
    Args:
        probe: A callable that takes in a dataset and returns probe scores (logits).
        baseline_model_name: The name of the baseline model to use (from the McKenzie et al. codebase)
        threshold: The threshold for the probe scores to decide when to use the baseline model.
        dataset: The dataset to run the cascade on.
        merge_strategy: The strategy to merge probe and baseline model scores ("avg" or "replace").
    """
    probe_scores = probe(dataset)
    to_call_baseline = np.logical_and(probe_scores < threshold, 1 - probe_scores < threshold)

    prompt_key = get_model_baseline_prompt(get_abbreviated_model_name(baseline_model_name))
    prompt_config = likelihood_continuation_prompts[prompt_key]
    model = LLMModel.load(LOCAL_MODELS[get_abbreviated_model_name(baseline_model_name)])
    baseline_model = LikelihoodContinuationBaseline(model, prompt_config=prompt_config)

    # Use boolean array to select examples where we need to call the baseline
    baseline_indices = np.where(to_call_baseline)[0].tolist()
    dataset_to_call_baseline = dataset[baseline_indices]
    baseline_results = baseline_model.likelihood_classify_dataset(
        dataset_to_call_baseline, batch_size=baseline_batch_size
    )
    baseline_high_stakes_prob = np.array(baseline_results.other_fields["high_stakes_score"])

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
