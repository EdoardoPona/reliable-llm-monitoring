"""Utilities for DV-based LTT cascade experiments.

Provides core functions for threshold-based delegation cascades with
Learn-then-Test budget guarantees, plus data loading and splitting helpers.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from activation_registry import compute_or_fetch_activations
from baseline_registry import compute_or_fetch_baseline
from mixed_dataset import (
    fetch_per_source_activations,
    fetch_per_source_baselines,
    has_mixed_config,
    load_mixed_dataset_with_baselines,
)
from sklearn.metrics import accuracy_score, roc_auc_score

from reliable_monitoring.cascade import CascadePredictionResults
from reliable_monitoring.dataset import ActivationConfig, load_dataset
from reliable_monitoring.learn_then_test import fixed_sequence_testing
from reliable_monitoring.probes import SequenceProbe
from reliable_monitoring.risks import BudgetCostRisk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core LTT / cascade functions
# ---------------------------------------------------------------------------


def ltt_budget_threshold(
    dv_scores: np.ndarray,
    alpha_budget: float,
    delta: float,
    tau_grid: np.ndarray,
) -> float | None:
    """Find smallest valid tau for budget guarantee via fixed-sequence testing.

    Hypotheses are ordered from safest (largest tau, lowest delegation rate)
    to most aggressive (smallest tau).  Uses the binomial bound from
    BudgetCostRisk.  Returns the smallest tau in the rejection chain,
    or None if no valid tau exists.
    """
    n = len(dv_scores)
    # Order from safest (largest tau) to most aggressive (smallest tau)
    ordered_taus = np.sort(tau_grid)[::-1]

    # P-values: binomial bound on empirical delegation rate at each tau
    bound_fn = BudgetCostRisk.p_value_bound_fn
    p_values = np.array([float(bound_fn(float((dv_scores > tau).mean()), n, alpha_budget)) for tau in ordered_taus])

    # Fixed-sequence testing: reject from safe end, stop at first failure
    rejected = fixed_sequence_testing(p_values, delta)
    logger.info(
        f"  LTT: alpha_budget={alpha_budget:.2f}, delta={delta:.2f}, rejected {len(rejected)}/{len(tau_grid)} taus"
    )
    if not rejected:
        return None
    # Last rejected index = most aggressive valid tau
    return float(ordered_taus[rejected[-1]])


def threshold_cascade(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    delegation_scores: np.ndarray,
    tau: float,
) -> CascadePredictionResults:
    """Delegate examples where delegation_scores > tau.

    The delegation_scores array can be DV scores, uncertainty, or any
    other per-example signal.  Higher score = more likely to delegate.
    """
    used_baseline = delegation_scores > tau
    final = np.where(used_baseline, baseline_scores, probe_scores)
    return CascadePredictionResults(
        probe_scores=probe_scores,
        baseline_scores=baseline_scores,
        used_baseline=used_baseline,
        final_scores=final,
    )


def cascade_metrics(
    results: CascadePredictionResults,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute AUC and accuracy from cascade results."""
    return {
        "auc": float(roc_auc_score(labels, results.final_scores)),
        "accuracy": float(accuracy_score(labels, (results.final_scores >= 0.5).astype(int))),
    }


# ---------------------------------------------------------------------------
# Data loading (supports mixed and single-dataset configs)
# ---------------------------------------------------------------------------


def _load_split(
    split: str,
    config,
    activation_config: ActivationConfig,
    safety_probe: SequenceProbe,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a data split, returning (probe_scores, baseline_scores, labels, activations, groups).

    Handles both mixed-dataset and single-dataset configs.  Uses the
    activation and baseline caches to avoid recomputation.
    """
    use_modal = getattr(config, "use_modal", False)
    modal_gpu = getattr(config, "modal_gpu", None)
    reduction = config.reduction_strategy
    activation_field = f"activations_{reduction}"

    if has_mixed_config(config):
        mixed_cfg = config.mixed_datasets
        sources = mixed_cfg["sources"]
        balance = mixed_cfg.get("balance_strategy", "min_size")

        logger.info(f"Fetching cached baselines for {split}...")
        per_source_bl = fetch_per_source_baselines(
            sources, split, config.baseline_model_name, local=not use_modal, gpu=modal_gpu
        )

        logger.info(f"Fetching cached activations for {split}...")
        per_source_acts = fetch_per_source_activations(
            sources,
            split,
            activation_config.model_name,
            activation_config.layer,
            reduction,
            local=not use_modal,
            gpu=modal_gpu,
        )

        logger.info(f"Loading {split} datasets with cached activations and baselines...")
        dataset, baseline_scores = load_mixed_dataset_with_baselines(
            sources,
            split,
            per_source_bl,
            activation_config=None,
            balance_strategy=balance,
            seed=config.seed,
            per_source_activations=per_source_acts,
            activation_field_name=activation_field,
        )
        groups = np.array(dataset.other_fields["group"]) if "group" in dataset.other_fields else None
    else:
        # Single-dataset mode
        path_attr = f"{split}_dataset_path"
        path = Path(getattr(config, path_attr))
        logger.info(f"Loading {split} dataset from {path}...")
        dataset = load_dataset(path, activation_config=None)

        baseline_scores = compute_or_fetch_baseline(
            model_name=config.baseline_model_name,
            dataset=dataset,
            dataset_path=str(path),
            local=not use_modal,
            gpu=modal_gpu,
        )

        acts = compute_or_fetch_activations(
            model_name=activation_config.model_name,
            layer=activation_config.layer,
            reduction=reduction,
            dataset=dataset,
            dataset_path=str(path),
            local=not use_modal,
            gpu=modal_gpu,
        )
        dataset = dataset.assign(**{activation_field: acts})
        groups = None

    logger.info(f"Computing safety probe scores on {split}...")
    probe_scores = safety_probe.predict(dataset)
    labels = dataset.labels_numpy()

    X = dataset.other_fields[activation_field]
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    X = np.asarray(X)

    logger.info(f"  {split}: n={len(labels)}, activations={X.shape}")
    if groups is not None:
        for g in np.unique(groups):
            logger.info(f"    {g}: n={int((groups == g).sum())}")

    return probe_scores, baseline_scores, labels, X, groups


def split_calib_eval(
    *arrays: np.ndarray | None,
    calib_fraction: float,
    seed: int,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """Split arrays into calibration and evaluation subsets.

    Returns (calib_arrays, eval_arrays) with the same order as input.
    None arrays are passed through as None.
    """
    # Determine n from the first non-None array
    n = next(len(a) for a in arrays if a is not None)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_calib = int(n * calib_fraction)
    calib_idx, eval_idx = perm[:n_calib], perm[n_calib:]

    calib_out = [a[calib_idx] if a is not None else None for a in arrays]
    eval_out = [a[eval_idx] if a is not None else None for a in arrays]
    return calib_out, eval_out
