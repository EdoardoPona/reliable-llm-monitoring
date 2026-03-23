"""Utilities for DV-based LTT cascade experiments.

Provides core functions for threshold-based delegation cascades with
Learn-then-Test budget guarantees, plus data loading, splitting, and
shared data-preparation helpers.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from activation_registry import compute_or_fetch_activations
from baseline_registry import compute_or_fetch_baseline
from delegation_value_probe import (
    compute_continuous_delegation_value,
    compute_delegation_value,
    predict_dv_scores,
    train_dv_probe,
)
from mixed_dataset import (
    fetch_per_source_activations,
    fetch_per_source_baselines,
    has_mixed_config,
    load_mixed_dataset_with_baselines,
)
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.linear_model._ridge import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score

from reliable_monitoring.cascade import CascadePredictionResults
from reliable_monitoring.dataset import ActivationConfig, load_dataset
from reliable_monitoring.learn_then_test import (
    ParetoFilterResult,
    fixed_sequence_testing,
    pareto_filter_thresholds,
)
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
    merge_strategy: str = "replace",
) -> CascadePredictionResults:
    """Delegate examples where delegation_scores > tau.

    The delegation_scores array can be DV scores, uncertainty, or any
    other per-example signal.  Higher score = more likely to delegate.
    """
    used_baseline = delegation_scores > tau
    if merge_strategy == "replace":
        final = np.where(used_baseline, baseline_scores, probe_scores)
    elif merge_strategy == "avg":
        final = np.where(used_baseline, (probe_scores + baseline_scores) / 2, probe_scores)
    else:
        raise ValueError(f"Unknown merge strategy: {merge_strategy}")
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
# Probe training
# ---------------------------------------------------------------------------


def train_probes(
    config,
    activation_config: ActivationConfig,
) -> tuple[SequenceProbe, LogisticRegression | Ridge, str]:
    """Train the safety probe (on train split) and DV probe (on dev split).

    Returns:
        (safety_probe, dv_clf, dv_target) where dv_target is "binary" or "continuous".
    """
    use_modal = getattr(config, "use_modal", False)
    modal_gpu = getattr(config, "modal_gpu", None)
    reduction = config.reduction_strategy

    logger.info("Loading training data and fitting safety probe...")
    train_dataset = load_dataset(Path(config.train_dataset_path), activation_config=None)
    train_acts = compute_or_fetch_activations(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
        reduction=reduction,
        dataset=train_dataset,
        dataset_path=config.train_dataset_path,
        local=not use_modal,
        gpu=modal_gpu,
    )
    train_dataset = train_dataset.assign(**{f"activations_{reduction}": train_acts})
    safety_probe = SequenceProbe(reduction_strategy=reduction)
    safety_probe.fit(train_dataset)
    del train_dataset, train_acts

    logger.info("Loading dev split for DV probe training...")
    dev_ps, dev_bs, dev_labels, X_dev, dev_groups = _load_split(
        "dev",
        config,
        activation_config,
        safety_probe,
    )
    dv_target = config.dv_target
    logger.info(f"DV target mode: {dv_target}")

    if dv_target == "continuous":
        v_dev = compute_continuous_delegation_value(dev_ps, dev_bs, dev_labels)
        logger.info(
            f"Dev delegation value: v>0 rate={float((v_dev > 0).mean()):.1%}, "
            f"mean={float(v_dev.mean()):.3f} (n={len(v_dev)})"
        )
    else:
        v_dev = compute_delegation_value(dev_ps, dev_bs, dev_labels).astype(float)
        logger.info(f"Dev delegation value: v=1 rate={v_dev.mean():.1%} (n={len(v_dev)})")

    logger.info("Training DV probe on dev split...")
    dv_clf: LogisticRegression | Ridge = train_dv_probe(X_dev, v_dev, mode=dv_target)

    return safety_probe, dv_clf, dv_target


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


# ---------------------------------------------------------------------------
# Pareto filtering for DV cascades (step 2)
# ---------------------------------------------------------------------------


def pareto_ht_opt_split(
    n: int,
    proportion: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into hypothesis-testing and optimisation subsets.

    Ensures a consistent, reproducible split across all DV cascade
    experiments (``dv_cascade_comparison``, ``dv_sgt_cascade``,
    ``ltt_coverage_validation``).

    Args:
        n: Total number of calibration examples.
        proportion: Fraction of data for the optimisation subset.
        seed: RNG seed.

    Returns:
        ``(ht_idx, opt_idx)`` index arrays.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_opt = int(n * proportion)
    return perm[n_opt:], perm[:n_opt]


def pareto_filter_dv_thresholds(
    calib_ps: np.ndarray,
    calib_bs: np.ndarray,
    calib_labels: np.ndarray,
    calib_dv: np.ndarray,
    tau_grid: np.ndarray,
    *,
    risks: list,
    merge_strategy: str = "replace",
    pareto_proportion: float = 0.3,
    seed: int = 42,
) -> tuple[ParetoFilterResult, np.ndarray, np.ndarray]:
    """Split calib into ht/opt and Pareto-filter DV thresholds.

    This is the shared Pareto filtering step for DV cascade experiments.
    Splits calibration data into hypothesis-testing and optimisation
    subsets, evaluates the given risks on the opt subset, and returns
    the Pareto-efficient thresholds.

    Args:
        calib_ps: Probe scores on the calibration set.
        calib_bs: Baseline scores on the calibration set.
        calib_labels: Ground-truth labels on the calibration set.
        calib_dv: DV delegation scores on the calibration set.
        tau_grid: Full grid of candidate thresholds.
        risks: List of Risk objects to Pareto-filter on.
        merge_strategy: Cascade merge strategy.
        pareto_proportion: Fraction of calib data for the opt split.
        seed: RNG seed for the ht/opt split.

    Returns:
        ``(pareto_result, ht_idx, opt_idx)`` where ``pareto_result``
        contains the filtered thresholds and ``ht_idx`` / ``opt_idx``
        are the indices into the original calib arrays.
    """
    ht_idx, opt_idx = pareto_ht_opt_split(len(calib_dv), pareto_proportion, seed)

    pf = pareto_filter_thresholds(
        calib_ps[opt_idx],
        calib_bs[opt_idx],
        calib_labels[opt_idx],
        calib_dv[opt_idx],
        tau_grid,
        risks=risks,
        merge_strategy=merge_strategy,
    )

    logger.info(
        f"Pareto filter: {len(pf.taus)}/{pf.n_original} thresholds retained (ht={len(ht_idx)}, opt={len(opt_idx)})"
    )

    return pf, ht_idx, opt_idx


# ---------------------------------------------------------------------------
# Shared data preparation pipeline (step 1)
# ---------------------------------------------------------------------------


@dataclass
class DVCascadeData:
    """Pre-computed scores for DV cascade experiments.

    Bundles the outputs of probe training, test-set scoring, and DV probe evaluation into a single object
    """

    safety_probe: SequenceProbe
    dv_clf: LogisticRegression | Ridge
    dv_target: str  # "binary" or "continuous"
    test_ps: np.ndarray  # probe scores on test
    test_bs: np.ndarray  # baseline scores on test
    test_labels: np.ndarray
    test_groups: np.ndarray | None
    dv_scores: np.ndarray  # DV probe predictions on test
    v_test: np.ndarray  # delegation value (continuous or binary float)
    dv_auc: float  # DV probe AUC (vs binarised v)
    dv_tau_grid: np.ndarray  # threshold grid adapted to score range


def prepare_dv_cascade_data(config, *, tau_steps: int | None = None) -> DVCascadeData:
    """Train probes, load test data, compute DV scores and tau grid.

    Args:
        config: Experiment configuration (from ``load_config``).
        tau_steps: Number of candidate DV thresholds.  Defaults to
            ``config.tau_steps`` (with fallback 30).

    Returns:
        A :class:`DVCascadeData` with all pre-computed arrays.
    """
    activation_config = ActivationConfig(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
    )

    # --- Train probes ---
    safety_probe, dv_clf, dv_target = train_probes(config, activation_config)

    # --- Load test split and compute scores ---
    logger.info("Loading test split...")
    test_ps, test_bs, test_labels, X_test, test_groups = _load_split("test", config, activation_config, safety_probe)

    # --- Delegation value ---
    if dv_target == "continuous":
        v_test = compute_continuous_delegation_value(test_ps, test_bs, test_labels)
        logger.info(
            f"Test delegation value: v>0 rate={float((v_test > 0).mean()):.1%}, "
            f"mean={float(v_test.mean()):.3f} (n={len(v_test)})"
        )
    else:
        v_test = compute_delegation_value(test_ps, test_bs, test_labels).astype(float)
        logger.info(f"Test delegation value: v=1 rate={v_test.mean():.1%} (n={len(v_test)})")

    # --- DV scores ---
    dv_scores = predict_dv_scores(dv_clf, X_test, mode=dv_target)
    del X_test

    v_test_binary = (v_test > 0).astype(int) if dv_target == "continuous" else v_test.astype(int)
    dv_auc = float(roc_auc_score(v_test_binary, dv_scores))
    logger.info(f"DV probe AUC on test: {dv_auc:.4f}")

    # --- DV threshold grid ---
    if tau_steps is None:
        tau_steps = config.tau_steps
    if dv_target == "continuous":
        dv_tau_grid = np.linspace(
            float(np.min(dv_scores)) - 0.01,
            float(np.max(dv_scores)) + 0.01,
            tau_steps,
        )
    else:
        dv_tau_grid = np.linspace(0.0, 1.0, tau_steps)

    return DVCascadeData(
        safety_probe=safety_probe,
        dv_clf=dv_clf,
        dv_target=dv_target,
        test_ps=test_ps,
        test_bs=test_bs,
        test_labels=test_labels,
        test_groups=test_groups,
        dv_scores=dv_scores,
        v_test=v_test,
        dv_auc=dv_auc,
        dv_tau_grid=dv_tau_grid,
    )
