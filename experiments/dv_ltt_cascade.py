"""Experiment 2: LTT threshold delegation with DV probe.

Trains a DV probe and uses LTT to find valid delegation thresholds at
multiple budget levels.  Compares DV-threshold cascade against fixed-k
uncertainty-ranked baseline at the same budget constraint.

Pipeline:

1. Train safety probe on training data.
2. Train DV probe on dev split (using delegation value as target).
3. Split test data into calib (LTT threshold selection) and eval (evaluation).
4. For each alpha_budget level, use LTT (binomial bound) to find the
   smallest valid tau on the calibration set.
5. Evaluate DV-threshold cascade vs fixed-k uncertainty on the eval set.
6. Generate figures and optionally log to ClearML.

See ``experiments/notes/delegation_value.md`` for theory (Exp 2).

Usage::

    uv run experiments/dv_ltt_cascade.py --config configs/dv_ltt_cascade.yaml
    uv run experiments/dv_ltt_cascade.py --config configs/dv_ltt_cascade.yaml --use-clearml
"""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from baseline_registry import compute_or_fetch_baseline
from config import load_config
from delegation_value_probe import compute_delegation_value, fetch_per_source_baselines, train_dv_probe
from dotenv import load_dotenv
from mixed_dataset import has_mixed_config, load_mixed_dataset_with_baselines
from sklearn.metrics import accuracy_score, roc_auc_score

from reliable_monitoring.cascade import CascadePredictionResults, offline_batch_cascade, run_offline_cascade
from reliable_monitoring.dataset import ActivationConfig, load_dataset, reduce_activations
from reliable_monitoring.learn_then_test import fixed_sequence_testing
from reliable_monitoring.probes import SequenceProbe
from reliable_monitoring.risks import BudgetCostRisk

load_dotenv()

logging.basicConfig(level=logging.INFO)
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

    Handles both mixed-dataset and single-dataset configs.
    """
    if has_mixed_config(config):
        mixed_cfg = config.mixed_datasets
        sources = mixed_cfg["sources"]
        balance = mixed_cfg.get("balance_strategy", "min_size")

        logger.info(f"Fetching cached baselines for {split}...")
        per_source_bl = fetch_per_source_baselines(sources, split, config.baseline_model_name)

        logger.info(f"Loading {split} datasets with activations...")
        dataset, baseline_scores = load_mixed_dataset_with_baselines(
            sources,
            split,
            per_source_bl,
            activation_config,
            balance,
            config.seed,
        )
        groups = np.array(dataset.other_fields["group"]) if "group" in dataset.other_fields else None
    else:
        # Single-dataset mode
        path_attr = f"{split}_dataset_path"
        path = Path(getattr(config, path_attr))
        logger.info(f"Loading {split} dataset from {path}...")
        dataset = load_dataset(path, activation_config)
        baseline_scores = compute_or_fetch_baseline(
            model_name=config.baseline_model_name,
            dataset=dataset,
            dataset_path=str(path),
        )
        groups = None

    logger.info(f"Computing safety probe scores on {split}...")
    probe_scores = safety_probe.predict(dataset)
    labels = dataset.labels_numpy()

    logger.info(f"Extracting mean-pooled activations for {split}...")
    reduced = reduce_activations(dataset, "mean")
    X = reduced.other_fields["activations_mean"]
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
# Figures
# ---------------------------------------------------------------------------


def plot_budget_and_performance(
    alpha_budgets: np.ndarray,
    results: list[dict],
    probe_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    output_dir: Path,
) -> plt.Figure:
    """Budget control + performance comparison across alpha_budget levels.

    Top row: target vs realized budget.
    Bottom rows: AUC and accuracy comparison.
    """
    valid = [r for r in results if r["valid"]]
    if not valid:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid thresholds found", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_dir / "budget_and_performance.png", dpi=150)
        return fig

    alphas = np.array([r["alpha_budget"] for r in valid])
    realized = np.array([r["dv_budget"] for r in valid])
    dv_auc = np.array([r["dv_metrics"]["auc"] for r in valid])
    dv_acc = np.array([r["dv_metrics"]["accuracy"] for r in valid])
    unc_auc = np.array([r["unc_metrics"]["auc"] for r in valid])
    unc_acc = np.array([r["unc_metrics"]["accuracy"] for r in valid])

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Budget control
    ax = axes[0]
    ax.plot(alphas, realized, "o-", label="DV realized", color="C0")
    ax.plot(alphas, alphas, "k--", alpha=0.4, label=r"$\alpha_{\mathrm{budget}}$ = realized")
    ax.fill_between(alphas, 0, alphas, alpha=0.08, color="green", label="Valid region")
    ax.set_ylabel("Delegation rate")
    ax.set_title("Budget control")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # AUC comparison
    ax = axes[1]
    ax.plot(alphas, dv_auc, "o-", label="DV threshold", color="C0")
    ax.plot(alphas, unc_auc, "s-", label="Fixed-k uncertainty", color="C1")
    ax.axhline(probe_metrics["auc"], color="gray", ls=":", alpha=0.5, label=f"Probe only ({probe_metrics['auc']:.3f})")
    ax.axhline(
        baseline_metrics["auc"],
        color="gray",
        ls="--",
        alpha=0.5,
        label=f"Baseline only ({baseline_metrics['auc']:.3f})",
    )
    ax.set_ylabel("Cascade ROC AUC")
    ax.set_title("Performance comparison (same budget constraint)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Accuracy comparison
    ax = axes[2]
    ax.plot(alphas, dv_acc, "o-", label="DV threshold", color="C0")
    ax.plot(alphas, unc_acc, "s-", label="Fixed-k uncertainty", color="C1")
    ax.axhline(
        probe_metrics["accuracy"],
        color="gray",
        ls=":",
        alpha=0.5,
        label=f"Probe only ({probe_metrics['accuracy']:.3f})",
    )
    ax.axhline(
        baseline_metrics["accuracy"],
        color="gray",
        ls="--",
        alpha=0.5,
        label=f"Baseline only ({baseline_metrics['accuracy']:.3f})",
    )
    ax.set_ylabel("Cascade Accuracy")
    ax.set_xlabel(r"Budget constraint $\alpha_{\mathrm{budget}}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "budget_and_performance.png", dpi=150)
    return fig


def plot_adaptivity(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    dv_scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    tau: float,
    alpha_budget: float,
    batch_size: int,
    output_dir: Path,
    baseline_cascade_fn: Callable[[np.ndarray, np.ndarray, float], CascadePredictionResults] | None = None,
) -> plt.Figure:
    """Per-batch delegation rate histogram for DV threshold vs fixed-k."""
    dv_result = threshold_cascade(probe_scores, baseline_scores, dv_scores, tau)
    if baseline_cascade_fn is None:
        unc_result = offline_batch_cascade(
            probe_scores,
            baseline_scores,
            batch_size,
            selection_strategy="fixed_budget_rate",
            merge_strategy="replace",
            rate=alpha_budget,
        )
    else:
        unc_result = baseline_cascade_fn(probe_scores, baseline_scores, alpha_budget)

    n = len(probe_scores)
    n_batches = n // batch_size
    n_used = n_batches * batch_size

    dv_batch_rates = []
    unc_batch_rates = []
    for i in range(n_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        dv_batch_rates.append(float(dv_result.used_baseline[s:e].mean()))
        unc_batch_rates.append(float(unc_result.used_baseline[s:e].mean()))

    n_cols = 2 if groups is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Histogram of per-batch delegation rates
    ax = axes[0]
    bins = np.linspace(0, max(max(dv_batch_rates), max(unc_batch_rates)) * 1.2, 20)
    ax.hist(dv_batch_rates, bins=bins, alpha=0.6, label="DV threshold", color="C0", edgecolor="black")
    ax.axvline(np.mean(dv_batch_rates), color="C0", ls="--", label=f"DV mean ({np.mean(dv_batch_rates):.1%})")
    ax.axvline(alpha_budget, color="red", ls=":", lw=2, label=f"Budget constraint ({alpha_budget:.0%})")
    ax.axvline(np.mean(unc_batch_rates), color="C1", ls="--", label=f"Fixed-k ({np.mean(unc_batch_rates):.1%})")
    ax.set_xlabel("Per-batch delegation rate")
    ax.set_ylabel("Count")
    ax.set_title(rf"Adaptivity at $\alpha_{{\mathrm{{budget}}}}$ = {alpha_budget:.0%}")
    ax.legend(fontsize=7)

    # Per-group delegation rates
    if groups is not None:
        ax = axes[1]
        unique_groups = np.unique(groups[:n_used])
        dv_group_rates = [float(dv_result.used_baseline[:n_used][groups[:n_used] == g].mean()) for g in unique_groups]
        unc_group_rates = [float(unc_result.used_baseline[:n_used][groups[:n_used] == g].mean()) for g in unique_groups]
        x = np.arange(len(unique_groups))
        w = 0.35
        ax.bar(x - w / 2, dv_group_rates, w, label="DV threshold", color="C0", alpha=0.7, edgecolor="black")
        ax.bar(x + w / 2, unc_group_rates, w, label="Fixed-k uncertainty", color="C1", alpha=0.7, edgecolor="black")
        ax.axhline(alpha_budget, color="red", ls=":", lw=2, label=f"Budget ({alpha_budget:.0%})")
        ax.set_xticks(x)
        ax.set_xticklabels(unique_groups)
        ax.set_ylabel("Delegation rate")
        ax.set_title("Per-group delegation rates")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "adaptivity.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="DV LTT Cascade (Experiment 2)")
    parser.add_argument("--config", type=str, default="configs/dv_ltt_cascade.yaml")
    parser.add_argument("--output-dir", type=str, default="results/dv_ltt_cascade")
    parser.add_argument("--use-clearml", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    seed = config.seed

    # --- ClearML init ---
    clearml_logger = None
    if args.use_clearml:
        import os

        from clearml_logger import ClearMLLogger

        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="dv_ltt_cascade",
            enabled=True,
        )
        clearml_logger.add_tags(["dv-ltt-cascade", "experiment-2"])
        clearml_logger.connect_configuration(
            {
                "config_path": args.config,
                "baseline_model": config.baseline_model_name,
                "activations_model": config.activations_model_name,
                "activations_layer": config.activations_layer,
            }
        )

    activation_config = ActivationConfig(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
    )

    # --- Train safety probe ---
    logger.info("Loading training data and fitting safety probe...")
    train_dataset = load_dataset(Path(config.train_dataset_path), activation_config)
    safety_probe = SequenceProbe(reduction_strategy=config.reduction_strategy)
    safety_probe.fit(train_dataset)
    del train_dataset

    # --- Load dev split and train DV probe ---
    logger.info("Loading dev split for DV probe training...")
    dev_ps, dev_bs, dev_labels, X_dev, dev_groups = _load_split(
        "dev",
        config,
        activation_config,
        safety_probe,
    )
    v_dev = compute_delegation_value(dev_ps, dev_bs, dev_labels)
    logger.info(f"Dev delegation value: v=1 rate={v_dev.mean():.1%} (n={len(v_dev)})")

    logger.info("Training DV probe on dev split...")
    dv_clf = train_dv_probe(X_dev, v_dev)
    del X_dev

    # --- Load test split (will be split into calib + eval) ---
    logger.info("Loading test split...")
    test_ps, test_bs, test_labels, X_test, test_groups = _load_split(
        "test",
        config,
        activation_config,
        safety_probe,
    )
    v_test = compute_delegation_value(test_ps, test_bs, test_labels)
    logger.info(f"Test delegation value: v=1 rate={v_test.mean():.1%} (n={len(v_test)})")

    # Compute DV scores on test (out-of-sample for DV probe)
    dv_scores_test = dv_clf.predict_proba(X_test)[:, 1]
    del X_test

    dv_auc = roc_auc_score(v_test, dv_scores_test)
    logger.info(f"DV probe AUC on test: {dv_auc:.4f}")

    # --- Split test into calib and eval ---
    calib_fraction = getattr(config, "calib_fraction", 0.5)
    calib_arrays, eval_arrays = split_calib_eval(
        test_ps,
        test_bs,
        test_labels,
        dv_scores_test,
        v_test,
        test_groups,
        calib_fraction=calib_fraction,
        seed=seed,
    )
    calib_ps, calib_bs, calib_labels, calib_dv, calib_v, calib_groups = calib_arrays
    eval_ps, eval_bs, eval_labels, eval_dv, eval_v, eval_groups = eval_arrays
    # Narrow types: only groups can be None
    assert calib_ps is not None and calib_bs is not None and calib_labels is not None
    assert calib_dv is not None and calib_v is not None
    assert eval_ps is not None and eval_bs is not None and eval_labels is not None
    assert eval_dv is not None and eval_v is not None
    logger.info(f"Calib: n={len(calib_labels)}, Eval: n={len(eval_labels)}")

    # --- Baseline cascade mode ---
    baseline_global_topk = getattr(config, "baseline_global_topk", False)
    batch_size = getattr(config, "cascade_batch_size", 128)

    def run_baseline_cascade(ps: np.ndarray, bs: np.ndarray, budget: float) -> CascadePredictionResults:
        if baseline_global_topk:
            return run_offline_cascade(
                ps, bs, selection_strategy="fixed_budget_rate", merge_strategy="replace", rate=budget
            )
        return offline_batch_cascade(
            ps, bs, batch_size, selection_strategy="fixed_budget_rate", merge_strategy="replace", rate=budget
        )

    baseline_label = "Global top-k uncertainty" if baseline_global_topk else "Batched top-k uncertainty"
    logger.info(f"Baseline mode: {baseline_label} (batch_size={batch_size})")

    # --- LTT: find valid tau at each alpha_budget ---
    alpha_budgets = np.linspace(config.alpha_budget_start, config.alpha_budget_end, config.alpha_budget_steps)
    delta = 1.0 - config.guarantee_probability
    tau_steps = getattr(config, "tau_steps", 200)
    # DV scores are probabilities in [0, 1]; search the full range
    tau_grid = np.linspace(0.0, 1.0, tau_steps)

    logger.info(
        f"Running LTT budget search: {len(alpha_budgets)} alpha levels, {len(tau_grid)} tau candidates, delta={delta}"
    )
    logger.info(f"  Calib DV score range: [{calib_dv.min():.4f}, {calib_dv.max():.4f}]")

    results: list[dict] = []
    for alpha_b in alpha_budgets:
        tau = ltt_budget_threshold(calib_dv, alpha_b, delta, tau_grid)
        if tau is None:
            logger.info(f"  alpha_budget={alpha_b:.2f}: no valid tau found")
            results.append({"alpha_budget": alpha_b, "valid": False})
            continue

        # DV threshold cascade on eval
        dv_result = threshold_cascade(eval_ps, eval_bs, eval_dv, tau)
        dv_budget = float(dv_result.used_baseline.mean())
        dv_met = cascade_metrics(dv_result, eval_labels)

        # Uncertainty baseline at same alpha_budget constraint
        unc_result = run_baseline_cascade(eval_ps, eval_bs, alpha_b)
        unc_budget = float(unc_result.used_baseline.mean())
        unc_met = cascade_metrics(unc_result, eval_labels)

        logger.info(
            f"  alpha_budget={alpha_b:.2f}: tau={tau:.4f}, "
            f"DV budget={dv_budget:.1%} AUC={dv_met['auc']:.4f} Acc={dv_met['accuracy']:.4f} | "
            f"Unc budget={unc_budget:.1%} AUC={unc_met['auc']:.4f} Acc={unc_met['accuracy']:.4f}"
        )

        results.append(
            {
                "alpha_budget": alpha_b,
                "valid": True,
                "tau": tau,
                "dv_budget": dv_budget,
                "dv_metrics": dv_met,
                "unc_budget": unc_budget,
                "unc_metrics": unc_met,
            }
        )

    # --- Reference metrics (probe-only, baseline-only) ---
    probe_met = {
        "auc": float(roc_auc_score(eval_labels, eval_ps)),
        "accuracy": float(accuracy_score(eval_labels, (eval_ps >= 0.5).astype(int))),
    }
    baseline_met = {
        "auc": float(roc_auc_score(eval_labels, eval_bs)),
        "accuracy": float(accuracy_score(eval_labels, (eval_bs >= 0.5).astype(int))),
    }
    logger.info(f"Probe only:    AUC={probe_met['auc']:.4f}, Acc={probe_met['accuracy']:.4f}")
    logger.info(f"Baseline only: AUC={baseline_met['auc']:.4f}, Acc={baseline_met['accuracy']:.4f}")

    # --- Per-group breakdown at a representative budget level ---
    valid_results = [r for r in results if r["valid"]]
    if valid_results and eval_groups is not None:
        mid_result = valid_results[len(valid_results) // 2]
        tau_mid = mid_result["tau"]
        logger.info(f"\nPer-group breakdown at alpha_budget={mid_result['alpha_budget']:.2f} (tau={tau_mid:.4f}):")
        dv_result = threshold_cascade(eval_ps, eval_bs, eval_dv, tau_mid)
        unc_result = run_baseline_cascade(eval_ps, eval_bs, mid_result["alpha_budget"])
        for g in np.unique(eval_groups):
            mask = eval_groups == g
            logger.info(
                f"  {g}: DV rate={dv_result.used_baseline[mask].mean():.1%} "
                f"AUC={roc_auc_score(eval_labels[mask], dv_result.final_scores[mask]):.4f} | "
                f"Unc rate={unc_result.used_baseline[mask].mean():.1%} "
                f"AUC={roc_auc_score(eval_labels[mask], unc_result.final_scores[mask]):.4f}"
            )

    # --- Figures ---
    logger.info("Generating figures...")
    figs: dict[str, plt.Figure | None] = {}

    figs["budget_and_performance"] = plot_budget_and_performance(
        alpha_budgets,
        results,
        probe_met,
        baseline_met,
        output_dir,
    )

    # Adaptivity figure at the median valid budget level
    if valid_results:
        mid = valid_results[len(valid_results) // 2]
        figs["adaptivity"] = plot_adaptivity(
            eval_ps,
            eval_bs,
            eval_dv,
            eval_labels,
            eval_groups,
            mid["tau"],
            mid["alpha_budget"],
            batch_size,
            output_dir,
            baseline_cascade_fn=run_baseline_cascade,
        )

    plt.close("all")
    logger.info(f"Figures saved to {output_dir}")

    # --- ClearML logging ---
    if clearml_logger is not None:
        scalars = {
            "dv_probe_auc": dv_auc,
            "n_valid_budgets": len(valid_results),
            "probe_only_auc": probe_met["auc"],
            "baseline_only_auc": baseline_met["auc"],
        }
        if valid_results:
            mid = valid_results[len(valid_results) // 2]
            scalars["mid_alpha_budget"] = mid["alpha_budget"]
            scalars["mid_tau"] = mid["tau"]
            scalars["mid_dv_auc"] = mid["dv_metrics"]["auc"]
            scalars["mid_unc_auc"] = mid["unc_metrics"]["auc"]
        clearml_logger.log_scalars(scalars)

        for name, fig in figs.items():
            if fig is not None:
                clearml_logger.log_figure("DV LTT Cascade", name, fig)

        clearml_logger.finalize()
        logger.info("Results logged to ClearML.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
