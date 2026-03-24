"""[Demo / Debug] Analyse cascade under uncertainty-stratified batching.

Sorts test examples by **probe uncertainty** to create batches with genuine
difficulty variation, then compares adaptive (threshold) vs fixed-rate
cascade performance.

.. note::

   This is a **demonstration script** — not part of the production pipeline.
   In a real deployment we do not have access to the probe uncertainty at
   batch-assignment time.  For the production group-stratified analysis
   (which uses dataset source labels), see ``analyse_grouped_cascade.py``.

Demonstrates:
1. Jensen's inequality explains why fixed-rate wins on i.i.d. (random) batches
2. Uncertainty-stratified batching creates genuine batch structure (high variance ratio)
3. The adaptive method outperforms fixed-rate when batch-level structure exists

Usage::

    python analyse_uncertainty_stratified_cascade.py --task-id <sgt_task_id> [--use-clearml]
    python analyse_uncertainty_stratified_cascade.py --results-pkl results.pkl

Figures are saved to ``--output-dir`` (default ``figures/uncertainty_stratified/``)
and optionally logged to a new ClearML task.
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cascade_utils import (
    load_results_from_clearml,
    load_results_from_pickle,
    run_both_methods_batched,
)
from scipy import stats

from reliable_monitoring.cascade import probe_uncertainty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Colours
_C_ADAPTIVE = "steelblue"
_C_FIXED = "orange"
_C_PROBE = "gray"


def parse_args():
    parser = argparse.ArgumentParser(description="[Demo] Analyse cascade under uncertainty-stratified batching")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-id", type=str, help="ClearML task ID of an SGT cascade run.")
    source.add_argument("--results-pkl", type=str, help="Path to a local results pickle file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/uncertainty_stratified",
        help="Output directory for figures (default: figures/uncertainty_stratified/).",
    )
    parser.add_argument("--use-clearml", action="store_true", help="Log figures to a new ClearML task.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Jensen's inequality verification
# ---------------------------------------------------------------------------


def compute_marginal_values(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean marginal value of escalation by uncertainty rank.

    Returns (mean_marginal, cumulative_value) arrays of length batch_size.
    """
    n_batches = len(probe_scores) // batch_size
    all_marginals = np.zeros((n_batches, batch_size))

    for i in range(n_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        ps, bs, lb = probe_scores[s:e], baseline_scores[s:e], labels[s:e]

        uncertainty = probe_uncertainty(ps)
        order = np.argsort(-uncertainty)  # most uncertain first

        probe_correct = (ps >= 0.5).astype(int) == lb
        baseline_correct = (bs >= 0.5).astype(int) == lb

        for rank, idx in enumerate(order):
            if baseline_correct[idx] and not probe_correct[idx]:
                all_marginals[i, rank] = 1.0
            elif probe_correct[idx] and not baseline_correct[idx]:
                all_marginals[i, rank] = -1.0

    mean_marginal = all_marginals.mean(axis=0)
    cumulative = np.cumsum(mean_marginal)
    return mean_marginal, cumulative


def plot_jensens_verification(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    threshold: float,
    fixed_rate: float,
) -> plt.Figure:
    """Verify Jensen's inequality: marginal value curve and V(K) concavity."""
    mean_marginal, cumulative = compute_marginal_values(probe_scores, baseline_scores, labels, batch_size)
    fixed_k = int(fixed_rate * batch_size)

    # Compute per-batch K for adaptive
    n_batches = len(probe_scores) // batch_size
    adaptive_ks = []
    for i in range(n_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        ps = probe_scores[s:e]
        unc = probe_uncertainty(ps)
        adaptive_ks.append((unc > -(1 - threshold)).sum())
    adaptive_ks = np.array(adaptive_ks)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Jensen's Inequality Verification (Random Batches)", fontsize=14, fontweight="bold")

    # Panel 1: Marginal value by rank
    ax = axes[0]
    ax.bar(range(len(mean_marginal)), mean_marginal, alpha=0.6, color=_C_ADAPTIVE, width=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(fixed_k, color=_C_FIXED, linestyle="--", linewidth=2, label=f"Fixed K={fixed_k}")
    ax.set_xlabel("Uncertainty rank (0 = most uncertain)")
    ax.set_ylabel("Mean marginal value")
    ax.set_title("Marginal benefit of escalation\n(+1 = helps, -1 = hurts)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Cumulative value V(K)
    ax = axes[1]
    ax.plot(range(1, len(cumulative) + 1), cumulative, "b-", linewidth=2)
    ax.axvline(fixed_k, color=_C_FIXED, linestyle="--", linewidth=2, label=f"Fixed K={fixed_k}")
    ax.axvline(
        adaptive_ks.mean(),
        color=_C_ADAPTIVE,
        linestyle="--",
        linewidth=2,
        label=f"Adaptive mean K={adaptive_ks.mean():.1f}",
    )
    ax.set_xlabel("K (examples escalated)")
    ax.set_ylabel("Cumulative value V(K)")
    ax.set_title("V(K) is concave\n=> E[V(K_b)] <= V(E[K_b])")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: Per-batch K distribution
    ax = axes[2]
    ax.hist(
        adaptive_ks,
        bins=range(adaptive_ks.min(), adaptive_ks.max() + 2),
        alpha=0.6,
        color=_C_ADAPTIVE,
        edgecolor="black",
        label="Adaptive K_b",
    )
    ax.axvline(fixed_k, color=_C_FIXED, linestyle="--", linewidth=2, label=f"Fixed K={fixed_k}")
    ax.set_xlabel("Examples escalated per batch (K)")
    ax.set_ylabel("Count")
    ax.set_title(f"Adaptive K_b: range [{adaptive_ks.min()}, {adaptive_ks.max()}]\nVar(K_b) = {adaptive_ks.var():.1f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Uncertainty-stratified vs random comparison plots
# ---------------------------------------------------------------------------


def plot_variance_ratio_comparison(random_data: dict, stratified_data: dict) -> plt.Figure:
    """Bar chart comparing variance ratios for random vs uncertainty-stratified batching."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Batch-Level Structure: Random vs Uncertainty-Stratified", fontsize=14, fontweight="bold")

    for ax, (data, label) in zip(axes, [(random_data, "Random"), (stratified_data, "Stratified")], strict=False):
        unc = data["uncertainty_mean"]
        budget = data["adaptive_budget_cost"]
        acc = data["adaptive_accuracy"]

        metrics = {"Uncertainty": unc, "Adaptive budget": budget, "Adaptive accuracy": acc}
        names, ratios = [], []
        for name, vals in metrics.items():
            obs_var = vals.var()
            names.append(name)
            ratios.append(obs_var)

        x = np.arange(len(names))
        bars = ax.bar(x, ratios, alpha=0.7, color=[_C_ADAPTIVE, _C_FIXED, "forestgreen"], edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("Variance of batch means")
        ax.set_title(f"{label} batching")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, ratios, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    return fig


def plot_budget_allocation(random_data: dict, stratified_data: dict) -> plt.Figure:
    """Show how each method allocates budget across batches of varying difficulty."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Budget Allocation vs Batch Difficulty", fontsize=14, fontweight="bold")

    for ax, (data, label) in zip(
        axes, [(random_data, "Random Batches"), (stratified_data, "Stratified Batches")], strict=False
    ):
        unc = data["uncertainty_mean"]
        order = np.argsort(unc)

        ax.scatter(
            unc[order],
            data["adaptive_budget_cost"][order],
            s=80,
            alpha=0.7,
            color=_C_ADAPTIVE,
            label="Adaptive (threshold)",
            zorder=3,
        )
        ax.scatter(
            unc[order], data["fixed_budget_cost"][order], s=80, alpha=0.7, color=_C_FIXED, label="Fixed-rate", zorder=3
        )

        # Trend lines
        x_line = np.linspace(unc.min(), unc.max(), 100)
        for vals, color in [(data["adaptive_budget_cost"], _C_ADAPTIVE), (data["fixed_budget_cost"], _C_FIXED)]:
            z = np.polyfit(unc, vals, 1)
            ax.plot(x_line, np.poly1d(z)(x_line), color=color, linestyle="--", linewidth=2, alpha=0.7)

        # Spearman correlation for adaptive
        rho_adaptive, p_adaptive = stats.spearmanr(unc, data["adaptive_budget_cost"])
        rho_fixed, p_fixed = stats.spearmanr(unc, data["fixed_budget_cost"])

        ax.set_xlabel("Batch mean uncertainty", fontweight="bold")
        ax.set_ylabel("Budget (fraction delegated)", fontweight="bold")
        ax.set_title(f"{label}\nAdaptive: rho={rho_adaptive:.3f} (p={p_adaptive:.3f}) | Fixed: rho={rho_fixed:.3f}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_performance_comparison(random_data: dict, stratified_data: dict) -> plt.Figure:
    """Per-batch accuracy and ROC-AUC for both methods under both batching schemes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Performance: Adaptive vs Fixed-Rate", fontsize=14, fontweight="bold")

    panels = [
        (0, 0, random_data, "accuracy", "Accuracy", "Random Batches"),
        (0, 1, stratified_data, "accuracy", "Accuracy", "Stratified Batches"),
        (1, 0, random_data, "roc_auc", "ROC-AUC", "Random Batches"),
        (1, 1, stratified_data, "roc_auc", "ROC-AUC", "Stratified Batches"),
    ]

    for row, col, data, metric, ylabel, title in panels:
        ax = axes[row, col]
        unc = data["uncertainty_mean"]
        a_vals = data[f"adaptive_{metric}"]
        f_vals = data[f"fixed_{metric}"]

        ax.scatter(unc, a_vals, s=80, alpha=0.6, color=_C_ADAPTIVE, label="Adaptive", zorder=3)
        ax.scatter(unc, f_vals, s=80, alpha=0.6, color=_C_FIXED, label="Fixed-rate", zorder=3)

        # Trend lines
        x_line = np.linspace(unc.min(), unc.max(), 100)
        for vals, color in [(a_vals, _C_ADAPTIVE), (f_vals, _C_FIXED)]:
            z = np.polyfit(unc, vals, 1)
            ax.plot(x_line, np.poly1d(z)(x_line), color=color, linestyle="--", linewidth=2, alpha=0.7)

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(a_vals, f_vals)
        a_wins = int((a_vals > f_vals).sum())
        n = len(a_vals)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        stats_text = (
            f"Adaptive: \u03bc={a_vals.mean():.4f}\n"
            f"Fixed:    \u03bc={f_vals.mean():.4f}\n"
            f"\u0394={a_vals.mean() - f_vals.mean():.4f}\n"
            f"t={t_stat:.2f}, p={p_val:.4f}{sig}\n"
            f"Adaptive wins: {a_wins}/{n}"
        )
        ax.text(
            0.02,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_xlabel("Batch mean uncertainty", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_adaptivity_advantage(random_data: dict, stratified_data: dict) -> plt.Figure:
    """Plot (adaptive - fixed) metric difference vs batch difficulty."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Adaptivity Advantage (Adaptive - Fixed)", fontsize=14, fontweight="bold")

    for ax, (data, label) in zip(
        axes, [(random_data, "Random Batches"), (stratified_data, "Stratified Batches")], strict=False
    ):
        unc = data["uncertainty_mean"]

        for metric, marker, mlabel in [("accuracy", "o", "Accuracy"), ("roc_auc", "s", "ROC-AUC")]:
            diff = data[f"adaptive_{metric}"] - data[f"fixed_{metric}"]
            ax.scatter(unc, diff, s=80, alpha=0.6, marker=marker, label=mlabel, zorder=3)

            # Trend line
            z = np.polyfit(unc, diff, 1)
            x_line = np.linspace(unc.min(), unc.max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), linestyle="--", linewidth=2, alpha=0.7)

        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Batch mean uncertainty", fontweight="bold")
        ax.set_ylabel("Adaptive - Fixed", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_paired_scatter(data: dict, title: str) -> plt.Figure:
    """Paired scatter: adaptive vs fixed per batch, with diagonal + t-test stats."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    metrics = [
        ("accuracy", "Accuracy"),
        ("f1_score", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
    ]

    batch_indices = np.arange(int(data["n_batches"]))

    for ax, (attr, label) in zip(axes, metrics, strict=False):
        a_vals = data[f"adaptive_{attr}"]
        f_vals = data[f"fixed_{attr}"]

        scatter = ax.scatter(
            a_vals,
            f_vals,
            c=batch_indices,
            s=100,
            alpha=0.6,
            cmap="viridis",
            edgecolors="black",
            linewidth=0.5,
        )

        # Diagonal
        lo = min(a_vals.min(), f_vals.min())
        hi = max(a_vals.max(), f_vals.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", linewidth=2, alpha=0.5, label="Equal")
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)

        ax.set_xlabel(f"Adaptive {label}", fontweight="bold")
        ax.set_ylabel(f"Fixed {label}", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_aspect("equal", adjustable="box")

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Batch index", fontsize=8)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(a_vals, f_vals)
        mean_diff = float((a_vals - f_vals).mean())
        a_wins = int((a_vals > f_vals).sum())
        n = len(a_vals)

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        stats_text = (
            f"Adaptive: \u03bc={a_vals.mean():.3f}\n"
            f"Fixed:    \u03bc={f_vals.mean():.3f}\n"
            f"Mean \u0394: {mean_diff:+.4f}\n"
            f"t={t_stat:.2f}, p={p_value:.4f}{sig}\n"
            f"Adaptive wins: {a_wins}/{n}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    return fig


def plot_overall_summary(random_data: dict, stratified_data: dict) -> plt.Figure:
    """Bar chart: overall metrics for both methods under both batching schemes.

    Uses GLOBAL metrics (computed over the full test set), not per-batch averages.
    Budget is the exception -- it's the mean of per-batch budgets (which equals the global rate).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Overall Comparison: Random vs Uncertainty-Stratified Batching (global metrics)",
        fontsize=14,
        fontweight="bold",
    )

    metrics = [
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("budget_cost", "Budget"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics, strict=False):
        if metric == "budget_cost":
            values = [
                random_data[f"adaptive_{metric}"].mean(),
                random_data[f"fixed_{metric}"].mean(),
                stratified_data[f"adaptive_{metric}"].mean(),
                stratified_data[f"fixed_{metric}"].mean(),
            ]
        else:
            values = [
                random_data["adaptive_global"][metric],
                random_data["fixed_global"][metric],
                stratified_data["adaptive_global"][metric],
                stratified_data["fixed_global"][metric],
            ]
        bar_labels = ["Random\nAdaptive", "Random\nFixed", "Unc-Strat\nAdaptive", "Unc-Strat\nFixed"]
        colors = [_C_ADAPTIVE, _C_FIXED, _C_ADAPTIVE, _C_FIXED]
        hatches = ["", "", "///", "///"]

        x = np.arange(len(bar_labels))
        bars = ax.bar(x, values, width=0.6, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
        for bar, hatch in zip(bars, hatches, strict=False):
            bar.set_hatch(hatch)

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.002,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, output_dir: Path, name: str, clearml_logger=None) -> None:
    path = output_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    logger.info(f"Saved {path}")
    if clearml_logger is not None:
        clearml_logger.log_figure(title="Uncertainty Stratified Analysis", series=name, figure=fig)


def run_uncertainty_stratified_analysis(sgt_results, output_dir: Path, clearml_logger=None) -> None:
    """Run uncertainty-stratified vs random batching analysis on SGT cascade results.

    Can be called with in-memory results, or standalone via ``main()``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = sgt_results.reliable_threshold
    batch_size = sgt_results.cascade_batch_size
    merge_strategy = sgt_results.config.get("cascade_merge_strategy", "replace")
    fixed_rate = sgt_results.mean_budget_cost  # match adaptive's overall budget

    ps = sgt_results.test_probe_scores
    bs = sgt_results.test_baseline_scores
    lb = sgt_results.test_labels

    logger.info(f"Threshold: {threshold}, batch_size: {batch_size}, fixed_rate: {fixed_rate:.4f}")
    logger.info(f"Merge strategy: {merge_strategy}")
    logger.info(f"Test set: {len(ps)} examples")

    # --- Random batching (original order) ---
    logger.info("Running cascade on random (original) batches...")
    random_data = run_both_methods_batched(ps, bs, lb, batch_size, threshold, fixed_rate, merge_strategy)

    # --- Uncertainty-stratified batching (sorted by uncertainty) ---
    logger.info("Sorting examples by probe uncertainty for stratified batching...")
    uncertainty = probe_uncertainty(ps)
    sort_order = np.argsort(uncertainty)  # ascending: easy batches first
    ps_strat, bs_strat, lb_strat = ps[sort_order], bs[sort_order], lb[sort_order]

    logger.info("Running cascade on uncertainty-stratified batches...")
    stratified_data = run_both_methods_batched(
        ps_strat, bs_strat, lb_strat, batch_size, threshold, fixed_rate, merge_strategy
    )

    # --- Variance ratio diagnostic ---
    unc_random = random_data["uncertainty_mean"]
    unc_strat = stratified_data["uncertainty_mean"]
    global_unc = probe_uncertainty(ps)
    expected_var_iid = global_unc.var() / batch_size
    vr_random = unc_random.var() / expected_var_iid
    vr_strat = unc_strat.var() / expected_var_iid

    logger.info(f"Variance ratio (random):     {vr_random:.3f}")
    logger.info(f"Variance ratio (stratified): {vr_strat:.3f}")

    # --- Print summary ---
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for label, data in [("RANDOM", random_data), ("UNCERTAINTY-STRATIFIED", stratified_data)]:
        logger.info(f"\n  {label} BATCHING (global metrics over full test set):")
        for method in ["adaptive", "fixed"]:
            g = data[f"{method}_global"]
            bud = data[f"{method}_budget_cost"].mean()
            logger.info(f"    {method:>8s}: Acc={g['accuracy']:.4f}  ROC-AUC={g['roc_auc']:.4f}  Budget={bud:.4f}")

        ag = data["adaptive_global"]
        fg = data["fixed_global"]
        diff_acc = ag["accuracy"] - fg["accuracy"]
        diff_roc = ag["roc_auc"] - fg["roc_auc"]
        logger.info(f"    Adaptive - Fixed: Acc={diff_acc:+.4f}  ROC-AUC={diff_roc:+.4f}")

        logger.info(f"  {label} BATCHING (per-batch paired tests):")
        _, p_acc = stats.ttest_rel(data["adaptive_accuracy"], data["fixed_accuracy"])
        _, p_roc = stats.ttest_rel(data["adaptive_roc_auc"], data["fixed_roc_auc"])
        a_wins_acc = int((data["adaptive_accuracy"] > data["fixed_accuracy"]).sum())
        a_wins_roc = int((data["adaptive_roc_auc"] > data["fixed_roc_auc"]).sum())
        n = int(random_data["n_batches"])
        logger.info(
            f"    Acc: mean_batch_diff={data['adaptive_accuracy'].mean() - data['fixed_accuracy'].mean():+.4f} (p={p_acc:.4f}, adaptive wins {a_wins_acc}/{n})"
        )
        logger.info(
            f"    ROC: mean_batch_diff={data['adaptive_roc_auc'].mean() - data['fixed_roc_auc'].mean():+.4f} (p={p_roc:.4f}, adaptive wins {a_wins_roc}/{n})"
        )

    # --- Generate figures ---
    logger.info("\nGenerating figures...")

    figures = {
        "jensens_verification": plot_jensens_verification(ps, bs, lb, batch_size, threshold, fixed_rate),
        "budget_allocation": plot_budget_allocation(random_data, stratified_data),
        "performance_comparison": plot_performance_comparison(random_data, stratified_data),
        "adaptivity_advantage": plot_adaptivity_advantage(random_data, stratified_data),
        "paired_random": plot_paired_scatter(random_data, "Paired Comparison: Adaptive vs Fixed (Random Batches)"),
        "paired_stratified": plot_paired_scatter(
            stratified_data, "Paired Comparison: Adaptive vs Fixed (Uncertainty-Stratified Batches)"
        ),
        "overall_summary": plot_overall_summary(random_data, stratified_data),
    }

    for name, fig in figures.items():
        _save_figure(fig, output_dir, name, clearml_logger)

    plt.close("all")
    logger.info("\nUncertainty-stratified analysis done!")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ClearML setup
    clearml_logger = None
    if args.use_clearml:
        from clearml_logger import ClearMLLogger

        source_label = args.task_id[:8] if args.task_id else Path(args.results_pkl).stem
        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name=f"uncertainty_stratified_{source_label}",
            enabled=True,
        )
        if args.task_id:
            clearml_logger.connect_configuration({"source_task_id": args.task_id})

    # Load results
    if args.task_id:
        logger.info(f"Loading results from ClearML task: {args.task_id}")
        results = load_results_from_clearml(args.task_id)
    else:
        logger.info(f"Loading results from pickle: {args.results_pkl}")
        results = load_results_from_pickle(args.results_pkl)

    run_uncertainty_stratified_analysis(results, output_dir, clearml_logger)

    if clearml_logger is not None:
        clearml_logger.finalize()


if __name__ == "__main__":
    main()
