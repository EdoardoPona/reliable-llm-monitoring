"""Analyse cascade performance under group-stratified batching.

When test data comes from multiple sources (e.g. anthropic, mt, mts, toolace),
this script groups examples by their source label and compares adaptive
(threshold) vs fixed-rate cascade performance across groups.

This is the **production** stratified analysis — unlike the uncertainty-based
demo (``analyse_uncertainty_stratified_cascade.py``), group labels are available
at deployment time.

Can be run standalone or called from the pipeline.

Usage::

    # From a local pickle (no recomputation needed)
    python analyse_grouped_cascade.py --results-pkl results/pipeline/<run>/sgt_cascade/results.pkl

    # From a ClearML task
    python analyse_grouped_cascade.py --task-id <sgt_task_id> [--use-clearml]

Figures are saved to ``--output-dir`` (default ``figures/grouped/``) and
optionally logged to a new ClearML task.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Colours
_C_ADAPTIVE = "steelblue"
_C_FIXED = "orange"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse cascade under group-stratified batching")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-id", type=str, help="ClearML task ID of an SGT cascade run.")
    source.add_argument("--results-pkl", type=str, help="Path to a local results pickle file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/grouped",
        help="Output directory for figures (default: figures/grouped/).",
    )
    parser.add_argument("--use-clearml", action="store_true", help="Log figures to a new ClearML task.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Group ordering
# ---------------------------------------------------------------------------


def order_by_group(
    ps: np.ndarray,
    bs: np.ndarray,
    lb: np.ndarray,
    groups: np.ndarray,
    group_purity: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Order examples by group with configurable purity.

    Args:
        ps: Probe scores.
        bs: Baseline scores.
        lb: Labels.
        groups: Group labels (string array).
        group_purity: 1.0 = fully grouped (contiguous by group),
            0.0 = fully random, in-between = partial mixing where a fraction
            (1 - group_purity) of examples are randomly repositioned.
        seed: Random seed.

    Returns:
        Reordered (ps, bs, lb, groups) arrays.
    """
    if group_purity <= 0.0:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ps))
    elif group_purity >= 1.0:
        order = np.argsort(groups, kind="stable")
    else:
        order = np.argsort(groups, kind="stable")
        rng = np.random.default_rng(seed)
        n_swap = int(len(ps) * (1 - group_purity))
        swap_indices = rng.choice(len(ps), size=n_swap, replace=False)
        order[swap_indices] = order[rng.permutation(swap_indices)]
    return ps[order], bs[order], lb[order], groups[order]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_group_performance(
    group_data: dict,
    groups_sorted: np.ndarray,
    batch_size: int,
) -> plt.Figure:
    """Per-group accuracy and budget comparison between adaptive and fixed."""
    n_batches = int(group_data["n_batches"])
    unique_groups = np.unique(groups_sorted)

    # Determine dominant group per batch
    batch_groups = []
    for i in range(n_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        bg = groups_sorted[s:e]
        vals, counts = np.unique(bg, return_counts=True)
        batch_groups.append(vals[np.argmax(counts)])
    batch_groups = np.array(batch_groups)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Group-Stratified: Adaptive vs Fixed per Group", fontsize=14, fontweight="bold")

    metrics = [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("budget_cost", "Budget")]

    for ax, (metric, ylabel) in zip(axes, metrics, strict=False):
        x = np.arange(len(unique_groups))
        adaptive_means = []
        fixed_means = []
        for g in unique_groups:
            mask = batch_groups == g
            adaptive_means.append(group_data[f"adaptive_{metric}"][mask].mean())
            fixed_means.append(group_data[f"fixed_{metric}"][mask].mean())

        width = 0.35
        ax.bar(x - width / 2, adaptive_means, width, label="Adaptive", color=_C_ADAPTIVE, alpha=0.8, edgecolor="black")
        ax.bar(x + width / 2, fixed_means, width, label="Fixed", color=_C_FIXED, alpha=0.8, edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(unique_groups, fontsize=10)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xlabel("Dataset group", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Annotate values
        for i, (a, f) in enumerate(zip(adaptive_means, fixed_means, strict=False)):
            ax.text(i - width / 2, a + 0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width / 2, f + 0.005, f"{f:.3f}", ha="center", va="bottom", fontsize=8)

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


def plot_overall_summary(random_data: dict, group_data: dict) -> plt.Figure:
    """Bar chart comparing overall metrics: random vs group-stratified batching."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "Overall Comparison: Random vs Group-Stratified Batching (global metrics)", fontsize=14, fontweight="bold"
    )

    metrics = [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("budget_cost", "Budget")]

    for ax, (metric, ylabel) in zip(axes, metrics, strict=False):
        if metric == "budget_cost":
            values = [
                random_data[f"adaptive_{metric}"].mean(),
                random_data[f"fixed_{metric}"].mean(),
                group_data[f"adaptive_{metric}"].mean(),
                group_data[f"fixed_{metric}"].mean(),
            ]
        else:
            values = [
                random_data["adaptive_global"][metric],
                random_data["fixed_global"][metric],
                group_data["adaptive_global"][metric],
                group_data["fixed_global"][metric],
            ]
        bar_labels = ["Random\nAdaptive", "Random\nFixed", "Group-Strat\nAdaptive", "Group-Strat\nFixed"]
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
# Main analysis
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, output_dir: Path, name: str, clearml_logger=None) -> None:
    path = output_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    logger.info(f"Saved {path}")
    if clearml_logger is not None:
        clearml_logger.log_figure(title="Grouped Analysis", series=name, figure=fig)


def run_grouped_analysis(sgt_results, output_dir: Path, clearml_logger=None) -> None:
    """Run group-stratified vs random batching analysis on SGT cascade results.

    Requires ``sgt_results.test_groups`` to be set (from a mixed-dataset run).
    Can be called from the pipeline with in-memory results, or standalone via ``main()``.
    """
    test_groups = getattr(sgt_results, "test_groups", None)
    if test_groups is None:
        logger.warning("No group labels found in results — cannot run grouped analysis.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = sgt_results.reliable_threshold
    batch_size = sgt_results.cascade_batch_size
    merge_strategy = sgt_results.config.get("cascade_merge_strategy", "replace")
    fixed_rate = sgt_results.mean_budget_cost  # match adaptive's overall budget
    group_purity = getattr(sgt_results, "group_purity", 1.0) or 1.0

    ps = sgt_results.test_probe_scores
    bs = sgt_results.test_baseline_scores
    lb = sgt_results.test_labels

    logger.info(f"Threshold: {threshold}, batch_size: {batch_size}, fixed_rate: {fixed_rate:.4f}")
    logger.info(f"Merge strategy: {merge_strategy}")
    logger.info(f"Test set: {len(ps)} examples")

    unique_groups, group_counts = np.unique(test_groups, return_counts=True)
    logger.info(f"Groups: {dict(zip(unique_groups, group_counts, strict=True))}")
    logger.info(f"Group purity: {group_purity}")

    # --- Random batching (baseline comparison) ---
    logger.info("Running cascade on random (original) batches...")
    random_data = run_both_methods_batched(ps, bs, lb, batch_size, threshold, fixed_rate, merge_strategy)

    # --- Group-stratified batching ---
    logger.info(f"Running group-stratified batching analysis (group_purity={group_purity})...")
    ps_group, bs_group, lb_group, groups_sorted = order_by_group(
        ps, bs, lb, test_groups, group_purity, seed=sgt_results.seed
    )
    group_data = run_both_methods_batched(
        ps_group, bs_group, lb_group, batch_size, threshold, fixed_rate, merge_strategy
    )

    # --- Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for label, data in [("RANDOM", random_data), ("GROUP-STRATIFIED", group_data)]:
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
        n = int(data["n_batches"])
        logger.info(
            f"    Acc: mean_batch_diff={data['adaptive_accuracy'].mean() - data['fixed_accuracy'].mean():+.4f} (p={p_acc:.4f}, adaptive wins {a_wins_acc}/{n})"
        )
        logger.info(
            f"    ROC: mean_batch_diff={data['adaptive_roc_auc'].mean() - data['fixed_roc_auc'].mean():+.4f} (p={p_roc:.4f}, adaptive wins {a_wins_roc}/{n})"
        )

    # --- Generate figures ---
    logger.info("\nGenerating figures...")

    figures = {
        "group_performance": plot_group_performance(group_data, groups_sorted, batch_size),
        "paired_random": plot_paired_scatter(random_data, "Paired Comparison: Adaptive vs Fixed (Random Batches)"),
        "paired_group": plot_paired_scatter(
            group_data, "Paired Comparison: Adaptive vs Fixed (Group-Stratified Batches)"
        ),
        "overall_summary": plot_overall_summary(random_data, group_data),
    }

    for name, fig in figures.items():
        _save_figure(fig, output_dir, name, clearml_logger)

    plt.close("all")
    logger.info("\nGrouped analysis done!")


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
            task_name=f"grouped_analysis_{source_label}",
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

    run_grouped_analysis(results, output_dir, clearml_logger)

    if clearml_logger is not None:
        clearml_logger.finalize()


if __name__ == "__main__":
    main()
