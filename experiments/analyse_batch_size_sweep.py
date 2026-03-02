"""Analyse cascade performance as a function of batch size.

Loads a previous SGT cascade experiment from ClearML (by task ID) or from a
local results pickle, then re-runs both adaptive (threshold) and fixed-rate
cascades at a range of batch sizes.  No LLM inference is needed — everything
is computed offline from the saved probe/baseline scores.

Usage::

    # From ClearML task
    uv run experiments/analyse_batch_size_sweep.py --task-id <id>

    # From local results pickle
    uv run experiments/analyse_batch_size_sweep.py --results-pkl results/pipeline/.../sgt_cascade/results.pkl

    # Custom batch sizes
    uv run experiments/analyse_batch_size_sweep.py --task-id <id> --batch-sizes 4 8 16 32 64 128 256

    # Custom output directory
    uv run experiments/analyse_batch_size_sweep.py --task-id <id> --output-dir results/batch_sweep
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cascade_utils import (
    _ExperimentUnpickler,
    compute_batch_statistics,
    compute_overall_metrics,
)
from clearml_logger import ClearMLLogger

from reliable_monitoring.cascade import offline_batch_cascade

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_C_ADAPTIVE = "steelblue"
_C_FIXED = "orange"
_C_PROBE = "gray"

# ---------------------------------------------------------------------------
# Batch size selection
# ---------------------------------------------------------------------------


def effective_fixed_budget_rate(rate: float, batch_size: int) -> float:
    """Compute the actual budget the ``fixed_budget_rate`` strategy produces.

    The strategy uses ``int()`` (floor) on rank boundaries, so the effective
    budget can differ from *rate* — especially at small batch sizes.
    """
    lower = int((1 - rate) / 2 * batch_size)
    upper = int((1 + rate) / 2 * batch_size)
    return (upper - lower) / batch_size


def find_best_rational(rate: float, max_denom: int = 20) -> tuple[int, int]:
    """Find the rational approximation *p/q* closest to *rate* with *q* <= *max_denom*.

    When the fixed-rate strategy uses rate = p/q exactly, every multiple of *q*
    is guaranteed to select exactly *p/q* of examples (the ``int()`` floors
    cancel), making the comparison fair across all batch sizes.
    """
    best_p, best_q, best_err = 1, round(1 / rate), float("inf")
    for q in range(2, max_denom + 1):
        p = round(rate * q)
        if p < 1:
            continue
        err = abs(p / q - rate)
        if err < best_err:
            best_p, best_q, best_err = p, q, err
    return best_p, best_q


def generate_fair_batch_sizes(
    q: int,
    max_bs: int = 256,
    target_count: int = 8,
) -> list[int]:
    """Return multiples of *q* up to *max_bs*, sub-sampled to ~*target_count*."""
    multiples = list(range(q, max_bs + 1, q))
    if not multiples:
        return [max_bs]

    if len(multiples) <= target_count:
        return multiples

    # Sub-sample: pick values roughly evenly spaced in log-space
    log_targets = np.linspace(np.log2(multiples[0]), np.log2(multiples[-1]), target_count)
    selected: list[int] = []
    for lt in log_targets:
        closest = min(multiples, key=lambda x, t=lt: abs(np.log2(x) - t))
        if closest not in selected:
            selected.append(closest)
    return sorted(selected)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results_from_clearml(task_id: str):
    """Load the full results object from a ClearML task by ID."""
    from clearml import Task

    task = Task.get_task(task_id=task_id)
    artifact = task.artifacts.get("results_object")
    if artifact is None:
        raise ValueError(
            f"ClearML task '{task_id}' has no 'results_object' artifact. "
            "Was it run with the updated code that saves full results?"
        )
    local_path = artifact.get_local_copy()
    with open(local_path, "rb") as f:
        return _ExperimentUnpickler(f).load()  # noqa: S301


def load_results_from_pickle(pkl_path: str):
    """Load a results object from a local pickle file."""
    with open(pkl_path, "rb") as f:
        return _ExperimentUnpickler(f).load()  # noqa: S301


# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------


def run_batch_size_sweep(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    fixed_rate: float,
    merge_strategy: str,
    batch_sizes: list[int],
) -> dict[str, list]:
    """Sweep over batch sizes, running both cascade methods at each size.

    Returns a dict with lists indexed by batch_size position:
        - batch_sizes: the batch sizes tested
        - adaptive/fixed global metrics (accuracy, f1_score, roc_auc)
        - adaptive/fixed mean budget cost
        - adaptive/fixed per-batch metric arrays (for variance analysis)
    """
    n_samples = len(probe_scores)

    results = {
        "batch_sizes": [],
        "n_batches": [],
        "effective_fixed_rates": [],  # actual budget the selection strategy produces
        # Global metrics (computed over concatenated scores)
        "adaptive_accuracy": [],
        "adaptive_f1_score": [],
        "adaptive_roc_auc": [],
        "adaptive_mean_budget": [],
        "fixed_accuracy": [],
        "fixed_f1_score": [],
        "fixed_roc_auc": [],
        "fixed_mean_budget": [],
        # Per-batch metric arrays (for variance/spread analysis)
        "adaptive_batch_accuracies": [],
        "adaptive_batch_roc_aucs": [],
        "adaptive_batch_f1_scores": [],
        "adaptive_batch_budgets": [],
        "fixed_batch_accuracies": [],
        "fixed_batch_f1_scores": [],
        "fixed_batch_roc_aucs": [],
        "fixed_batch_budgets": [],
    }

    for bs in batch_sizes:
        if bs > n_samples:
            logger.warning(f"Batch size {bs} > test set size {n_samples}, skipping.")
            continue

        n_batches = n_samples // bs
        n_used = n_batches * bs

        ps = probe_scores[:n_used]
        bls = baseline_scores[:n_used]
        lb = labels[:n_used]

        eff_rate = effective_fixed_budget_rate(fixed_rate, bs)
        logger.info(
            f"Batch size {bs}: {n_batches} batches, {n_used}/{n_samples} examples used "
            f"(effective fixed budget: {eff_rate:.4f}, target: {fixed_rate:.4f})"
        )

        # Run both methods
        adaptive_result = offline_batch_cascade(
            ps,
            bls,
            batch_size=bs,
            selection_strategy="fixed_threshold",
            merge_strategy=merge_strategy,
            threshold=threshold,
        )
        fixed_result = offline_batch_cascade(
            ps,
            bls,
            batch_size=bs,
            selection_strategy="fixed_budget_rate",
            merge_strategy=merge_strategy,
            rate=fixed_rate,
        )

        # Global metrics
        adaptive_m = compute_overall_metrics(adaptive_result.final_scores, lb)
        fixed_m = compute_overall_metrics(fixed_result.final_scores, lb)

        # Per-batch statistics
        adaptive_batch_stats = []
        fixed_batch_stats = []
        for i in range(n_batches):
            s, e = i * bs, (i + 1) * bs
            adaptive_batch_stats.append(
                compute_batch_statistics(
                    i,
                    ps[s:e],
                    adaptive_result.baseline_scores[s:e],
                    adaptive_result.used_baseline[s:e],
                    adaptive_result.final_scores[s:e],
                    lb[s:e],
                )
            )
            fixed_batch_stats.append(
                compute_batch_statistics(
                    i,
                    ps[s:e],
                    fixed_result.baseline_scores[s:e],
                    fixed_result.used_baseline[s:e],
                    fixed_result.final_scores[s:e],
                    lb[s:e],
                )
            )

        results["batch_sizes"].append(bs)
        results["n_batches"].append(n_batches)
        results["effective_fixed_rates"].append(eff_rate)

        results["adaptive_accuracy"].append(adaptive_m["accuracy"])
        results["adaptive_f1_score"].append(adaptive_m["f1_score"])
        results["adaptive_roc_auc"].append(adaptive_m["roc_auc"])
        results["adaptive_mean_budget"].append(np.mean([b.budget_cost for b in adaptive_batch_stats]))

        results["fixed_accuracy"].append(fixed_m["accuracy"])
        results["fixed_f1_score"].append(fixed_m["f1_score"])
        results["fixed_roc_auc"].append(fixed_m["roc_auc"])
        results["fixed_mean_budget"].append(np.mean([b.budget_cost for b in fixed_batch_stats]))

        results["adaptive_batch_accuracies"].append(np.array([b.accuracy for b in adaptive_batch_stats]))
        results["adaptive_batch_f1_scores"].append(np.array([b.f1_score for b in adaptive_batch_stats]))
        results["adaptive_batch_roc_aucs"].append(np.array([b.roc_auc for b in adaptive_batch_stats]))
        results["adaptive_batch_budgets"].append(np.array([b.budget_cost for b in adaptive_batch_stats]))
        results["fixed_batch_accuracies"].append(np.array([b.accuracy for b in fixed_batch_stats]))
        results["fixed_batch_f1_scores"].append(np.array([b.f1_score for b in fixed_batch_stats]))
        results["fixed_batch_roc_aucs"].append(np.array([b.roc_auc for b in fixed_batch_stats]))
        results["fixed_batch_budgets"].append(np.array([b.budget_cost for b in fixed_batch_stats]))

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_global_metrics(sweep: dict) -> plt.Figure:
    """Line plots of global accuracy, ROC-AUC, and budget vs batch size."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Global Cascade Metrics vs Batch Size", fontsize=14, fontweight="bold")

    bs = sweep["batch_sizes"]

    panels = [
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("mean_budget", "Mean Budget"),
    ]

    for ax, (metric, ylabel) in zip(axes, panels, strict=False):
        a_vals = sweep[f"adaptive_{metric}"]
        f_vals = sweep[f"fixed_{metric}"]

        ax.plot(bs, a_vals, "o-", color=_C_ADAPTIVE, linewidth=2, markersize=7, label="Adaptive (threshold)")
        ax.plot(bs, f_vals, "s-", color=_C_FIXED, linewidth=2, markersize=7, label="Fixed-rate")
        ax.set_xlabel("Batch size", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(bs)
        ax.set_xticklabels([str(b) for b in bs])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_batch_metric_spread(sweep: dict) -> plt.Figure:
    """Box plots showing per-batch metric spread at each batch size."""
    bs = sweep["batch_sizes"]
    n = len(bs)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Per-Batch Metric Distributions vs Batch Size", fontsize=14, fontweight="bold")

    metrics = [
        ("batch_accuracies", "Accuracy"),
        ("batch_roc_aucs", "ROC-AUC"),
        ("batch_budgets", "Budget Cost"),
    ]

    for col, (metric_key, ylabel) in enumerate(metrics):
        for row, (method, color, label) in enumerate(
            [
                ("adaptive", _C_ADAPTIVE, "Adaptive"),
                ("fixed", _C_FIXED, "Fixed-rate"),
            ]
        ):
            ax = axes[row, col]
            data = sweep[f"{method}_{metric_key}"]

            bp = ax.boxplot(
                data,
                positions=range(n),
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=".", markersize=3, alpha=0.5),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_xticks(range(n))
            ax.set_xticklabels([str(b) for b in bs])
            ax.set_xlabel("Batch size", fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_title(f"{label} — {ylabel}", fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_adaptive_vs_fixed_delta(sweep: dict) -> plt.Figure:
    """Plot the difference (adaptive - fixed) for each metric vs batch size."""
    bs = sweep["batch_sizes"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Adaptive - Fixed Difference vs Batch Size", fontsize=14, fontweight="bold")

    metrics = [
        ("accuracy", "Accuracy Diff"),
        ("roc_auc", "ROC-AUC Diff"),
        ("mean_budget", "Budget Diff"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics, strict=False):
        a_vals = np.array(sweep[f"adaptive_{metric}"])
        f_vals = np.array(sweep[f"fixed_{metric}"])
        diff = a_vals - f_vals

        ax.bar(
            range(len(bs)),
            diff,
            color=[_C_ADAPTIVE if d >= 0 else _C_FIXED for d in diff],
            alpha=0.7,
            edgecolor="black",
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([str(b) for b in bs])
        ax.set_xlabel("Batch size", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Annotate values
        for i, (v, _b) in enumerate(zip(diff, bs, strict=False)):
            ax.text(i, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    plt.tight_layout()
    return fig


def plot_budget_variance(sweep: dict) -> plt.Figure:
    """Show how per-batch budget variance changes with batch size."""
    bs = sweep["batch_sizes"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Per-Batch Budget Variance vs Batch Size", fontsize=14, fontweight="bold")

    for ax, (method, color, label) in zip(
        axes,
        [
            ("adaptive", _C_ADAPTIVE, "Adaptive (threshold)"),
            ("fixed", _C_FIXED, "Fixed-rate"),
        ],
        strict=False,
    ):
        budgets = sweep[f"{method}_batch_budgets"]
        means = [b.mean() for b in budgets]
        stds = [b.std() for b in budgets]
        mins = [b.min() for b in budgets]
        maxs = [b.max() for b in budgets]

        ax.fill_between(range(len(bs)), mins, maxs, alpha=0.15, color=color)
        ax.fill_between(
            range(len(bs)),
            [m - s for m, s in zip(means, stds, strict=False)],
            [m + s for m, s in zip(means, stds, strict=False)],
            alpha=0.3,
            color=color,
        )
        ax.plot(range(len(bs)), means, "o-", color=color, linewidth=2, markersize=7, label="Mean")

        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([str(b) for b in bs])
        ax.set_xlabel("Batch size", fontweight="bold")
        ax.set_ylabel("Budget cost", fontweight="bold")
        ax.set_title(f"{label}\n(shaded: min-max, darker: mean +/- std)", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def _distribution_stats(arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Compute mean, std, median, min, max across a list of per-batch arrays."""
    means = np.array([a.mean() for a in arrays])
    stds = np.array([a.std() for a in arrays])
    medians = np.array([np.median(a) for a in arrays])
    mins = np.array([a.min() for a in arrays])
    maxs = np.array([a.max() for a in arrays])
    return {"mean": means, "std": stds, "median": medians, "min": mins, "max": maxs}


def plot_within_batch_distributions(sweep: dict) -> plt.Figure:
    """Line plots with shaded 1- and 2-std regions for within-batch metric distributions.

    For each metric (accuracy, F1, ROC-AUC, budget) and each cascade method,
    plots the mean across batches as a solid line, with darker shading for
    +/- 1 std and lighter shading for +/- 2 std.
    """
    bs = sweep["batch_sizes"]
    x = np.arange(len(bs))

    metrics = [
        ("batch_accuracies", "Accuracy"),
        ("batch_f1_scores", "F1 Score"),
        ("batch_roc_aucs", "ROC-AUC"),
        ("batch_budgets", "Budget Cost"),
    ]

    fig, axes = plt.subplots(2, len(metrics), figsize=(6 * len(metrics), 10), sharey="col")
    fig.suptitle(
        "Within-Batch Performance Distributions vs Batch Size",
        fontsize=14,
        fontweight="bold",
    )

    methods = [
        ("adaptive", _C_ADAPTIVE, "Adaptive (threshold)"),
        ("fixed", _C_FIXED, "Fixed-rate"),
    ]

    for row, (method, color, label) in enumerate(methods):
        for col, (metric_key, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            data = sweep[f"{method}_{metric_key}"]
            stats = _distribution_stats(data)

            m = stats["mean"]
            s = stats["std"]
            med = stats["median"]

            # 2-std band
            ax.fill_between(x, m - 2 * s, m + 2 * s, alpha=0.12, color=color, label="+/- 2 std")
            # 1-std band
            ax.fill_between(x, m - s, m + s, alpha=0.25, color=color, label="+/- 1 std")
            # Mean line
            ax.plot(x, m, "o-", color=color, linewidth=2, markersize=6, label="Mean")
            # Median line
            ax.plot(x, med, "x--", color=color, linewidth=1, markersize=5, alpha=0.7, label="Median")

            ax.set_xticks(x)
            ax.set_xticklabels([str(b) for b in bs])
            ax.set_xlabel("Batch size", fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{label}\n{ylabel}", fontweight="bold")
            else:
                ax.set_ylabel(ylabel, fontweight="bold")
            if row == 0:
                ax.set_title(ylabel, fontweight="bold")
            ax.grid(alpha=0.3)

            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    return fig


def build_distribution_summary_dataframe(sweep: dict):
    """Build a DataFrame with within-batch distribution statistics per batch size."""
    import pandas as pd

    metrics = [
        ("batch_accuracies", "Accuracy"),
        ("batch_f1_scores", "F1"),
        ("batch_roc_aucs", "ROC-AUC"),
        ("batch_budgets", "Budget"),
    ]

    rows = []
    for i, bs_val in enumerate(sweep["batch_sizes"]):
        for method in ["adaptive", "fixed"]:
            for metric_key, metric_label in metrics:
                arr = sweep[f"{method}_{metric_key}"][i]
                rows.append(
                    {
                        "Batch Size": bs_val,
                        "Method": method.title(),
                        "Metric": metric_label,
                        "Mean": round(float(arr.mean()), 4),
                        "Std": round(float(arr.std()), 4),
                        "Median": round(float(np.median(arr)), 4),
                        "Min": round(float(arr.min()), 4),
                        "Max": round(float(arr.max()), 4),
                        "N Batches": len(arr),
                    }
                )
    return pd.DataFrame(rows)


def build_summary_dataframe(sweep: dict, threshold: float, fixed_rate: float):
    """Build a pandas DataFrame summarising the sweep results."""
    import pandas as pd

    rows = []
    for i, bs in enumerate(sweep["batch_sizes"]):
        d_acc = sweep["adaptive_accuracy"][i] - sweep["fixed_accuracy"][i]
        d_roc = sweep["adaptive_roc_auc"][i] - sweep["fixed_roc_auc"][i]
        d_bud = sweep["adaptive_mean_budget"][i] - sweep["fixed_mean_budget"][i]
        rows.append(
            {
                "Batch Size": bs,
                "N Batches": sweep["n_batches"][i],
                "Eff. Fixed Rate": round(sweep["effective_fixed_rates"][i], 4),
                "Adapt. Acc": round(sweep["adaptive_accuracy"][i], 4),
                "Fixed Acc": round(sweep["fixed_accuracy"][i], 4),
                "Diff Acc": round(d_acc, 4),
                "Adapt. ROC": round(sweep["adaptive_roc_auc"][i], 4),
                "Fixed ROC": round(sweep["fixed_roc_auc"][i], 4),
                "Diff ROC": round(d_roc, 4),
                "Adapt. Budget": round(sweep["adaptive_mean_budget"][i], 4),
                "Fixed Budget": round(sweep["fixed_mean_budget"][i], 4),
                "Diff Budget": round(d_bud, 4),
            }
        )
    return pd.DataFrame(rows)


def plot_summary_table(sweep: dict, threshold: float, fixed_rate: float) -> plt.Figure:
    """Summary table of all metrics at each batch size."""
    df = build_summary_dataframe(sweep, threshold, fixed_rate)
    n = len(df)

    fig, ax = plt.subplots(figsize=(18, 0.6 * n + 2.5))
    ax.axis("off")
    ax.set_title(
        f"Batch Size Sweep Summary\nThreshold={threshold:.4f}, Target Fixed Rate={fixed_rate:.4f}",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    cell_data = df.values.tolist()
    columns = list(df.columns)

    table = ax.table(
        cellText=[
            [
                f"{v:+.4f}"
                if isinstance(v, float) and columns[j].startswith("Diff")
                else f"{v:.4f}"
                if isinstance(v, float)
                else str(v)
                for j, v in enumerate(row)
            ]
            for row in cell_data
        ],
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color diff columns
    diff_cols = [i for i, c in enumerate(columns) if c.startswith("Diff")]
    for row_idx in range(n):
        for col_idx in diff_cols:
            cell = table[row_idx + 1, col_idx]
            val = float(cell_data[row_idx][col_idx])
            if val > 0.001:
                cell.set_facecolor("#d4edda")
            elif val < -0.001:
                cell.set_facecolor("#f8d7da")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse cascade performance vs batch size")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-id", type=str, help="ClearML task ID of an SGT cascade run.")
    group.add_argument("--results-pkl", type=str, help="Path to a local results pickle file.")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to sweep. If not specified, auto-selects sizes where "
        "the fixed-rate budget discretizes fairly (error < 2%%).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/batch_size_sweep",
        help="Output directory for figures and results (default: results/batch_size_sweep).",
    )
    parser.add_argument(
        "--use-clearml",
        action="store_true",
        help="Log figures and scalars to a new ClearML task.",
    )
    return parser.parse_args()


def _log_sweep_to_clearml(
    clearml_logger: ClearMLLogger,
    sweep: dict,
    figures: dict[str, plt.Figure],
    threshold: float,
    fixed_rate: float,
    merge_strategy: str,
    test_size: int,
    source_id: str,
) -> None:
    """Log sweep scalars, summary table, and figures to ClearML."""
    clearml_logger.connect_configuration(
        {
            "source_id": source_id,
            "threshold": threshold,
            "fixed_rate": fixed_rate,
            "merge_strategy": merge_strategy,
            "test_size": test_size,
            "batch_sizes": sweep["batch_sizes"],
        }
    )
    clearml_logger.add_tags(["batch_size_sweep", "analysis"])

    if not (clearml_logger.enabled and clearml_logger.task is not None):
        return

    cl = clearml_logger.task.get_logger()

    # Log per-batch-size scalars (one iteration per batch size)
    for i, bs in enumerate(sweep["batch_sizes"]):
        for method in ["adaptive", "fixed"]:
            for metric in ["accuracy", "f1_score", "roc_auc", "mean_budget"]:
                cl.report_scalar(
                    title=f"{method.title()} {metric}",
                    series=method,
                    value=float(sweep[f"{method}_{metric}"][i]),
                    iteration=bs,
                )
        # Delta
        for series, metric in [("accuracy", "accuracy"), ("roc_auc", "roc_auc"), ("mean_budget", "mean_budget")]:
            cl.report_scalar(
                title="Delta (adaptive - fixed)",
                series=series,
                value=float(sweep[f"adaptive_{metric}"][i] - sweep[f"fixed_{metric}"][i]),
                iteration=bs,
            )

    # Log summary as a ClearML table (matplotlib tables don't render in ClearML)
    df = build_summary_dataframe(sweep, threshold, fixed_rate)
    cl.report_table(
        title="Batch Size Sweep",
        series="Summary",
        table_plot=df,
        iteration=0,
    )

    # Log within-batch distribution summary table
    dist_df = build_distribution_summary_dataframe(sweep)
    cl.report_table(
        title="Within-Batch Distributions",
        series="Summary",
        table_plot=dist_df,
        iteration=0,
    )

    # Log within-batch distribution scalars (mean +/- std per batch size)
    dist_metrics = [
        ("batch_accuracies", "Accuracy"),
        ("batch_f1_scores", "F1"),
        ("batch_roc_aucs", "ROC-AUC"),
        ("batch_budgets", "Budget"),
    ]
    for i, bs in enumerate(sweep["batch_sizes"]):
        for method in ["adaptive", "fixed"]:
            for metric_key, metric_label in dist_metrics:
                arr = sweep[f"{method}_{metric_key}"][i]
                cl.report_scalar(
                    title=f"Within-Batch {metric_label} Std",
                    series=f"{method.title()}",
                    value=float(arr.std()),
                    iteration=bs,
                )

    # Log figures (skip the matplotlib summary_table — the ClearML table replaces it)
    for name, fig in figures.items():
        if name == "summary_table":
            continue
        clearml_logger.log_figure(title="Batch Size Sweep", series=name, figure=fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ClearML setup
    clearml_logger = None
    if args.use_clearml:
        source_label = args.task_id[:8] if args.task_id else Path(args.results_pkl).stem
        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name=f"batch_size_sweep_{source_label}",
            enabled=True,
        )

    # Load results
    if args.task_id:
        logger.info(f"Loading results from ClearML task: {args.task_id}")
        sgt_results = load_results_from_clearml(args.task_id)
    else:
        logger.info(f"Loading results from pickle: {args.results_pkl}")
        sgt_results = load_results_from_pickle(args.results_pkl)

    # Extract what we need
    probe_scores = sgt_results.test_probe_scores
    baseline_scores = sgt_results.test_baseline_scores
    labels = sgt_results.test_labels
    threshold = sgt_results.reliable_threshold
    empirical_rate = sgt_results.mean_budget_cost
    merge_strategy = sgt_results.config.get("cascade_merge_strategy", "replace")

    # Find the best rational approximation p/q for the empirical budget rate.
    # Using rate = p/q exactly guarantees that every multiple of q selects
    # exactly p/q of examples (the int() floors in fixed_budget_rate cancel),
    # making the budget identical at every batch size.
    p, q = find_best_rational(empirical_rate)
    fixed_rate = p / q
    logger.info(
        f"Empirical budget rate: {empirical_rate:.4f} -> rational approximation: "
        f"{p}/{q} = {fixed_rate:.4f} (error: {abs(fixed_rate - empirical_rate):.4f})"
    )

    # Determine batch sizes
    if args.batch_sizes is not None:
        batch_sizes = sorted(args.batch_sizes)
    else:
        batch_sizes = generate_fair_batch_sizes(q)
        logger.info(f"Auto-selected batch sizes (multiples of {q}): {batch_sizes}")

    logger.info(f"Test set: {len(probe_scores)} examples")
    logger.info(f"Threshold: {threshold:.4f}")
    logger.info(f"Fixed rate: {fixed_rate:.4f} ({p}/{q})")
    logger.info(f"Merge strategy: {merge_strategy}")
    logger.info(f"Batch sizes: {batch_sizes}")

    # Run sweep
    sweep = run_batch_size_sweep(
        probe_scores,
        baseline_scores,
        labels,
        threshold,
        fixed_rate,
        merge_strategy,
        batch_sizes=batch_sizes,
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH SIZE SWEEP RESULTS")
    logger.info("=" * 80)
    for i, bs in enumerate(sweep["batch_sizes"]):
        d_acc = sweep["adaptive_accuracy"][i] - sweep["fixed_accuracy"][i]
        d_roc = sweep["adaptive_roc_auc"][i] - sweep["fixed_roc_auc"][i]
        eff = sweep["effective_fixed_rates"][i]
        logger.info(
            f"  bs={bs:>4d} ({sweep['n_batches'][i]:>4d} batches, eff. fixed rate={eff:.4f}): "
            f"Adaptive Acc={sweep['adaptive_accuracy'][i]:.4f} ROC={sweep['adaptive_roc_auc'][i]:.4f} "
            f"Budget={sweep['adaptive_mean_budget'][i]:.4f}  |  "
            f"Fixed Acc={sweep['fixed_accuracy'][i]:.4f} ROC={sweep['fixed_roc_auc'][i]:.4f} "
            f"Budget={sweep['fixed_mean_budget'][i]:.4f}  |  "
            f"dAcc={d_acc:+.4f} dROC={d_roc:+.4f}"
        )

    # Generate and save figures
    logger.info("\nGenerating figures...")
    figures = {
        "global_metrics": plot_global_metrics(sweep),
        "batch_metric_spread": plot_batch_metric_spread(sweep),
        "within_batch_distributions": plot_within_batch_distributions(sweep),
        "adaptive_vs_fixed_delta": plot_adaptive_vs_fixed_delta(sweep),
        "budget_variance": plot_budget_variance(sweep),
        "summary_table": plot_summary_table(sweep, threshold, fixed_rate),
    }

    for name, fig in figures.items():
        path = output_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved {path}")

    # Save raw sweep data as JSON (scalars only)
    scalars_json = {
        "batch_sizes": sweep["batch_sizes"],
        "n_batches": sweep["n_batches"],
        "fixed_rate": fixed_rate,
        "effective_fixed_rates": [float(v) for v in sweep["effective_fixed_rates"]],
        "threshold": threshold,
        "merge_strategy": merge_strategy,
        "test_size": len(probe_scores),
    }
    for key in [
        "adaptive_accuracy",
        "adaptive_f1_score",
        "adaptive_roc_auc",
        "adaptive_mean_budget",
        "fixed_accuracy",
        "fixed_f1_score",
        "fixed_roc_auc",
        "fixed_mean_budget",
    ]:
        scalars_json[key] = [float(v) for v in sweep[key]]

    (output_dir / "sweep_results.json").write_text(json.dumps(scalars_json, indent=2))
    logger.info(f"Saved {output_dir / 'sweep_results.json'}")

    # Log to ClearML
    if clearml_logger is not None:
        source_id = args.task_id or args.results_pkl
        _log_sweep_to_clearml(
            clearml_logger,
            sweep,
            figures,
            threshold,
            fixed_rate,
            merge_strategy,
            len(probe_scores),
            source_id,
        )
        clearml_logger.finalize()

    plt.close("all")
    logger.info("\nBatch size sweep complete!")


if __name__ == "__main__":
    main()
