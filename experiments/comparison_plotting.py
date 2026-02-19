"""Generic pairwise comparison plotting for any two cascade experiments.

All functions accept two objects satisfying :class:`CascadeExperimentResults`
and return matplotlib figures.  Labels default to ``"A"``/``"B"`` but can be
overridden (e.g. ``"Adaptive (SGT)"`` / ``"Fixed"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from cascade_utils import CascadeExperimentResults


_COLOR_A = "steelblue"
_COLOR_B = "orange"


def plot_comparison_overall(
    a: CascadeExperimentResults,
    b: CascadeExperimentResults,
    label_a: str = "A",
    label_b: str = "B",
) -> Figure:
    """Bar chart comparing overall metrics across probe / baseline / cascade_a / cascade_b."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Overall Performance Comparison", fontsize=14, fontweight="bold")

    metrics = [
        (
            "Accuracy",
            [a.probe_only_accuracy, a.baseline_only_accuracy, a.cascade_accuracy, b.cascade_accuracy],
        ),
        (
            "F1 Score",
            [a.probe_only_f1_score, a.baseline_only_f1_score, a.cascade_f1_score, b.cascade_f1_score],
        ),
        (
            "ROC-AUC",
            [a.probe_only_roc_auc, a.baseline_only_roc_auc, a.cascade_roc_auc, b.cascade_roc_auc],
        ),
    ]

    colors = [_COLOR_A, "coral", "forestgreen", _COLOR_B]
    bar_labels = ["Probe Only", "Baseline Only", label_a, label_b]

    for ax, (metric_name, values) in zip(axes.flat, metrics, strict=False):
        x = np.arange(len(bar_labels))
        bars = ax.bar(x, values, width=0.6, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
        ax.set_ylabel(metric_name, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    return fig


def plot_comparison_batch_distributions(
    a: CascadeExperimentResults,
    b: CascadeExperimentResults,
    label_a: str = "A",
    label_b: str = "B",
) -> dict[str, Figure]:
    """Overlaid histograms of per-batch metrics for two experiments."""
    figures: dict[str, Figure] = {}

    def _extract(batches, attr):
        return np.array([getattr(batch, attr) for batch in batches])

    metrics_data = [
        ("budget_cost", "budget_cost", "Budget Cost"),
        ("accuracy", "accuracy", "Accuracy"),
        ("f1_score", "f1_score", "F1 Score"),
        ("roc_auc", "roc_auc", "ROC-AUC"),
        ("probe_uncertainty", "probe_uncertainty_mean", "Probe Uncertainty"),
    ]

    for key, attr, title in metrics_data:
        data_a = _extract(a.batches, attr)
        data_b = _extract(b.batches, attr)

        fig, ax = plt.subplots(figsize=(10, 6))
        bins = max(5, len(data_a) // 2)
        ax.hist(data_a, bins=bins, alpha=0.5, label=label_a, color=_COLOR_A, edgecolor="black")
        ax.hist(data_b, bins=bins, alpha=0.5, label=label_b, color=_COLOR_B, edgecolor="black")

        ax.set_xlabel(title, fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title(f"Distribution of {title} Across Batches", fontweight="bold")
        ax.legend(loc="best", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        stats_text = (
            f"{label_a}: \u03bc={data_a.mean():.3f}, \u03c3={data_a.std():.3f}\n"
            f"{label_b}: \u03bc={data_b.mean():.3f}, \u03c3={data_b.std():.3f}"
        )
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        figures[key] = fig

    return figures


def plot_comparison_boxplots(
    a: CascadeExperimentResults,
    b: CascadeExperimentResults,
    label_a: str = "A",
    label_b: str = "B",
) -> Figure:
    """Side-by-side box plots for budget / accuracy / f1 / roc_auc."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Metric Range Comparison", fontsize=16, fontweight="bold")

    def _extract(batches, attr):
        return np.array([getattr(batch, attr) for batch in batches])

    metrics = [
        ("budget_cost", "Budget Cost"),
        ("accuracy", "Accuracy"),
        ("f1_score", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
    ]

    for ax, (attr, title) in zip(axes.flat, metrics, strict=False):
        data_a = _extract(a.batches, attr)
        data_b = _extract(b.batches, attr)

        bp = ax.boxplot(
            [data_a, data_b],
            labels=[label_a, label_b],
            patch_artist=True,
            widths=0.6,
        )
        for patch, color in zip(bp["boxes"], [_COLOR_A, _COLOR_B], strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set(color="red", linewidth=2)

        ax.set_ylabel(title, fontweight="bold")
        ax.set_title(f"{title} Distribution", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        stats_text = (
            f"{label_a}: \u03bc={data_a.mean():.3f}, \u03c3={data_a.std():.3f}\n"
            f"{label_b}: \u03bc={data_b.mean():.3f}, \u03c3={data_b.std():.3f}"
        )
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    return fig


def plot_comparison_uncertainty_vs_metrics(
    a: CascadeExperimentResults,
    b: CascadeExperimentResults,
    label_a: str = "A",
    label_b: str = "B",
) -> Figure:
    """Scatter: probe uncertainty vs budget / accuracy / f1 / roc_auc with trend lines."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Probe Uncertainty vs Performance\n(Probe Uncertainty = min(p, 1-p))",
        fontsize=16,
        fontweight="bold",
    )

    def _extract(batches, attr):
        return np.array([getattr(batch, attr) for batch in batches])

    unc_a = _extract(a.batches, "probe_uncertainty_mean")
    unc_b = _extract(b.batches, "probe_uncertainty_mean")

    metrics = [
        ("budget_cost", "Budget Cost", "Lower is better"),
        ("accuracy", "Accuracy", "Higher is better"),
        ("f1_score", "F1 Score", "Higher is better"),
        ("roc_auc", "ROC-AUC", "Higher is better"),
    ]

    for ax, (attr, ylabel, note) in zip(axes.flat, metrics, strict=False):
        y_a = _extract(a.batches, attr)
        y_b = _extract(b.batches, attr)

        ax.scatter(unc_a, y_a, alpha=0.6, s=100, label=label_a, color=_COLOR_A)
        ax.scatter(unc_b, y_b, alpha=0.6, s=100, label=label_b, color=_COLOR_B)

        # Trend lines
        x_min = min(unc_a.min(), unc_b.min())
        x_max = max(unc_a.max(), unc_b.max())
        x_line = np.linspace(x_min, x_max, 100)

        z_a = np.polyfit(unc_a, y_a, 1)
        ax.plot(x_line, np.poly1d(z_a)(x_line), _COLOR_A, linestyle="--", linewidth=2, alpha=0.7)

        z_b = np.polyfit(unc_b, y_b, 1)
        ax.plot(x_line, np.poly1d(z_b)(x_line), _COLOR_B, linestyle="--", linewidth=2, alpha=0.7)

        ax.set_xlabel("Probe Uncertainty", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(f"{ylabel} vs Probe Uncertainty ({note})", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_comparison_paired_scatter(
    a: CascadeExperimentResults,
    b: CascadeExperimentResults,
    label_a: str = "A",
    label_b: str = "B",
) -> Figure:
    """Paired scatter: metric_a vs metric_b per batch with diagonal + t-test stats."""
    from scipy import stats

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Paired Comparison: {label_a} vs {label_b} (per batch)", fontsize=16, fontweight="bold")

    def _extract(batches, attr):
        return np.array([getattr(batch, attr) for batch in batches])

    metrics = [
        ("accuracy", "Accuracy"),
        ("f1_score", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
    ]

    batch_indices = np.arange(len(a.batches))

    for ax, (attr, title) in zip(axes, metrics, strict=False):
        data_a = _extract(a.batches, attr)
        data_b = _extract(b.batches, attr)

        scatter = ax.scatter(
            data_a,
            data_b,
            c=batch_indices,
            s=120,
            alpha=0.6,
            cmap="viridis",
            edgecolors="black",
            linewidth=0.5,
        )

        # Diagonal
        val_min = min(data_a.min(), data_b.min())
        val_max = max(data_a.max(), data_b.max())
        ax.plot([val_min, val_max], [val_min, val_max], "r--", linewidth=2, alpha=0.5, label="Equal")

        ax.set_xlabel(f"{label_a} {title}", fontweight="bold")
        ax.set_ylabel(f"{label_b} {title}", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Batch Index", fontweight="bold")

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(data_a, data_b)
        mean_diff = float((data_a - data_b).mean())
        a_wins = int((data_a > data_b).sum())
        win_pct = 100 * a_wins / len(data_a)

        sig = ""
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"

        stats_text = (
            f"{label_a}: \u03bc={data_a.mean():.3f}, \u03c3={data_a.std():.3f}\n"
            f"{label_b}: \u03bc={data_b.mean():.3f}, \u03c3={data_b.std():.3f}\n"
            f"Mean \u0394: {mean_diff:.4f}\n"
            f"Paired t: t={t_stat:.3f}, p={p_value:.4f}{sig}\n"
            f"{label_a} wins: {a_wins}/{len(data_a)} ({win_pct:.1f}%)"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            family="monospace",
        )

    plt.tight_layout()
    return fig
