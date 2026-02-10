"""Plotting utilities for the guaranteed-risk cascade experiment.

Experiment-specific plots take ``GuaranteedRiskCascadeResults``; shared
primitives (score histograms, reliability diagrams, ROC curves, Pareto
frontier) are delegated to ``plot_utils``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from guaranteed_risk_cascade import GuaranteedRiskCascadeResults


# ---------------------------------------------------------------------------
# Experiment-specific plots
# ---------------------------------------------------------------------------


def plot_overall_performance(results: GuaranteedRiskCascadeResults) -> Figure:
    """Grouped bar chart: probe / baseline / cascade across accuracy, F1, ROC-AUC."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Overall Performance: Probe vs Baseline vs Cascade", fontsize=14, fontweight="bold")

    metrics = [
        ("Accuracy", [results.probe_only_accuracy, results.baseline_only_accuracy, results.cascade_accuracy]),
        ("F1 Score", [results.probe_only_f1_score, results.baseline_only_f1_score, results.cascade_f1_score]),
        ("ROC-AUC", [results.probe_only_roc_auc, results.baseline_only_roc_auc, results.cascade_roc_auc]),
    ]

    colors = ["steelblue", "coral", "forestgreen"]
    bar_labels = ["Probe Only", "Baseline Only", "Cascade"]

    for ax, (metric_name, values) in zip(axes.flat, metrics, strict=False):
        x = np.arange(len(bar_labels))
        bars = ax.bar(x, values, width=0.6, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)

        ax.set_ylabel(metric_name, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=10)
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


def plot_batch_distributions(results: GuaranteedRiskCascadeResults) -> dict[str, Figure]:
    """Histograms of per-batch metrics (single cascade, no comparison)."""
    figures: dict[str, Figure] = {}

    metrics_data = [
        ("budget_cost", [b.budget_cost for b in results.batches], "Budget Cost"),
        ("accuracy", [b.accuracy for b in results.batches], "Accuracy"),
        ("f1_score", [b.f1_score for b in results.batches], "F1 Score"),
        ("roc_auc", [b.roc_auc for b in results.batches], "ROC-AUC"),
        ("probe_uncertainty", [b.probe_uncertainty_mean for b in results.batches], "Probe Uncertainty"),
    ]

    for key, values_list, title in metrics_data:
        data = np.array(values_list)
        fig, ax = plt.subplots(figsize=(10, 6))

        bins = max(5, len(data) // 2)
        ax.hist(data, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")

        ax.set_xlabel(title, fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title(f"Distribution of {title} Across Batches", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        stats_text = f"\u03bc={data.mean():.3f}, \u03c3={data.std():.3f}"
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


def plot_batch_uncertainty_vs_metrics(results: GuaranteedRiskCascadeResults) -> Figure:
    """4-panel scatter: probe uncertainty vs budget / accuracy / F1 / ROC-AUC."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Probe Uncertainty vs Performance\n(Probe Uncertainty = min(p, 1-p))",
        fontsize=16,
        fontweight="bold",
    )

    uncertainty = np.array([b.probe_uncertainty_mean for b in results.batches])

    metrics = [
        (np.array([b.budget_cost for b in results.batches]), "Budget Cost", "Lower is better"),
        (np.array([b.accuracy for b in results.batches]), "Accuracy", "Higher is better"),
        (np.array([b.f1_score for b in results.batches]), "F1 Score", "Higher is better"),
        (np.array([b.roc_auc for b in results.batches]), "ROC-AUC", "Higher is better"),
    ]

    for ax, (y_data, ylabel, note) in zip(axes.flat, metrics, strict=False):
        ax.scatter(uncertainty, y_data, alpha=0.6, s=100, color="steelblue")

        # Trend line
        z = np.polyfit(uncertainty, y_data, 1)
        p = np.poly1d(z)
        x_line = np.linspace(uncertainty.min(), uncertainty.max(), 100)
        ax.plot(x_line, p(x_line), "steelblue", linestyle="--", linewidth=2, alpha=0.7)

        ax.set_xlabel("Probe Uncertainty", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(f"{ylabel} vs Probe Uncertainty ({note})", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_batch_metric_boxplots(results: GuaranteedRiskCascadeResults) -> Figure:
    """Box plots for per-batch budget cost, accuracy, F1, ROC-AUC."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Batch Metric Distributions", fontsize=16, fontweight="bold")

    metrics = [
        (np.array([b.budget_cost for b in results.batches]), "Budget Cost"),
        (np.array([b.accuracy for b in results.batches]), "Accuracy"),
        (np.array([b.f1_score for b in results.batches]), "F1 Score"),
        (np.array([b.roc_auc for b in results.batches]), "ROC-AUC"),
    ]

    for ax, (data, title) in zip(axes.flat, metrics, strict=False):
        bp = ax.boxplot([data], labels=["Cascade"], patch_artist=True, widths=0.5)

        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][0].set_alpha(0.7)
        for whisker in bp["whiskers"]:
            whisker.set(linewidth=1.5)
        for cap in bp["caps"]:
            cap.set(linewidth=1.5)
        for median in bp["medians"]:
            median.set(color="red", linewidth=2)

        ax.set_ylabel(title, fontweight="bold")
        ax.set_title(f"{title} Distribution", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        stats_text = f"\u03bc={data.mean():.3f}, \u03c3={data.std():.3f}"
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


def plot_cascade_vs_probe(results: GuaranteedRiskCascadeResults) -> Figure:
    """Scatter: cascade metric vs probe metric per batch (3 subplots)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cascade vs Probe Performance (per batch)", fontsize=16, fontweight="bold")

    metrics = [
        ("accuracy", "probe_accuracy", "Accuracy"),
        ("f1_score", "probe_f1_score", "F1 Score"),
        ("roc_auc", "probe_roc_auc", "ROC-AUC"),
    ]

    for ax, (cascade_attr, probe_attr, label) in zip(axes, metrics, strict=False):
        probe_vals = np.array([getattr(b, probe_attr) for b in results.batches])
        cascade_vals = np.array([getattr(b, cascade_attr) for b in results.batches])

        ax.scatter(probe_vals, cascade_vals, alpha=0.6, s=100, color="steelblue")

        # Trend line
        z = np.polyfit(probe_vals, cascade_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(probe_vals.min(), probe_vals.max(), 100)
        ax.plot(x_line, p(x_line), "steelblue", linestyle="--", linewidth=2, alpha=0.7)

        # y = x reference
        all_vals = np.concatenate([probe_vals, cascade_vals])
        diag_min, diag_max = all_vals.min(), all_vals.max()
        ax.plot([diag_min, diag_max], [diag_min, diag_max], "r--", linewidth=1, alpha=0.5, label="No improvement")

        ax.set_xlabel(f"Probe {label}", fontweight="bold")
        ax.set_ylabel(f"Cascade {label}", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
