"""Plotting utilities for SGT cascade experiment results.

Generates visualizations specific to the Sequential Graphical Testing
cascade experiment. All functions are pure and return matplotlib figures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from sgt_cascade import SGTCascadeResults


def plot_rejection_heatmap(results: SGTCascadeResults) -> Figure:
    """Heatmap of the (threshold x alpha) grid showing rejected hypotheses.

    Rejected cells are green, non-rejected are red. The selected best
    pair is highlighted with a star marker.
    """
    n_t = results.n_thresholds
    n_a = results.n_alphas
    grid = np.zeros((n_t, n_a))
    for t_idx, a_idx in results.rejected_pairs:
        grid[t_idx, a_idx] = 1.0

    fig, ax = plt.subplots(figsize=(max(8, n_a * 0.6), max(6, n_t * 0.4)))

    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")

    # Axis labels
    alpha_labels = [f"{a:.3f}" for a in results.ordered_alphas]
    threshold_labels = [f"{t:.3f}" for t in results.ordered_thresholds]

    step_a = max(1, n_a // 10)
    step_t = max(1, n_t // 10)
    ax.set_xticks(range(0, n_a, step_a))
    ax.set_xticklabels(alpha_labels[::step_a], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(0, n_t, step_t))
    ax.set_yticklabels(threshold_labels[::step_t], fontsize=8)

    ax.set_xlabel("Alpha (risk bound)", fontweight="bold")
    ax.set_ylabel("Threshold", fontweight="bold")
    ax.set_title(
        f"SGT Rejection Map ({results.n_rejected}/{results.n_hypotheses} rejected)\n"
        f"Graph: {results.sgt_graph_type}, Risk: {results.guaranteed_risk_name}",
        fontweight="bold",
    )

    # Mark the selected best pair
    best_t_idx = int(np.argmin(np.abs(results.ordered_thresholds - results.reliable_threshold)))
    best_a_idx = int(np.argmin(np.abs(results.ordered_alphas - results.achieved_alpha)))
    ax.plot(best_a_idx, best_t_idx, marker="*", markersize=18, color="gold", markeredgecolor="black", linewidth=1.5)
    ax.annotate(
        f"Best: t={results.reliable_threshold:.3f}, a={results.achieved_alpha:.3f}",
        xy=(best_a_idx, best_t_idx),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    plt.colorbar(im, ax=ax, label="Rejected", ticks=[0, 1], format=lambda x, _: "Yes" if x > 0.5 else "No")
    plt.tight_layout()
    return fig


def plot_budget_cost_distribution(results: SGTCascadeResults) -> Figure:
    """Histogram of per-batch budget costs with summary statistics."""
    budget_costs = np.array([b.budget_cost for b in results.batches])

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = max(5, len(budget_costs) // 3)
    ax.hist(budget_costs, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")

    ax.axvline(budget_costs.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {budget_costs.mean():.3f}")
    ax.axvline(
        np.median(budget_costs),
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Median: {np.median(budget_costs):.3f}",
    )

    ax.set_xlabel("Budget Cost (fraction using baseline)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(
        f"Budget Cost Distribution (threshold={results.reliable_threshold:.3f})",
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    stats_text = f"n={len(budget_costs)}, std={budget_costs.std():.3f}\nmin={budget_costs.min():.3f}, max={budget_costs.max():.3f}"
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
    return fig


def plot_performance_comparison(results: SGTCascadeResults) -> Figure:
    """Bar chart comparing probe-only, baseline-only, and cascade performance."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Performance Comparison: Probe vs Baseline vs SGT Cascade", fontsize=14, fontweight="bold")

    metrics = [
        ("Accuracy", results.probe_only_accuracy, results.baseline_only_accuracy, results.cascade_accuracy),
        ("F1 Score", results.probe_only_f1_score, results.baseline_only_f1_score, results.cascade_f1_score),
        ("ROC-AUC", results.probe_only_roc_auc, results.baseline_only_roc_auc, results.cascade_roc_auc),
    ]

    colors = ["steelblue", "coral", "forestgreen"]
    labels = ["Probe Only", "Baseline Only", "SGT Cascade"]

    for ax, (metric_name, probe_val, baseline_val, cascade_val) in zip(axes.flat, metrics, strict=False):
        x = np.arange(len(labels))
        values = [probe_val, baseline_val, cascade_val]
        bars = ax.bar(x, values, 0.6, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)

        ax.set_ylabel(metric_name, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        for bar, value in zip(bars, values, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    return fig


def plot_probe_score_histograms(results: SGTCascadeResults) -> dict[str, Figure]:
    """Probe score histograms per dataset split, colored by label."""
    figures: dict[str, Figure] = {}

    datasets = {
        "train": (results.train_probe_scores, results.train_labels),
        "calibration": (results.calib_probe_scores, results.calib_labels),
        "test": (results.test_probe_scores, results.test_labels),
    }

    for name, (scores, labels) in datasets.items():
        labels = np.asarray(labels)
        scores = np.asarray(scores)

        fig, ax = plt.subplots(figsize=(10, 6))
        scores_0 = scores[labels == 0]
        scores_1 = scores[labels == 1]
        bins = max(10, int(np.sqrt(len(scores)) / 2)) if len(scores) > 0 else 10

        ax.hist(scores_0, bins=bins, alpha=0.6, label="Target 0", color="steelblue", edgecolor="black")
        ax.hist(scores_1, bins=bins, alpha=0.6, label="Target 1", color="red", edgecolor="black")

        ax.set_title(f"Probe Score Distribution ({name.title()} Set)", fontweight="bold")
        ax.set_xlabel("Probe Score", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        figures[name] = fig

    return figures


def plot_roc_curves(results: SGTCascadeResults) -> dict[str, Figure]:
    """ROC curves for probe, baseline, and cascade scores on the test set."""
    from sklearn.metrics import roc_auc_score, roc_curve

    figures: dict[str, Figure] = {}
    labels = np.asarray(results.test_labels)

    score_sets = {
        "probe": (results.test_probe_scores, "Probe Only"),
        "baseline": (results.test_baseline_scores, "Baseline Only"),
        "cascade": (results.cascade_final_scores, "SGT Cascade"),
    }

    for key, (scores, title) in score_sets.items():
        scores = np.asarray(scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.2, color="steelblue")

        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.set_title(f"ROC Curve - {title}", fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        figures[key] = fig

    return figures


def plot_empirical_risk_vs_threshold(results: SGTCascadeResults) -> Figure:
    """Plot empirical risk as a function of threshold from calibration data.

    Shows the risk curve with the selected threshold marked, and the
    achieved alpha bound as a horizontal line.
    """
    eval_result = results.calib_evaluation_risks
    risk_name = results.guaranteed_risk_name
    thresholds = eval_result.thresholds
    risks = eval_result[risk_name]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, risks, "o-", color="steelblue", linewidth=2, markersize=6, label=f"Empirical {risk_name}")

    ax.axvline(
        results.reliable_threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Selected threshold: {results.reliable_threshold:.3f}",
    )
    ax.axhline(
        results.achieved_alpha,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Achieved alpha: {results.achieved_alpha:.3f}",
    )

    ax.set_xlabel("Threshold", fontweight="bold")
    ax.set_ylabel(f"Empirical {risk_name}", fontweight="bold")
    ax.set_title(f"Calibration Risk Curve ({risk_name})", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_performance_heatmaps(results: SGTCascadeResults) -> dict[str, Figure]:
    """Heatmaps of cascade performance metrics across the (threshold x alpha) grid.

    One heatmap per metric (accuracy, f1_score, roc_auc, budget_cost).
    Rejected cells are colored by metric value; non-rejected cells are dark grey.
    The selected best pair is highlighted with a star marker.

    Requires ``results.threshold_results`` (per-threshold cascade results).
    """
    if not hasattr(results, "threshold_results") or not results.threshold_results:
        return {}

    n_t = results.n_thresholds
    n_a = results.n_alphas

    # Build lookup: threshold value -> ThresholdCascadeResult
    from cascade_utils import ThresholdCascadeResult

    tr_lookup: dict[float, ThresholdCascadeResult] = {}
    for tr in results.threshold_results:
        tr_lookup[round(tr.threshold, 6)] = tr

    # Build rejection mask
    rejected = np.zeros((n_t, n_a), dtype=bool)
    for t_idx, a_idx in results.rejected_pairs:
        rejected[t_idx, a_idx] = True

    metrics = {
        "accuracy": ("Cascade Accuracy", "YlGn"),
        "f1_score": ("Cascade F1 Score", "YlGn"),
        "roc_auc": ("Cascade ROC-AUC", "YlGn"),
        "budget_cost": ("Budget Cost", "YlOrRd"),
    }

    # Selected pair indices for star marker
    best_t_idx = int(np.argmin(np.abs(results.ordered_thresholds - results.reliable_threshold)))
    best_a_idx = int(np.argmin(np.abs(results.ordered_alphas - results.achieved_alpha)))

    alpha_labels = [f"{a:.3f}" for a in results.ordered_alphas]
    threshold_labels = [f"{t:.3f}" for t in results.ordered_thresholds]
    step_a = max(1, n_a // 10)
    step_t = max(1, n_t // 10)

    figures: dict[str, Figure] = {}
    for metric_key, (title, cmap_name) in metrics.items():
        grid = np.full((n_t, n_a), np.nan)

        for t_idx in range(n_t):
            thr_val = round(float(results.ordered_thresholds[t_idx]), 6)
            tr = tr_lookup.get(thr_val)
            if tr is None:
                continue
            # Fill all valid alpha cells for this threshold with the metric value
            if metric_key == "budget_cost":
                val = tr.mean_budget_cost
            else:
                val = getattr(tr, f"cascade_{metric_key}")
            for a_idx in range(n_a):
                if rejected[t_idx, a_idx]:
                    grid[t_idx, a_idx] = val

        fig, ax = plt.subplots(figsize=(max(8, n_a * 0.6), max(6, n_t * 0.4)))

        # Background: dark grey for non-rejected
        bg = np.ones((n_t, n_a, 3)) * 0.2
        ax.imshow(bg, aspect="auto", origin="lower")

        # Overlay: metric values for rejected cells
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color=(0.2, 0.2, 0.2))  # NaN = dark grey
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, aspect="auto", cmap=cmap, origin="lower")

        ax.set_xticks(range(0, n_a, step_a))
        ax.set_xticklabels(alpha_labels[::step_a], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(0, n_t, step_t))
        ax.set_yticklabels(threshold_labels[::step_t], fontsize=8)
        ax.set_xlabel("Alpha (risk bound)", fontweight="bold")
        ax.set_ylabel("Threshold", fontweight="bold")
        ax.set_title(f"{title} across SGT Grid", fontweight="bold")

        # Mark the selected best pair
        ax.plot(best_a_idx, best_t_idx, marker="*", markersize=18, color="gold", markeredgecolor="black", linewidth=1.5)

        plt.colorbar(im, ax=ax, label=title)
        plt.tight_layout()
        figures[metric_key] = fig

    return figures


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def make_sgt_figures(results: SGTCascadeResults) -> dict[str, Figure | None | dict[str, Figure]]:
    """Generate all SGT cascade figures."""
    figures: dict[str, Figure | None | dict[str, Figure]] = {}

    figures["rejection_heatmap"] = plot_rejection_heatmap(results)
    figures["budget_distribution"] = plot_budget_cost_distribution(results)
    figures["performance_comparison"] = plot_performance_comparison(results)
    figures["risk_vs_threshold"] = plot_empirical_risk_vs_threshold(results)
    figures["probe_score_hists"] = plot_probe_score_histograms(results)
    figures["roc_curves"] = plot_roc_curves(results)
    figures["performance_heatmaps"] = plot_performance_heatmaps(results)

    # Pareto frontier (only when Pareto testing was used)
    if results.opt_evaluation_risks is not None and results.pareto_mask is not None:
        from plot_utils import plot_pareto_frontier

        opt_risk_name = results.config.get("opt_risk", "budget")
        figures["pareto"] = plot_pareto_frontier(
            results.opt_evaluation_risks,
            results.pareto_mask,
            results.guaranteed_risk_name,
            opt_risk_name,
        )
    else:
        figures["pareto"] = None

    return figures


def log_sgt_figures_to_clearml(clearml_logger, figures: dict[str, Figure | None | dict[str, Figure]]) -> None:
    """Log all SGT figures to ClearML."""
    clearml_logger.log_figure(title="SGT", series="Rejection Heatmap", figure=figures["rejection_heatmap"])
    clearml_logger.log_figure(title="SGT", series="Budget Cost Distribution", figure=figures["budget_distribution"])
    clearml_logger.log_figure(title="SGT", series="Performance Comparison", figure=figures["performance_comparison"])
    clearml_logger.log_figure(title="SGT", series="Risk vs Threshold", figure=figures["risk_vs_threshold"])

    for dataset_name, fig in figures["probe_score_hists"].items():  # type: ignore[union-attr]
        clearml_logger.log_figure(title="Probe Score Histograms", series=f"Probe Scores ({dataset_name})", figure=fig)

    for score_type, fig in figures["roc_curves"].items():  # type: ignore[union-attr]
        clearml_logger.log_figure(title="ROC Curves", series=f"ROC ({score_type.title()})", figure=fig)

    for metric_name, fig in figures.get("performance_heatmaps", {}).items():  # type: ignore[union-attr]
        clearml_logger.log_figure(title="Performance Heatmaps", series=metric_name, figure=fig)

    if figures.get("pareto") is not None:
        clearml_logger.log_figure(title="Pareto Frontier", series="Pareto Frontier", figure=figures["pareto"])

    plt.close("all")
