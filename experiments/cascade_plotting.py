"""Plotting utilities for cascade comparison experiment results.

This module generates visualizations for easy comparison of adaptive vs fixed
budget cascade strategies. All plotting functions are pure (no side effects)
and take CascadeComparisonResults as input.

Functions generate matplotlib figures ready for logging to ClearML.
"""

import matplotlib.pyplot as plt
import numpy as np
from cascade_comparison import CascadeComparisonResults
from matplotlib.figure import Figure


def plot_pareto_frontier(results: CascadeComparisonResults) -> Figure:
    """Plot Pareto frontier from optimization set evaluation.

    Note: This function should only be called when results.opt_evaluation_risks
    and results.pareto_mask are not None.
    """
    assert results.opt_evaluation_risks is not None, "opt_evaluation_risks must not be None"
    assert results.pareto_mask is not None, "pareto_mask must not be None"

    fig, ax = plt.subplots(figsize=(8, 6))

    # Get 2D empirical risks array (n_thresholds, 2)
    empirical_risks_2d = results.opt_evaluation_risks.get_empirical_risks_array()
    pareto_mask = results.pareto_mask

    # Plot all points
    ax.scatter(
        empirical_risks_2d[:, 0],
        empirical_risks_2d[:, 1],
        alpha=0.5,
        s=80,
        c="gray",
        label="Dominated",
    )

    # Highlight Pareto-efficient points
    ax.scatter(
        empirical_risks_2d[pareto_mask, 0],
        empirical_risks_2d[pareto_mask, 1],
        s=100,
        c="red",
        label="Pareto-efficient",
    )

    ax.set_xlabel("Budget Cost")
    ax.set_ylabel("1-Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_overall_performance_comparison(results: CascadeComparisonResults) -> Figure:
    """Generate bar chart comparing overall performance metrics across all approaches.

    Shows performance comparison for four approaches:
    - Probe only (baseline using just probe)
    - Baseline only (all examples use baseline)
    - Adaptive cascade (adaptive threshold selection)
    - Fixed cascade (fixed budget selection)

    Metrics plotted:
    - Accuracy (higher is better)
    - F1 Score (higher is better)
    - ROC-AUC (higher is better)

    Args:
        results: CascadeComparisonResults from experiment

    Returns:
        Matplotlib figure with grouped bar chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Overall Performance Comparison: Probe vs Baseline vs Cascades", fontsize=14, fontweight="bold")

    metrics = [
        (
            "Accuracy",
            ["probe_only_accuracy", "baseline_only_accuracy", "adaptive_overall_accuracy", "fixed_overall_accuracy"],
        ),
        (
            "F1 Score",
            ["probe_only_f1_score", "baseline_only_f1_score", "adaptive_overall_f1_score", "fixed_overall_f1_score"],
        ),
        (
            "ROC-AUC",
            ["probe_only_roc_auc", "baseline_only_roc_auc", "adaptive_overall_roc_auc", "fixed_overall_roc_auc"],
        ),
    ]

    colors = ["steelblue", "coral", "forestgreen", "orange"]
    labels = ["Probe Only", "Baseline Only", "Adaptive Cascade", "Fixed Cascade"]

    for ax, (metric_name, field_names) in zip(axes.flat, metrics, strict=False):
        x = np.arange(len(labels))
        width = 0.6
        values = [getattr(results, field_name) for field_name in field_names]

        bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)

        # Labels and formatting
        ax.set_ylabel(metric_name, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        # Add value labels on bars
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


def plot_summary_comparison(results: CascadeComparisonResults) -> Figure:
    """Generate grouped bar chart comparing summary statistics.

    Shows min/avg/max for each metric side-by-side for adaptive vs fixed.

    Metrics plotted:
    - Budget Cost (%, lower is better for efficiency)
    - Accuracy (%, higher is better)
    - F1 Score (%, higher is better)
    - ROC-AUC (%, higher is better)

    Args:
        results: CascadeComparisonResults from experiment

    Returns:
        Matplotlib figure with grouped bar chart
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cascade Comparison: Summary Statistics", fontsize=16, fontweight="bold")

    metrics = [
        (
            "Budget Cost",
            [
                "adaptive_min_budget_cost",
                "adaptive_mean_budget_cost",
                "adaptive_max_budget_cost",
                "fixed_min_budget_cost",
                "fixed_mean_budget_cost",
                "fixed_max_budget_cost",
            ],
        ),
        (
            "Accuracy",
            [
                "adaptive_best_batch_accuracy",
                "adaptive_worst_batch_accuracy",
                "fixed_best_batch_accuracy",
                "fixed_worst_batch_accuracy",
            ],
        ),
        (
            "F1 Score",
            [
                "adaptive_best_batch_f1_score",
                "adaptive_worst_batch_f1_score",
                "fixed_best_batch_f1_score",
                "fixed_worst_batch_f1_score",
            ],
        ),
        (
            "ROC-AUC",
            [
                "adaptive_best_batch_roc_auc",
                "adaptive_worst_batch_roc_auc",
                "fixed_best_batch_roc_auc",
                "fixed_worst_batch_roc_auc",
            ],
        ),
    ]

    for _idx, (metric_name, ax) in enumerate(zip([m[0] for m in metrics], axes.flat, strict=False)):
        # Extract data for grouped bar chart
        if metric_name == "Budget Cost":
            x = np.arange(3)  # min, avg, max
            width = 0.35
            adaptive = [
                results.adaptive_min_budget_cost,
                results.adaptive_mean_budget_cost,
                results.adaptive_max_budget_cost,
            ]
            fixed = [results.fixed_min_budget_cost, results.fixed_mean_budget_cost, results.fixed_max_budget_cost]
            labels = ["Min", "Avg", "Max"]
        else:
            # For performance metrics, use best/worst
            x = np.arange(2)  # best, worst
            width = 0.35
            if metric_name == "Accuracy":
                adaptive = [results.adaptive_best_batch_accuracy, results.adaptive_worst_batch_accuracy]
                fixed = [results.fixed_best_batch_accuracy, results.fixed_worst_batch_accuracy]
            elif metric_name == "F1 Score":
                adaptive = [results.adaptive_best_batch_f1_score, results.adaptive_worst_batch_f1_score]
                fixed = [results.fixed_best_batch_f1_score, results.fixed_worst_batch_f1_score]
            else:  # ROC-AUC
                adaptive = [results.adaptive_best_batch_roc_auc, results.adaptive_worst_batch_roc_auc]
                fixed = [results.fixed_best_batch_roc_auc, results.fixed_worst_batch_roc_auc]
            labels = ["Best", "Worst"]

        # Plot bars
        ax.bar(x - width / 2, adaptive, width, label="Adaptive", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, fixed, width, label="Fixed", color="orange", alpha=0.8)

        # Labels and formatting
        ax.set_ylabel(metric_name, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        # Add value labels on bars
        for bars in [ax.patches[i::2] for i in range(2)]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom", fontsize=9
                )

    plt.tight_layout()
    return fig


def plot_batch_distributions(results: CascadeComparisonResults) -> dict[str, Figure]:
    """Generate overlaying histograms for batch metric distributions.

    Creates histograms showing adaptive (blue) and fixed (orange) distributions
    overlaid on the same chart for easy comparison.

    Metrics plotted:
    - budget_cost: Distribution of budget costs per batch
    - accuracy: Distribution of accuracies per batch
    - f1_score: Distribution of F1 scores per batch
    - roc_auc: Distribution of ROC-AUC scores per batch
    - probe_uncertainty: Distribution of probe uncertainties per batch

    Args:
        results: CascadeComparisonResults from experiment

    Returns:
        Dictionary mapping metric names to matplotlib figures
    """
    figures = {}

    # Extract per-batch metrics
    adaptive_budget = np.array([b.budget_cost for b in results.adaptive_batches])
    fixed_budget = np.array([b.budget_cost for b in results.fixed_batches])

    adaptive_accuracy = np.array([b.accuracy for b in results.adaptive_batches])
    fixed_accuracy = np.array([b.accuracy for b in results.fixed_batches])

    adaptive_f1 = np.array([b.f1_score for b in results.adaptive_batches])
    fixed_f1 = np.array([b.f1_score for b in results.fixed_batches])

    adaptive_roc_auc = np.array([b.roc_auc for b in results.adaptive_batches])
    fixed_roc_auc = np.array([b.roc_auc for b in results.fixed_batches])

    adaptive_uncertainty = np.array([b.probe_uncertainty_mean for b in results.adaptive_batches])
    fixed_uncertainty = np.array([b.probe_uncertainty_mean for b in results.fixed_batches])

    # Metrics to plot
    metrics_data = [
        ("budget_cost", adaptive_budget, fixed_budget, "Budget Cost", "Fraction"),
        ("accuracy", adaptive_accuracy, fixed_accuracy, "Accuracy", "Score"),
        ("f1_score", adaptive_f1, fixed_f1, "F1 Score", "Score"),
        ("roc_auc", adaptive_roc_auc, fixed_roc_auc, "ROC-AUC", "Score"),
        ("probe_uncertainty", adaptive_uncertainty, fixed_uncertainty, "Probe Uncertainty", "Mean Value"),
    ]

    for metric_key, adaptive_data, fixed_data, title, _ylabel in metrics_data:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot overlaying histograms
        bins = max(5, len(adaptive_data) // 2)
        ax.hist(adaptive_data, bins=bins, alpha=0.5, label="Adaptive", color="steelblue", edgecolor="black")
        ax.hist(fixed_data, bins=bins, alpha=0.5, label="Fixed", color="orange", edgecolor="black")

        # Labels and formatting
        ax.set_xlabel(title, fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title(f"Distribution of {title} Across Batches", fontweight="bold")
        ax.legend(loc="best", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        # Add statistics text
        stats_text = (
            f"Adaptive: μ={adaptive_data.mean():.3f}, σ={adaptive_data.std():.3f}\n"
            f"Fixed:    μ={fixed_data.mean():.3f}, σ={fixed_data.std():.3f}"
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
        figures[metric_key] = fig

    return figures


def plot_probe_uncertainty_vs_metrics(results: CascadeComparisonResults) -> Figure:
    """Generate scatter plots showing relationship between probe uncertainty and metrics.

    Shows 4 subplots:
    - Budget cost vs probe_uncertainty: Adaptive should correlate, fixed should be flat
    - Accuracy vs probe_uncertainty
    - F1 score vs probe_uncertainty
    - ROC-AUC vs probe_uncertainty

    Probe uncertainty is defined as min(p, 1-p), measuring closeness to decision boundary.
    Higher values indicate higher classification uncertainty (closer to 0.5 decision threshold).

    Args:
        results: CascadeComparisonResults from experiment

    Returns:
        Matplotlib figure with 2x2 subplot grid
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Probe Uncertainty vs Performance: Adaptive vs Fixed Strategy\n(Probe Uncertainty = min(p, 1-p))",
        fontsize=16,
        fontweight="bold",
    )

    # Extract data
    adaptive_probe_uncertainty = np.array([b.probe_uncertainty_mean for b in results.adaptive_batches])
    fixed_probe_uncertainty = np.array([b.probe_uncertainty_mean for b in results.fixed_batches])

    adaptive_budget = np.array([b.budget_cost for b in results.adaptive_batches])
    fixed_budget = np.array([b.budget_cost for b in results.fixed_batches])

    adaptive_accuracy = np.array([b.accuracy for b in results.adaptive_batches])
    fixed_accuracy = np.array([b.accuracy for b in results.fixed_batches])

    adaptive_f1 = np.array([b.f1_score for b in results.adaptive_batches])
    fixed_f1 = np.array([b.f1_score for b in results.fixed_batches])

    adaptive_roc_auc = np.array([b.roc_auc for b in results.adaptive_batches])
    fixed_roc_auc = np.array([b.roc_auc for b in results.fixed_batches])

    # Scatter plots with best-fit lines
    metrics = [
        (adaptive_budget, fixed_budget, "Budget Cost", "Lower is better"),
        (adaptive_accuracy, fixed_accuracy, "Accuracy", "Higher is better"),
        (adaptive_f1, fixed_f1, "F1 Score", "Higher is better"),
        (adaptive_roc_auc, fixed_roc_auc, "ROC-AUC", "Higher is better"),
    ]

    for ax, (adaptive_y, fixed_y, ylabel, note) in zip(axes.flat, metrics, strict=False):
        # Scatter plots
        ax.scatter(adaptive_probe_uncertainty, adaptive_y, alpha=0.6, s=100, label="Adaptive", color="steelblue")
        ax.scatter(fixed_probe_uncertainty, fixed_y, alpha=0.6, s=100, label="Fixed", color="orange")

        # Best-fit lines
        z_adaptive = np.polyfit(adaptive_probe_uncertainty, adaptive_y, 1)
        p_adaptive = np.poly1d(z_adaptive)
        x_line = np.linspace(adaptive_probe_uncertainty.min(), adaptive_probe_uncertainty.max(), 100)
        ax.plot(x_line, p_adaptive(x_line), "steelblue", linestyle="--", linewidth=2, alpha=0.7)

        z_fixed = np.polyfit(fixed_probe_uncertainty, fixed_y, 1)
        p_fixed = np.poly1d(z_fixed)
        ax.plot(x_line, p_fixed(x_line), "orange", linestyle="--", linewidth=2, alpha=0.7)

        # Labels and formatting
        ax.set_xlabel("Probe Uncertainty", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(f"{ylabel} vs Probe Uncertainty ({note})", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_metric_boxplots(results: CascadeComparisonResults) -> Figure:
    """Generate box plots comparing metric distributions.

    Shows quartiles, whiskers, and outliers for adaptive vs fixed strategies.

    Metrics plotted:
    - Budget Cost
    - Accuracy
    - F1 Score
    - ROC-AUC

    Args:
        results: CascadeComparisonResults from experiment

    Returns:
        Matplotlib figure with box plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Metric Range Comparison: Adaptive vs Fixed", fontsize=16, fontweight="bold")

    # Extract per-batch metrics
    adaptive_budget = np.array([b.budget_cost for b in results.adaptive_batches])
    fixed_budget = np.array([b.budget_cost for b in results.fixed_batches])

    adaptive_accuracy = np.array([b.accuracy for b in results.adaptive_batches])
    fixed_accuracy = np.array([b.accuracy for b in results.fixed_batches])

    adaptive_f1 = np.array([b.f1_score for b in results.adaptive_batches])
    fixed_f1 = np.array([b.f1_score for b in results.fixed_batches])

    adaptive_roc_auc = np.array([b.roc_auc for b in results.adaptive_batches])
    fixed_roc_auc = np.array([b.roc_auc for b in results.fixed_batches])

    # Metrics to plot
    metrics = [
        (adaptive_budget, fixed_budget, "Budget Cost"),
        (adaptive_accuracy, fixed_accuracy, "Accuracy"),
        (adaptive_f1, fixed_f1, "F1 Score"),
        (adaptive_roc_auc, fixed_roc_auc, "ROC-AUC"),
    ]

    for ax, (adaptive_data, fixed_data, title) in zip(axes.flat, metrics, strict=False):
        # Create box plots
        bp = ax.boxplot(
            [adaptive_data, fixed_data],
            labels=["Adaptive", "Fixed"],
            patch_artist=True,
            widths=0.6,
        )

        # Color the boxes
        colors = ["steelblue", "orange"]
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Format whiskers and caps
        for whisker in bp["whiskers"]:
            whisker.set(linewidth=1.5)
        for cap in bp["caps"]:
            cap.set(linewidth=1.5)
        for median in bp["medians"]:
            median.set(color="red", linewidth=2)

        # Labels and formatting
        ax.set_ylabel(title, fontweight="bold")
        ax.set_title(f"{title} Distribution", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        # Add statistics
        stats_text = (
            f"Adaptive: μ={adaptive_data.mean():.3f}, σ={adaptive_data.std():.3f}\n"
            f"Fixed:    μ={fixed_data.mean():.3f}, σ={fixed_data.std():.3f}"
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


def plot_paired_method_comparison(results: CascadeComparisonResults) -> Figure:
    """Generate scatter plots comparing paired performance: adaptive vs fixed on same batches.

    Shows subplots where each dot represents a batch:
    - X-axis: metric value for adaptive method
    - Y-axis: metric value for fixed method
    - Diagonal line: where both methods perform equally
    - P-values from paired t-tests displayed on plots

    Metrics plotted:
    - accuracy: Accuracy score (higher is better)
    - f1_score: F1 score (higher is better)
    - roc_auc: ROC-AUC score (higher is better)

    Args:
        results: CascadeComparisonResults from experiment (including t-test results)

    Returns:
        Matplotlib figure with 1x3 subplot grid
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Paired Method Comparison: Adaptive vs Fixed (per batch)", fontsize=16, fontweight="bold")

    # Extract metrics from paired batches
    adaptive_accuracy = np.array([b.accuracy for b in results.adaptive_batches])
    fixed_accuracy = np.array([b.accuracy for b in results.fixed_batches])

    adaptive_f1 = np.array([b.f1_score for b in results.adaptive_batches])
    fixed_f1 = np.array([b.f1_score for b in results.fixed_batches])

    adaptive_roc_auc = np.array([b.roc_auc for b in results.adaptive_batches])
    fixed_roc_auc = np.array([b.roc_auc for b in results.fixed_batches])

    # Metrics to plot: (adaptive_data, fixed_data, title, better_direction, t_stat_field, p_value_field, mean_diff_field)
    metrics = [
        (
            adaptive_accuracy,
            fixed_accuracy,
            "Accuracy",
            "higher",
            results.accuracy_t_stat,
            results.accuracy_p_value,
            results.accuracy_mean_diff,
        ),
        (
            adaptive_f1,
            fixed_f1,
            "F1 Score",
            "higher",
            results.f1_score_t_stat,
            results.f1_score_p_value,
            results.f1_score_mean_diff,
        ),
        (
            adaptive_roc_auc,
            fixed_roc_auc,
            "ROC-AUC",
            "higher",
            results.roc_auc_t_stat,
            results.roc_auc_p_value,
            results.roc_auc_mean_diff,
        ),
    ]

    # Batch indices for coloring
    batch_indices = np.arange(len(results.adaptive_batches))

    for ax, (adaptive_data, fixed_data, title, better_dir, t_stat, p_value, mean_diff) in zip(
        axes, metrics, strict=False
    ):
        # Scatter plot colored by batch index
        scatter = ax.scatter(
            adaptive_data,
            fixed_data,
            c=batch_indices,
            s=120,
            alpha=0.6,
            cmap="viridis",
            edgecolors="black",
            linewidth=0.5,
        )

        # Diagonal line where both methods are equal
        min_val = min(adaptive_data.min(), fixed_data.min())
        max_val = max(adaptive_data.max(), fixed_data.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, alpha=0.5, label="Equal performance")

        # Labels and formatting
        ax.set_xlabel(f"Adaptive {title}", fontweight="bold")
        ax.set_ylabel(f"Fixed {title}", fontweight="bold")
        ax.set_title(f"{title} Pairing ({better_dir} is better)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9)

        # Add colorbar for batch indices
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Batch Index", fontweight="bold")

        # Add statistics
        if better_dir == "higher":
            adaptive_wins = (adaptive_data > fixed_data).sum()
        else:  # lower is better
            adaptive_wins = (adaptive_data < fixed_data).sum()

        win_pct = 100 * adaptive_wins / len(adaptive_data)

        # Determine significance level
        sig_stars = ""
        if p_value < 0.001:
            sig_stars = "***"
        elif p_value < 0.01:
            sig_stars = "**"
        elif p_value < 0.05:
            sig_stars = "*"

        stats_text = (
            f"Adaptive: μ={adaptive_data.mean():.3f}, σ={adaptive_data.std():.3f}\n"
            f"Fixed:    μ={fixed_data.mean():.3f}, σ={fixed_data.std():.3f}\n"
            f"Mean Δ: {mean_diff:.4f}\n"
            f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}{sig_stars}\n"
            f"Adaptive wins: {int(adaptive_wins)}/{len(adaptive_data)} ({win_pct:.1f}%)"
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


def plot_cascade_vs_probe_performance(results: CascadeComparisonResults) -> Figure:
    """Generate scatter plots comparing cascade performance vs probe performance.

    Shows 3 sublots, one for each metric (accuracy, f1_score, roc_auc)
    For each metric subplot:
    - X-axis: Probe metric
    - Y-axis: Cascade metric
    Each subplot has a scatterplot for adaptive (blue) and fixed (orange) strategies with best-fit lines.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cascade vs Probe Performance", fontsize=16, fontweight="bold")

    metrics = [
        ("accuracy", "probe_accuracy", "Accuracy"),
        ("f1_score", "probe_f1_score", "F1 Score"),
        ("roc_auc", "probe_roc_auc", "ROC-AUC"),
    ]

    for ax, (cascade_attr, probe_attr, label) in zip(axes, metrics, strict=False):
        # Extract data
        adaptive_probe = np.array([getattr(b, probe_attr) for b in results.adaptive_batches])
        adaptive_cascade = np.array([getattr(b, cascade_attr) for b in results.adaptive_batches])
        fixed_probe = np.array([getattr(b, probe_attr) for b in results.fixed_batches])
        fixed_cascade = np.array([getattr(b, cascade_attr) for b in results.fixed_batches])

        # Scatter plots
        ax.scatter(adaptive_probe, adaptive_cascade, alpha=0.6, s=100, color="steelblue", label="Adaptive")
        ax.scatter(fixed_probe, fixed_cascade, alpha=0.6, s=100, color="orange", label="Fixed")

        # Best-fit lines
        z_adaptive = np.polyfit(adaptive_probe, adaptive_cascade, 1)
        p_adaptive = np.poly1d(z_adaptive)
        z_fixed = np.polyfit(fixed_probe, fixed_cascade, 1)
        p_fixed = np.poly1d(z_fixed)

        x_min = min(adaptive_probe.min(), fixed_probe.min())
        x_max = max(adaptive_probe.max(), fixed_probe.max())
        x_line = np.linspace(x_min, x_max, 100)

        ax.plot(x_line, p_adaptive(x_line), "steelblue", linestyle="--", linewidth=2, alpha=0.7)
        ax.plot(x_line, p_fixed(x_line), "orange", linestyle="--", linewidth=2, alpha=0.7)

        # Diagonal line (no improvement)
        y_min = min(adaptive_cascade.min(), fixed_cascade.min())
        y_max = max(adaptive_cascade.max(), fixed_cascade.max())
        diag_min = min(x_min, y_min)
        diag_max = max(x_max, y_max)
        ax.plot([diag_min, diag_max], [diag_min, diag_max], "r--", linewidth=1, alpha=0.5, label="No improvement")

        # Labels
        ax.set_xlabel(f"Probe {label}", fontweight="bold")
        ax.set_ylabel(f"Cascade {label}", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_gain_per_budget(results: CascadeComparisonResults, metric: str = "accuracy") -> Figure:
    """Generate scatter plots showing performance gain per unit budget spent.

    Shows efficiency: how much improvement per unit of budget cost.
    X-axis: Adaptive gain/budget vs Y-axis: Fixed gain/budget
    Points above diagonal show where adaptive is more efficient.

    Args:
        results: CascadeComparisonResults from experiment
        metric: Performance metric to use ('accuracy', 'f1_score', or 'roc_auc')

    Returns:
        Matplotlib figure with efficiency comparison
    """
    # Validate metric
    valid_metrics = ["accuracy", "f1_score", "roc_auc"]
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got {metric}")

    # Map metric names to attribute names
    metric_map = {
        "accuracy": ("accuracy", "probe_accuracy"),
        "f1_score": ("f1_score", "probe_f1_score"),
        "roc_auc": ("roc_auc", "probe_roc_auc"),
    }
    cascade_attr, probe_attr = metric_map[metric]

    # Extract data
    adaptive_cascade = np.array([getattr(b, cascade_attr) for b in results.adaptive_batches])
    fixed_cascade = np.array([getattr(b, cascade_attr) for b in results.fixed_batches])
    adaptive_probe = np.array([getattr(b, probe_attr) for b in results.adaptive_batches])
    fixed_probe = np.array([getattr(b, probe_attr) for b in results.fixed_batches])
    adaptive_budget = np.array([b.budget_cost for b in results.adaptive_batches])
    fixed_budget = np.array([b.budget_cost for b in results.fixed_batches])

    # Compute gains
    adaptive_gain = adaptive_cascade - adaptive_probe
    fixed_gain = fixed_cascade - fixed_probe

    # Compute efficiency (gain per unit budget)
    # Avoid division by zero
    adaptive_efficiency = np.divide(
        adaptive_gain, adaptive_budget, where=adaptive_budget > 0, out=np.zeros_like(adaptive_gain)
    )
    fixed_efficiency = np.divide(fixed_gain, fixed_budget, where=fixed_budget > 0, out=np.zeros_like(fixed_gain))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    metric_display = metric.replace("_", " ").title()
    fig.suptitle(f"{metric_display} Gain per Unit Budget: Efficiency Comparison", fontsize=14, fontweight="bold")

    # Plot 1: Gain per budget vs Probe performance
    ax = axes[0]
    ax.scatter(
        adaptive_probe,
        adaptive_efficiency,
        alpha=0.6,
        s=100,
        color="steelblue",
        edgecolors="black",
        linewidth=0.5,
        label="Adaptive",
    )
    ax.scatter(
        fixed_probe,
        fixed_efficiency,
        alpha=0.6,
        s=100,
        color="orange",
        edgecolors="black",
        linewidth=0.5,
        label="Fixed",
    )
    z_adaptive = np.polyfit(adaptive_probe, adaptive_efficiency, 1)
    p_adaptive = np.poly1d(z_adaptive)
    x_line = np.linspace(
        min(adaptive_probe.min(), fixed_probe.min()), max(adaptive_probe.max(), fixed_probe.max()), 100
    )
    ax.plot(x_line, p_adaptive(x_line), "steelblue", linestyle="--", linewidth=2, alpha=0.7)

    z_fixed = np.polyfit(fixed_probe, fixed_efficiency, 1)
    p_fixed = np.poly1d(z_fixed)
    ax.plot(x_line, p_fixed(x_line), "orange", linestyle="--", linewidth=2, alpha=0.7)

    ax.axhline(y=0, color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel(f"Probe {metric_display} (Batch Difficulty)", fontweight="bold")
    ax.set_ylabel(f"{metric_display} Gain / Budget Cost", fontweight="bold")
    ax.set_title("Efficiency vs Batch Difficulty", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Statistics
    stats_text = (
        f"Adaptive: μ={adaptive_efficiency.mean():.4f}, σ={adaptive_efficiency.std():.4f}\n"
        f"Fixed:    μ={fixed_efficiency.mean():.4f}, σ={fixed_efficiency.std():.4f}"
    )
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )

    # Plot 2: Paired efficiency comparison
    ax = axes[1]
    batch_indices = np.arange(len(results.adaptive_batches))
    scatter = ax.scatter(
        adaptive_efficiency,
        fixed_efficiency,
        c=batch_indices,
        s=120,
        alpha=0.6,
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
    )

    # Diagonal line (equal efficiency)
    min_val = min(adaptive_efficiency.min(), fixed_efficiency.min())
    max_val = max(adaptive_efficiency.max(), fixed_efficiency.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, alpha=0.5, label="Equal efficiency")

    ax.set_xlabel("Adaptive Efficiency (Gain / Budget)", fontweight="bold")
    ax.set_ylabel("Fixed Efficiency (Gain / Budget)", fontweight="bold")
    ax.set_title("Efficiency Pairing: Adaptive vs Fixed", fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Batch Index", fontweight="bold")

    # Statistics
    adaptive_wins = (adaptive_efficiency > fixed_efficiency).sum()
    win_pct = 100 * adaptive_wins / len(adaptive_efficiency)
    stats_text = (
        f"Adaptive Mean Efficiency: {adaptive_efficiency.mean():.4f}\n"
        f"Fixed Mean Efficiency: {fixed_efficiency.mean():.4f}\n"
        f"Adaptive wins: {int(adaptive_wins)}/{len(adaptive_efficiency)} ({win_pct:.1f}%)"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )

    plt.tight_layout()
    return fig


def plot_probe_score_histograms(results: CascadeComparisonResults) -> dict[str, Figure]:
    """Plot probe score histograms for train, calibration, and test datasets.

    Each figure overlays two histograms by target label:
    - Target 0 in blue
    - Target 1 in red

    Returns:
        Dictionary mapping dataset name to matplotlib figure
    """
    figures: dict[str, Figure] = {}

    datasets = {
        "train": (results.train_probe_scores, results.train_labels),
        "calibration": (results.calib_probe_scores, results.calib_labels),
        "test": (results.test_probe_scores, results.test_labels),
    }

    for name, (scores, labels) in datasets.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = np.asarray(labels)
        scores = np.asarray(scores)

        scores_0 = scores[labels == 0]
        scores_1 = scores[labels == 1]

        bins = max(10, int(np.sqrt(len(scores))/2)) if len(scores) > 0 else 10

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


def plot_reliability_diagrams(results: CascadeComparisonResults, n_bins: int = 10) -> dict[str, Figure]:
    """Plot reliability diagrams (calibration plots) for probe scores.

    Generates one plot per dataset (train/calibration/test) with:
    - Reliability curve (mean predicted vs fraction positive)
    - Diagonal line for perfect calibration

    Args:
        results: CascadeComparisonResults
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary mapping dataset name to matplotlib figure
    """
    from sklearn.calibration import calibration_curve

    figures: dict[str, Figure] = {}

    datasets = {
        "train": (results.train_probe_scores, results.train_labels),
        "calibration": (results.calib_probe_scores, results.calib_labels),
        "test": (results.test_probe_scores, results.test_labels),
    }

    for name, (scores, labels) in datasets.items():
        scores = np.asarray(scores)
        labels = np.asarray(labels)

        frac_pos, mean_pred = calibration_curve(labels, scores, n_bins=n_bins, strategy="uniform")

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, color="steelblue", label="Probe")

        ax.set_title(f"Reliability Diagram ({name.title()} Set)", fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
        ax.set_ylabel("Fraction of Positives", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        plt.tight_layout()
        figures[name] = fig

    return figures
