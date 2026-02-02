"""Plotting utilities for probe evaluation experiment results.

This module generates visualizations for evaluating probe performance and
calibration. All plotting functions are pure (no side effects).

Functions generate matplotlib figures ready for logging to ClearML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from probe_eval import ProbeEvalResults


def plot_performance_summary(results: ProbeEvalResults) -> Figure:
    """Generate bar chart comparing dev vs test performance metrics.

    Args:
        results: ProbeEvalResults from experiment

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ["Accuracy", "F1 Score", "ROC-AUC"]
    dev_values = [results.dev_accuracy, results.dev_f1_score, results.dev_roc_auc]
    test_values = [results.test_accuracy, results.test_f1_score, results.test_roc_auc]

    x = np.arange(len(metrics))
    width = 0.35

    bars_dev = ax.bar(x - width / 2, dev_values, width, label="Dev", color="steelblue", alpha=0.8)
    bars_test = ax.bar(x + width / 2, test_values, width, label="Test", color="coral", alpha=0.8)

    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Performance Metrics: Dev vs Test", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim((0, 1.05))
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels
    for bar in list(bars_dev) + list(bars_test):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def plot_calibration_summary(results: ProbeEvalResults) -> Figure:
    """Generate bar chart comparing ECE across train/dev/test.

    Args:
        results: ProbeEvalResults from experiment

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    datasets = ["Train", "Dev", "Test"]
    ece_values = [results.train_ece, results.dev_ece, results.test_ece]
    colors = ["forestgreen", "steelblue", "coral"]

    bars = ax.bar(datasets, ece_values, color=colors, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Expected Calibration Error (ECE)", fontweight="bold")
    ax.set_title("Calibration Quality (Lower is Better)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def plot_reliability_diagram(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str = "Reliability Diagram",
    n_bins: int = 10,
) -> Figure:
    """Plot a reliability diagram (calibration curve).

    Args:
        scores: Predicted probabilities
        labels: True binary labels
        title: Plot title
        n_bins: Number of bins

    Returns:
        Matplotlib figure
    """
    from sklearn.calibration import calibration_curve

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    frac_pos, mean_pred = calibration_curve(labels, scores, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfectly Calibrated")

    # Calibration curve
    ax.plot(mean_pred, frac_pos, "o-", color="steelblue", linewidth=2, markersize=8, label="Probe")

    # Fill between for visual emphasis
    ax.fill_between(mean_pred, mean_pred, frac_pos, alpha=0.2, color="steelblue")

    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_score_histogram(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str = "Score Distribution",
) -> Figure:
    """Plot score distribution histogram split by label.

    Args:
        scores: Predicted probabilities
        labels: True binary labels
        title: Plot title

    Returns:
        Matplotlib figure
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    scores_0 = scores[labels == 0]
    scores_1 = scores[labels == 1]

    bins = list(np.linspace(0, 1, 21))

    ax.hist(scores_0, bins=bins, alpha=0.6, label=f"Negative (n={len(scores_0)})", color="steelblue", edgecolor="black")
    ax.hist(scores_1, bins=bins, alpha=0.6, label=f"Positive (n={len(scores_1)})", color="coral", edgecolor="black")

    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Decision Threshold")

    ax.set_xlabel("Probe Score", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig
