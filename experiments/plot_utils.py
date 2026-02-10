"""Shared plotting primitives for experiment visualisations.

All functions accept raw numpy arrays (no dependency on any results dataclass)
and return matplotlib figures. They can be called from any experiment's plotting
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from reliable_monitoring.risks import ThresholdEvaluationResult


# ---------------------------------------------------------------------------
# Low-level single-figure primitives
# ---------------------------------------------------------------------------


def plot_score_histogram(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str = "Score Distribution",
) -> Figure:
    """Overlaid histograms of scores split by binary label.

    Shows label-0 and label-1 distributions with a decision-threshold line
    at 0.5.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    scores_0 = scores[labels == 0]
    scores_1 = scores[labels == 1]

    bins = max(10, int(np.sqrt(len(scores)) / 2)) if len(scores) > 0 else 10

    ax.hist(scores_0, bins=bins, alpha=0.6, label="Target 0", color="steelblue", edgecolor="black")
    ax.hist(scores_1, bins=bins, alpha=0.6, label="Target 1", color="red", edgecolor="black")

    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Decision Threshold")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Probe Score", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_reliability_diagram(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str = "Reliability Diagram",
    n_bins: int = 10,
) -> Figure:
    """Calibration plot using sklearn's calibration_curve."""
    from sklearn.calibration import calibration_curve

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    frac_pos, mean_pred = calibration_curve(labels, scores, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, color="steelblue", label="Probe")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    return fig


def plot_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str = "ROC Curve",
) -> Figure:
    """Single ROC curve with AUC annotation and filled area."""
    from sklearn.metrics import roc_auc_score, roc_curve

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.2, color="steelblue")

    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-split convenience wrappers
# ---------------------------------------------------------------------------


def plot_score_histograms_by_split(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Figure]:
    """Plot probe-score histograms for multiple dataset splits.

    Args:
        splits: Mapping of split name to ``(scores, labels)`` tuple.
                 Example: ``{"train": (train_scores, train_labels), ...}``

    Returns:
        Dictionary mapping split name to matplotlib figure.
    """
    return {
        name: plot_score_histogram(scores, labels, title=f"Probe Score Distribution ({name.title()} Set)")
        for name, (scores, labels) in splits.items()
    }


def plot_reliability_diagrams_by_split(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    n_bins: int = 10,
) -> dict[str, Figure]:
    """Plot reliability diagrams for multiple dataset splits.

    Args:
        splits: Mapping of split name to ``(scores, labels)`` tuple.
        n_bins: Number of calibration bins.

    Returns:
        Dictionary mapping split name to matplotlib figure.
    """
    return {
        name: plot_reliability_diagram(scores, labels, title=f"Reliability Diagram ({name.title()} Set)", n_bins=n_bins)
        for name, (scores, labels) in splits.items()
    }


def plot_roc_curves_by_score_set(
    labels: np.ndarray,
    score_sets: dict[str, np.ndarray],
) -> dict[str, Figure]:
    """Plot ROC curves for multiple score sets against the same labels.

    Args:
        labels: True binary labels (shared across all score sets).
        score_sets: Mapping of name to score array.
                    Example: ``{"probe": probe_scores, "cascade": final_scores}``

    Returns:
        Dictionary mapping name to matplotlib figure.
    """
    return {
        name: plot_roc_curve(scores, labels, title=f"ROC Curve - {name.title()}") for name, scores in score_sets.items()
    }


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def plot_pareto_frontier(
    eval_result: ThresholdEvaluationResult,
    pareto_mask: np.ndarray,
    guaranteed_risk_name: str,
    opt_risk_name: str,
) -> Figure:
    """Plot Pareto frontier from a two-risk optimisation evaluation.

    Args:
        eval_result: Threshold evaluation containing empirical risks for both risks.
        pareto_mask: Boolean mask over thresholds indicating Pareto-efficient points.
        guaranteed_risk_name: Registry name of the guaranteed risk (x-axis).
        opt_risk_name: Registry name of the optimisation risk (y-axis).

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    empirical_risks_2d = eval_result.get_empirical_risks_array()

    # Column order from get_empirical_risks_array is alphabetical by risk name
    risk_names_sorted = sorted(eval_result.empirical_risks.keys())
    guaranteed_col = risk_names_sorted.index(guaranteed_risk_name)
    opt_col = risk_names_sorted.index(opt_risk_name)

    x_vals = empirical_risks_2d[:, guaranteed_col]
    y_vals = empirical_risks_2d[:, opt_col]

    # All points
    ax.scatter(x_vals, y_vals, alpha=0.5, s=80, c="gray", label="Dominated")

    # Pareto-efficient points
    ax.scatter(x_vals[pareto_mask], y_vals[pareto_mask], s=100, c="red", label="Pareto-efficient")

    guaranteed_desc = eval_result.get_risk(guaranteed_risk_name).description
    opt_desc = eval_result.get_risk(opt_risk_name).description

    ax.set_xlabel(guaranteed_desc, fontweight="bold")
    ax.set_ylabel(opt_desc, fontweight="bold")
    ax.set_title("Pareto Frontier", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
