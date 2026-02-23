"""Diagnose why the adaptive cascade may not outperform the fixed baseline.

Checks three failure modes:
  1. Probe makes confident errors (errors at extremes, not near 0.5)
  2. Escalation doesn't help (baseline no better than probe on escalated examples)
  3. Probe uncertainty is uninformative (doesn't predict probe errors)

Usage::

    uv run experiments/diagnose_cascade.py --task-id <clearml_task_id>
    uv run experiments/diagnose_cascade.py --task-id <id> --output-dir figures/diagnostics
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cascade_utils import load_results_from_clearml
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic computations
# ---------------------------------------------------------------------------


def compute_diagnostics(results) -> dict:
    """Compute all diagnostic statistics from a cascade results object."""
    probe = results.test_probe_scores
    baseline = results.test_baseline_scores
    labels = results.test_labels
    threshold = results.reliable_threshold

    probe_preds = (probe >= 0.5).astype(int)
    baseline_preds = (baseline >= 0.5).astype(int)
    probe_correct = probe_preds == labels
    probe_wrong = ~probe_correct

    # Uncertainty: distance from decision boundary (0 = confident, 0.5 = maximally uncertain)
    uncertainty = np.minimum(probe, 1 - probe)

    # Which examples fall inside the escalation band [1-threshold, threshold]
    in_band = (probe >= (1 - threshold)) & (probe <= threshold)

    # --- Failure mode 1: where do probe errors live? ---
    scores_when_correct = probe[probe_correct]
    scores_when_wrong = probe[probe_wrong]
    uncertainty_when_correct = uncertainty[probe_correct]
    uncertainty_when_wrong = uncertainty[probe_wrong]

    # Fraction of errors that fall inside vs outside the escalation band
    errors_in_band = probe_wrong & in_band
    errors_outside_band = probe_wrong & ~in_band
    n_errors = int(probe_wrong.sum())
    frac_errors_caught = float(errors_in_band.sum()) / n_errors if n_errors > 0 else 0.0
    frac_errors_missed = float(errors_outside_band.sum()) / n_errors if n_errors > 0 else 0.0

    # --- Failure mode 2: does escalation help? ---
    # For examples inside the band: compare probe vs baseline correctness
    band_probe_correct = probe_correct[in_band]
    band_baseline_correct = baseline_preds[in_band] == labels[in_band]
    n_in_band = int(in_band.sum())

    # Correction: probe wrong, baseline right
    corrected = (~band_probe_correct) & band_baseline_correct
    # Broken: probe right, baseline wrong
    broken = band_probe_correct & (~band_baseline_correct)
    # Both wrong
    both_wrong = (~band_probe_correct) & (~band_baseline_correct)

    n_corrected = int(corrected.sum())
    n_broken = int(broken.sum())
    n_both_wrong = int(both_wrong.sum())
    n_both_right = int((band_probe_correct & band_baseline_correct).sum())
    net_corrections = n_corrected - n_broken

    # --- Failure mode 3: is uncertainty discriminative? ---
    # AUC of uncertainty as a predictor of "probe is wrong"
    # Higher uncertainty should predict higher error rate
    if n_errors > 0 and n_errors < len(probe):
        uncertainty_auc = float(roc_auc_score(probe_wrong.astype(int), uncertainty))
    else:
        uncertainty_auc = float("nan")

    # Binned analysis: split examples into uncertainty quintiles
    n_bins = 5
    bin_edges = np.quantile(uncertainty, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9  # include max
    bins = []
    for i in range(n_bins):
        mask = (uncertainty >= bin_edges[i]) & (uncertainty < bin_edges[i + 1])
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        error_rate = float(probe_wrong[mask].mean())
        mean_unc = float(uncertainty[mask].mean())
        baseline_error_rate = float((baseline_preds[mask] != labels[mask]).mean())
        bins.append(
            {
                "bin": i,
                "uncertainty_range": (float(bin_edges[i]), float(bin_edges[i + 1])),
                "mean_uncertainty": mean_unc,
                "n_examples": n_bin,
                "probe_error_rate": error_rate,
                "baseline_error_rate": baseline_error_rate,
                "in_band_frac": float(in_band[mask].mean()),
            }
        )

    return {
        "threshold": threshold,
        "n_test": len(probe),
        "n_errors": n_errors,
        "probe_error_rate": float(probe_wrong.mean()),
        # FM1
        "scores_when_correct": scores_when_correct,
        "scores_when_wrong": scores_when_wrong,
        "uncertainty_when_correct": uncertainty_when_correct,
        "uncertainty_when_wrong": uncertainty_when_wrong,
        "frac_errors_caught_by_band": frac_errors_caught,
        "frac_errors_missed_by_band": frac_errors_missed,
        "n_in_band": n_in_band,
        "band_frac": float(in_band.mean()),
        # FM2
        "n_corrected": n_corrected,
        "n_broken": n_broken,
        "n_both_wrong": n_both_wrong,
        "n_both_right": n_both_right,
        "net_corrections": net_corrections,
        "correction_rate": n_corrected / n_in_band if n_in_band > 0 else 0.0,
        "breakage_rate": n_broken / n_in_band if n_in_band > 0 else 0.0,
        # FM3
        "uncertainty_auc": uncertainty_auc,
        "uncertainty_bins": bins,
        # Raw arrays for plotting
        "probe_scores": probe,
        "baseline_scores": baseline,
        "labels": labels,
        "probe_correct": probe_correct,
        "uncertainty": uncertainty,
        "in_band": in_band,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_fm1_error_distribution(diag: dict) -> Figure:
    """FM1: Where do probe errors live in score space?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    threshold = diag["threshold"]

    # Left: probe score histograms for correct vs wrong
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(diag["scores_when_correct"], bins=bins, alpha=0.5, label="Correct", color="steelblue", density=True)
    ax.hist(diag["scores_when_wrong"], bins=bins, alpha=0.6, label="Wrong", color="crimson", density=True)
    ax.axvspan(
        1 - threshold,
        threshold,
        alpha=0.15,
        color="gold",
        label=f"Escalation band [{1 - threshold:.2f}, {threshold:.2f}]",
    )
    ax.set_xlabel("Probe Score", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Probe Score Distribution: Correct vs Wrong", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: uncertainty histograms for correct vs wrong
    ax = axes[1]
    bins_unc = np.linspace(0, 0.5, 26)
    ax.hist(
        diag["uncertainty_when_correct"], bins=bins_unc, alpha=0.5, label="Correct", color="steelblue", density=True
    )
    ax.hist(diag["uncertainty_when_wrong"], bins=bins_unc, alpha=0.6, label="Wrong", color="crimson", density=True)
    unc_threshold = min(threshold, 1 - threshold)
    ax.axvline(
        unc_threshold, color="gold", linewidth=2, linestyle="--", label=f"Escalation cutoff ({unc_threshold:.2f})"
    )
    ax.set_xlabel("Probe Uncertainty = min(p, 1-p)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Uncertainty Distribution: Correct vs Wrong", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"FM1: Do errors concentrate near 0.5?  "
        f"({diag['frac_errors_caught_by_band']:.0%} of errors caught by band, "
        f"{diag['frac_errors_missed_by_band']:.0%} missed)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_fm2_escalation_impact(diag: dict) -> Figure:
    """FM2: Does sending examples to the baseline help?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: correction/breakage counts
    ax = axes[0]
    categories = ["Corrected\n(P wrong, B right)", "Broken\n(P right, B wrong)", "Both wrong", "Both right"]
    counts = [diag["n_corrected"], diag["n_broken"], diag["n_both_wrong"], diag["n_both_right"]]
    colors = ["forestgreen", "crimson", "gray", "steelblue"]
    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor="black")
    for bar, count in zip(bars, counts, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center", fontweight="bold")
    ax.set_ylabel("Number of examples", fontweight="bold")
    ax.set_title(
        f"Escalation Outcomes (n={diag['n_in_band']} in band)\nNet corrections: {diag['net_corrections']:+d}",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Right: scatter of probe vs baseline score for in-band examples, colored by label
    ax = axes[1]
    probe = diag["probe_scores"][diag["in_band"]]
    baseline = diag["baseline_scores"][diag["in_band"]]
    label = diag["labels"][diag["in_band"]]
    scatter = ax.scatter(probe, baseline, c=label, cmap="coolwarm", alpha=0.4, s=30, edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Probe Score", fontweight="bold")
    ax.set_ylabel("Baseline Score", fontweight="bold")
    ax.set_title("Probe vs Baseline (escalated examples)", fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("True Label", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fm3_uncertainty_discrimination(diag: dict) -> Figure:
    """FM3: Is probe uncertainty informative about errors?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: binned error rates
    ax = axes[0]
    bins_data = diag["uncertainty_bins"]
    x = [b["mean_uncertainty"] for b in bins_data]
    probe_err = [b["probe_error_rate"] for b in bins_data]
    baseline_err = [b["baseline_error_rate"] for b in bins_data]
    n_examples = [b["n_examples"] for b in bins_data]
    in_band_frac = [b["in_band_frac"] for b in bins_data]

    width = 0.015
    ax.bar([xi - width / 2 for xi in x], probe_err, width=width, label="Probe error rate", color="steelblue", alpha=0.8)
    ax.bar(
        [xi + width / 2 for xi in x], baseline_err, width=width, label="Baseline error rate", color="coral", alpha=0.8
    )

    # Annotate counts
    for xi, n in zip(x, n_examples, strict=False):
        ax.text(xi, max(probe_err[x.index(xi)], baseline_err[x.index(xi)]) + 0.01, f"n={n}", ha="center", fontsize=8)

    ax.set_xlabel("Mean Probe Uncertainty (per quintile)", fontweight="bold")
    ax.set_ylabel("Error Rate", fontweight="bold")
    ax.set_title(
        f"Error Rate by Uncertainty Quintile\n(Uncertainty AUC for predicting errors: {diag['uncertainty_auc']:.3f})",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Right: fraction of each bin that falls in escalation band
    ax = axes[1]
    ax.bar(x, in_band_frac, width=width * 2, color="gold", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Mean Probe Uncertainty (per quintile)", fontweight="bold")
    ax.set_ylabel("Fraction in Escalation Band", fontweight="bold")
    ax.set_title("Escalation Coverage by Uncertainty Quintile", fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_fm1_error_locations_detailed(diag: dict) -> Figure:
    """Supplementary: where exactly are the confident errors?"""
    fig, ax = plt.subplots(figsize=(10, 6))

    probe = diag["probe_scores"]
    correct = diag["probe_correct"]
    in_band = diag["in_band"]

    # Four categories
    categories = {
        "Correct, not escalated": correct & ~in_band,
        "Correct, escalated": correct & in_band,
        "Wrong, escalated (caught)": ~correct & in_band,
        "Wrong, not escalated (missed)": ~correct & ~in_band,
    }
    colors = {
        "Correct, not escalated": "steelblue",
        "Correct, escalated": "lightblue",
        "Wrong, escalated (caught)": "orange",
        "Wrong, not escalated (missed)": "crimson",
    }

    bins = np.linspace(0, 1, 41)
    bottom = np.zeros(len(bins) - 1)
    for label, mask in categories.items():
        counts, _ = np.histogram(probe[mask], bins=bins)
        ax.bar(
            (bins[:-1] + bins[1:]) / 2,
            counts,
            width=bins[1] - bins[0],
            bottom=bottom,
            label=f"{label} (n={int(mask.sum())})",
            color=colors[label],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += counts

    threshold = diag["threshold"]
    ax.axvline(1 - threshold, color="gold", linewidth=2, linestyle="--")
    ax.axvline(threshold, color="gold", linewidth=2, linestyle="--", label="Escalation band edges")

    ax.set_xlabel("Probe Score", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title("Example Classification by Probe Score", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(diag: dict) -> None:
    """Print a text summary of all diagnostics."""
    print("\n" + "=" * 70)
    print("CASCADE DIAGNOSTIC REPORT")
    print("=" * 70)

    print(f"\nDataset: {diag['n_test']} examples, {diag['n_errors']} probe errors ({diag['probe_error_rate']:.1%})")
    print(
        f"Threshold: {diag['threshold']:.4f} → escalation band: [{1 - diag['threshold']:.4f}, {diag['threshold']:.4f}]"
    )
    print(f"Examples in band: {diag['n_in_band']} ({diag['band_frac']:.1%})")

    print("\n--- FM1: Error Location ---")
    print(f"  Errors caught by band:  {diag['frac_errors_caught_by_band']:.1%}")
    print(f"  Errors missed (confident): {diag['frac_errors_missed_by_band']:.1%}")
    verdict = "PROBLEM" if diag["frac_errors_missed_by_band"] > 0.5 else "OK"
    print(f"  Verdict: {verdict}")

    print("\n--- FM2: Escalation Impact ---")
    print(f"  Corrected (P wrong → B right): {diag['n_corrected']} ({diag['correction_rate']:.1%} of band)")
    print(f"  Broken (P right → B wrong):    {diag['n_broken']} ({diag['breakage_rate']:.1%} of band)")
    print(f"  Both wrong:                    {diag['n_both_wrong']}")
    print(f"  Both right:                    {diag['n_both_right']}")
    print(f"  Net corrections:               {diag['net_corrections']:+d}")
    net_rate = diag["net_corrections"] / diag["n_in_band"] if diag["n_in_band"] > 0 else 0
    verdict = "PROBLEM" if net_rate <= 0 else ("WEAK" if net_rate < 0.05 else "OK")
    print(f"  Verdict: {verdict} (net correction rate: {net_rate:.1%})")

    print("\n--- FM3: Uncertainty Discrimination ---")
    print(f"  Uncertainty AUC (predicting probe error): {diag['uncertainty_auc']:.3f}")
    verdict = "PROBLEM" if diag["uncertainty_auc"] < 0.6 else ("WEAK" if diag["uncertainty_auc"] < 0.7 else "OK")
    print(f"  Verdict: {verdict}")
    print("\n  Error rate by uncertainty quintile:")
    print(f"  {'Uncertainty':>14s}  {'N':>6s}  {'Probe Err':>10s}  {'Baseline Err':>12s}  {'In Band':>8s}")
    for b in diag["uncertainty_bins"]:
        lo, hi = b["uncertainty_range"]
        print(
            f"  [{lo:.3f}, {hi:.3f}]  {b['n_examples']:>6d}  {b['probe_error_rate']:>10.1%}"
            f"  {b['baseline_error_rate']:>12.1%}  {b['in_band_frac']:>8.1%}"
        )

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose cascade failure modes")
    parser.add_argument("--task-id", type=str, required=True, help="ClearML task ID of SGT cascade experiment")
    parser.add_argument("--output-dir", type=str, default="figures/diagnostics", help="Directory to save figures")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading results from ClearML task: {args.task_id}")
    results = load_results_from_clearml(args.task_id)

    logger.info("Computing diagnostics...")
    diag = compute_diagnostics(results)

    print_report(diag)

    logger.info(f"Saving figures to {output_dir}")
    figures = {
        "fm1_error_distribution": plot_fm1_error_distribution(diag),
        "fm1_error_locations": plot_fm1_error_locations_detailed(diag),
        "fm2_escalation_impact": plot_fm2_escalation_impact(diag),
        "fm3_uncertainty_discrimination": plot_fm3_uncertainty_discrimination(diag),
    }
    for name, fig in figures.items():
        path = output_dir / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"  Saved {path}")

    plt.close("all")
    logger.info("Done.")


if __name__ == "__main__":
    main()
