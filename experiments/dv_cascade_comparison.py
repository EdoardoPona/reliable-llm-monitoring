"""DV cascade comparison experiment.

Compares delegation strategies for the probe-baseline safety cascade:

1. **Batched top-k** (three ranking signals):
   - Probe uncertainty: McKenzie et al. signal — proximity to median probe score
   - DV probe score:    d(x) = P(v=1 | z)
   - Oracle:            v(x) = 1[probe wrong AND baseline correct]

2. **LTT threshold** (global, with PAC budget guarantee):
   - CTD (calibrated threshold): delegate where d(x) > tau, with tau calibrated
     via Pareto-filtered Learn-then-Test for each target budget alpha.

Outputs:
  - ranking_comparison_B{batch_size}.pdf  (per batch size)
  - ranking_comparison_grid.pdf           (all batch sizes side by side)
  - adaptivity.pdf                        (per-batch/group delegation rates)

Usage::

    uv run experiments/dv_cascade_comparison.py --config configs/dv_cascade_comparison.yaml
"""

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import load_config
from dotenv import load_dotenv
from dv_ltt_cascade import (
    cascade_metrics,
    pareto_ht_opt_split,
    prepare_dv_cascade_data,
    split_calib_eval,
    threshold_cascade,
)
from sklearn.metrics import accuracy_score, roc_auc_score

from reliable_monitoring.cascade import offline_batch_cascade, probe_uncertainty
from reliable_monitoring.learn_then_test import ParetoLTTResult, build_pareto_ltt
from reliable_monitoring.risks import RISK_RGISTRY, BudgetCostRisk

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: batched top-k sweep using library cascade infrastructure
# ---------------------------------------------------------------------------


def batched_topk_sweep(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    ranking_scores: np.ndarray | None,
    batch_size: int,
    k_values: np.ndarray,
    merge_strategy: str = "replace",
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep over k values and compute AUC and accuracy for each.

    Uses ``offline_batch_cascade`` with the ``topk`` selection strategy,
    which ranks by ``ranking_scores`` within each batch and delegates the
    top-k to the baseline.

    Returns:
        (aucs, accs) arrays of shape (len(k_values),).
    """
    aucs = np.empty(len(k_values))
    accs = np.empty(len(k_values))

    for i, k in enumerate(k_values):
        result = offline_batch_cascade(
            probe_scores,
            baseline_scores,
            batch_size,
            selection_strategy="fixed_budget_amount",
            merge_strategy=merge_strategy,
            amount=int(k),
            ranking_scores=ranking_scores,
        )
        aucs[i] = roc_auc_score(labels, result.final_scores)
        accs[i] = accuracy_score(labels, (result.final_scores >= 0.5).astype(int))

    return aucs, accs


def global_oracle_sweep(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    delegation_values: np.ndarray,
    alpha_budgets: np.ndarray,
    merge_strategy: str = "replace",
) -> list[dict]:
    """Compute the global (non-batched) oracle cascade at each budget level.

    Ranks all examples by ground-truth delegation value v(x, y) and delegates
    the top fraction, capping at v > 0 (never delegates when harmful).

    Returns a list of per-alpha result dicts with keys:
    alpha, k, realized_budget, auc, accuracy.
    """
    n = len(probe_scores)
    max_beneficial = int((delegation_values > 0).sum())
    order = np.argsort(-delegation_values)  # descending by v

    results: list[dict] = []
    for alpha in alpha_budgets:
        k = min(int(np.ceil(alpha * n)), max_beneficial)
        delegated = np.zeros(n, dtype=bool)
        delegated[order[:k]] = True

        if merge_strategy == "replace":
            final = np.where(delegated, baseline_scores, probe_scores)
        elif merge_strategy == "avg":
            final = np.where(delegated, (probe_scores + baseline_scores) / 2, probe_scores)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

        results.append(
            {
                "alpha": float(alpha),
                "k": k,
                "realized_budget": float(delegated.mean()),
                "auc": float(roc_auc_score(labels, final)),
                "accuracy": float(accuracy_score(labels, (final >= 0.5).astype(int))),
            }
        )
    return results


def compute_faic(
    ctd_results: list[dict],
    oracle_results: list[dict],
    probe_value: float,
    metric: str = "auc",
) -> float:
    """Compute Fraction of Achievable Improvement Captured.

    Integrates CTD and oracle curves (relative to probe-only baseline)
    over the budget range up to the oracle's effective delegation capacity,
    then returns the ratio.

    Args:
        ctd_results: Per-alpha dicts from ``pareto_ltt_sweep`` (keys: alpha, auc, accuracy).
        oracle_results: Per-alpha dicts from ``global_oracle_sweep``.
        probe_value: Probe-only metric (AUC or accuracy, matching *metric*).
        metric: Which metric to integrate ("auc" or "accuracy").

    Returns:
        FAIC in [0, 1], or 0.0 if the oracle achieves no improvement.
    """
    # Find the effective delegation capacity: last alpha where oracle still improves
    capacity_alpha = 0.0
    for r in oracle_results:
        if r["realized_budget"] > 0 and r[metric] > probe_value:
            capacity_alpha = r["alpha"]
    if capacity_alpha == 0.0:
        return 0.0

    # Filter both curves to [0, capacity_alpha]
    oracle_alphas = np.array([r["alpha"] for r in oracle_results if r["alpha"] <= capacity_alpha])
    oracle_vals = np.array([r[metric] for r in oracle_results if r["alpha"] <= capacity_alpha])
    ctd_alphas = np.array([r["alpha"] for r in ctd_results if r["alpha"] <= capacity_alpha])
    ctd_vals = np.array([r[metric] for r in ctd_results if r["alpha"] <= capacity_alpha])

    if len(oracle_alphas) < 2 or len(ctd_alphas) < 2:
        return 0.0

    oracle_area = float(np.trapz(oracle_vals - probe_value, oracle_alphas))
    ctd_area = float(np.trapz(ctd_vals - probe_value, ctd_alphas))

    if oracle_area <= 0:
        return 0.0
    return max(0.0, ctd_area / oracle_area)


@dataclass
class LTTSignalConfig:
    """Per-signal inputs for Pareto LTT construction and evaluation."""

    ht_scores: np.ndarray  # delegation signal (e.g. d(x) or uncertainty) on the hypothesis-testing split
    opt_scores: np.ndarray  # delegation signal on the optimisation split
    tau_grid: np.ndarray  # candidate thresholds, adapted to this signal's range
    eval_scores: np.ndarray  # delegation signal on the eval split (used for cascade at test time)


def pareto_ltt_sweep(
    pareto_ltt: ParetoLTTResult,
    alpha_budgets: np.ndarray,
    delta: float,
    eval_ps: np.ndarray,
    eval_bs: np.ndarray,
    eval_labels: np.ndarray,
    eval_delegation_scores: np.ndarray,
    merge_strategy: str = "replace",
) -> list[dict]:
    """Sweep budget levels for a Pareto-calibrated LTT threshold cascade.

    Returns a list of per-alpha result dicts with keys:
    alpha, tau, realized_budget, auc, accuracy.
    """
    results: list[dict] = []
    for alpha_b in alpha_budgets:
        tau = pareto_ltt.select_threshold(alpha_b, delta)
        if tau is None:
            logger.warning(f"  No valid threshold at alpha={alpha_b:.3f} (delta={delta:.3f})")
            continue
        result = threshold_cascade(eval_ps, eval_bs, eval_delegation_scores, tau, merge_strategy=merge_strategy)
        met = cascade_metrics(result, eval_labels)
        results.append(
            {
                "alpha": float(alpha_b),
                "tau": tau,
                "realized_budget": float(result.used_baseline.mean()),
                "auc": met["auc"],
                "accuracy": met["accuracy"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "Unc. top-k": "C0",  # blue
    "DV top-k": "C1",  # orange
    "Oracle top-k": "C2",  # green
    "CTD": "C1",  # orange
    "Unc. calibrated": "C0",  # blue
}
MARKERS = {
    "Unc. top-k": "D",
    "DV top-k": "D",
    "Oracle top-k": "D",
    "CTD": "s",
    "Unc. calibrated": "s",
}
STYLES = {
    "Unc. top-k": "--",
    "DV top-k": "--",
    "Oracle top-k": "--",
    "CTD": "-",
    "Unc. calibrated": "-",
}


def _plot_single_batch_size(
    ax_auc: plt.Axes,
    ax_acc: plt.Axes,
    budget_fractions: np.ndarray,
    signal_results: dict[str, tuple[np.ndarray, np.ndarray]],
    probe_auc: float,
    probe_acc: float,
    baseline_auc: float,
    baseline_acc: float,
    batch_size: int,
    show_legend: bool = True,
    ltt_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Plot AUC and accuracy vs budget for one batch size on given axes.

    Args:
        ltt_results: Optional dict mapping line name to
            (alpha_budgets, ltt_aucs, ltt_accs) for LTT-calibrated threshold
            cascades.  Plotted as additional lines; independent of batch size.
    """
    for name, (aucs, accs) in signal_results.items():
        ax_auc.plot(
            budget_fractions,
            aucs,
            label=name,
            color=COLORS[name],
            marker=MARKERS[name],
            linestyle=STYLES[name],
            markersize=3,
        )
        ax_acc.plot(
            budget_fractions,
            accs,
            label=name,
            color=COLORS[name],
            marker=MARKERS[name],
            linestyle=STYLES[name],
            markersize=3,
        )

    # LTT calibrated lines (global threshold, not batched)
    if ltt_results is not None:
        for ltt_name, (alphas, ltt_aucs, ltt_accs) in ltt_results.items():
            ax_auc.plot(
                alphas,
                ltt_aucs,
                label=ltt_name,
                color=COLORS[ltt_name],
                marker=MARKERS[ltt_name],
                linestyle=STYLES[ltt_name],
                markersize=3,
            )
            ax_acc.plot(
                alphas,
                ltt_accs,
                label=ltt_name,
                color=COLORS[ltt_name],
                marker=MARKERS[ltt_name],
                linestyle=STYLES[ltt_name],
                markersize=3,
            )

    for ax, ref_probe, ref_base, ylabel in [
        (ax_auc, probe_auc, baseline_auc, "Cascade ROC AUC"),
        (ax_acc, probe_acc, baseline_acc, "Cascade Accuracy"),
    ]:
        ax.axhline(ref_probe, color="gray", ls=":", alpha=0.5, label=f"Probe only ({ref_probe:.3f})")
        ax.axhline(ref_base, color="gray", ls="--", alpha=0.5, label=f"Expert only ({ref_base:.3f})")
        ax.set_xlabel("Budget fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Batch size $B = {batch_size}$")
        ax.grid(alpha=0.3)
        if show_legend:
            # Order legend by novelty
            handles, labels = ax.get_legend_handles_labels()
            legend_order = [
                "CTD",
                "DV top-k",
                "Unc. calibrated",
                "Unc. top-k",
                "Oracle top-k",
            ]
            # Add reference lines (Probe only, Expert only) at the end
            legend_order += [lab for lab in labels if lab not in legend_order]
            order = [labels.index(name) for name in legend_order if name in labels]
            ax.legend(
                [handles[i] for i in order],
                [labels[i] for i in order],
                fontsize=7,
                loc="lower right",
            )


def plot_single_batch_size(
    budget_fractions: np.ndarray,
    signal_results: dict[str, tuple[np.ndarray, np.ndarray]],
    probe_auc: float,
    probe_acc: float,
    baseline_auc: float,
    baseline_acc: float,
    batch_size: int,
    output_dir: Path,
    ltt_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    file_prefix: str = "",
) -> plt.Figure:
    """Main-paper figure: 1x2 (AUC, accuracy) for a single batch size."""
    fig, (ax_auc, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    _plot_single_batch_size(
        ax_auc,
        ax_acc,
        budget_fractions,
        signal_results,
        probe_auc,
        probe_acc,
        baseline_auc,
        baseline_acc,
        batch_size,
        show_legend=True,
        ltt_results=ltt_results,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}ranking_comparison_B{batch_size}.pdf", bbox_inches="tight")
    return fig


def plot_grid(
    all_results: dict[int, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    probe_auc: float,
    probe_acc: float,
    baseline_auc: float,
    baseline_acc: float,
    output_dir: Path,
    ltt_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    file_prefix: str = "",
) -> plt.Figure:
    """Appendix figure: 2-row x N-col grid (rows: AUC/accuracy, cols: batch sizes)."""
    batch_sizes = sorted(all_results.keys())
    n_cols = len(batch_sizes)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7), squeeze=False)

    for j, bs in enumerate(batch_sizes):
        budget_fractions, signal_results = all_results[bs]
        _plot_single_batch_size(
            axes[0, j],
            axes[1, j],
            budget_fractions,
            signal_results,
            probe_auc,
            probe_acc,
            baseline_auc,
            baseline_acc,
            bs,
            show_legend=(j == n_cols - 1),
            ltt_results=ltt_results,
        )
        # Only put y-label on leftmost column
        if j > 0:
            axes[0, j].set_ylabel("")
            axes[1, j].set_ylabel("")

    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}ranking_comparison_grid.pdf", bbox_inches="tight")
    return fig


def plot_adaptivity(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    dv_scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    tau: float,
    alpha_budget: float,
    batch_size: int,
    output_dir: Path,
    merge_strategy: str = "replace",
    file_prefix: str = "",
) -> plt.Figure:
    """Per-batch delegation rate histogram for DV threshold vs fixed-k."""
    dv_result = threshold_cascade(probe_scores, baseline_scores, dv_scores, tau, merge_strategy=merge_strategy)
    unc_result = offline_batch_cascade(
        probe_scores,
        baseline_scores,
        batch_size,
        selection_strategy="fixed_budget_rate",
        merge_strategy=merge_strategy,
        rate=alpha_budget,
    )

    n = len(probe_scores)
    n_batches = n // batch_size
    n_used = n_batches * batch_size

    dv_batch_rates = []
    unc_batch_rates = []
    for i in range(n_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        dv_batch_rates.append(float(dv_result.used_baseline[s:e].mean()))
        unc_batch_rates.append(float(unc_result.used_baseline[s:e].mean()))

    n_cols = 2 if groups is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Histogram of per-batch delegation rates
    ax = axes[0]
    bins = np.linspace(0, max(max(dv_batch_rates), max(unc_batch_rates)) * 1.2, 20)
    ax.hist(dv_batch_rates, bins=bins, alpha=0.6, label="DV threshold", color="C0", edgecolor="black")
    ax.axvline(np.mean(dv_batch_rates), color="C0", ls="--", label=f"DV mean ({np.mean(dv_batch_rates):.1%})")
    ax.axvline(alpha_budget, color="red", ls=":", lw=2, label=f"Budget constraint ({alpha_budget:.0%})")
    ax.axvline(np.mean(unc_batch_rates), color="C1", ls="--", label=f"Fixed-k ({np.mean(unc_batch_rates):.1%})")
    ax.set_xlabel("Per-batch delegation rate")
    ax.set_ylabel("Count")
    ax.set_title(rf"Adaptivity at $\alpha_{{\mathrm{{budget}}}}$ = {alpha_budget:.0%}")
    ax.legend(fontsize=7)

    # Per-group delegation rates
    if groups is not None:
        ax = axes[1]
        unique_groups = np.unique(groups[:n_used])
        dv_group_rates = [float(dv_result.used_baseline[:n_used][groups[:n_used] == g].mean()) for g in unique_groups]
        unc_group_rates = [float(unc_result.used_baseline[:n_used][groups[:n_used] == g].mean()) for g in unique_groups]
        x = np.arange(len(unique_groups))
        w = 0.35
        ax.bar(x - w / 2, dv_group_rates, w, label="DV threshold", color="C0", alpha=0.7, edgecolor="black")
        ax.bar(x + w / 2, unc_group_rates, w, label="Fixed-k uncertainty", color="C1", alpha=0.7, edgecolor="black")
        ax.axhline(alpha_budget, color="red", ls=":", lw=2, label=f"Budget ({alpha_budget:.0%})")
        ax.set_xticks(x)
        ax.set_xticklabels(unique_groups)
        ax.set_ylabel("Delegation rate")
        ax.set_title("Per-group delegation rates")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}adaptivity.pdf", bbox_inches="tight")
    return fig


def run_ltt_calibration(
    calib_ps: np.ndarray,
    calib_bs: np.ndarray,
    calib_labels: np.ndarray,
    calib_dv: np.ndarray,
    eval_ps: np.ndarray,
    eval_bs: np.ndarray,
    eval_labels: np.ndarray,
    eval_dv: np.ndarray,
    eval_uncertainty: np.ndarray,
    dv_scores_full: np.ndarray,
    dv_target: str,
    config,
    merge_strategy: str,
) -> tuple[dict[str, list[dict]], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None]:
    """Build Pareto LTTs for each delegation signal, sweep budgets, and assemble plot data.

    Returns:
        (ltt_sweep_results, ltt_plot_data) where ltt_sweep_results maps signal name
        to a list of per-alpha result dicts, and ltt_plot_data is formatted for plotting
        (or None if no results).
    """
    guarantee_probability = config.guarantee_probability
    delta = 1.0 - guarantee_probability
    tau_steps = config.tau_steps
    n_alpha_steps = config.n_alpha_steps
    alpha_budgets = np.linspace(0.05, 1, n_alpha_steps)

    pareto_testing = config.pareto_testing
    if not pareto_testing:
        return {}, None

    opt_risk_name = config.opt_risk
    OptRisk = RISK_RGISTRY.get(opt_risk_name)
    if OptRisk is None:
        raise ValueError(f"Unknown opt_risk: '{opt_risk_name}'. Available: {list(RISK_RGISTRY.keys())}")

    # Split calib into ht/opt (shared across both delegation signals)
    pareto_proportion = config.pareto_split_proportion
    ht_idx, opt_idx = pareto_ht_opt_split(len(calib_dv), pareto_proportion, config.seed)
    opt_ps, opt_bs, opt_labels = calib_ps[opt_idx], calib_bs[opt_idx], calib_labels[opt_idx]

    # Build Pareto LTT for each delegation signal
    calib_uncertainty = probe_uncertainty(calib_ps)

    if dv_target == "continuous":
        dv_tau_min = float(np.min(dv_scores_full)) - 0.01
        dv_tau_max = float(np.max(dv_scores_full)) + 0.01
        dv_tau_grid = np.linspace(dv_tau_min, dv_tau_max, tau_steps)
    else:
        dv_tau_grid = np.linspace(0.0, 1.0, tau_steps)

    unc_tau_min = float(np.min(calib_uncertainty)) - 0.01
    unc_tau_max = float(np.max(calib_uncertainty)) + 0.01
    unc_tau_grid = np.linspace(unc_tau_min, unc_tau_max, tau_steps)

    signal_configs: dict[str, LTTSignalConfig] = {
        "CTD": LTTSignalConfig(
            ht_scores=calib_dv[ht_idx],
            opt_scores=calib_dv[opt_idx],
            tau_grid=dv_tau_grid,
            eval_scores=eval_dv,
        ),
        "Unc. calibrated": LTTSignalConfig(
            ht_scores=calib_uncertainty[ht_idx],
            opt_scores=calib_uncertainty[opt_idx],
            tau_grid=unc_tau_grid,
            eval_scores=eval_uncertainty,
        ),
    }

    ltt_sweep_results: dict[str, list[dict]] = {}
    for name, sig in signal_configs.items():
        logger.info(f"Pareto testing: ht={len(sig.ht_scores)}, opt={len(sig.opt_scores)}, opt_risk={opt_risk_name}")
        pltt = build_pareto_ltt(
            ht_delegation_scores=sig.ht_scores,
            opt_probe_scores=opt_ps,
            opt_baseline_scores=opt_bs,
            opt_labels=opt_labels,
            opt_delegation_scores=sig.opt_scores,
            tau_grid=sig.tau_grid,
            opt_risk=OptRisk,
            budget_risk=BudgetCostRisk,
            merge_strategy=merge_strategy,
        )
        logger.info(f"Pareto frontier ({name}): {len(pltt.taus)}/{len(sig.tau_grid)} thresholds retained")

        rows = pareto_ltt_sweep(
            pltt, alpha_budgets, delta, eval_ps, eval_bs, eval_labels, sig.eval_scores, merge_strategy
        )
        if rows:
            ltt_sweep_results[name] = rows

    # Assemble plot data
    ltt_plot_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, rows in ltt_sweep_results.items():
        ltt_plot_data[name] = (
            np.array([r["alpha"] for r in rows]),
            np.array([r["auc"] for r in rows]),
            np.array([r["accuracy"] for r in rows]),
        )

    for name, rows in ltt_sweep_results.items():
        for r in rows:
            logger.info(
                f"  [{name}] alpha={r['alpha']:.2f}: tau={r['tau']:.4f}, "
                f"realized={r['realized_budget']:.1%}, AUC={r['auc']:.4f}, Acc={r['accuracy']:.4f}"
            )

    return ltt_sweep_results, ltt_plot_data or None


def save_results_json(
    output_dir: Path,
    file_prefix: str,
    config,
    merge_strategy: str,
    dv_target: str,
    probe_auc: float,
    probe_acc: float,
    baseline_auc: float,
    baseline_acc: float,
    dv_auc: float,
    topk_results: dict[int, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    ltt_sweep_results: dict[str, list[dict]],
    oracle_results: list[dict],
    faic: dict[str, dict[str, float]],
) -> Path:
    """Serialize experiment results to JSON."""
    results_json: dict = {
        "config": {
            "dv_target": dv_target,
            "baseline_model_name": config.baseline_model_name,
            "activations_model_name": config.activations_model_name,
            "batch_sizes": sorted(topk_results.keys()),
            "merge_strategy": merge_strategy,
        },
        "reference": {
            "probe_auc": probe_auc,
            "probe_acc": probe_acc,
            "baseline_auc": baseline_auc,
            "baseline_acc": baseline_acc,
            "dv_probe_auc": dv_auc,
        },
        "faic": faic,
        "oracle": oracle_results,
        "topk": {},
        "ltt": {
            name: [{k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in r.items()} for r in rows]
            for name, rows in ltt_sweep_results.items()
        },
    }
    for bs, (budget_fractions, signal_results) in topk_results.items():
        results_json["topk"][str(bs)] = {
            "budget_fractions": budget_fractions.tolist(),
            "signals": {
                name: {"auc": aucs.tolist(), "accuracy": accs.tolist()} for name, (aucs, accs) in signal_results.items()
            },
        }
    results_path = output_dir / f"{file_prefix}results.json"
    results_path.write_text(json.dumps(results_json, indent=2))
    logger.info(f"Results JSON saved to {results_path}")
    return results_path


# ---------------------------------------------------------------------------
# Core experiment runner (importable by sweep scripts)
# ---------------------------------------------------------------------------


@dataclass
class DVCascadeExperimentResults:
    """Collected outputs from a single DV cascade comparison run."""

    probe_auc: float
    probe_acc: float
    baseline_auc: float
    baseline_acc: float
    dv_auc: float
    dv_target: str
    merge_strategy: str
    topk_results: dict[int, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]]
    ltt_sweep_results: dict[str, list[dict]]
    oracle_results: list[dict]
    faic: dict[str, dict[str, float]]  # method_name -> {"auc": ..., "acc": ...}
    figs: dict[str, plt.Figure]


def run_dv_cascade_experiment(
    config,
    output_dir: Path,
    file_prefix: str = "",
) -> DVCascadeExperimentResults:
    """Run the full DV cascade comparison experiment.

    This is the core experiment logic, separated from CLI parsing and
    ClearML logging so that sweep scripts can call it directly.

    Args:
        config: Experiment configuration (``SimpleNamespace`` from ``load_config``
            or constructed programmatically).
        output_dir: Directory for plots and results.json.
        file_prefix: Optional prefix for output filenames.

    Returns:
        Experiment results for downstream aggregation or ClearML logging.
    """
    merge_strategy = config.merge_strategy

    # --- Train probes & load test data ---
    data = prepare_dv_cascade_data(config)
    dv_target = data.dv_target
    dv_scores_full = data.dv_scores
    dv_auc = data.dv_auc

    # --- Split test into calib / eval ---
    # v_test is passed through for oracle ranking
    calib_arrays, eval_arrays = split_calib_eval(
        data.test_ps,
        data.test_bs,
        data.test_labels,
        data.dv_scores,
        data.v_test,
        data.test_groups,
        calib_fraction=config.calib_fraction,
        seed=config.seed,
    )
    calib_ps, calib_bs, calib_labels, calib_dv, calib_v, calib_groups = calib_arrays
    eval_ps, eval_bs, eval_labels, eval_dv, eval_v, eval_groups = eval_arrays
    assert eval_ps is not None and eval_bs is not None and eval_labels is not None
    assert eval_dv is not None and eval_v is not None
    assert calib_dv is not None
    assert calib_ps is not None and calib_bs is not None and calib_labels is not None
    logger.info(f"Calib: n={len(calib_dv)}, Eval: n={len(eval_labels)}")

    # --- Ranking signals and reference metrics (on eval split) ---
    # For the batched top-k uncertainty baseline, pass ranking_scores=None so
    # that select_fixed_budget_amount uses its default within-batch median
    # ranking (McKenzie et al. applied per batch).  No precomputed scores needed.
    #
    # For the uncertainty calibrated threshold (LTT), we need a per-instance
    # signal in the same space as the calibration scores: rank of each eval
    # score within the sorted calib distribution.  This makes the tau found
    # during calibration transfer directly and requires no pool at test time.
    eval_uncertainty_ltt = probe_uncertainty(eval_ps, reference=calib_ps)
    signals = {
        "Unc. top-k": None,  # default: within-batch median ranking (McKenzie)
        "DV top-k": eval_dv,
        "Oracle top-k": eval_v.astype(float),
    }
    probe_auc = float(roc_auc_score(eval_labels, eval_ps))
    probe_acc = float(accuracy_score(eval_labels, (eval_ps >= 0.5).astype(int)))
    baseline_auc = float(roc_auc_score(eval_labels, eval_bs))
    baseline_acc = float(accuracy_score(eval_labels, (eval_bs >= 0.5).astype(int)))
    logger.info(f"Probe only:    AUC={probe_auc:.4f}, Acc={probe_acc:.4f}")
    logger.info(f"Baseline only: AUC={baseline_auc:.4f}, Acc={baseline_acc:.4f}")

    # --- LTT calibration ---
    ltt_sweep_results, ltt_plot_data = run_ltt_calibration(
        calib_ps,
        calib_bs,
        calib_labels,
        calib_dv,
        eval_ps,
        eval_bs,
        eval_labels,
        eval_dv,
        eval_uncertainty_ltt,
        dv_scores_full,
        dv_target,
        config,
        merge_strategy,
    )

    # --- Global oracle ---
    alpha_budgets_oracle = np.linspace(0.05, 1, config.n_alpha_steps)
    oracle_results = global_oracle_sweep(
        eval_ps,
        eval_bs,
        eval_labels,
        eval_v.astype(float),
        alpha_budgets_oracle,
        merge_strategy=merge_strategy,
    )

    # --- Batched top-k sweeps ---
    batch_sizes = config.batch_sizes
    n_k_steps = config.n_k_steps

    topk_results: dict[int, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
    for bs in batch_sizes:
        logger.info(f"\n--- Batch size B={bs} ---")
        k_values = np.unique(np.linspace(0, bs, n_k_steps + 1).astype(int))
        budget_fractions = k_values / bs

        signal_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, ranking in signals.items():
            aucs, accs = batched_topk_sweep(
                eval_ps,
                eval_bs,
                eval_labels,
                ranking,
                bs,
                k_values,
                merge_strategy=merge_strategy,
            )
            signal_results[name] = (aucs, accs)
            for frac in [0.1, 0.2, 0.3, 0.5]:
                idx = np.argmin(np.abs(budget_fractions - frac))
                logger.info(
                    f"  {name:>18s} @ k/B={budget_fractions[idx]:.2f}: AUC={aucs[idx]:.4f}, Acc={accs[idx]:.4f}"
                )

        topk_results[bs] = (budget_fractions, signal_results)

    # --- FAIC for all methods ---
    def _curve_to_rows(alphas, aucs, accs):
        return [
            {"alpha": float(a), "auc": float(u), "accuracy": float(c)}
            for a, u, c in zip(alphas, aucs, accs, strict=True)
        ]

    faic: dict[str, dict[str, float]] = {}

    # LTT methods (CTD, Unc. calibrated)
    for name, rows in ltt_sweep_results.items():
        faic[name] = {
            "auc": compute_faic(rows, oracle_results, probe_auc, metric="auc"),
            "acc": compute_faic(rows, oracle_results, probe_acc, metric="accuracy"),
        }

    # Top-k methods (use largest batch size)
    largest_bs = max(batch_sizes)
    topk_fracs, topk_signals = topk_results[largest_bs]
    for name, (aucs, accs) in topk_signals.items():
        rows = _curve_to_rows(topk_fracs, aucs, accs)
        faic[name] = {
            "auc": compute_faic(rows, oracle_results, probe_auc, metric="auc"),
            "acc": compute_faic(rows, oracle_results, probe_acc, metric="accuracy"),
        }

    logger.info("FAIC scores:")
    for name, scores in faic.items():
        logger.info(f"  {name:>18s}: AUC={scores['auc']:.4f}, Acc={scores['acc']:.4f}")

    # --- Generate plots ---
    logger.info("\nGenerating plots...")
    figs: dict[str, plt.Figure] = {}

    for bs in batch_sizes:
        budget_fractions, signal_results = topk_results[bs]
        figs[f"B{bs}"] = plot_single_batch_size(
            budget_fractions,
            signal_results,
            probe_auc,
            probe_acc,
            baseline_auc,
            baseline_acc,
            bs,
            output_dir,
            ltt_results=ltt_plot_data,
            file_prefix=file_prefix,
        )

    figs["grid"] = plot_grid(
        topk_results,
        probe_auc,
        probe_acc,
        baseline_auc,
        baseline_acc,
        output_dir,
        ltt_results=ltt_plot_data,
        file_prefix=file_prefix,
    )

    dv_ltt_rows = ltt_sweep_results.get("CTD", [])
    representative = dv_ltt_rows[len(dv_ltt_rows) // 2] if dv_ltt_rows else None
    if representative is not None:
        figs["adaptivity"] = plot_adaptivity(
            eval_ps,
            eval_bs,
            eval_dv,
            eval_labels,
            eval_groups,
            representative["tau"],
            representative["alpha"],
            batch_sizes[-1],
            output_dir,
            merge_strategy=merge_strategy,
            file_prefix=file_prefix,
        )

    plt.close("all")
    logger.info(f"Plots saved to {output_dir}")

    # --- Save results ---
    save_results_json(
        output_dir,
        file_prefix,
        config,
        merge_strategy,
        dv_target,
        probe_auc,
        probe_acc,
        baseline_auc,
        baseline_acc,
        dv_auc,
        topk_results,
        ltt_sweep_results,
        oracle_results,
        faic,
    )

    return DVCascadeExperimentResults(
        probe_auc=probe_auc,
        probe_acc=probe_acc,
        baseline_auc=baseline_auc,
        baseline_acc=baseline_acc,
        dv_auc=dv_auc,
        dv_target=dv_target,
        merge_strategy=merge_strategy,
        topk_results=topk_results,
        ltt_sweep_results=ltt_sweep_results,
        oracle_results=oracle_results,
        faic=faic,
        figs=figs,
    )


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="DV cascade comparison experiment")
    parser.add_argument("--config", type=str, default="configs/dv_cascade_comparison.yaml")
    default_output = os.path.join(os.environ.get("RESULTS_DIR", "results"), "dv_cascade_comparison")
    parser.add_argument("--output-dir", type=str, default=default_output)
    parser.add_argument("--file-prefix", type=str, default="", help="Prefix for output filenames (e.g. 'llama1b_')")
    parser.add_argument("--use-clearml", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / Path(args.config).name)

    config = load_config(args.config)

    # --- ClearML init ---
    clearml_logger = None
    if args.use_clearml:
        import os

        from clearml_logger import ClearMLLogger

        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="dv_cascade_comparison",
            enabled=True,
        )
        clearml_logger.add_tags(
            [
                "dv-cascade-comparison",
                f"baseline:{config.baseline_model_name}",
                f"activations:{config.activations_model_name}",
                f"layer:{config.activations_layer}",
                f"reduction:{config.reduction_strategy}",
                f"batches:{','.join(str(b) for b in config.batch_sizes)}",
                f"pareto:{config.pareto_testing}",
                f"opt_risk:{config.opt_risk}",
            ]
        )

    # --- Run experiment ---
    results = run_dv_cascade_experiment(config, output_dir, file_prefix=args.file_prefix)

    # --- ClearML logging ---
    if clearml_logger is not None:
        scalars = {
            "dv_probe_auc": results.dv_auc,
            "probe_only_auc": results.probe_auc,
            "baseline_only_auc": results.baseline_auc,
        }
        for bs in config.batch_sizes:
            budget_fractions, signal_results = results.topk_results[bs]
            idx_20 = np.argmin(np.abs(budget_fractions - 0.2))
            dv_auc_20 = signal_results["DV top-k"][0][idx_20]
            unc_auc_20 = signal_results["Unc. top-k"][0][idx_20]
            scalars[f"dv_advantage_auc_B{bs}_at_20pct"] = dv_auc_20 - unc_auc_20

        clearml_logger.log_scalars(scalars)
        for name, fig in results.figs.items():
            clearml_logger.log_figure("DV Cascade Comparison", name, fig)
        clearml_logger.finalize()
        logger.info("Results logged to ClearML.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
