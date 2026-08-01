"""Recompute probe-ablation metrics and figures from saved score artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from aaai_plot_style import AAAI_COLUMN_WIDTH, AAAI_TEXT_WIDTH, apply_aaai_plot_style
from config import load_config
from dv_cascade_comparison import (
    _add_figure_legend,
    _plot_single_batch_size,
    batched_topk_sweep,
    global_oracle_sweep,
    plot_single_batch_size,
    run_ltt_calibration,
)
from dv_ltt_cascade import split_calib_eval
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score

from reliable_monitoring.cascade import probe_uncertainty

apply_aaai_plot_style()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUMMARY_BUDGETS = (0.1, 0.2, 0.3, 0.5)
CELL_ORDER = (
    "mean_ridge",
    "attention_attention",
    "attention_ridge",
    "softmax_softmax",
    "softmax_ridge",
    "mlp_mlp",
    "mlp_ridge",
)
CELL_LABELS = {
    "mean_ridge": "Mean + ridge",
    "attention_attention": "Attention + matched",
    "attention_ridge": "Attention + ridge",
    "softmax_softmax": "Softmax + matched",
    "softmax_ridge": "Softmax + ridge",
    "mlp_mlp": "MLP + matched",
    "mlp_ridge": "MLP + ridge",
}
MATCHED_CELLS = ("mean_ridge", "attention_attention", "softmax_softmax", "mlp_mlp")
MATCHED_LABELS = {
    "mean_ridge": "Mean",
    "attention_attention": "Attention",
    "softmax_softmax": "Softmax",
    "mlp_mlp": "MLP",
}
ARCHITECTURE_COLORS = dict(zip(MATCHED_CELLS, ("C0", "C1", "C2", "C3"), strict=True))
ARCHITECTURE_MARKERS = dict(zip(MATCHED_CELLS, ("o", "s", "D", "^"), strict=True))
PANEL_OPT_RISKS = ("accuracy_error", "roc_auc_error")
METHOD_SPECIFIC_MERGE_STRATEGIES = {
    "CTD": "replace",
    "DV top-k": "replace",
    "Oracle top-k": "replace",
    "Unc. calibrated": "avg",
    "Unc. top-k": "avg",
}


def _nearest(rows: list[dict], alpha: float) -> dict | None:
    return min(rows, key=lambda row: abs(row["alpha"] - alpha)) if rows else None


def evaluate_artifact(
    artifact_path: Path,
    *,
    opt_risk: str | None = None,
    merge_strategy: str | None = None,
    method_merge_strategies: Mapping[str, str] | None = None,
    results_filename: str = "results.json",
    write_cell_figure: bool = True,
) -> dict:
    """Evaluate one saved score artifact under a chosen Pareto objective.

    The default arguments reproduce the original accuracy-optimised analysis
    and its filenames. Passing ``opt_risk="roc_auc_error"`` recalibrates only
    the Pareto/LTT threshold from the saved scores; it does not retrain probes
    or load model activations.
    """
    metadata = json.loads((artifact_path.parent / "metadata.json").read_text())
    # This entry point reconstructs the original budget-guarantee ablation.
    # Old score artifacts predate the explicit guaranteed_risk config field.
    config_dict = {**metadata["config"], "guaranteed_risk": "budget"}
    if opt_risk is not None:
        config_dict["opt_risk"] = opt_risk
    if merge_strategy is not None:
        config_dict["merge_strategy"] = merge_strategy
    if merge_strategy is not None and method_merge_strategies is not None:
        raise ValueError("Specify either merge_strategy or method_merge_strategies, not both")
    config = SimpleNamespace(**config_dict)
    default_merge_strategy = config.merge_strategy
    merge_strategies = {method: default_merge_strategy for method in METHOD_SPECIFIC_MERGE_STRATEGIES}
    if method_merge_strategies is not None:
        unknown = set(method_merge_strategies) - set(merge_strategies)
        if unknown:
            raise ValueError(f"Unknown methods in method_merge_strategies: {sorted(unknown)}")
        merge_strategies.update(method_merge_strategies)
    invalid = {method: strategy for method, strategy in merge_strategies.items() if strategy not in {"avg", "replace"}}
    if invalid:
        raise ValueError(f"Invalid merge strategies: {invalid}")
    arrays = np.load(artifact_path)

    calib, evaluation = split_calib_eval(
        arrays["test_probe"],
        arrays["test_expert"],
        arrays["test_labels"],
        arrays["test_dv"],
        arrays["test_value"],
        arrays["test_groups"],
        calib_fraction=config.calib_fraction,
        seed=config.seed,
    )
    calib_ps, calib_expert, calib_labels, calib_dv, _, _ = calib
    eval_ps, eval_expert, eval_labels, eval_dv, eval_value, _ = evaluation
    assert calib_ps is not None
    assert calib_expert is not None
    assert calib_labels is not None
    assert calib_dv is not None
    assert eval_ps is not None
    assert eval_expert is not None
    assert eval_labels is not None
    assert eval_dv is not None
    assert eval_value is not None

    eval_uncertainty = probe_uncertainty(eval_ps, reference=calib_ps)
    ltt_by_merge = {}
    ltt_plot_by_merge = {}
    for strategy in {
        merge_strategies["CTD"],
        merge_strategies["Unc. calibrated"],
    }:
        strategy_ltt, strategy_plot = run_ltt_calibration(
            calib_ps,
            calib_expert,
            calib_labels,
            calib_dv,
            eval_ps,
            eval_expert,
            eval_labels,
            eval_dv,
            eval_uncertainty,
            arrays["test_dv"],
            "continuous",
            config,
            strategy,
        )
        ltt_by_merge[strategy] = strategy_ltt
        ltt_plot_by_merge[strategy] = strategy_plot or {}

    ltt = {
        name: ltt_by_merge[merge_strategies[name]][name]
        for name in ("CTD", "Unc. calibrated")
        if name in ltt_by_merge[merge_strategies[name]]
    }
    ltt_plot = {
        name: ltt_plot_by_merge[merge_strategies[name]][name]
        for name in ("CTD", "Unc. calibrated")
        if name in ltt_plot_by_merge[merge_strategies[name]]
    }

    batch_size = config.batch_size
    k_values = np.unique(np.linspace(0, batch_size, config.n_k_steps + 1).astype(int))
    fractions = k_values / batch_size
    signals = {}
    for name, ranking in {"Unc. top-k": None, "DV top-k": eval_dv, "Oracle top-k": eval_value}.items():
        signals[name] = batched_topk_sweep(
            eval_ps,
            eval_expert,
            eval_labels,
            ranking,
            batch_size,
            k_values,
            merge_strategy=merge_strategies[name],
        )

    probe_auc = float(roc_auc_score(eval_labels, eval_ps))
    probe_acc = float(accuracy_score(eval_labels, eval_ps >= 0.5))
    expert_auc = float(roc_auc_score(eval_labels, eval_expert))
    expert_acc = float(accuracy_score(eval_labels, eval_expert >= 0.5))
    dv_spearman = float(spearmanr(eval_value, eval_dv).statistic)
    dv_auc = float(roc_auc_score(eval_value > 0, eval_dv))
    capacity = float((eval_value > 0).mean())
    ranking_quality = {}
    rng = np.random.default_rng(config.seed)
    random_order = rng.permutation(len(eval_value))
    dv_order = np.argsort(-eval_dv)
    uncertainty_order = np.argsort(-eval_uncertainty)
    for fraction in SUMMARY_BUDGETS:
        k = max(1, int(np.ceil(fraction * len(eval_value))))
        ranking_quality[str(fraction)] = {
            "dv_mean_value_at_k": float(eval_value[dv_order[:k]].mean()),
            "uncertainty_mean_value_at_k": float(eval_value[uncertainty_order[:k]].mean()),
            "random_mean_value_at_k": float(eval_value[random_order[:k]].mean()),
        }

    if write_cell_figure:
        plot_single_batch_size(
            fractions,
            signals,
            probe_auc,
            probe_acc,
            expert_auc,
            expert_acc,
            batch_size,
            artifact_path.parent,
            ltt_results=ltt_plot,
            file_prefix="ablation_",
        )
        plt.close("all")

    ctd_rows = ltt.get("CTD", [])
    unc_rows = ltt.get("Unc. calibrated", [])
    gains = {}
    for budget in SUMMARY_BUDGETS:
        ctd, unc = _nearest(ctd_rows, budget), _nearest(unc_rows, budget)
        gains[str(budget)] = {
            "ctd_auc": ctd["auc"] if ctd else None,
            "uncertainty_auc": unc["auc"] if unc else None,
            "auc_gain": ctd["auc"] - unc["auc"] if ctd and unc else None,
            "ctd_accuracy": ctd["accuracy"] if ctd else None,
            "uncertainty_accuracy": unc["accuracy"] if unc else None,
            "accuracy_gain": ctd["accuracy"] - unc["accuracy"] if ctd and unc else None,
            "realized_budget": ctd["realized_budget"] if ctd else None,
        }

    oracle = global_oracle_sweep(
        eval_ps,
        eval_expert,
        eval_labels,
        eval_value,
        np.linspace(0.05, 1.0, config.n_alpha_steps),
        merge_strategies["Oracle top-k"],
    )
    result = {
        "cell": config.cell_name,
        "expert": config.expert_name,
        "seed": config.seed,
        "opt_risk": config.opt_risk,
        "batch_size": batch_size,
        "probe": config.probe,
        "dv_probe": config.dv_probe,
        "probe_auc": probe_auc,
        "probe_accuracy": probe_acc,
        "expert_auc": expert_auc,
        "expert_accuracy": expert_acc,
        "dv_auc": dv_auc,
        "dv_spearman": dv_spearman,
        "delegation_capacity": capacity,
        "mean_delegation_value": float(eval_value.mean()),
        "ranking_quality": ranking_quality,
        "gains": gains,
        "ltt": ltt,
        "oracle": oracle,
        "topk": {
            "budget_fractions": fractions.tolist(),
            "signals": {name: {"auc": auc.tolist(), "accuracy": acc.tolist()} for name, (auc, acc) in signals.items()},
        },
    }
    if merge_strategy is not None:
        result["merge_strategy"] = config.merge_strategy
    if method_merge_strategies is not None:
        result["method_merge_strategies"] = merge_strategies
    (artifact_path.parent / results_filename).write_text(json.dumps(result, indent=2))
    return result


def _summary_row(result: dict) -> dict:
    row = {
        key: result[key]
        for key in (
            "cell",
            "expert",
            "seed",
            "probe_auc",
            "probe_accuracy",
            "expert_auc",
            "expert_accuracy",
            "dv_auc",
            "dv_spearman",
            "delegation_capacity",
            "mean_delegation_value",
        )
    }
    for budget, values in result["gains"].items():
        suffix = f"{round(float(budget) * 100)}pct"
        for metric, value in values.items():
            row[f"{metric}_{suffix}"] = value
        for metric, value in result["ranking_quality"][budget].items():
            row[f"{metric}_{suffix}"] = value
    return row


def plot_summary(results: list[dict], output_dir: Path, budget: float = 0.2) -> None:
    cells = [cell for cell in CELL_ORDER if any(result["cell"] == cell for result in results)]
    experts = [expert for expert in ("strong", "weak") if any(result["expert"] == expert for result in results)]
    fig, axes = plt.subplots(1, len(experts), figsize=(6 * len(experts), 4), squeeze=False)
    for ax, expert in zip(axes[0], experts, strict=True):
        means, errors = [], []
        for cell in cells:
            values = [
                r["gains"][str(budget)]["auc_gain"]
                for r in results
                if r["expert"] == expert and r["cell"] == cell and r["gains"][str(budget)]["auc_gain"] is not None
            ]
            means.append(float(np.mean(values)) if values else np.nan)
            errors.append(float(np.std(values)) if len(values) > 1 else 0.0)
        colors = ["tab:blue" if cell in MATCHED_CELLS else "0.65" for cell in cells]
        ax.bar(np.arange(len(cells)), means, yerr=errors, capsize=3, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(cells)), [CELL_LABELS[cell] for cell in cells], rotation=35, ha="right")
        ax.set_title(f"{expert.title()} expert")
        ax.set_ylabel("ROC AUC gain over uncertainty")
        ax.set_xlabel("Safety / DV architecture")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    stem = output_dir / f"summary_gain_{round(budget * 100)}pct"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_curves(results: list[dict], output_dir: Path) -> None:
    """Compare the four matched architecture families across target budgets."""
    experts = [expert for expert in ("strong", "weak") if any(result["expert"] == expert for result in results)]
    fig, axes = plt.subplots(1, len(experts), figsize=(6 * len(experts), 4), squeeze=False, sharex=True)
    colors = dict(zip(MATCHED_CELLS, ("tab:gray", "tab:blue", "tab:orange", "tab:green"), strict=True))
    for ax, expert in zip(axes[0], experts, strict=True):
        for cell in MATCHED_CELLS:
            runs = [result for result in results if result["expert"] == expert and result["cell"] == cell]
            if not runs:
                continue
            alpha = np.asarray([row["alpha"] for row in runs[0]["ltt"]["CTD"]])
            auc = np.asarray([[row["auc"] for row in run["ltt"]["CTD"]] for run in runs])
            mean = auc.mean(axis=0)
            std = auc.std(axis=0)
            ax.plot(alpha, mean, label=MATCHED_LABELS[cell], color=colors[cell], linewidth=2)
            ax.fill_between(alpha, mean - std, mean + std, color=colors[cell], alpha=0.15, linewidth=0)
        ax.set_title(f"{expert.title()} expert")
        ax.set_xlabel("Target delegation budget")
        ax.set_ylabel("CTD ROC AUC")
        ax.set_xlim(0.05, 0.95)
        ax.grid(alpha=0.25)
    axes[0, -1].legend(frameon=False)
    fig.tight_layout()
    stem = output_dir / "summary_matched_architectures"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_aggregate_summary(
    results: list[dict],
    output_dir: Path,
    *,
    filename: str = "summary_aggregate.csv",
) -> None:
    """Write mean and standard deviation across seeds for the paper table."""
    rows = []
    for expert in ("strong", "weak"):
        for cell in CELL_ORDER:
            runs = [result for result in results if result["expert"] == expert and result["cell"] == cell]
            if not runs:
                continue
            row = {"expert": expert, "cell": cell, "n_seeds": len(runs)}
            values = {
                "probe_auc": [run["probe_auc"] for run in runs],
                "probe_accuracy": [run["probe_accuracy"] for run in runs],
                "dv_spearman": [run["dv_spearman"] for run in runs],
                "delegation_capacity": [run["delegation_capacity"] for run in runs],
            }
            for budget in SUMMARY_BUDGETS:
                suffix = f"{round(budget * 100)}pct"
                values[f"ctd_auc_{suffix}"] = [run["gains"][str(budget)]["ctd_auc"] for run in runs]
                values[f"auc_gain_{suffix}"] = [run["gains"][str(budget)]["auc_gain"] for run in runs]
            for metric, metric_values in values.items():
                array = np.asarray(metric_values, dtype=float)
                row[f"{metric}_mean"] = float(array.mean())
                row[f"{metric}_std"] = float(array.std())
            rows.append(row)
    with (output_dir / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_topk_results(runs: list[dict]) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    fractions = np.asarray(runs[0]["topk"]["budget_fractions"])
    signals = {}
    for name in runs[0]["topk"]["signals"]:
        auc = np.asarray([run["topk"]["signals"][name]["auc"] for run in runs], dtype=float)
        accuracy = np.asarray([run["topk"]["signals"][name]["accuracy"] for run in runs], dtype=float)
        signals[name] = (auc.mean(axis=0), accuracy.mean(axis=0))
    return fractions, signals


def _mean_ltt_results(
    auc_runs: list[dict],
    accuracy_runs: list[dict],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Combine separately calibrated policies into the two plotting panels."""
    curves = {}
    for name in ("CTD", "Unc. calibrated"):
        auc_alpha = np.asarray([row["alpha"] for row in auc_runs[0]["ltt"][name]], dtype=float)
        accuracy_alpha = np.asarray([row["alpha"] for row in accuracy_runs[0]["ltt"][name]], dtype=float)
        if not np.array_equal(auc_alpha, accuracy_alpha):
            raise ValueError(f"Target budgets differ between AUROC and accuracy runs for {name}")
        auc = np.asarray([[row["auc"] for row in run["ltt"][name]] for run in auc_runs], dtype=float)
        accuracy = np.asarray(
            [[row["accuracy"] for row in run["ltt"][name]] for run in accuracy_runs],
            dtype=float,
        )
        curves[name] = (auc_alpha, auc.mean(axis=0), accuracy.mean(axis=0))
    return curves


def _realized_budget_curve(runs: list[dict], method: str = "CTD") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target budgets and the mean/std realised budget across runs."""
    alpha = np.asarray([row["alpha"] for row in runs[0]["ltt"][method]], dtype=float)
    realized = []
    for run in runs:
        run_alpha = np.asarray([row["alpha"] for row in run["ltt"][method]], dtype=float)
        if not np.array_equal(alpha, run_alpha):
            raise ValueError(f"Target budgets differ between runs for {method}")
        realized.append([row["realized_budget"] for row in run["ltt"][method]])
    values = np.asarray(realized, dtype=float)
    return alpha, values.mean(axis=0), values.std(axis=0)


def plot_budget_utilisation(
    accuracy_results: list[dict],
    auc_results: list[dict],
    output_dir: Path,
) -> None:
    """Plot how much of each target budget the attention CTD policy uses."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(AAAI_COLUMN_WIDTH, 2.35),
        sharex=True,
        sharey=True,
    )
    styles = (
        ("AUROC objective", auc_results, "C1", "-"),
        ("Accuracy objective", accuracy_results, "C1", "--"),
    )
    for ax, expert in zip(axes, ("strong", "weak"), strict=True):
        for label, results, color, linestyle in styles:
            runs = _runs_for_cell(results, expert, "attention_attention")
            if not runs:
                continue
            alpha, mean, std = _realized_budget_curve(runs)
            ax.plot(alpha, mean, color=color, linestyle=linestyle, linewidth=1, label=label)
            ax.fill_between(alpha, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.plot([0, 1], [0, 1], color="0.35", linestyle=":", linewidth=1, label="Fixed-quota top-$k$")
        ax.set_title(f"{expert.title()} expert")
        ax.set_xlabel(r"Target budget $\alpha_{\mathrm{B}}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Realised delegation rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.89, bottom=0.42, wspace=0.24)
    stem = output_dir / "attention_budget_utilisation"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def plot_paper_cell_figures(
    accuracy_results: list[dict],
    auc_results: list[dict],
    output_dir: Path,
) -> None:
    """Reproduce the paper's two-panel cascade figure for each main architecture."""
    for expert in ("strong", "weak"):
        for cell in MATCHED_CELLS:
            accuracy_runs = [
                result for result in accuracy_results if result["expert"] == expert and result["cell"] == cell
            ]
            auc_runs = [result for result in auc_results if result["expert"] == expert and result["cell"] == cell]
            if not accuracy_runs:
                continue
            if {run["seed"] for run in accuracy_runs} != {run["seed"] for run in auc_runs}:
                raise ValueError(f"AUROC and accuracy runs differ for {expert}/{cell}")
            fractions, signals = _mean_topk_results(accuracy_runs)
            batch_size = int(accuracy_runs[0].get("batch_size", 128))
            fig = plot_single_batch_size(
                fractions,
                signals,
                float(np.mean([run["probe_auc"] for run in accuracy_runs])),
                float(np.mean([run["probe_accuracy"] for run in accuracy_runs])),
                float(np.mean([run["expert_auc"] for run in accuracy_runs])),
                float(np.mean([run["expert_accuracy"] for run in accuracy_runs])),
                batch_size,
                output_dir,
                ltt_results=_mean_ltt_results(auc_runs, accuracy_runs),
                file_prefix=f"{expert}_{cell}_",
                layout="column",
            )
            fig.savefig(
                output_dir / f"{expert}_{cell}_ranking_comparison_B{batch_size}.png",
                dpi=200,
                bbox_inches="tight",
            )
            paper_name = f"{MATCHED_LABELS[cell].lower()}_{expert}_B{batch_size}"
            fig.savefig(output_dir / f"{paper_name}.pdf", bbox_inches="tight")
            fig.savefig(output_dir / f"{paper_name}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)


def _plot_architecture_panel(
    ax: plt.Axes,
    results: list[dict],
    expert: str,
    metric: str,
    *,
    show_expert_value: bool = True,
) -> None:
    probe_key = "probe_auc" if metric == "auc" else "probe_accuracy"
    expert_key = "expert_auc" if metric == "auc" else "expert_accuracy"
    for cell in MATCHED_CELLS:
        runs = [result for result in results if result["expert"] == expert and result["cell"] == cell]
        if not runs:
            continue
        alpha = np.asarray([0.0, *[row["alpha"] for row in runs[0]["ltt"]["CTD"]]])
        values = np.asarray(
            [[run[probe_key], *[row[metric] for row in run["ltt"]["CTD"]]] for run in runs],
            dtype=float,
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(
            alpha,
            mean,
            label=MATCHED_LABELS[cell],
            color=ARCHITECTURE_COLORS[cell],
            marker=ARCHITECTURE_MARKERS[cell],
            linewidth=0.8,
            markersize=2,
            markevery=2,
        )
        ax.fill_between(
            alpha,
            mean - std,
            mean + std,
            color=ARCHITECTURE_COLORS[cell],
            alpha=0.12,
            linewidth=0,
        )
    expert_value = float(np.mean([result[expert_key] for result in results if result["expert"] == expert]))
    expert_label = f"Expert only ({expert_value:.3f})" if show_expert_value else "Expert only"
    ax.axhline(
        expert_value,
        color="gray",
        ls="--",
        lw=0.8,
        alpha=0.5,
        label=expert_label,
    )
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Cascade ROC AUC" if metric == "auc" else "Cascade Accuracy")
    ax.set_title(r"Batch size $B = 128$")
    ax.grid(alpha=0.3)


def _add_architecture_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=9,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.18)


def plot_paper_architecture_comparisons(
    accuracy_results: list[dict],
    auc_results: list[dict],
    output_dir: Path,
) -> None:
    """Compare all main architecture families using the paper's metric panels."""
    for expert in ("strong", "weak"):
        fig, (ax_auc, ax_acc) = plt.subplots(1, 2, figsize=(AAAI_TEXT_WIDTH, 3.1))
        _plot_architecture_panel(ax_auc, auc_results, expert, "auc")
        _plot_architecture_panel(ax_acc, accuracy_results, expert, "accuracy")
        _add_architecture_legend(fig, ax_auc)
        stem = output_dir / f"{expert}_architecture_comparison_B128"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(AAAI_TEXT_WIDTH, 5.8), squeeze=False)
    for row, expert in enumerate(("strong", "weak")):
        _plot_architecture_panel(axes[row, 0], auc_results, expert, "auc", show_expert_value=False)
        _plot_architecture_panel(
            axes[row, 1],
            accuracy_results,
            expert,
            "accuracy",
            show_expert_value=False,
        )
        axes[row, 0].set_title(f"{expert.title()} expert")
        axes[row, 1].set_title(f"{expert.title()} expert")
    _add_architecture_legend(fig, axes[0, 0])
    fig.subplots_adjust(bottom=0.12, hspace=0.48, wspace=0.36)
    stem = output_dir / "architecture_comparison_grid_B128"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary_files(results: list[dict], root: Path, stem: str) -> None:
    rows = [_summary_row(result) for result in results]
    (root / f"{stem}.json").write_text(json.dumps(rows, indent=2))
    with (root / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse_root(
    root: Path,
    *,
    merge_strategy: str | None = None,
    method_merge_strategies: Mapping[str, str] | None = None,
) -> list[dict]:
    """Analyse saved scores under both metric-specific Pareto objectives."""
    if merge_strategy is not None and method_merge_strategies is not None:
        raise ValueError("Specify either merge_strategy or method_merge_strategies, not both")
    artifacts = sorted(root.rglob("scores.npz"))
    if not artifacts:
        raise FileNotFoundError(f"No scores.npz files below {root}")

    if method_merge_strategies is not None:
        suffix = "_method_specific"
        output_root = root / "fusion_method_specific"
    elif merge_strategy is not None:
        suffix = f"_{merge_strategy}"
        output_root = root / f"merge_{merge_strategy}"
    else:
        suffix = ""
        output_root = root
    output_root.mkdir(exist_ok=True)

    results_by_risk = {}
    for opt_risk in PANEL_OPT_RISKS:
        results = [
            evaluate_artifact(
                path,
                opt_risk=opt_risk,
                merge_strategy=merge_strategy,
                method_merge_strategies=method_merge_strategies,
                results_filename=f"results_{opt_risk}{suffix}.json",
                write_cell_figure=(
                    merge_strategy is None and method_merge_strategies is None and opt_risk == "accuracy_error"
                ),
            )
            for path in artifacts
        ]
        results_by_risk[opt_risk] = results
        _write_summary_files(results, output_root, f"summary_{opt_risk}")
        write_aggregate_summary(
            results,
            output_root,
            filename=f"summary_aggregate_{opt_risk}.csv",
        )

    accuracy_results = results_by_risk["accuracy_error"]
    auc_results = results_by_risk["roc_auc_error"]

    if merge_strategy is None and method_merge_strategies is None:
        # Preserve the original filenames exactly for downstream legacy analyses.
        for artifact, result in zip(artifacts, accuracy_results, strict=True):
            (artifact.parent / "results.json").write_text(json.dumps(result, indent=2))
        _write_summary_files(accuracy_results, root, "summary")
        write_aggregate_summary(accuracy_results, root)

    for budget in SUMMARY_BUDGETS:
        plot_summary(auc_results, output_root, budget)
    plot_architecture_curves(auc_results, output_root)
    paper_output = output_root / "paper_figures"
    paper_output.mkdir(exist_ok=True)
    plot_paper_cell_figures(accuracy_results, auc_results, paper_output)
    plot_paper_architecture_comparisons(accuracy_results, auc_results, paper_output)
    if method_merge_strategies is not None:
        plot_budget_utilisation(accuracy_results, auc_results, paper_output)
    return accuracy_results


def _runs_for_cell(results: list[dict], expert: str, cell: str) -> list[dict]:
    return [result for result in results if result["expert"] == expert and result["cell"] == cell]


def plot_fusion_control_figures(
    replace_accuracy: list[dict],
    replace_auc: list[dict],
    average_accuracy: list[dict],
    average_auc: list[dict],
    output_dir: Path,
) -> None:
    """Compare fixed replacement and averaging protocols for Attention probes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    protocols = (
        ("All methods replace", replace_accuracy, replace_auc),
        ("All methods average", average_accuracy, average_auc),
    )
    for expert in ("strong", "weak"):
        fig, axes = plt.subplots(2, 2, figsize=(AAAI_TEXT_WIDTH, 5.4), squeeze=False)
        for row, (label, accuracy_results, auc_results) in enumerate(protocols):
            accuracy_runs = _runs_for_cell(accuracy_results, expert, "attention_attention")
            auc_runs = _runs_for_cell(auc_results, expert, "attention_attention")
            if not accuracy_runs or not auc_runs:
                continue
            fractions, signals = _mean_topk_results(accuracy_runs)
            _plot_single_batch_size(
                axes[row, 0],
                axes[row, 1],
                fractions,
                signals,
                float(np.mean([run["probe_auc"] for run in accuracy_runs])),
                float(np.mean([run["probe_accuracy"] for run in accuracy_runs])),
                float(np.mean([run["expert_auc"] for run in accuracy_runs])),
                float(np.mean([run["expert_accuracy"] for run in accuracy_runs])),
                int(accuracy_runs[0].get("batch_size", 128)),
                show_legend=False,
                ltt_results=_mean_ltt_results(auc_runs, accuracy_runs),
            )
            axes[row, 0].set_title(f"{label}: AUROC")
            axes[row, 1].set_title(f"{label}: accuracy")
        _add_figure_legend(fig, axes[0, 0])
        fig.subplots_adjust(bottom=0.13, hspace=0.42, wspace=0.36)
        stem = output_dir / f"attention_{expert}_fusion_controls_B128"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # The main paper also uses the weak-expert overwrite row as a standalone
    # stress test of fixed-quota delegation.
    accuracy_runs = _runs_for_cell(replace_accuracy, "weak", "attention_attention")
    auc_runs = _runs_for_cell(replace_auc, "weak", "attention_attention")
    if accuracy_runs and auc_runs:
        fractions, signals = _mean_topk_results(accuracy_runs)
        fig = plot_single_batch_size(
            fractions,
            signals,
            float(np.mean([run["probe_auc"] for run in accuracy_runs])),
            float(np.mean([run["probe_accuracy"] for run in accuracy_runs])),
            float(np.mean([run["expert_auc"] for run in accuracy_runs])),
            float(np.mean([run["expert_accuracy"] for run in accuracy_runs])),
            int(accuracy_runs[0].get("batch_size", 128)),
            output_dir,
            ltt_results=_mean_ltt_results(auc_runs, accuracy_runs),
            file_prefix="attention_weak_replacement_stress_",
            layout="column",
        )
        stem = output_dir / "attention_weak_replacement_stress_B128"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def _load_protocol_results(root: Path, protocol: str, opt_risk: str) -> list[dict]:
    if protocol == "replace":
        filename = f"results_{opt_risk}_replace.json"
    elif protocol == "avg":
        filename = f"results_{opt_risk}_avg.json"
    else:
        raise ValueError(f"Unknown fusion protocol: {protocol}")
    paths = sorted(root.rglob(filename))
    if not paths:
        raise FileNotFoundError(f"No {filename} files below {root}")
    return [json.loads(path.read_text()) for path in paths]


def analyse_fusion_comparison(root: Path) -> list[dict]:
    """Generate method-specific primary results and fixed-fusion controls."""
    analyse_root(root, merge_strategy="replace")
    analyse_root(root, merge_strategy="avg")
    primary = analyse_root(
        root,
        method_merge_strategies=METHOD_SPECIFIC_MERGE_STRATEGIES,
    )
    plot_fusion_control_figures(
        _load_protocol_results(root, "replace", "accuracy_error"),
        _load_protocol_results(root, "replace", "roc_auc_error"),
        _load_protocol_results(root, "avg", "accuracy_error"),
        _load_protocol_results(root, "avg", "roc_auc_error"),
        root / "fusion_method_specific" / "paper_figures",
    )
    return primary


def evaluate_accuracy_guarantee_artifact(
    artifact_path: Path,
    *,
    guarantee_performance_levels: list[float],
    shared_config: dict,
) -> dict:
    """Run accuracy-guarantee LTT using one saved probe-ablation artifact."""
    metadata = json.loads((artifact_path.parent / "metadata.json").read_text())
    config_dict = {
        **metadata["config"],
        **shared_config,
        "guaranteed_risk": "accuracy_error",
        "opt_risk": "budget",
        "guarantee_performance_levels": guarantee_performance_levels,
    }
    config = SimpleNamespace(**config_dict)
    arrays = np.load(artifact_path)

    calib, evaluation = split_calib_eval(
        arrays["test_probe"],
        arrays["test_expert"],
        arrays["test_labels"],
        arrays["test_dv"],
        calib_fraction=config.calib_fraction,
        seed=config.seed,
    )
    calib_ps, calib_expert, calib_labels, calib_dv = calib
    eval_ps, eval_expert, eval_labels, eval_dv = evaluation
    assert calib_ps is not None and calib_expert is not None
    assert calib_labels is not None and calib_dv is not None
    assert eval_ps is not None and eval_expert is not None
    assert eval_labels is not None and eval_dv is not None

    eval_uncertainty = probe_uncertainty(eval_ps, reference=calib_ps)
    ltt, _ = run_ltt_calibration(
        calib_ps,
        calib_expert,
        calib_labels,
        calib_dv,
        eval_ps,
        eval_expert,
        eval_labels,
        eval_dv,
        eval_uncertainty,
        arrays["test_dv"],
        "continuous",
        config,
        config.merge_strategy,
    )
    return {
        "cell": config.cell_name,
        "expert": config.expert_name,
        "seed": config.seed,
        "guaranteed_risk": config.guaranteed_risk,
        "opt_risk": config.opt_risk,
        "ltt": ltt,
    }


def plot_accuracy_guarantee_results(results: list[dict], output_dir: Path) -> None:
    """Plot delegation required for each certified minimum accuracy."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(AAAI_TEXT_WIDTH, 3.1),
        squeeze=False,
        sharey=True,
    )
    for ax, expert in zip(axes[0], ("strong", "weak"), strict=True):
        for cell in MATCHED_CELLS:
            runs = [result for result in results if result["expert"] == expert and result["cell"] == cell]
            if not runs:
                continue
            targets = sorted({row["target_performance"] for run in runs for row in run["ltt"].get("CTD", [])})
            means, errors, valid_targets = [], [], []
            for target in targets:
                budgets = [
                    row["realized_budget"]
                    for run in runs
                    for row in run["ltt"].get("CTD", [])
                    if row["target_performance"] == target
                ]
                if budgets:
                    valid_targets.append(target)
                    means.append(float(np.mean(budgets)))
                    errors.append(float(np.std(budgets)))
            ax.errorbar(
                valid_targets,
                means,
                yerr=errors,
                label=MATCHED_LABELS[cell],
                color=ARCHITECTURE_COLORS[cell],
                marker=ARCHITECTURE_MARKERS[cell],
                markersize=4,
                capsize=2,
            )
        ax.set_title(f"{expert.title()} expert")
        ax.set_xlabel("Guaranteed minimum accuracy")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Realized delegation rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.2, wspace=0.15)
    stem = output_dir / "budget_vs_guaranteed_accuracy"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyse_accuracy_guarantees(config_path: Path, output_dir: Path) -> list[dict]:
    """Run the reverse CTD formulation over saved architecture scores."""
    config = load_config(config_path)
    root = Path(config.artifact_root)
    results = []
    for expert in config.experts:
        levels = list(config.guarantee_performance_levels[expert])
        for cell in config.cells:
            for seed in config.seeds:
                artifact = root / expert / cell / f"seed_{seed}" / "scores.npz"
                if not artifact.exists():
                    raise FileNotFoundError(f"Missing score artifact: {artifact}")
                results.append(
                    evaluate_accuracy_guarantee_artifact(
                        artifact,
                        guarantee_performance_levels=levels,
                        shared_config=config.shared,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": vars(config), "results": results}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
    plot_accuracy_guarantee_results(results, output_dir)
    return results


def plot_saved_accuracy_guarantees(output_dir: Path) -> None:
    saved = json.loads((output_dir / "results.json").read_text())
    plot_accuracy_guarantee_results(saved["results"], output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse saved probe-ablation scores")
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--merge-strategy",
        choices=("avg", "replace", "method-specific"),
        help="Fusion protocol for saved-score analysis",
    )
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config is not None:
        if args.plot_only:
            if args.output_dir is None:
                raise SystemExit("--plot-only requires --output-dir")
            plot_saved_accuracy_guarantees(args.output_dir)
            logger.info("Figures saved to %s", args.output_dir.resolve())
        else:
            destination = args.output_dir or Path(
                "results/probe_ablation_accuracy_guarantee"
            ) / datetime.now().strftime("%Y%m%d_%H%M%S")
            analyse_accuracy_guarantees(args.config, destination)
            logger.info("Results saved to %s", destination.resolve())
    elif args.root is not None:
        if args.plot_only:
            raise SystemExit("--plot-only is only available with --config")
        if args.merge_strategy == "method-specific":
            analyse_fusion_comparison(args.root)
            destination = args.root / "fusion_method_specific"
        else:
            analyse_root(args.root, merge_strategy=args.merge_strategy)
            destination = args.root if args.merge_strategy is None else args.root / f"merge_{args.merge_strategy}"
        logger.info("Results saved to %s", destination.resolve())
    else:
        raise SystemExit("Provide an ablation root or --config")
