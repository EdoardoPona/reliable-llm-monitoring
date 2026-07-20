"""Recompute probe-ablation metrics and figures from saved score artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from dv_cascade_comparison import (
    batched_topk_sweep,
    global_oracle_sweep,
    plot_single_batch_size,
    run_ltt_calibration,
)
from dv_ltt_cascade import split_calib_eval
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score

from reliable_monitoring.cascade import probe_uncertainty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUMMARY_BUDGETS = (0.1, 0.2, 0.3, 0.5)


def _nearest(rows: list[dict], alpha: float) -> dict | None:
    return min(rows, key=lambda row: abs(row["alpha"] - alpha)) if rows else None


def evaluate_artifact(artifact_path: Path) -> dict:
    metadata = json.loads((artifact_path.parent / "metadata.json").read_text())
    config_dict = metadata["config"]
    config = SimpleNamespace(**config_dict)
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
    ltt, ltt_plot = run_ltt_calibration(
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
            merge_strategy=config.merge_strategy,
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
        config.merge_strategy,
    )
    result = {
        "cell": config.cell_name,
        "expert": config.expert_name,
        "seed": config.seed,
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
    (artifact_path.parent / "results.json").write_text(json.dumps(result, indent=2))
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
    cells = list(dict.fromkeys(result["cell"] for result in results))
    experts = list(dict.fromkeys(result["expert"] for result in results))
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
        ax.bar(np.arange(len(cells)), means, yerr=errors, capsize=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(cells)), cells, rotation=35, ha="right")
        ax.set_title(expert)
        ax.set_ylabel("CTD − uncertainty ROC AUC")
        ax.set_xlabel("Safety / DV architecture")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"summary_gain_{round(budget * 100)}pct.pdf", bbox_inches="tight")
    plt.close(fig)


def analyse_root(root: Path) -> list[dict]:
    artifacts = sorted(root.rglob("scores.npz"))
    if not artifacts:
        raise FileNotFoundError(f"No scores.npz files below {root}")
    results = [evaluate_artifact(path) for path in artifacts]
    rows = [_summary_row(result) for result in results]
    (root / "summary.json").write_text(json.dumps(rows, indent=2))
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for budget in SUMMARY_BUDGETS:
        plot_summary(results, root, budget)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse saved probe-ablation scores")
    parser.add_argument("root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    analyse_root(parse_args().root)
