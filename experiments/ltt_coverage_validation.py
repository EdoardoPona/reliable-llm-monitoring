"""LTT coverage validation experiment.

Empirically validates the PAC budget guarantee Pr(R_budget(λ*) ≤ α) ≥ 1 - δ
by running many random calib/eval splits and measuring violation rates.

For each trial:
  1. Randomly split test data into calib and eval (with a different seed)
  2. Calibrate threshold via LTT (budget-only) and LTT + Pareto
  3. Measure realized delegation rate on eval
  4. Check if budget constraint α is violated

The budget-only LTT should be tight: violation rate ≈ δ.
Pareto LTT may be more conservative (violation rate ≤ δ) since
the Pareto filter discards thresholds that hurt performance.

Outputs:
  - coverage_histogram.pdf: overlaid histograms of realized delegation rates
  - coverage_results.json: per-trial results and summary statistics

Usage::

    cd experiments && uv run ltt_coverage_validation.py \\
        --config configs/ltt_coverage/strong_expert.yaml --use-clearml
"""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import load_config
from dotenv import load_dotenv
from dv_ltt_cascade import (
    ltt_budget_threshold,
    pareto_ht_opt_split,
    prepare_dv_cascade_data,
    split_calib_eval,
    threshold_cascade,
)
from tqdm import tqdm

from reliable_monitoring.learn_then_test import build_pareto_ltt
from reliable_monitoring.risks import RISK_RGISTRY, BudgetCostRisk

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: single trial
# ---------------------------------------------------------------------------


def run_single_trial(
    test_ps: np.ndarray,
    test_bs: np.ndarray,
    test_labels: np.ndarray,
    dv_scores: np.ndarray,
    v_test: np.ndarray,
    test_groups: np.ndarray | None,
    alpha: float,
    delta: float,
    dv_tau_grid: np.ndarray,
    seed: int,
    calib_fraction: float,
    pareto_split_proportion: float,
    opt_risk_name: str,
    merge_strategy: str,
) -> dict:
    """Run one trial: split, calibrate, evaluate.

    Returns a dict with realized budget rates for each method,
    or None for methods that failed to find a valid threshold.
    """
    calib_arrays, eval_arrays = split_calib_eval(
        test_ps,
        test_bs,
        test_labels,
        dv_scores,
        v_test,
        test_groups,
        calib_fraction=calib_fraction,
        seed=seed,
    )
    calib_ps, calib_bs, calib_labels, calib_dv, _calib_v, _calib_groups = calib_arrays
    eval_ps, eval_bs, eval_labels, eval_dv, _eval_v, _eval_groups = eval_arrays
    assert calib_ps is not None and calib_bs is not None and calib_labels is not None and calib_dv is not None
    assert eval_ps is not None and eval_bs is not None and eval_labels is not None and eval_dv is not None

    result: dict = {"seed": seed}

    # --- Budget-only LTT ---
    tau_budget = ltt_budget_threshold(calib_dv, alpha, delta, dv_tau_grid)
    if tau_budget is not None:
        cascade = threshold_cascade(eval_ps, eval_bs, eval_dv, tau_budget, merge_strategy=merge_strategy)
        result["budget_only_rate"] = float(cascade.used_baseline.mean())
        result["budget_only_tau"] = tau_budget
    else:
        result["budget_only_rate"] = None
        result["budget_only_tau"] = None

    # --- Pareto LTT ---
    OptRisk = RISK_RGISTRY.get(opt_risk_name)
    if OptRisk is None:
        raise ValueError(f"Unknown opt_risk: '{opt_risk_name}'")

    ht_idx, opt_idx = pareto_ht_opt_split(len(calib_dv), pareto_split_proportion, seed)

    pareto_ltt = build_pareto_ltt(
        ht_delegation_scores=calib_dv[ht_idx],
        opt_probe_scores=calib_ps[opt_idx],
        opt_baseline_scores=calib_bs[opt_idx],
        opt_labels=calib_labels[opt_idx],
        opt_delegation_scores=calib_dv[opt_idx],
        tau_grid=dv_tau_grid,
        opt_risk=OptRisk,
        budget_risk=BudgetCostRisk,
        merge_strategy=merge_strategy,
    )

    tau_pareto = pareto_ltt.select_threshold(alpha, delta)
    if tau_pareto is not None:
        cascade = threshold_cascade(eval_ps, eval_bs, eval_dv, tau_pareto, merge_strategy=merge_strategy)
        result["pareto_rate"] = float(cascade.used_baseline.mean())
        result["pareto_tau"] = tau_pareto
    else:
        result["pareto_rate"] = None
        result["pareto_tau"] = None

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_coverage_histogram(
    trials: list[dict],
    alpha: float,
    delta: float,
    output_path: Path,
) -> plt.Figure:
    """Overlaid histograms of realized delegation rates across trials."""
    budget_only_rates = np.array([t["budget_only_rate"] for t in trials if t["budget_only_rate"] is not None])
    pareto_rates = np.array([t["pareto_rate"] for t in trials if t["pareto_rate"] is not None])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Fit bins to the data range with a small margin around alpha
    all_rates = np.concatenate([budget_only_rates, pareto_rates]) if len(pareto_rates) else budget_only_rates
    lo = min(float(all_rates.min()), alpha) - 0.02
    hi = max(float(all_rates.max()), alpha) + 0.02
    bins = np.linspace(lo, hi, 50).tolist()

    if len(budget_only_rates):
        violation_bo = float(np.mean(budget_only_rates > alpha))
        ax.hist(
            budget_only_rates,
            bins=bins,
            alpha=0.6,
            color="#6baed6",
            edgecolor="#2171b5",
            linewidth=0.5,
            label=f"LTT budget-only (violation: {violation_bo:.1%})",
        )

    if len(pareto_rates):
        violation_pareto = float(np.mean(pareto_rates > alpha))
        ax.hist(
            pareto_rates,
            bins=bins,
            alpha=0.6,
            color="#fdae6b",
            edgecolor="#e6550d",
            linewidth=0.5,
            label=f"LTT + Pareto (violation: {violation_pareto:.1%})",
        )

    ax.axvline(alpha, color="red", ls="--", lw=2, label=rf"$\alpha = {alpha}$")

    ax.set_xlabel("Realized delegation rate on eval split")
    ax.set_ylabel("Count")
    ax.set_title(rf"Budget coverage validation ($\alpha = {alpha}$, $\delta = {delta}$, $n = {len(trials)}$ trials)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="LTT coverage validation experiment")
    parser.add_argument("--config", type=str, required=True)
    default_output = os.path.join(os.environ.get("RESULTS_DIR", "results"), "ltt_coverage_validation")
    parser.add_argument("--output-dir", type=str, default=default_output)
    parser.add_argument("--alpha", type=float, default=None, help="Budget level (overrides config)")
    parser.add_argument("--n-trials", type=int, default=None, help="Number of random splits (overrides config)")
    parser.add_argument("--file-prefix", type=str, default="")
    parser.add_argument("--use-clearml", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / Path(args.config).name)

    config = load_config(args.config)
    merge_strategy = config.merge_strategy
    alpha = args.alpha if args.alpha is not None else config.alpha
    delta = config.delta
    n_trials = args.n_trials if args.n_trials is not None else config.n_trials

    # --- ClearML init ---
    clearml_logger = None
    if args.use_clearml:
        import os

        from clearml_logger import ClearMLLogger

        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="ltt_coverage_validation",
            enabled=True,
        )
        clearml_logger.add_tags(
            [
                "ltt-coverage",
                f"baseline:{config.baseline_model_name}",
                f"alpha:{alpha}",
                f"delta:{delta}",
                f"n_trials:{n_trials}",
            ]
        )

    # --- Train probes & load test data ---
    data = prepare_dv_cascade_data(config)
    test_ps = data.test_ps
    test_bs = data.test_bs
    test_labels = data.test_labels
    test_groups = data.test_groups
    dv_scores = data.dv_scores
    v_test = data.v_test
    dv_auc = data.dv_auc
    dv_tau_grid = data.dv_tau_grid

    # --- Run coverage trials ---
    calib_fraction = config.calib_fraction
    pareto_split_proportion = config.pareto_split_proportion
    opt_risk_name = config.opt_risk
    base_seed = config.seed

    logger.info(f"\n--- Running {n_trials} coverage trials (alpha={alpha}, delta={delta:.2f}) ---")

    # Suppress per-trial LTT logging
    logging.getLogger("dv_ltt_cascade").setLevel(logging.WARNING)

    trials: list[dict] = []
    for i in tqdm(range(n_trials), desc="Coverage trials"):
        trial = run_single_trial(
            test_ps,
            test_bs,
            test_labels,
            dv_scores,
            v_test,
            test_groups,
            alpha=alpha,
            delta=delta,
            dv_tau_grid=dv_tau_grid,
            seed=base_seed + i,
            calib_fraction=calib_fraction,
            pareto_split_proportion=pareto_split_proportion,
            opt_risk_name=opt_risk_name,
            merge_strategy=merge_strategy,
        )
        trials.append(trial)

    # --- Summary ---
    bo_rates = [t["budget_only_rate"] for t in trials if t["budget_only_rate"] is not None]
    pareto_rates = [t["pareto_rate"] for t in trials if t["pareto_rate"] is not None]

    bo_violation = float(np.mean(np.array(bo_rates) > alpha)) if bo_rates else None
    pareto_violation = float(np.mean(np.array(pareto_rates) > alpha)) if pareto_rates else None

    logger.info("\n--- Coverage summary ---")
    logger.info(f"  Budget-only LTT: {len(bo_rates)}/{n_trials} valid, violation rate = {bo_violation}")
    logger.info(f"  Pareto LTT:      {len(pareto_rates)}/{n_trials} valid, violation rate = {pareto_violation}")
    logger.info(f"  Target:          δ = {delta:.2f}")

    # --- Plot ---
    file_prefix = args.file_prefix
    fig = plot_coverage_histogram(
        trials,
        alpha,
        delta,
        output_dir / f"{file_prefix}coverage_histogram.pdf",
    )
    logger.info(f"Histogram saved to {output_dir}")

    # --- Save results ---
    results_json = {
        "config": {
            "alpha": alpha,
            "delta": delta,
            "n_trials": n_trials,
            "dv_target": data.dv_target,
            "baseline_model_name": config.baseline_model_name,
            "activations_model_name": config.activations_model_name,
            "merge_strategy": merge_strategy,
            "opt_risk": opt_risk_name,
        },
        "summary": {
            "budget_only_violation_rate": bo_violation,
            "pareto_violation_rate": pareto_violation,
            "budget_only_valid_trials": len(bo_rates),
            "pareto_valid_trials": len(pareto_rates),
            "budget_only_mean_rate": float(np.mean(bo_rates)) if bo_rates else None,
            "pareto_mean_rate": float(np.mean(pareto_rates)) if pareto_rates else None,
            "dv_probe_auc": dv_auc,
        },
        "trials": trials,
    }
    results_path = output_dir / f"{file_prefix}coverage_results.json"
    results_path.write_text(json.dumps(results_json, indent=2))
    logger.info(f"Results saved to {results_path}")

    # --- ClearML logging ---
    if clearml_logger is not None:
        clearml_logger.log_scalars(
            {
                "budget_only_violation_rate": bo_violation or 0.0,
                "pareto_violation_rate": pareto_violation or 0.0,
                "budget_only_mean_rate": float(np.mean(bo_rates)) if bo_rates else 0.0,
                "pareto_mean_rate": float(np.mean(pareto_rates)) if pareto_rates else 0.0,
                "dv_probe_auc": dv_auc,
                "alpha": alpha,
                "delta": delta,
            }
        )
        clearml_logger.log_figure("LTT Coverage", "coverage_histogram", fig)
        clearml_logger.finalize()
        logger.info("Results logged to ClearML.")

    plt.close(fig)
    logger.info("Done.")


if __name__ == "__main__":
    main()
