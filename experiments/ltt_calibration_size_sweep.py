"""LTT calibration-set-size sensitivity sweep.

Quantifies how the certified threshold and downstream cascade performance
change as the calibration set ``D_cal`` shrinks.  Mirrors the random-split
pattern of ``ltt_coverage_validation.py`` but with a calibration-size
knob: at each trial we redraw a calibration/evaluation split, subsample the
calibration half to a target size ``n_cal``, run **Pareto-LTT** (the paper's
main pipeline), and evaluate the certified cascade on this trial's
held-out evaluation half.  Reporting is aggregated per ``n_cal`` with
mean +/- std for tau*, realised budget, cascade AUC, and cascade accuracy.

The expected behaviour (and the rebuttal answer to 4hoA Q4): the formal LTT
guarantee remains valid at every ``n_cal``, but smaller calibration sets
give less statistical power, which manifests as **more conservative
thresholds** (realised budget falls further below the target ``alpha``)
rather than guarantee violations.

Usage::

    cd experiments && uv run ltt_calibration_size_sweep.py \\
        --config configs/ltt_cal_size_sweep/strong_expert.yaml
"""

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
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
from tqdm import tqdm

from reliable_monitoring.learn_then_test import build_pareto_ltt
from reliable_monitoring.risks import RISK_RGISTRY, BudgetCostRisk

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trial / aggregation dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """One Pareto-LTT call on a random calib-subsample/eval split."""

    seed: int
    tau: float | None
    realised_budget: float | None  # None when LTT certifies no tau
    cascade_auc: float | None
    cascade_accuracy: float | None


@dataclass
class SizeResult:
    """Aggregated stats for a given calibration-set size."""

    n_cal: int
    n_trials: int
    n_certified: int  # trials where Pareto-LTT found a valid tau
    mean_tau: float | None
    std_tau: float | None
    mean_realised_budget: float | None
    std_realised_budget: float | None
    violation_rate: float  # fraction of certified trials with realised_budget > alpha
    mean_cascade_auc: float | None
    std_cascade_auc: float | None
    mean_cascade_accuracy: float | None
    std_cascade_accuracy: float | None
    trials: list[TrialResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(n_cal: int, trials: list[TrialResult], alpha: float) -> SizeResult:
    certified = [t for t in trials if t.tau is not None]
    n_cert = len(certified)
    if n_cert == 0:
        return SizeResult(
            n_cal=n_cal,
            n_trials=len(trials),
            n_certified=0,
            mean_tau=None,
            std_tau=None,
            mean_realised_budget=None,
            std_realised_budget=None,
            violation_rate=0.0,
            mean_cascade_auc=None,
            std_cascade_auc=None,
            mean_cascade_accuracy=None,
            std_cascade_accuracy=None,
            trials=trials,
        )
    taus = np.array([t.tau for t in certified], dtype=float)
    budgets = np.array([t.realised_budget for t in certified], dtype=float)
    aucs = np.array([t.cascade_auc for t in certified], dtype=float)
    accs = np.array([t.cascade_accuracy for t in certified], dtype=float)
    return SizeResult(
        n_cal=n_cal,
        n_trials=len(trials),
        n_certified=n_cert,
        mean_tau=float(taus.mean()),
        std_tau=float(taus.std()),
        mean_realised_budget=float(budgets.mean()),
        std_realised_budget=float(budgets.std()),
        violation_rate=float((budgets > alpha).mean()),
        mean_cascade_auc=float(aucs.mean()),
        std_cascade_auc=float(aucs.std()),
        mean_cascade_accuracy=float(accs.mean()),
        std_cascade_accuracy=float(accs.std()),
        trials=trials,
    )


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------


def _run_trial(
    *,
    data,
    alpha: float,
    delta: float,
    calib_fraction: float,
    pareto_split_proportion: float,
    opt_risk,
    merge_strategy: str,
    n_cal: int,
    trial_seed: int,
) -> TrialResult:
    """One Pareto-LTT trial at a target ``n_cal``.

    Steps:
      1. Random calibration/evaluation split of the full test scores via
         ``split_calib_eval`` (fresh split per trial, both sides redrawn).
      2. Subsample the calibration half uniformly without replacement to
         size ``n_cal``.
      3. Inside the subsample: split into hypothesis-testing (HT) and
         optimisation (OPT) subsets, build the Pareto-LTT certifier, and
         select the threshold at level ``(alpha, delta)``.
      4. Evaluate the certified cascade on this trial's eval half.
    """
    calib_arrays, eval_arrays = split_calib_eval(
        data.test_ps,
        data.test_bs,
        data.test_labels,
        data.dv_scores,
        data.test_groups,
        calib_fraction=calib_fraction,
        seed=trial_seed,
    )
    calib_ps, calib_bs, calib_labels, calib_dv, _ = calib_arrays
    eval_ps, eval_bs, eval_labels, eval_dv, _ = eval_arrays
    assert calib_ps is not None and calib_bs is not None and calib_labels is not None and calib_dv is not None
    assert eval_ps is not None and eval_bs is not None and eval_labels is not None and eval_dv is not None

    # Subsample the calibration half to size n_cal
    n_pool = len(calib_dv)
    if n_cal < n_pool:
        sub_rng = np.random.default_rng(trial_seed)
        sub_idx = sub_rng.choice(n_pool, size=n_cal, replace=False)
        calib_ps = calib_ps[sub_idx]
        calib_bs = calib_bs[sub_idx]
        calib_labels = calib_labels[sub_idx]
        calib_dv = calib_dv[sub_idx]

    # Pareto-LTT inside the subsample (HT / OPT split + Pareto filter + FST)
    ht_idx, opt_idx = pareto_ht_opt_split(len(calib_dv), pareto_split_proportion, trial_seed)
    if len(ht_idx) < 1 or len(opt_idx) < 1:
        return TrialResult(seed=trial_seed, tau=None, realised_budget=None, cascade_auc=None, cascade_accuracy=None)

    pareto_ltt = build_pareto_ltt(
        ht_delegation_scores=calib_dv[ht_idx],
        opt_probe_scores=calib_ps[opt_idx],
        opt_baseline_scores=calib_bs[opt_idx],
        opt_labels=calib_labels[opt_idx],
        opt_delegation_scores=calib_dv[opt_idx],
        tau_grid=data.dv_tau_grid,
        opt_risk=opt_risk,
        budget_risk=BudgetCostRisk,
        merge_strategy=merge_strategy,
    )
    tau = pareto_ltt.select_threshold(alpha, delta)
    if tau is None:
        return TrialResult(seed=trial_seed, tau=None, realised_budget=None, cascade_auc=None, cascade_accuracy=None)

    # Evaluate on this trial's eval half
    cascade = threshold_cascade(eval_ps, eval_bs, eval_dv, tau, merge_strategy=merge_strategy)
    met = cascade_metrics(cascade, eval_labels)
    return TrialResult(
        seed=trial_seed,
        tau=float(tau),
        realised_budget=float(cascade.used_baseline.mean()),
        cascade_auc=met["auc"],
        cascade_accuracy=met["accuracy"],
    )


def run_calibration_size_sweep(config, output_dir: Path) -> dict:
    """Sweep |D_cal| with Pareto-LTT and fresh random splits per trial.

    Pipeline:
      1. Load DV cascade data once (probe + DV + baseline scores on the test split).
      2. Determine the maximum calibration-pool size from ``calib_fraction``.
      3. For each n_cal in ``config.cal_sizes``:
           For trial in 1..``config.n_trials``:
             - draw a fresh calib/eval split,
             - subsample the calib half to n_cal,
             - run Pareto-LTT (HT/OPT inside the subsample),
             - evaluate the cascade on this trial's eval half.
           Aggregate mean +/- std for tau*, realised budget, AUC, accuracy.
    """
    seed = config.seed
    np.random.seed(seed)

    # ---- Stage 1: Load data once ----
    logger.info("=== Stage 1: Load DV cascade data ===")
    data = prepare_dv_cascade_data(config, tau_steps=config.n_thresholds)
    n_test = len(data.dv_scores)

    # ---- Stage 2: Setup ----
    alpha = float(config.alpha_budget)
    delta = 1.0 - float(config.guarantee_probability)
    merge_strategy = config.merge_strategy
    calib_fraction = float(config.calib_fraction)
    pareto_split_proportion = float(config.pareto_split_proportion)
    opt_risk_name = config.opt_risk
    opt_risk = RISK_RGISTRY.get(opt_risk_name)
    if opt_risk is None:
        raise ValueError(f"Unknown opt_risk: '{opt_risk_name}'. Available: {list(RISK_RGISTRY.keys())}")

    n_pool_max = int(n_test * calib_fraction)  # max size of the calibration half
    cal_sizes_raw: list = list(config.cal_sizes)
    cal_sizes: list[int] = [n_pool_max if str(s).lower() == "full" else int(s) for s in cal_sizes_raw]
    cal_sizes = sorted({s for s in cal_sizes if 1 <= s <= n_pool_max})
    n_trials = int(config.n_trials)

    logger.info(
        f"n_test={n_test}, n_pool_max={n_pool_max} (calib_fraction={calib_fraction}), "
        f"alpha={alpha}, delta={delta:.3f}, opt_risk={opt_risk_name}, "
        f"cal_sizes={cal_sizes}, n_trials={n_trials}"
    )

    # Suppress per-trial LTT logging (very chatty otherwise)
    logging.getLogger("dv_ltt_cascade").setLevel(logging.WARNING)
    logging.getLogger("reliable_monitoring.learn_then_test").setLevel(logging.WARNING)

    # ---- Reference metrics on the full test set ----
    probe_auc = float(roc_auc_score(data.test_labels, data.test_ps))
    probe_acc = float(accuracy_score(data.test_labels, (data.test_ps >= 0.5).astype(int)))
    baseline_auc = float(roc_auc_score(data.test_labels, data.test_bs))
    baseline_acc = float(accuracy_score(data.test_labels, (data.test_bs >= 0.5).astype(int)))

    # ---- Stage 3: Sweep ----
    rng = np.random.default_rng(seed)
    size_results: list[SizeResult] = []
    for n_cal in cal_sizes:
        logger.info(f"=== n_cal={n_cal} ({n_trials} trials) ===")
        trials: list[TrialResult] = []
        for _trial in tqdm(range(n_trials), desc=f"n_cal={n_cal}", leave=False):
            trial_seed = int(rng.integers(0, 2**31 - 1))
            trials.append(
                _run_trial(
                    data=data,
                    alpha=alpha,
                    delta=delta,
                    calib_fraction=calib_fraction,
                    pareto_split_proportion=pareto_split_proportion,
                    opt_risk=opt_risk,
                    merge_strategy=merge_strategy,
                    n_cal=n_cal,
                    trial_seed=trial_seed,
                )
            )

        agg = _aggregate(n_cal, trials, alpha)
        if agg.mean_tau is None:
            logger.info(f"  n_cal={n_cal}: NO trials certified ({agg.n_trials} attempted)")
        else:
            logger.info(
                f"  n_cal={n_cal}: certified {agg.n_certified}/{agg.n_trials}; "
                f"tau* = {agg.mean_tau:.4f} +/- {agg.std_tau:.4f}; "
                f"realised budget = {agg.mean_realised_budget:.4f} +/- {agg.std_realised_budget:.4f} "
                f"(target <= {alpha}, violation rate {agg.violation_rate:.3f}); "
                f"AUC = {agg.mean_cascade_auc:.4f} +/- {agg.std_cascade_auc:.4f}; "
                f"acc = {agg.mean_cascade_accuracy:.4f} +/- {agg.std_cascade_accuracy:.4f}"
            )
        size_results.append(agg)

    # ---- Stage 4: Persist + plot ----
    summary = {
        "config": vars(config) if hasattr(config, "__dict__") else dict(config),
        "alpha_budget": alpha,
        "guarantee_probability": float(config.guarantee_probability),
        "delta": delta,
        "calib_fraction": calib_fraction,
        "pareto_split_proportion": pareto_split_proportion,
        "opt_risk": opt_risk_name,
        "n_test": n_test,
        "n_pool_max": n_pool_max,
        "n_trials": n_trials,
        "cal_sizes": cal_sizes,
        "probe_only_auc": probe_auc,
        "probe_only_accuracy": probe_acc,
        "baseline_only_auc": baseline_auc,
        "baseline_only_accuracy": baseline_acc,
        "dv_probe_auc": data.dv_auc,
        "results": [
            {
                "n_cal": r.n_cal,
                "n_trials": r.n_trials,
                "n_certified": r.n_certified,
                "mean_tau": r.mean_tau,
                "std_tau": r.std_tau,
                "mean_realised_budget": r.mean_realised_budget,
                "std_realised_budget": r.std_realised_budget,
                "violation_rate": r.violation_rate,
                "mean_cascade_auc": r.mean_cascade_auc,
                "std_cascade_auc": r.std_cascade_auc,
                "mean_cascade_accuracy": r.mean_cascade_accuracy,
                "std_cascade_accuracy": r.std_cascade_accuracy,
                "trials": [
                    {
                        "seed": t.seed,
                        "tau": t.tau,
                        "realised_budget": t.realised_budget,
                        "cascade_auc": t.cascade_auc,
                        "cascade_accuracy": t.cascade_accuracy,
                    }
                    for t in r.trials
                ],
            }
            for r in size_results
        ],
    }
    out_json = output_dir / "results.json"
    out_json.write_text(json.dumps(summary, indent=2, default=float))
    logger.info(f"Results JSON saved to {out_json}")

    _plot_summary(size_results, alpha=alpha, probe_acc=probe_acc, baseline_acc=baseline_acc, output_dir=output_dir)

    return summary


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_summary(
    size_results: list[SizeResult],
    alpha: float,
    probe_acc: float,
    baseline_acc: float,
    output_dir: Path,
) -> None:
    """Plot realised budget and cascade accuracy vs calibration-set size."""
    sizes = np.array([r.n_cal for r in size_results])
    rb_mean = np.array([r.mean_realised_budget if r.mean_realised_budget is not None else np.nan for r in size_results])
    rb_std = np.array([r.std_realised_budget if r.std_realised_budget is not None else 0.0 for r in size_results])
    acc_mean = np.array(
        [r.mean_cascade_accuracy if r.mean_cascade_accuracy is not None else np.nan for r in size_results]
    )
    acc_std = np.array([r.std_cascade_accuracy if r.std_cascade_accuracy is not None else 0.0 for r in size_results])
    auc_mean = np.array([r.mean_cascade_auc if r.mean_cascade_auc is not None else np.nan for r in size_results])
    auc_std = np.array([r.std_cascade_auc if r.std_cascade_auc is not None else 0.0 for r in size_results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].errorbar(sizes, rb_mean, yerr=rb_std, marker="o", capsize=4, label="Realised budget (mean +/- std)")
    axes[0].axhline(alpha, ls="--", color="red", alpha=0.6, label=f"Target alpha = {alpha}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Calibration set size  |D_cal|")
    axes[0].set_ylabel("Realised delegation rate")
    axes[0].set_title("LTT becomes more conservative as |D_cal| shrinks")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(sizes, acc_mean, yerr=acc_std, marker="o", capsize=4, label="Cascade accuracy")
    axes[1].errorbar(sizes, auc_mean, yerr=auc_std, marker="s", capsize=4, label="Cascade AUC")
    axes[1].axhline(probe_acc, ls="--", color="gray", alpha=0.6, label=f"Probe-only acc = {probe_acc:.3f}")
    axes[1].axhline(baseline_acc, ls="--", color="green", alpha=0.6, label=f"Baseline-only acc = {baseline_acc:.3f}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Calibration set size  |D_cal|")
    axes[1].set_ylabel("Cascade performance (mean +/- std across trials)")
    axes[1].set_title("Downstream performance vs |D_cal|")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = output_dir / "calibration_size_sweep.pdf"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Plot saved to {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    default_config = Path(__file__).parent / "configs" / "ltt_cal_size_sweep" / "strong_expert.yaml"
    parser = argparse.ArgumentParser(description="LTT calibration-size sweep")
    parser.add_argument("--config", type=str, default=str(default_config))
    default_output = os.path.join(os.environ.get("RESULTS_DIR", "results"), "ltt_calibration_size_sweep")
    parser.add_argument("--output-dir", type=str, default=default_output)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / Path(args.config).name)

    config = load_config(args.config)
    run_calibration_size_sweep(config, output_dir)
    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
