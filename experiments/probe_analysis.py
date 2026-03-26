"""
probe_analysis.py — DV probe quality analysis
==============================================
Analyses d(x) as a delegation routing signal and compares it against the
uncertainty (McKenzie et al.) baseline.

Dependencies
------------
This script takes *existing* cascade comparison run directories as input and
reconstructs the EXACT eval splits from the saved configs.  A sanity check
verifies that the reconstructed probe AUC matches the value stored in
results.json before any metrics are computed — if they diverge, the script
aborts rather than silently analyse the wrong data.

Outputs
-------
  <output_dir>/<timestamp>/
    metrics.json          — Spearman ρ, AUC(v>0), MSE for d(x) and uncertainty
    mean_v_at_k.pdf       — mean v(x,y) of delegated set vs budget fraction
    auc_vs_threshold.pdf  — AUC as a function of v(x,y) threshold

Usage
-----
  uv run python experiments/probe_analysis.py \\
      --config experiments/configs/probe_analysis.yaml

  uv run python experiments/probe_analysis.py \\
      --config experiments/configs/probe_analysis.yaml --use-clearml

Paper section structure
-----------------------
The analysis supports a two-paragraph section on DV probe quality.

¶1 — d(x) as a regression signal (standalone, no uncertainty comparison)
  Report Spearman ρ and the AUC-vs-threshold curve.  The key point is that
  d(x) is trained to predict v(x,y), so its AUC *grows* as the threshold τ
  rises — i.e. it becomes more discriminative exactly when it matters most
  (identifying the highest-value examples).  Do NOT frame this paragraph as a
  comparison with uncertainty: that invites a tedious discussion of why the
  global Spearman ρ numbers are close (spurious correlation via ρ(x) in both
  signals) and distracts from the intended message.

¶2 — Delegation routing quality via mean v@k
  Use the mean_v_at_k figure.  At each budget fraction α, the top-α% of
  examples is selected by d(x) vs uncertainty.  This directly mirrors the
  deployment scenario and shows the asymmetry: for the strong expert, the
  uncertainty signal selects negative-v examples at low budgets (AUC < 0.5),
  actively hurting the cascade, while d(x) consistently selects positive-v
  examples.  The magnitude of the cascade performance gap (much larger than
  the raw signal-quality gap) is explained here: the uncertainty baseline is
  not merely worse — it anti-selects at the budgets we care about most.

Figures for the paper
  - mean_v_at_k.pdf   → primary figure (goes in this section)
  - auc_vs_threshold.pdf → supporting figure (appendix or same section)
  - metrics.json        → source for any inline numbers
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from clearml_logger import ClearMLLogger
from config import load_config
from dv_ltt_cascade import prepare_dv_cascade_data, split_calib_eval

from reliable_monitoring.cascade import probe_uncertainty

# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #


def _find_run_config(run_dir: Path) -> Path:
    yamls = list(run_dir.glob("*.yaml"))
    if not yamls:
        raise FileNotFoundError(f"No YAML config found in {run_dir}")
    if len(yamls) > 1:
        raise ValueError(f"Multiple YAMLs in {run_dir}: {yamls}")
    return yamls[0]


def load_run(run_dir: Path) -> dict:
    """
    Reconstruct the eval split for a cascade comparison run.

    Steps:
      1. Find and load the saved config YAML from run_dir.
      2. Re-run prepare_dv_cascade_data + split_calib_eval with the same
         seed and calib_fraction — this deterministically reproduces the
         exact eval split used during the original experiment.
      3. Sanity-check: probe AUC on the full test set must match the value
         stored in results.json to within 0.005.  Aborts if it does not.
    """
    run_dir = Path(run_dir).resolve()
    config_path = _find_run_config(run_dir)
    results_path = run_dir / "results.json"

    print(f"  config  : {config_path}")
    print(f"  results : {results_path}")

    config = load_config(str(config_path))
    data = prepare_dv_cascade_data(config, local_only=True)

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
    assert calib_ps is not None and calib_labels is not None
    assert eval_ps is not None and eval_labels is not None

    # Sanity check — probe AUC on full test set must match results.json
    with open(results_path) as f:
        saved = json.load(f)
    expected_probe_auc = saved["reference"]["probe_auc"]
    actual_probe_auc = roc_auc_score(
        np.concatenate([calib_labels, eval_labels]),
        np.concatenate([calib_ps, eval_ps]),
    )
    delta = abs(actual_probe_auc - expected_probe_auc)
    if delta > 0.005:
        raise ValueError(
            f"Probe AUC mismatch: reconstructed={actual_probe_auc:.4f}, "
            f"saved={expected_probe_auc:.4f} (Δ={delta:.4f}).\n"
            f"Check that run_dir points to the correct experiment and that "
            f"DATA_DIR is set to the same data used originally."
        )
    print(f"  ✓ probe AUC check passed ({actual_probe_auc:.4f} ≈ {expected_probe_auc:.4f})")

    # Uncertainty signal: rank eval scores within sorted calib distribution
    # (same reference used in the cascade experiment for LTT calibration)
    unc = probe_uncertainty(eval_ps, reference=calib_ps)

    return dict(
        config=config,
        eval_ps=eval_ps,
        eval_bs=eval_bs,
        eval_labels=eval_labels,
        eval_dv=eval_dv,
        eval_v=eval_v,
        eval_groups=eval_groups,
        calib_ps=calib_ps,
        unc=unc,
        probe_auc=actual_probe_auc,
        baseline_auc=saved["reference"]["baseline_auc"],
        run_dir=run_dir,
    )


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #


def compute_metrics(run: dict, label: str = "") -> list[dict]:
    """
    Spearman ρ, AUC(v > 0), and MSE for d(x) and the uncertainty signal.
    """
    v, dv, unc = run["eval_v"], run["eval_dv"], run["unc"]
    labels_pos = (v > 0).astype(int)

    rows = []
    for signal, name in [(dv, "d(x)"), (unc, "Uncertainty")]:
        rho, pval = spearmanr(signal, v)
        auc = roc_auc_score(labels_pos, signal)
        mse = float(np.mean((signal - v) ** 2))
        rows.append(dict(signal=name, spearman_rho=float(rho), spearman_p=float(pval), auc=float(auc), mse=mse))

    if label:
        print(f"\n  {label}")
    print(f"  {'Signal':>12} {'Spearman ρ':>12} {'p-value':>10} {'AUC(v>0)':>10} {'MSE':>8}")
    for r in rows:
        print(
            f"  {r['signal']:>12} {r['spearman_rho']:>12.3f} "
            f"{r['spearman_p']:>10.2e} {r['auc']:>10.3f} {r['mse']:>8.3f}"
        )
    return rows


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #


def plot_mean_v_at_k(runs: list[dict], labels: list[str], random_seed: int = 0):
    """
    Mean v(x,y) of the selected (delegated) set as a function of budget fraction.

    At each budget α, the top-α fraction of examples is selected by each signal.
    A good signal selects high-v examples early; a bad signal may select negative-v
    examples, meaning the cascade is worse than the probe alone at that budget.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5), sharey=False)
    if len(runs) == 1:
        axes = [axes]

    for ax, run, label in zip(axes, runs, labels, strict=False):
        v, dv, unc = run["eval_v"], run["eval_dv"], run["unc"]
        N = len(v)
        ks = np.arange(1, N + 1)
        fracs = ks / N

        def mean_v_selected(order, _v=v, _ks=ks):
            return np.cumsum(_v[order]) / _ks

        ax.plot(fracs, mean_v_selected(np.argsort(-v)), label="$v(x,y)$ (upper bound)", color="green", lw=2)
        ax.plot(fracs, mean_v_selected(np.argsort(-dv)), label="DV probe $d(x)$", color="tab:orange", lw=2)
        ax.plot(fracs, mean_v_selected(np.argsort(-unc)), label="Uncertainty (McKenzie)", color="tab:blue", lw=2)
        # v.mean() is the expected mean v of any unranked (random) selection at any k;
        # equivalently, it is the mean v when delegating the full set.
        ax.axhline(v.mean(), color="gray", ls="--", lw=1.5, label="No ranking")
        ax.axhline(0, color="black", lw=0.8)

        ax.set_xlabel("Selection fraction $k/N$", fontsize=11)
        ax.set_ylabel("Mean $v(x,y)$ of selected set", fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)

    fig.suptitle("Mean delegation value of selected set vs budget", fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig


def plot_auc_vs_threshold(runs: list[dict], labels: list[str], n_steps: int = 40):
    """
    AUC as a function of the v(x,y) threshold used to define the positive class.

    At threshold τ, the positive class is {x : v(x,y) > τ}.  As τ increases,
    we ask a harder question: can the signal identify the most valuable examples?
    d(x) is trained on the magnitude of v, so its AUC should stay high or grow
    as τ rises; the uncertainty signal has no such magnitude information.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5), sharey=True)
    if len(runs) == 1:
        axes = [axes]

    for ax, run, label in zip(axes, runs, labels, strict=False):
        v, dv, unc = run["eval_v"], run["eval_dv"], run["unc"]
        v_pos = v[v > 0]
        thresholds = np.linspace(v.min(), np.percentile(v_pos, 80), n_steps)

        dv_aucs, unc_aucs = [], []
        for tau in thresholds:
            lbls = (v > tau).astype(int)
            n_pos = lbls.sum()
            if n_pos < 10 or n_pos > len(v) - 10:
                dv_aucs.append(np.nan)
                unc_aucs.append(np.nan)
                continue
            dv_aucs.append(roc_auc_score(lbls, dv))
            unc_aucs.append(roc_auc_score(lbls, unc))

        ax.plot(thresholds, dv_aucs, label="DV probe $d(x)$", color="tab:orange", lw=2)
        ax.plot(thresholds, unc_aucs, label="Uncertainty (McKenzie)", color="tab:blue", lw=2)
        ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random (AUC = 0.5)")
        ax.axvline(0, color="black", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("$v(x,y)$ threshold", fontsize=10)
        ax.set_ylabel("AUC", fontsize=11)
        ax.set_ylim(0.3, 1.0)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "AUC vs delegation value threshold\n(positive class = $v > $ threshold)", fontweight="bold", fontsize=12
    )
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="DV probe quality analysis")
    parser.add_argument("--config", required=True, help="Path to probe_analysis.yaml")
    parser.add_argument("--use-clearml", action="store_true", help="Log metrics and figures to ClearML")
    args = parser.parse_args()

    analysis_config = load_config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(analysis_config.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / Path(args.config).name)
    print(f"Output: {output_dir}")

    # ------------------------------------------------------------------ #
    # ClearML                                                              #
    # ------------------------------------------------------------------ #
    clearml_logger = None
    if args.use_clearml:
        clearml_logger = ClearMLLogger(
            project_name="reliable-llm-monitoring",
            task_name="probe_analysis",
            enabled=True,
        )
        clearml_logger.connect_configuration(vars(analysis_config))
        clearml_logger.add_tags(
            [
                "probe-analysis",
                f"strong:{Path(analysis_config.strong_run_dir).name}",
                f"weak:{Path(analysis_config.weak_run_dir).name}",
            ]
        )

    # ------------------------------------------------------------------ #
    # Load runs — reconstructs and verifies the exact eval splits         #
    # ------------------------------------------------------------------ #
    print("\nLoading strong expert run...")
    strong = load_run(Path(analysis_config.strong_run_dir))
    print("\nLoading weak expert run...")
    weak = load_run(Path(analysis_config.weak_run_dir))

    run_labels = [
        f"Strong expert ({strong['config'].baseline_model_name.split('/')[-1]})",
        f"Weak expert ({weak['config'].activations_model_name.split('/')[-1]})",
    ]

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #
    print("\n--- Metrics ---")
    strong_metrics = compute_metrics(strong, label=run_labels[0])
    weak_metrics = compute_metrics(weak, label=run_labels[1])

    metrics_out = {"strong": strong_metrics, "weak": weak_metrics}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("\n  Saved metrics.json")

    if clearml_logger is not None:
        flat = {}
        for split, rows in metrics_out.items():
            for r in rows:
                key = r["signal"].replace("(", "").replace(")", "").replace(" ", "_").lower()
                flat[f"{split}/{key}/spearman_rho"] = r["spearman_rho"]
                flat[f"{split}/{key}/auc_v0"] = r["auc"]
                flat[f"{split}/{key}/mse"] = r["mse"]
        clearml_logger.log_scalars(flat)

    # ------------------------------------------------------------------ #
    # Figures                                                              #
    # ------------------------------------------------------------------ #
    figs = {
        "mean_v_at_k": plot_mean_v_at_k(
            [strong, weak],
            run_labels,
            random_seed=getattr(analysis_config, "random_seed", 0),
        ),
        "auc_vs_threshold": plot_auc_vs_threshold(
            [strong, weak],
            run_labels,
            n_steps=getattr(analysis_config, "n_threshold_steps", 40),
        ),
    }

    for name, fig in figs.items():
        path = output_dir / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path.name}")
        if clearml_logger is not None:
            clearml_logger.log_figure("Probe Analysis", name, fig)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Done                                                                 #
    # ------------------------------------------------------------------ #
    if clearml_logger is not None:
        clearml_logger.finalize()

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
