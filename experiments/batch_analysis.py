"""
batch_analysis.py — Per-batch comparison of DV cascade vs uncertainty top-k
============================================================================
Compares two delegation strategies at the per-batch level:

  DV LTT (threshold cascade)
    A global threshold τ is applied to d(x).  The number of examples delegated
    per batch is *variable*: a batch where all d(x) < τ gets 0 delegations.

  Uncertainty top-k (McKenzie et al.)
    Within each batch of size B, the k = round(α * B) examples closest to the
    within-batch median probe score are delegated.  Delegation rate is *fixed*
    at exactly k/B per batch regardless of signal quality.

The central question: does the DV cascade learn to withhold delegation on
"hard" batches (mean v(x,y) < 0, where the expert is globally unhelpful) while
the top-k method blindly delegates k examples regardless?

Batch reconstruction
--------------------
Both methods operate on the same eval split produced by split_calib_eval (seed
fixed in the run config).  offline_batch_cascade slices the eval array
sequentially: [0:B], [B:2B], etc.  No additional shuffling — batches are
deterministic and identical across methods.

Note: eval examples are in the order perm[n_calib:] from split_calib_eval's
random permutation.  Batches are therefore random mixtures of the 4 source
datasets; "hard" and "easy" batches arise from sampling variance in v(x,y),
not from dataset-level effects.

Outputs
-------
  <output_dir>/<timestamp>/
    metrics.json           — paired t-test results, per-batch summary stats
    delegation_rate.pdf    — DV delegation rate vs top-k (fixed k/B) per batch
    accuracy_scatter.pdf   — per-batch accuracy: DV vs top-k, coloured by mean v
    adaptivity.pdf         — delegation rate vs mean batch v(x,y): shows DV
                             adapts to signal quality, top-k does not

Usage
-----
  uv run python experiments/batch_analysis.py \\
      --config experiments/configs/batch_analysis.yaml

  uv run python experiments/batch_analysis.py \\
      --config experiments/configs/batch_analysis.yaml --use-clearml
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from clearml_logger import ClearMLLogger
from config import load_config
from dv_ltt_cascade import threshold_cascade
from probe_analysis import load_run  # reuse the same loading + sanity check

from reliable_monitoring.cascade import offline_batch_cascade

# ---------------------------------------------------------------------------
# Data assumptions (read before modifying)
# ---------------------------------------------------------------------------
#
# Eval array ordering
#   split_calib_eval(seed=42) produces eval examples in the order perm[n_calib:]
#   where perm = np.random.default_rng(seed).permutation(n).  This is a fixed
#   but arbitrary permutation of the full test set — NOT the original dataset
#   order, NOT grouped by source dataset.  Batches are therefore random mixtures
#   of the 4 source datasets (anthropic, mt, mts, toolace).  "Hard" and "easy"
#   batches arise from sampling variance in v(x,y), not from dataset structure.
#
# Batch identity between methods
#   offline_batch_cascade slices the eval array sequentially: [0:B], [B:2B], ...
#   Both methods receive the same eval arrays in the same order, so the batch
#   assignments are identical.  make_batches() below replicates this slicing.
#
# Calibrated τ
#   The DV threshold τ is read from results.json["ltt"]["DV calibrated threshold"],
#   which stores the PAC-guaranteed threshold found during the original LTT
#   calibration on the calib split.  Using it here on the eval split gives the
#   same deployment semantics as the original experiment.  The realized budget
#   (fraction actually delegated) will be ≤ α with probability ≥ 1−δ by the
#   LTT guarantee — it may be substantially less than α at low budgets.
#
# Comparison asymmetry
#   DV LTT: realized budget ≤ α (PAC guarantee, can be much less)
#   Uncertainty top-k: exactly k/B = round(α*B)/B per batch (hard budget)
#   These are not at exactly the same delegation rate.  The asymmetry is the
#   point: the DV guarantee is on risk, not on budget consumed.  To compare
#   fairly at matched budgets, restrict to entries in results.json where
#   realized_budget ≈ k/B, or compare at a fixed α and acknowledge the
#   difference in realized rates.
#
# Per-batch accuracy noise
#   With B=32 examples, batch accuracy is discrete in steps of 1/32 ≈ 3.1%.
#   The paired t-test is valid (paired differences ≈ normal by CLT across
#   ~30 batches) but effect sizes should be interpreted carefully — a 1-example
#   improvement corresponds to 1/32 ≈ 3.1% accuracy.

# --------------------------------------------------------------------------- #
# Batch partitioning                                                           #
# --------------------------------------------------------------------------- #


def make_batches(n: int, batch_size: int) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for contiguous batches.

    Matches the partitioning used by offline_batch_cascade: each batch is a
    sequential slice; the final partial batch is included if n % batch_size != 0.
    """
    return [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]


# --------------------------------------------------------------------------- #
# Finding the calibrated τ for a given α                                      #
# --------------------------------------------------------------------------- #


def get_tau_for_alpha(results_path: Path, target_alpha: float, signal: str = "DV") -> tuple[float, float]:
    """Read the calibrated threshold closest to target_alpha from results.json.

    Args:
        results_path: Path to results.json from a cascade comparison run.
        target_alpha: Desired budget fraction α.
        signal: "DV" or "Uncertainty" — which LTT entry to use.

    Returns:
        (tau, realized_budget) for the closest stored α.
    """
    key = "DV calibrated threshold" if signal == "DV" else "Uncertainty calibrated threshold"
    with open(results_path) as f:
        saved = json.load(f)
    entries = saved["ltt"][key]
    closest = min(entries, key=lambda e: abs(e["alpha"] - target_alpha))
    return closest["tau"], closest["realized_budget"]


# --------------------------------------------------------------------------- #
# Per-batch cascade results                                                    #
# --------------------------------------------------------------------------- #


def run_batch_comparison(
    run: dict,
    batch_size: int,
    target_alpha: float,
) -> tuple[list[dict], dict]:
    """Compute per-batch results for both methods at a given α.

    For each batch:
      - DV LTT: apply global threshold τ (from results.json, closest to α)
      - Uncertainty top-k: delegate k = round(α * |batch|) examples by McKenzie

    Returns a list of per-batch dicts with keys:
        batch_idx, n, start, end, mean_v, groups,
        dv_delegation_rate, unc_delegation_rate,
        dv_accuracy, unc_accuracy, probe_accuracy,
        dv_delegated, unc_delegated.
    """
    results_path = run["run_dir"] / "results.json"
    tau, realized_budget = get_tau_for_alpha(results_path, target_alpha, signal="DV")

    eval_ps = run["eval_ps"]
    eval_bs = run["eval_bs"]
    eval_labels = run["eval_labels"]
    eval_dv = run["eval_dv"]
    eval_v = run["eval_v"]
    eval_groups = run["eval_groups"]
    merge_strategy = run["config"].merge_strategy

    # --- DV threshold cascade (global, no batch structure) ---
    # threshold_cascade applies τ globally: delegate example i iff d(x_i) > τ.
    # There is no per-batch constraint — some batches may get 0 delegations,
    # others may get many.  τ is read from results.json (LTT-calibrated on the
    # calib split) to replicate the original experiment's deployment decision.
    dv_result = threshold_cascade(eval_ps, eval_bs, eval_dv, tau, merge_strategy=merge_strategy)

    # --- Uncertainty top-k cascade (per batch, McKenzie: k nearest to median) ---
    # ranking_scores=None → select_fixed_budget_amount selects the k examples
    # with probe scores closest to the within-batch median (McKenzie et al.
    # criterion: "most uncertain = nearest decision boundary").  This is the
    # exact strategy used in dv_cascade_comparison.py for "Uncertainty top-k".
    # Note: this is NOT the same as probe_uncertainty() used in LTT calibration;
    # the top-k baseline does not need a reference calibration distribution.
    #
    # k = round(α * B) so that both methods target the same nominal budget.
    # amount is a scalar (not per-batch array) — offline_batch_cascade applies
    # the same k to every batch, including the final partial batch.  This is
    # identical to the original experiment's behaviour.
    k = max(1, round(target_alpha * batch_size))
    unc_result = offline_batch_cascade(
        eval_ps,
        eval_bs,
        batch_size=batch_size,
        selection_strategy="fixed_budget_amount",
        merge_strategy=merge_strategy,
        amount=k,
        ranking_scores=None,  # within-batch median proximity (McKenzie)
    )

    # --- Per-batch breakdown ---
    n = len(eval_ps)
    batches = make_batches(n, batch_size)
    rows = []
    for i, (s, e) in enumerate(batches):
        lbl = eval_labels[s:e]
        v = eval_v[s:e]
        dv_used = dv_result.used_baseline[s:e]
        unc_used = unc_result.used_baseline[s:e]
        dv_scores_batch = dv_result.final_scores[s:e]
        unc_scores_batch = unc_result.final_scores[s:e]
        probe_scores_batch = eval_ps[s:e]

        rows.append(
            dict(
                batch_idx=i,
                n=e - s,
                start=s,
                end=e,
                mean_v=float(v.mean()),
                groups=list(eval_groups[s:e]) if eval_groups is not None else None,
                dv_delegation_rate=float(dv_used.mean()),
                unc_delegation_rate=float(unc_used.mean()),
                dv_delegated=int(dv_used.sum()),
                unc_delegated=int(unc_used.sum()),
                dv_accuracy=float(accuracy_score(lbl, (dv_scores_batch >= 0.5).astype(int))),
                unc_accuracy=float(accuracy_score(lbl, (unc_scores_batch >= 0.5).astype(int))),
                probe_accuracy=float(accuracy_score(lbl, (probe_scores_batch >= 0.5).astype(int))),
            )
        )

    return rows, dict(tau=tau, realized_budget=realized_budget, k=k, target_alpha=target_alpha)


# --------------------------------------------------------------------------- #
# Paired t-test summary                                                        #
# --------------------------------------------------------------------------- #


def paired_ttest(batch_rows: list[dict]) -> dict:
    """Paired t-test: DV accuracy vs uncertainty top-k accuracy across batches.

    The pairing is by batch index — each batch is one observation.  The null
    hypothesis is that the mean per-batch accuracy difference (DV − top-k) is
    zero.  A one-sided alternative (DV > top-k) is what we expect; the returned
    p_val is two-sided so divide by 2 for a one-sided interpretation.

    Caveat: batch accuracy is discrete (multiples of 1/B), so the normality
    assumption of the t-test is approximate.  With ~15 batches (1000 eval
    examples / B=64), the CLT provides adequate approximation for the mean
    difference.  A Wilcoxon signed-rank test is a non-parametric alternative
    if the normality assumption is a concern.
    """
    dv_accs = np.array([r["dv_accuracy"] for r in batch_rows])
    unc_accs = np.array([r["unc_accuracy"] for r in batch_rows])
    diff = dv_accs - unc_accs
    t_stat, p_val = stats.ttest_rel(dv_accs, unc_accs)
    return dict(
        mean_dv_acc=float(dv_accs.mean()),
        mean_unc_acc=float(unc_accs.mean()),
        mean_diff=float(diff.mean()),
        std_diff=float(diff.std()),
        t_stat=float(t_stat),
        p_val=float(p_val),
        n_batches=len(batch_rows),
        n_dv_wins=int((diff > 0).sum()),
        n_unc_wins=int((diff < 0).sum()),
        n_ties=int((diff == 0).sum()),
    )


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #


def plot_delegation_rate(batch_rows: list[dict], cascade_meta: dict, label: str) -> plt.Figure:
    """Histogram of per-batch delegation rates: DV (variable) vs top-k (fixed).

    The top-k method always delegates exactly k/B per batch; the DV cascade
    delegates 0 on easy-to-judge batches and more on harder ones.
    This figure makes the contrast in strategy design immediately visible.
    """
    dv_rates = [r["dv_delegation_rate"] for r in batch_rows]
    fixed_rate = cascade_meta["k"] / batch_rows[0]["n"]  # k / B (constant)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = list(np.linspace(0, 1, 25))
    ax.hist(dv_rates, bins=bins, alpha=0.7, color="tab:orange", label="DV LTT (variable)")
    ax.axvline(fixed_rate, color="tab:blue", lw=2, ls="--", label=f"Uncertainty top-k (fixed = {fixed_rate:.0%})")
    ax.set_xlabel("Per-batch delegation rate", fontsize=11)
    ax.set_ylabel("Number of batches", fontsize=11)
    ax.set_title(f"{label}\nPer-batch delegation rate  (α = {cascade_meta['target_alpha']:.2f})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_accuracy_scatter(batch_rows: list[dict], ttest: dict, label: str) -> plt.Figure:
    """Per-batch accuracy scatter: DV (y) vs uncertainty top-k (x).

    Points above the diagonal = DV wins that batch.  Coloured by mean v(x,y)
    so that the pattern is visible: DV tends to win on low-v batches (where
    it wisely delegates less) and is comparable on high-v batches.
    """
    dv_accs = np.array([r["dv_accuracy"] for r in batch_rows])
    unc_accs = np.array([r["unc_accuracy"] for r in batch_rows])
    mean_vs = np.array([r["mean_v"] for r in batch_rows])

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(
        unc_accs,
        dv_accs,
        c=mean_vs,
        cmap="RdYlGn",
        alpha=0.6,
        s=25,
        vmin=np.percentile(mean_vs, 5),
        vmax=np.percentile(mean_vs, 95),
    )
    plt.colorbar(sc, ax=ax, label="Mean $v(x,y)$ of batch")

    lo = min(unc_accs.min(), dv_accs.min()) - 0.02
    hi = max(unc_accs.max(), dv_accs.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)

    ax.set_xlabel("Uncertainty top-k accuracy", fontsize=11)
    ax.set_ylabel("DV LTT accuracy", fontsize=11)
    ax.set_title(
        f"{label}\nDV wins {ttest['n_dv_wins']}/{ttest['n_batches']} batches  (p = {ttest['p_val']:.3f})",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_adaptivity(batch_rows: list[dict], cascade_meta: dict, label: str) -> plt.Figure:
    """Delegation rate vs mean batch v(x,y).

    DV cascade: scatter — should slope upward (delegate more on high-v batches).
    Uncertainty top-k: horizontal line at k/B — blind to batch quality.

    This is the key figure for the paper story: DV discovers which batches
    are worth delegating; top-k cannot.
    """
    mean_vs = np.array([r["mean_v"] for r in batch_rows])
    dv_rates = np.array([r["dv_delegation_rate"] for r in batch_rows])
    fixed_rate = cascade_meta["k"] / batch_rows[0]["n"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean_vs, dv_rates, color="tab:orange", alpha=0.5, s=20, label="DV LTT (per batch)")
    ax.axhline(fixed_rate, color="tab:blue", lw=2, ls="--", label=f"Uncertainty top-k (fixed = {fixed_rate:.0%})")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Mean $v(x,y)$ of batch", fontsize=11)
    ax.set_ylabel("Delegation rate", fontsize=11)
    ax.set_title(f"{label}\nDelegation rate vs batch delegation value", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Per-batch DV cascade vs uncertainty top-k")
    parser.add_argument("--config", required=True, help="Path to batch_analysis.yaml")
    parser.add_argument("--use-clearml", action="store_true")
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
            task_name="batch_analysis",
            enabled=True,
        )
        clearml_logger.connect_configuration(vars(analysis_config))
        clearml_logger.add_tags(
            [
                "batch-analysis",
                f"strong:{Path(analysis_config.strong_run_dir).name}",
                f"weak:{Path(analysis_config.weak_run_dir).name}",
                f"B:{analysis_config.batch_size}",
                f"alpha:{analysis_config.target_alpha}",
            ]
        )

    # ------------------------------------------------------------------ #
    # Load runs                                                            #
    # ------------------------------------------------------------------ #
    print("\nLoading strong expert run...")
    strong = load_run(Path(analysis_config.strong_run_dir))
    print("\nLoading weak expert run...")
    weak = load_run(Path(analysis_config.weak_run_dir))

    run_labels = [
        f"Strong expert ({strong['config'].baseline_model_name.split('/')[-1]})",
        f"Weak expert ({weak['config'].activations_model_name.split('/')[-1]})",
    ]

    batch_size = analysis_config.batch_size
    target_alpha = analysis_config.target_alpha

    # ------------------------------------------------------------------ #
    # Per-batch comparison                                                 #
    # ------------------------------------------------------------------ #
    print(f"\n--- Batch analysis (B={batch_size}, α={target_alpha}) ---")
    all_results = {}
    all_ttests = {}
    for run, label, key in [(strong, run_labels[0], "strong"), (weak, run_labels[1], "weak")]:
        batch_rows, meta = run_batch_comparison(run, batch_size, target_alpha)
        ttest = paired_ttest(batch_rows)
        all_results[key] = {"batches": batch_rows, "meta": meta}
        all_ttests[key] = ttest
        print(f"\n  {label}")
        print(
            f"    τ = {meta['tau']:.4f}  (α = {meta['target_alpha']:.2f}, "
            f"realized = {meta['realized_budget']:.1%},  k = {meta['k']})"
        )
        print(f"    DV mean acc:  {ttest['mean_dv_acc']:.4f}")
        print(f"    Unc mean acc: {ttest['mean_unc_acc']:.4f}")
        print(f"    Diff (DV−Unc): {ttest['mean_diff']:+.4f}  (p = {ttest['p_val']:.4f})")
        print(f"    DV wins {ttest['n_dv_wins']}/{ttest['n_batches']} batches")

    metrics_out = {
        "strong": all_ttests["strong"],
        "weak": all_ttests["weak"],
        "meta": {"batch_size": batch_size, "target_alpha": target_alpha},
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("\n  Saved metrics.json")

    if clearml_logger is not None:
        flat = {}
        for split, t in all_ttests.items():
            for metric in ["mean_dv_acc", "mean_unc_acc", "mean_diff", "p_val"]:
                flat[f"{split}/{metric}"] = t[metric]
        clearml_logger.log_scalars(flat)

    # ------------------------------------------------------------------ #
    # Figures                                                              #
    # ------------------------------------------------------------------ #
    for key, _run, label in [("strong", strong, run_labels[0]), ("weak", weak, run_labels[1])]:
        batch_rows = all_results[key]["batches"]
        meta = all_results[key]["meta"]
        ttest = all_ttests[key]

        figs = {
            f"delegation_rate_{key}": plot_delegation_rate(batch_rows, meta, label),
            f"accuracy_scatter_{key}": plot_accuracy_scatter(batch_rows, ttest, label),
            f"adaptivity_{key}": plot_adaptivity(batch_rows, meta, label),
        }
        for name, fig in figs.items():
            path = output_dir / f"{name}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"  Saved {path.name}")
            if clearml_logger is not None:
                clearml_logger.log_figure("Batch Analysis", name, fig)
            plt.close(fig)

    if clearml_logger is not None:
        clearml_logger.finalize()

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
