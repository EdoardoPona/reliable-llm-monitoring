"""
group_analysis.py — Per-group analysis of the DV cascade
=========================================================
Investigates whether the DV probe, trained on the mixed dataset pool,
implicitly learns per-group delegation patterns — concentrating budget toward
datasets where the expert is genuinely beneficial and avoiding those where
delegation would hurt.

Paper section outline (§5.4 — Analysis across groups)
------------------------------------------------------
We evaluate over a test set mixing four datasets (Anthropic HH, MTSamples,
MTS-Dialog, ToolACE). The DV probe is trained on this mixture with no access
to group labels. We ask: does it nonetheless discover which datasets are worth
delegating to?

¶1 — Setup
  Four datasets, globally mixed eval set.  DV probe trained without group
  labels.  Per-group mean v(x,y) establishes ground truth for which datasets
  the expert is locally useful.

¶2 — Strong expert (Figure 6a)
  All four groups have positive mean v, but not equally so.  The DV cascade
  over-allocates to the highest-v datasets and under-allocates to the lowest,
  recovering the group-level ordering of delegation benefit without ever
  seeing group labels.  The scatter (Figure 5) makes the correlation between
  mean v and mean d(x) explicit.

¶3 — Weak expert (Figure 6b) — the key result
  Globally, the expert is worse than the probe (mean v < 0), yet there are
  individual datasets where delegation is still beneficial.  The DV cascade
  discovers this implicitly: it concentrates its budget on the few groups with
  positive mean v while largely avoiding the rest.  This is the strongest
  demonstration of the no-harm property at the group level: even with a weak
  expert, the cascade routes selectively rather than degrading uniformly.

¶4 — Budget evolution (Figure 7, appendix)
  As the budget α grows, the DV top-k delegated set shifts composition: it
  starts concentrated on the highest-v group and gradually includes lower-v
  groups.  The uncertainty top-k has no such ordering and closely tracks the
  base rate of each dataset at every budget level.

Figures
  - group_scatter.pdf        → Figure 5 (main text): scatter mean v vs mean d(x), both experts
  - group_summary_strong.pdf → Figure 6a (main text): strong expert summary
  - group_summary_weak.pdf   → Figure 6b (main text): weak expert summary
  - delegation_composition.pdf → Figure 7 (appendix): composition by α

Usage
-----
  uv run python experiments/group_analysis.py \\
      --config experiments/configs/group_analysis.yaml

  uv run python experiments/group_analysis.py \\
      --config experiments/configs/group_analysis.yaml --use-clearml
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from clearml_logger import ClearMLLogger
from config import load_config
from dv_ltt_cascade import prepare_dv_cascade_data, split_calib_eval, threshold_cascade

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
    Reconstruct the eval and calibration splits for a cascade run.

    Mirrors probe_analysis.load_run: deterministically reproduces the exact
    eval split from the saved config, then sanity-checks against results.json.
    Also returns calibration probe scores (needed for uncertainty signal) and
    the full LTT table (needed for per-alpha tau values).
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
    assert eval_groups is not None, "group labels missing — check that test_groups is populated"

    # Sanity check: probe AUC must match results.json
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
    unc = probe_uncertainty(eval_ps, reference=calib_ps)

    return dict(
        config=config,
        eval_ps=eval_ps,
        eval_bs=eval_bs,
        eval_labels=eval_labels,
        eval_dv=eval_dv,
        eval_v=eval_v,
        eval_groups=eval_groups,
        unc=unc,
        probe_auc=actual_probe_auc,
        baseline_auc=saved["reference"]["baseline_auc"],
        ltt=saved["ltt"],
        run_dir=run_dir,
    )


# --------------------------------------------------------------------------- #
# Per-group metrics                                                            #
# --------------------------------------------------------------------------- #


def per_group_metrics(run: dict, groups: list[str]) -> dict:
    """Probe AUC, expert AUC, mean v(x,y), and mean d(x) per dataset group."""
    ps = run["eval_ps"]
    bs = run["eval_bs"]
    labels = run["eval_labels"]
    dv = run["eval_dv"]
    v = run["eval_v"]
    grps = run["eval_groups"]

    out = {}
    for g in groups:
        mask = grps == g
        out[g] = dict(
            n=int(mask.sum()),
            probe_auc=float(roc_auc_score(labels[mask], ps[mask])),
            expert_auc=float(roc_auc_score(labels[mask], bs[mask])),
            mean_v=float(v[mask].mean()),
            mean_dv=float(dv[mask].mean()),
        )
    return out


def get_tau_for_alpha(ltt: dict, method: str, target_alpha: float) -> tuple[float, float]:
    """Return (tau, realized_budget) for the entry in ltt[method] closest to target_alpha."""
    rows = ltt[method]
    best = min(rows, key=lambda r: abs(r["alpha"] - target_alpha))
    return best["tau"], best["realized_budget"]


def group_topk_stats(run: dict, alpha: float, groups: list[str], scores: np.ndarray) -> dict[str, dict]:
    """
    Per-group delegation rate and mean v(x,y) of the top-k delegated subset.

    Delegates the top ceil(alpha * N) examples globally, ranked descending by
    `scores`.  Unlike group_delegation_stats this is not threshold-based: it
    always uses exactly the requested budget.
    """
    N = len(scores)
    k = max(1, int(round(alpha * N)))
    del_mask = np.zeros(N, dtype=bool)
    del_mask[np.argsort(-scores)[:k]] = True
    grps = run["eval_groups"]
    v = run["eval_v"]
    out = {}
    for g in groups:
        group_mask = grps == g
        del_in_group = del_mask & group_mask
        n_del = int(del_in_group.sum())
        out[g] = {
            "rate": float(del_in_group.sum() / group_mask.sum()),
            "mean_v_delegated": float(v[del_in_group].mean()) if n_del > 0 else float("nan"),
        }
    return out


def group_delegation_stats(
    run: dict, tau: float, groups: list[str], scores: np.ndarray | None = None
) -> dict[str, dict]:
    """
    Per-group delegation rate and mean v(x,y) of the delegated subset.

    `scores` is the signal used by threshold_cascade to decide delegation
    (defaults to eval_dv).  Pass run["unc"] when computing stats for the
    uncertainty calibrated threshold.

    Even if a group has negative mean v overall, the delegated subset may still
    have positive mean v — the probe selects the positive-v examples within each
    group rather than routing by group membership.
    """
    routing_scores = run["eval_dv"] if scores is None else scores
    result = threshold_cascade(run["eval_ps"], run["eval_bs"], routing_scores, tau)
    del_mask = result.used_baseline
    grps = run["eval_groups"]
    v = run["eval_v"]
    out = {}
    for g in groups:
        group_mask = grps == g
        del_in_group = del_mask & group_mask
        n_del = int(del_in_group.sum())
        out[g] = {
            "rate": float(del_in_group.sum() / group_mask.sum()),
            "mean_v_delegated": float(v[del_in_group].mean()) if n_del > 0 else float("nan"),
        }
    return out


def delegation_composition(
    groups_arr: np.ndarray, ranking_scores: np.ndarray, alphas: np.ndarray, groups: list[str]
) -> dict:
    """
    Pool-level top-k composition sweep.

    At each alpha, sort descending by ranking_scores and take the top-alpha
    fraction; return the per-group fraction of that delegated set.
    """
    N = len(groups_arr)
    order = np.argsort(-ranking_scores)
    comp = {}
    for alpha in alphas:
        k = max(1, int(round(alpha * N)))
        delegated = groups_arr[order[:k]]
        comp[alpha] = {g: float((delegated == g).sum() / k) for g in groups}
    return comp


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #

GROUP_COLORS = {"anthropic": "C0", "mt": "C1", "mts": "C2", "toolace": "C3"}


def plot_group_scatter(strong_m: dict, weak_m: dict, groups: list[str]) -> plt.Figure:
    """
    Figure 5: scatter of mean v(x,y) vs mean d(x), one point per dataset group.

    Confirms that the DV probe's group-level predictions correlate with the
    ground-truth delegation benefit at the group level.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (title, m) in zip(
        axes,
        [
            ("Strong expert (Gemma-3-27B-IT)", strong_m),
            ("Weak expert (Llama-3.2-1B-Instruct)", weak_m),
        ],
        strict=False,
    ):
        mean_v = [m[g]["mean_v"] for g in groups]
        mean_dv = [m[g]["mean_dv"] for g in groups]
        colors = [GROUP_COLORS[g] for g in groups]

        ax.scatter(mean_v, mean_dv, s=120, c=colors, zorder=3)
        for g, v, dv in zip(groups, mean_v, mean_dv, strict=False):
            ax.annotate(g, (v, dv), textcoords="offset points", xytext=(6, 4), fontsize=10)
        ax.axvline(0, color="gray", ls="--", lw=0.8)
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("Mean ground-truth $v(x,y)$", fontsize=11)
        ax.set_ylabel("Mean DV probe score $d(x)$", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Group-level DV probe prediction vs ground-truth delegation benefit",
        fontweight="bold",
        fontsize=12,
    )
    plt.tight_layout()
    return fig


def _annotate_bars(ax, bars, vals, fontsize=7, offset=0.003):
    for bar, val in zip(bars, vals, strict=False):
        if np.isnan(val):
            continue
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (offset if val >= 0 else -offset),
            f"{val:.3f}",
            ha="center",
            va=va,
            fontsize=fontsize,
        )


def _plot_group_summary_single(
    axes,
    metrics: dict,
    dv_stats: dict,
    unc_stats: dict,
    unc_topk_stats: dict,
    dv_realized: float,
    unc_realized: float,
    target_alpha: float,
    groups: list[str],
) -> None:
    """
    Two-panel summary for one expert.

    Left: three bars per group — mean v(x,y) of all examples (green/red), mean v
    of the DV calibrated threshold delegated subset (orange), and mean v of the
    uncertainty top-k delegated subset at the same budget (blue).

    Right: per-group delegation rates for DV calibrated threshold (orange) vs
    uncertainty top-k (blue), with target α and DV realized budget as reference lines.

    To swap uncertainty top-k for the uncertainty calibrated threshold, replace the
    unc_topk_stats references with unc_stats (lines marked # [unc-ltt]).
    """
    ax_left, ax_right = axes
    x = np.arange(len(groups))
    bw = 0.25  # three bars per group on the left

    vals_v = [metrics[g]["mean_v"] for g in groups]
    vals_dv = [dv_stats[g]["mean_v_delegated"] for g in groups]
    vals_unc_topk = [unc_topk_stats[g]["mean_v_delegated"] for g in groups]
    # vals_unc_ltt = [unc_stats[g]["mean_v_delegated"] for g in groups]  # [unc-ltt]

    colors_v = ["forestgreen" if v > 0 else "crimson" for v in vals_v]

    b1 = ax_left.bar(
        x - bw,
        vals_v,
        bw,
        color=colors_v,
        alpha=0.70,
        edgecolor="black",
        label="Mean $v(x,y)$ — all examples",
    )
    b2 = ax_left.bar(
        x,
        vals_dv,
        bw,
        color="tab:orange",
        alpha=0.85,
        edgecolor="black",
        label="Mean $v$ — DV delegated",
    )
    b3 = ax_left.bar(
        x + bw,
        vals_unc_topk,
        bw,
        color="tab:blue",
        alpha=0.85,
        edgecolor="black",
        label="Mean $v$ — Unc. top-$k$",
    )
    # replace b3 above with the following for LTT uncertainty:  # [unc-ltt]
    # b3 = ax_left.bar(x + bw, vals_unc_ltt, bw, color="tab:blue", alpha=0.85,  # [unc-ltt]
    #                  edgecolor="black", label="Mean $v$ — Uncertainty LTT")     # [unc-ltt]

    ax_left.axhline(0, color="black", lw=0.8)
    global_mean_v = float(np.mean(vals_v))
    ax_left.axhline(global_mean_v, color="gray", ls="--", lw=1.2, label=f"Global mean $v$ = {global_mean_v:.3f}")

    _annotate_bars(ax_left, b1, vals_v)
    _annotate_bars(ax_left, b2, vals_dv)
    _annotate_bars(ax_left, b3, vals_unc_topk)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(groups)
    ax_left.set_ylabel("Mean $v(x,y)$", fontsize=11)
    ax_left.set_title("Mean $v(x,y)$: all examples vs delegated subset", fontsize=11)
    ax_left.legend(fontsize=8)
    ax_left.grid(axis="y", alpha=0.3)

    # Right: per-group delegation rates
    dv_rates = [dv_stats[g]["rate"] for g in groups]
    unc_topk_rates = [unc_topk_stats[g]["rate"] for g in groups]
    # unc_ltt_rates = [unc_stats[g]["rate"] for g in groups]  # [unc-ltt]

    b4 = ax_right.bar(
        x - bw / 2,
        dv_rates,
        bw,
        color="tab:orange",
        alpha=0.85,
        edgecolor="black",
        label="CTD",
    )
    b5 = ax_right.bar(
        x + bw / 2,
        unc_topk_rates,
        bw,
        color="tab:blue",
        alpha=0.85,
        edgecolor="black",
        label="Unc. top-$k$",
    )
    # replace b5 above with the following for LTT uncertainty:   # [unc-ltt]
    # b5 = ax_right.bar(x + bw / 2, unc_ltt_rates, bw, color="tab:blue", alpha=0.85,  # [unc-ltt]
    #                   edgecolor="black", label="Uncertainty calibrated threshold")     # [unc-ltt]
    ax_right.axhline(target_alpha, color="black", ls="--", lw=1.2, label=f"Target $\\alpha$ = {target_alpha:.2f}")
    ax_right.axhline(dv_realized, color="tab:orange", ls=":", lw=1.2, label=f"DV realized = {dv_realized:.2f}")

    _annotate_bars(ax_right, b4, dv_rates)
    _annotate_bars(ax_right, b5, unc_topk_rates)

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(groups)
    ax_right.set_ylabel("Delegation rate", fontsize=11)
    ax_right.set_title("Per-group delegation rate", fontsize=11)
    ax_right.legend(fontsize=8)
    ax_right.grid(axis="y", alpha=0.3)
    ax_right.set_ylim(0, None)


def plot_group_summary(
    metrics: dict,
    dv_stats: dict,
    unc_stats: dict,
    unc_topk_stats: dict,
    dv_realized: float,
    unc_realized: float,
    target_alpha: float,
    groups: list[str],
    suptitle: str,
) -> plt.Figure:
    """Figure 6a or 6b: two-panel summary for one expert."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _plot_group_summary_single(
        axes, metrics, dv_stats, unc_stats, unc_topk_stats, dv_realized, unc_realized, target_alpha, groups
    )
    fig.suptitle(suptitle, fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig


def plot_delegation_composition(strong: dict, weak: dict, groups: list[str], alphas: np.ndarray) -> plt.Figure:
    """
    Figure 7: 2x2 stacked bar charts of delegation composition by budget α.

    Rows: strong / weak expert.  Columns: DV top-k / Uncertainty top-k.
    White dashed lines mark cumulative base rates (where a uniform method
    would sit at every α).
    """
    base_rates = {g: float((strong["eval_groups"] == g).sum() / len(strong["eval_groups"])) for g in groups}

    dv_comp = {
        "strong": delegation_composition(strong["eval_groups"], strong["eval_dv"], alphas, groups),
        "weak": delegation_composition(weak["eval_groups"], weak["eval_dv"], alphas, groups),
    }
    unc_comp = {
        "strong": delegation_composition(strong["eval_groups"], strong["unc"], alphas, groups),
        "weak": delegation_composition(weak["eval_groups"], weak["unc"], alphas, groups),
    }

    def _stacked_bars(ax, comp, title):
        bottom = np.zeros(len(alphas))
        bar_w = (alphas[1] - alphas[0]) * 0.85
        for g in groups:
            fracs = np.array([comp[a][g] for a in alphas])
            ax.bar(alphas, fracs, bottom=bottom, width=bar_w, color=GROUP_COLORS[g], alpha=0.85, label=g)
            bottom += fracs
        # Dashed lines at cumulative base rates (reference for uniform routing)
        cum = 0.0
        for g in groups[:-1]:
            cum += base_rates[g]
            ax.axhline(cum, color="white", ls="--", lw=0.9, alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_xlim(alphas[0] - 0.03, alphas[-1] + 0.03)
        ax.set_xlabel("Budget $\\alpha$", fontsize=10)
        ax.set_ylabel("Fraction of delegated set", fontsize=10)
        ax.set_title(title, fontsize=11)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    _stacked_bars(axes[0, 0], dv_comp["strong"], "Strong expert — DV top-$k$")
    _stacked_bars(axes[0, 1], unc_comp["strong"], "Strong expert — Unc. top-$k$")
    _stacked_bars(axes[1, 0], dv_comp["weak"], "Weak expert — DV top-$k$")
    _stacked_bars(axes[1, 1], unc_comp["weak"], "Weak expert — Unc. top-$k$")

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g], alpha=0.85) for g in groups]
    fig.legend(
        handles, groups, title="Dataset", loc="lower center", ncol=len(groups), bbox_to_anchor=(0.5, 0.01), fontsize=10
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Per-group analysis of the DV cascade")
    parser.add_argument("--config", required=True, help="Path to group_analysis.yaml")
    parser.add_argument("--use-clearml", action="store_true", help="Log figures to ClearML")
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
            task_name="group_analysis",
            enabled=True,
        )
        clearml_logger.connect_configuration(vars(analysis_config))
        clearml_logger.add_tags(
            [
                "group-analysis",
                f"strong:{Path(analysis_config.strong_run_dir).name}",
                f"weak:{Path(analysis_config.weak_run_dir).name}",
            ]
        )

    # ------------------------------------------------------------------ #
    # Load runs                                                            #
    # ------------------------------------------------------------------ #
    print("\nLoading strong expert run...")
    strong = load_run(Path(analysis_config.strong_run_dir))
    print("\nLoading weak expert run...")
    weak = load_run(Path(analysis_config.weak_run_dir))

    groups = sorted(np.unique(strong["eval_groups"]).tolist())
    print(f"\nGroups: {groups}")
    print(f"Eval size: {len(strong['eval_ps'])} examples per run")

    target_alpha = getattr(analysis_config, "target_alpha", 0.2)

    # ------------------------------------------------------------------ #
    # Per-group metrics                                                    #
    # ------------------------------------------------------------------ #
    print("\n--- Per-group metrics ---")
    strong_m = per_group_metrics(strong, groups)
    weak_m = per_group_metrics(weak, groups)

    for label, m in [("STRONG", strong_m), ("WEAK", weak_m)]:
        print(f"\n  {label} EXPERT")
        print(f"  {'Group':>12} {'n':>5} {'Probe AUC':>10} {'Expert AUC':>10} {'Mean v':>10} {'Mean d(x)':>10}")
        for g in groups:
            mm = m[g]
            print(
                f"  {g:>12} {mm['n']:>5} {mm['probe_auc']:>10.3f} "
                f"{mm['expert_auc']:>10.3f} {mm['mean_v']:>10.3f} {mm['mean_dv']:>10.3f}"
            )

    metrics_out = {"strong": strong_m, "weak": weak_m}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("\n  Saved metrics.json")

    if clearml_logger is not None:
        flat = {}
        for split, m in metrics_out.items():
            for g, mm in m.items():
                flat[f"{split}/{g}/probe_auc"] = mm["probe_auc"]
                flat[f"{split}/{g}/expert_auc"] = mm["expert_auc"]
                flat[f"{split}/{g}/mean_v"] = mm["mean_v"]
                flat[f"{split}/{g}/mean_dv"] = mm["mean_dv"]
        clearml_logger.log_scalars(flat)

    # ------------------------------------------------------------------ #
    # Delegation rates at target_alpha                                     #
    # ------------------------------------------------------------------ #
    strong_dv_tau, strong_dv_realized = get_tau_for_alpha(strong["ltt"], "CTD", target_alpha)
    strong_unc_tau, strong_unc_realized = get_tau_for_alpha(strong["ltt"], "Unc. calibrated", target_alpha)
    weak_dv_tau, weak_dv_realized = get_tau_for_alpha(weak["ltt"], "CTD", target_alpha)
    weak_unc_tau, weak_unc_realized = get_tau_for_alpha(weak["ltt"], "Unc. calibrated", target_alpha)
    print(f"\nTarget α = {target_alpha}")
    print(f"  Strong DV:  τ = {strong_dv_tau:.4f},  realized = {strong_dv_realized:.3f}")
    print(f"  Strong Unc: τ = {strong_unc_tau:.4f}, realized = {strong_unc_realized:.3f}")
    print(f"  Weak DV:    τ = {weak_dv_tau:.4f},    realized = {weak_dv_realized:.3f}")
    print(f"  Weak Unc:   τ = {weak_unc_tau:.4f},   realized = {weak_unc_realized:.3f}")

    strong_dv_stats = group_delegation_stats(strong, strong_dv_tau, groups)
    strong_unc_stats = group_delegation_stats(strong, strong_unc_tau, groups, scores=strong["unc"])
    strong_unc_topk_stats = group_topk_stats(strong, target_alpha, groups, scores=strong["unc"])
    weak_dv_stats = group_delegation_stats(weak, weak_dv_tau, groups)
    weak_unc_stats = group_delegation_stats(weak, weak_unc_tau, groups, scores=weak["unc"])
    weak_unc_topk_stats = group_topk_stats(weak, target_alpha, groups, scores=weak["unc"])

    # ------------------------------------------------------------------ #
    # Figures                                                              #
    # ------------------------------------------------------------------ #
    alphas = np.linspace(0.05, 0.95, 19)

    figs = {
        "group_scatter": plot_group_scatter(strong_m, weak_m, groups),
        "group_summary_strong": plot_group_summary(
            strong_m,
            strong_dv_stats,
            strong_unc_stats,
            strong_unc_topk_stats,
            strong_dv_realized,
            strong_unc_realized,
            target_alpha,
            groups,
            suptitle="Strong expert: DV cascade concentrates budget on highest-value datasets",
        ),
        "group_summary_weak": plot_group_summary(
            weak_m,
            weak_dv_stats,
            weak_unc_stats,
            weak_unc_topk_stats,
            weak_dv_realized,
            weak_unc_realized,
            target_alpha,
            groups,
            suptitle="Weak expert: DV cascade selects positive-value examples across groups",
        ),
        "delegation_composition": plot_delegation_composition(strong, weak, groups, alphas),
    }

    for name, fig in figs.items():
        path = output_dir / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path.name}")
        if clearml_logger is not None:
            clearml_logger.log_figure("Group Analysis", name, fig)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Done                                                                 #
    # ------------------------------------------------------------------ #
    if clearml_logger is not None:
        clearml_logger.finalize()

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
