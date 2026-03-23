"""Experiment: SGT over DV cascade with joint budget + safety guarantees.

Applies Sequential Graphical Testing (Bretz et al. 2009) to a DV-scored
cascade, providing formal guarantees on **both** budget and a safety risk
(e.g., 1-AUROC or 1-accuracy) simultaneously.

Each hypothesis (tau, alpha_safety) is tested using a combined p-value
p = max(p_safety, p_budget), following the multi-risk LTT framework
(Angelopoulos et al. 2022, Proposition 6). The budget risk level is
fixed from config; the safety risk level is swept over a grid.

Main idea: if the target safety alpha is unreachable, the procedure
discovers the tightest achievable alpha that is jointly feasible with
the budget constraint.

Usage::

    uv run experiments/dv_sgt_cascade.py --config configs/dv_sgt/weak_expert_acc.yaml --use-clearml
    uv run experiments/dv_sgt_cascade.py --config configs/dv_sgt/strong_expert_acc.yaml --use-clearml
"""

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import load_config
from dotenv import load_dotenv
from dv_ltt_cascade import (
    cascade_metrics,
    pareto_filter_dv_thresholds,
    prepare_dv_cascade_data,
    split_calib_eval,
    threshold_cascade,
)

from reliable_monitoring.graphical_test_graphs import lattice_graph, row_chain_graph, uniform_lattice_graph
from reliable_monitoring.learn_then_test import (
    GraphicalTestResult,
    Hypothesis,
    compute_p_values,
    graphical_testing,
)
from reliable_monitoring.risks import (
    RISK_RGISTRY,
    BudgetCostRisk,
    evaluate_threshold_risks,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_FACTORIES = {
    "lattice": lattice_graph,
    "uniform_lattice": uniform_lattice_graph,
    "row_chain": row_chain_graph,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DVSGTCascadeResults:
    """Results from the DV SGT cascade experiment."""

    # Experiment metadata
    config: dict
    seed: int

    # Dataset info
    n_calib: int
    n_eval: int

    # Risk configuration
    guaranteed_safety_risk_name: str
    budget_guarantee_level: float
    guarantee_probability: float

    # SGT configuration
    sgt_graph_type: str
    sgt_row_dimension: str
    n_thresholds: int
    n_alphas: int
    n_hypotheses: int

    # SGT results
    n_rejected: int
    rejected_pairs: list[tuple[int, int]]
    ordered_thresholds: np.ndarray
    ordered_alphas: np.ndarray

    # Per-threshold cascade results on eval
    threshold_results: list[dict]

    # Selection
    selection_mode: str

    # Selected best
    reliable_threshold: float
    achieved_alpha: float

    # Cascade metrics at selected threshold (eval split)
    cascade_auc: float
    cascade_accuracy: float
    mean_budget_cost: float

    # Reference metrics (eval split)
    probe_only_auc: float
    probe_only_accuracy: float
    baseline_only_auc: float
    baseline_only_accuracy: float
    dv_probe_auc: float

    # Pareto testing
    n_original_thresholds: int
    n_pareto_thresholds: int | None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_rejection_heatmap(
    results: DVSGTCascadeResults,
    output_dir: Path,
) -> plt.Figure:
    """Heatmap of the (DV threshold x alpha_safety) grid showing rejected hypotheses.

    Uses the same two-layer imshow pattern as ``sgt_cascade_plotting.plot_performance_heatmaps``
    (background + masked overlay) which renders correctly on ClearML.
    """
    n_t = results.n_thresholds
    n_a = results.n_alphas

    # Human-readable risk names
    risk_display = {
        "accuracy_error": "1 - accuracy",
        "roc_auc_error": "1 - AUROC",
    }
    risk_label = risk_display.get(results.guaranteed_safety_risk_name, results.guaranteed_safety_risk_name)

    grid = np.zeros((n_t, n_a))
    for t_idx, a_idx in results.rejected_pairs:
        grid[t_idx, a_idx] = 1.0

    fig, ax = plt.subplots(figsize=(max(8, n_a * 0.6), max(6, n_t * 0.4)))

    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")

    # Axis labels
    alpha_labels = [f"{a:.3f}" for a in results.ordered_alphas]
    threshold_labels = [f"{t:.3f}" for t in results.ordered_thresholds]

    step_a = max(1, n_a // 10)
    step_t = max(1, n_t // 10)
    ax.set_xticks(range(0, n_a, step_a))
    ax.set_xticklabels(alpha_labels[::step_a], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(0, n_t, step_t))
    ax.set_yticklabels(threshold_labels[::step_t], fontsize=8)

    ax.set_xlabel(f"Target safety risk ({risk_label})", fontweight="bold")
    ax.set_ylabel("DV probe threshold (budget control)", fontweight="bold")
    ax.set_title(
        f"DV-SGT Rejection Map ({results.n_rejected}/{results.n_hypotheses} rejected)\n"
        f"Risk: {risk_label}, Budget: {results.budget_guarantee_level:.0%}, "
        f"Graph: {results.sgt_graph_type} ({results.sgt_row_dimension})",
        fontweight="bold",
    )

    # Mark the selected best pair
    best_t_idx = int(np.argmin(np.abs(results.ordered_thresholds - results.reliable_threshold)))
    best_a_idx = int(np.argmin(np.abs(results.ordered_alphas - results.achieved_alpha)))
    ax.plot(best_a_idx, best_t_idx, marker="*", markersize=18, color="gold", markeredgecolor="black", linewidth=1.5)

    plt.tight_layout()

    # Save PDF first, without colorbar (clean version for the paper)
    fig.savefig(output_dir / "rejection_heatmap.pdf", bbox_inches="tight")

    # Add colorbar AFTER saving the PDF. Without a colorbar, ClearML's
    # matplotlib-to-plotly conversion silently produces a blank white plot.
    # The colorbar forces ClearML to fall back to rasterized image mode,
    # which renders the imshow correctly.
    plt.colorbar(im, ax=ax, label="Rejected", ticks=[0, 1], format=lambda x, _: "Yes" if x > 0.5 else "No")
    return fig


def plot_alpha_discovery_curve(
    results: DVSGTCascadeResults,
    output_dir: Path,
) -> plt.Figure:
    """For each valid threshold, plot tightest achievable safety alpha vs budget cost."""
    if not results.threshold_results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid thresholds", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_dir / "alpha_discovery.pdf", bbox_inches="tight")
        return fig

    budgets = [r["mean_budget_cost"] for r in results.threshold_results]
    alphas = [r["best_alpha"] for r in results.threshold_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(budgets, alphas, "o-", color="steelblue", markersize=8, linewidth=2)

    # Budget guarantee level
    ax.axvline(
        results.budget_guarantee_level,
        color="red",
        ls=":",
        lw=2,
        label=rf"Budget guarantee: {results.budget_guarantee_level:.0%}",
    )

    # Highlight selected point
    ax.plot(
        results.mean_budget_cost,
        results.achieved_alpha,
        "*",
        color="gold",
        markersize=18,
        markeredgecolor="black",
        zorder=5,
    )
    ax.annotate(
        f"Selected: alpha_safety={results.achieved_alpha:.3f}, budget={results.mean_budget_cost:.1%}",
        xy=(results.mean_budget_cost, results.achieved_alpha),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    ax.set_xlabel("Budget cost (delegation rate)", fontweight="bold")
    ax.set_ylabel("Tightest achievable alpha_safety", fontweight="bold")
    ax.set_title(f"Safety Alpha Discovery ({results.guaranteed_safety_risk_name})", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower alpha = tighter = better

    fig.tight_layout()
    fig.savefig(output_dir / "alpha_discovery.pdf", bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_dv_sgt_cascade(config, output_dir: Path) -> DVSGTCascadeResults | None:
    """Run the DV SGT cascade experiment with joint budget + safety guarantees.

    Stages:
    1. Train probes & load data
    2. Evaluate empirical risks (safety + budget) on calib
    3. Build SGT grid with combined p-values and run graphical testing
    4. Evaluate cascade on eval split
    5. Generate plots & save results
    """
    seed = config.seed
    np.random.seed(seed)

    n_thresholds = config.n_thresholds

    # ---- Stage 1: Train probes & load data ----
    logger.info("=== Stage 1: Train probes & load data ===")
    data = prepare_dv_cascade_data(config, tau_steps=n_thresholds)

    calib_fraction = config.calib_fraction
    calib_arrays, eval_arrays = split_calib_eval(
        data.test_ps,
        data.test_bs,
        data.test_labels,
        data.dv_scores,
        data.test_groups,
        calib_fraction=calib_fraction,
        seed=seed,
    )
    calib_ps, calib_bs, calib_labels, calib_dv, calib_groups = calib_arrays
    eval_ps, eval_bs, eval_labels, eval_dv, eval_groups = eval_arrays
    assert calib_ps is not None and calib_bs is not None and calib_labels is not None
    assert calib_dv is not None and eval_ps is not None and eval_bs is not None
    assert eval_labels is not None and eval_dv is not None

    n_calib = len(calib_labels)
    n_eval = len(eval_labels)
    logger.info(f"Calib: n={n_calib}, Eval: n={n_eval}")

    # ---- Risk setup ----
    safety_risk_name = config.guaranteed_safety_risk
    SafetyRisk = RISK_RGISTRY.get(safety_risk_name)
    if SafetyRisk is None:
        raise ValueError(
            f"Invalid guaranteed_safety_risk: '{safety_risk_name}'. Available: {list(RISK_RGISTRY.keys())}"
        )

    budget_guarantee_level = config.budget_guarantee_level
    merge_strategy = config.merge_strategy
    thresholds = data.dv_tau_grid

    # ---- Stage 2: Evaluate empirical risks on calib ----
    logger.info("=== Stage 2: Evaluate empirical risks on calib ===")

    # Evaluate both safety and budget risks
    calib_eval_result = evaluate_threshold_risks(
        calib_ps,
        calib_bs,
        thresholds,
        risks=[SafetyRisk, BudgetCostRisk],
        labels=calib_labels,
        merge_strategy=merge_strategy,
        delegation_scores=calib_dv,
    )

    logger.info(f"Empirical risks computed on {n_calib} calib samples.")
    for thr, s_risk, b_risk in zip(
        calib_eval_result.thresholds,
        calib_eval_result[safety_risk_name],
        calib_eval_result["budget"],
        strict=True,
    ):
        logger.info(f"  tau={thr:.4f}, {safety_risk_name}={s_risk:.4f}, budget={b_risk:.4f}")

    n_original_thresholds = len(thresholds)

    # ---- Optional: Pareto pre-filtering ----
    pareto_testing = config.pareto_testing
    n_pareto = None

    if pareto_testing:
        pareto_proportion = config.pareto_split_proportion
        pf, ht_idx, _opt_idx = pareto_filter_dv_thresholds(
            calib_ps,
            calib_bs,
            calib_labels,
            calib_dv,
            thresholds,
            risks=[SafetyRisk, BudgetCostRisk],
            merge_strategy=merge_strategy,
            pareto_proportion=pareto_proportion,
            seed=seed,
        )
        thresholds = pf.taus
        n_pareto = len(thresholds)

        # Re-evaluate both risks on ht split (independent of Pareto selection)
        calib_eval_result = evaluate_threshold_risks(
            calib_ps[ht_idx],
            calib_bs[ht_idx],
            thresholds,
            risks=[SafetyRisk, BudgetCostRisk],
            labels=calib_labels[ht_idx],
            merge_strategy=merge_strategy,
            delegation_scores=calib_dv[ht_idx],
        )

    calib_safety_empirical = calib_eval_result[safety_risk_name]
    calib_budget_empirical = calib_eval_result["budget"]

    # ---- Optional: Deduplicate thresholds ----
    deduplicate_thresholds = config.deduplicate_thresholds
    if deduplicate_thresholds:
        # Deduplicate on safety risk (the varying dimension)
        _, unique_indices = np.unique(calib_safety_empirical, return_index=True)
        unique_indices = np.sort(unique_indices)
        n_before = len(thresholds)
        thresholds = thresholds[unique_indices]
        calib_safety_empirical = calib_safety_empirical[unique_indices]
        calib_budget_empirical = calib_budget_empirical[unique_indices]
        logger.info(f"Deduplicated thresholds: {n_before} -> {len(thresholds)}")

    # ---- Stage 3: Build SGT grid and run graphical testing ----
    logger.info("=== Stage 3: SGT graphical testing (joint budget + safety) ===")

    alpha_grid = np.linspace(config.alpha_start, config.alpha_end, config.alpha_steps)
    guarantee_probability = config.guarantee_probability
    delta = 1 - guarantee_probability

    n_t, n_a = len(thresholds), len(alpha_grid)
    logger.info(f"SGT grid: {n_t} thresholds x {n_a} safety alphas = {n_t * n_a} hypotheses")
    logger.info(
        f"Safety alpha range: [{alpha_grid.min():.3f}, {alpha_grid.max():.3f}], "
        f"Budget guarantee: {budget_guarantee_level:.3f}, FWER delta={delta:.3f}"
    )

    # Order thresholds: large tau = conservative = lowest risk -> safest first
    threshold_order = np.argsort(-thresholds)
    # Order safety alphas: most permissive (largest) first
    alpha_order = np.argsort(-alpha_grid)

    ordered_thresholds = thresholds[threshold_order]
    ordered_alphas = alpha_grid[alpha_order]
    ordered_safety_empirical = calib_safety_empirical[threshold_order]
    ordered_budget_empirical = calib_budget_empirical[threshold_order]

    n_samples = calib_eval_result.n_samples
    safety_bound_fn = SafetyRisk.p_value_bound_fn
    budget_bound_fn = BudgetCostRisk.p_value_bound_fn

    # Row dimension: controls graph orientation
    #   "safety_first" (default): rows=thresholds, cols=safety_alphas
    #       → chains across safety alphas per threshold, surplus flows to next threshold
    #       → good when some thresholds are budget-infeasible (they waste only their row)
    #   "budget_first": rows=safety_alphas, cols=thresholds
    #       → chains across thresholds per safety alpha, surplus flows to next alpha
    #       → good after Pareto filtering when all thresholds are budget-feasible
    row_dim = config.sgt_row_dimension

    # Pre-convert to Python floats to avoid numpy deprecation warnings in lambdas
    ordered_safety_floats = [float(x) for x in ordered_safety_empirical]
    ordered_budget_floats = [float(x) for x in ordered_budget_empirical]

    if row_dim == "safety_first":
        n_rows, n_cols = n_t, n_a
        hypotheses = [
            Hypothesis(
                p_value_fn=lambda s_risk=s_emp, b_risk=b_emp, a=alpha: float(
                    max(
                        safety_bound_fn(s_risk, n_samples, a),
                        budget_bound_fn(b_risk, n_samples, budget_guarantee_level),
                    )
                ),
                params={"threshold": float(ordered_thresholds[t_idx]), "alpha_safety": float(alpha)},
            )
            for t_idx, (s_emp, b_emp) in enumerate(zip(ordered_safety_floats, ordered_budget_floats, strict=True))
            for alpha in ordered_alphas
        ]
    elif row_dim == "budget_first":
        n_rows, n_cols = n_a, n_t
        hypotheses = [
            Hypothesis(
                p_value_fn=lambda s_risk=s_emp, b_risk=b_emp, a=alpha: float(
                    max(
                        safety_bound_fn(s_risk, n_samples, a),
                        budget_bound_fn(b_risk, n_samples, budget_guarantee_level),
                    )
                ),
                params={"threshold": float(ordered_thresholds[t_idx]), "alpha_safety": float(alpha)},
            )
            for alpha in ordered_alphas
            for t_idx, (s_emp, b_emp) in enumerate(zip(ordered_safety_floats, ordered_budget_floats, strict=True))
        ]
    else:
        raise ValueError(f"Invalid sgt_row_dimension: '{row_dim}'. Use 'safety_first' or 'budget_first'.")

    logger.info(f"SGT row dimension: {row_dim} ({n_rows} rows x {n_cols} cols)")

    flat_p_values = compute_p_values(hypotheses)

    # Build graph
    graph_type = config.sgt_graph_type
    if graph_type not in GRAPH_FACTORIES:
        raise ValueError(f"Unknown sgt_graph_type: '{graph_type}'. Available: {list(GRAPH_FACTORIES.keys())}")
    weights, transitions = GRAPH_FACTORIES[graph_type](n_rows, n_cols)

    logger.info(f"Running graphical testing (graph={graph_type})...")
    sgt_result: GraphicalTestResult = graphical_testing(flat_p_values, weights, transitions, delta=delta)

    if not sgt_result.rejected:
        logger.warning("SGT: No hypotheses rejected! Cannot run cascade.")
        return None

    # Map flat indices back to (threshold_idx, alpha_idx) pairs
    if row_dim == "safety_first":
        rejected_pairs = [(idx // n_cols, idx % n_cols) for idx in sgt_result.rejected]
    else:  # budget_first
        rejected_pairs = [(idx % n_cols, idx // n_cols) for idx in sgt_result.rejected]
    logger.info(f"SGT rejected {len(rejected_pairs)}/{n_t * n_a} hypotheses")

    # Log achievable guarantees per safety alpha level
    for a_idx, alpha_val in enumerate(ordered_alphas):
        valid_t = [t for (t, a) in rejected_pairs if a == a_idx]
        if valid_t:
            t_range = f"{ordered_thresholds[min(valid_t)]:.4f}-{ordered_thresholds[max(valid_t)]:.4f}"
            logger.info(f"  alpha_safety={alpha_val:.3f}: {len(valid_t)} valid thresholds ({t_range})")

    # ---- Stage 4: Evaluate cascade on eval split ----
    logger.info("=== Stage 4: Evaluate cascade on eval split ===")

    valid_by_threshold: dict[int, list[int]] = defaultdict(list)
    for t_idx, a_idx in rejected_pairs:
        valid_by_threshold[t_idx].append(a_idx)

    threshold_results: list[dict] = []
    for t_idx in sorted(valid_by_threshold):
        thr = float(ordered_thresholds[t_idx])
        alpha_indices = sorted(valid_by_threshold[t_idx])
        # ordered_alphas is sorted descending, so largest index = tightest alpha
        best_alpha = float(ordered_alphas[max(alpha_indices)])

        cascade_result = threshold_cascade(eval_ps, eval_bs, eval_dv, thr, merge_strategy=merge_strategy)
        met = cascade_metrics(cascade_result, eval_labels)

        budget_cost = float(cascade_result.used_baseline.mean())
        threshold_results.append(
            {
                "threshold": thr,
                "best_alpha": best_alpha,
                "valid_alpha_indices": alpha_indices,
                "auc": met["auc"],
                "accuracy": met["accuracy"],
                "mean_budget_cost": budget_cost,
            }
        )
        logger.info(
            f"  tau={thr:.4f}: alpha_safety<={best_alpha:.4f}, "
            f"budget={budget_cost:.4f}, auc={met['auc']:.4f}, acc={met['accuracy']:.4f}"
        )

    # ---- Select headline (threshold, alpha) pair ----
    selection_mode = config.selection_mode

    if selection_mode == "best_alpha":
        # Tightest safety alpha, break ties by lowest budget
        selected = min(threshold_results, key=lambda r: (r["best_alpha"], r["mean_budget_cost"]))
    else:
        raise ValueError(f"Unknown selection_mode: '{selection_mode}'")

    reliable_threshold = selected["threshold"]
    achieved_alpha = selected["best_alpha"]
    mean_budget = selected["mean_budget_cost"]

    logger.info(
        f"Selected ({selection_mode}): tau={reliable_threshold:.4f}, "
        f"alpha_safety={achieved_alpha:.4f}, budget={mean_budget:.4f}"
    )

    # ---- Reference metrics (eval split) ----
    from sklearn.metrics import accuracy_score, roc_auc_score

    probe_auc = float(roc_auc_score(eval_labels, eval_ps))
    probe_acc = float(accuracy_score(eval_labels, (eval_ps >= 0.5).astype(int)))
    baseline_auc = float(roc_auc_score(eval_labels, eval_bs))
    baseline_acc = float(accuracy_score(eval_labels, (eval_bs >= 0.5).astype(int)))

    dv_auc = data.dv_auc

    logger.info("\n=== OVERALL PERFORMANCE METRICS ===")
    logger.info(f"Probe Only:    AUC={probe_auc:.4f}, Acc={probe_acc:.4f}")
    logger.info(f"Baseline Only: AUC={baseline_auc:.4f}, Acc={baseline_acc:.4f}")
    logger.info(f"Cascade:       AUC={selected['auc']:.4f}, Acc={selected['accuracy']:.4f}")
    logger.info(f"Guaranteed:    {safety_risk_name} <= {achieved_alpha:.4f}, budget <= {budget_guarantee_level:.4f}")
    logger.info(f"DV probe AUC:  {dv_auc:.4f}")
    logger.info("===================================\n")

    results = DVSGTCascadeResults(
        config=vars(config) if hasattr(config, "__dict__") else dict(config),
        seed=seed,
        n_calib=n_calib,
        n_eval=n_eval,
        guaranteed_safety_risk_name=safety_risk_name,
        budget_guarantee_level=budget_guarantee_level,
        guarantee_probability=guarantee_probability,
        sgt_graph_type=graph_type,
        sgt_row_dimension=row_dim,
        n_thresholds=n_t,
        n_alphas=n_a,
        n_hypotheses=n_t * n_a,
        n_rejected=len(rejected_pairs),
        rejected_pairs=rejected_pairs,
        ordered_thresholds=ordered_thresholds,
        ordered_alphas=ordered_alphas,
        threshold_results=threshold_results,
        selection_mode=selection_mode,
        reliable_threshold=reliable_threshold,
        achieved_alpha=achieved_alpha,
        cascade_auc=selected["auc"],
        cascade_accuracy=selected["accuracy"],
        mean_budget_cost=mean_budget,
        probe_only_auc=probe_auc,
        probe_only_accuracy=probe_acc,
        baseline_only_auc=baseline_auc,
        baseline_only_accuracy=baseline_acc,
        dv_probe_auc=dv_auc,
        n_original_thresholds=n_original_thresholds,
        n_pareto_thresholds=n_pareto,
    )

    # ---- Stage 5: Save results JSON ----
    results_json = {
        "guaranteed_safety_risk": safety_risk_name,
        "budget_guarantee_level": budget_guarantee_level,
        "guarantee_probability": guarantee_probability,
        "sgt_graph_type": graph_type,
        "sgt_row_dimension": row_dim,
        "n_thresholds": n_t,
        "n_alphas": n_a,
        "n_hypotheses": n_t * n_a,
        "n_rejected": len(rejected_pairs),
        "selection_mode": selection_mode,
        "reliable_threshold": reliable_threshold,
        "achieved_alpha": achieved_alpha,
        "cascade_auc": selected["auc"],
        "cascade_accuracy": selected["accuracy"],
        "mean_budget_cost": mean_budget,
        "probe_only_auc": probe_auc,
        "probe_only_accuracy": probe_acc,
        "baseline_only_auc": baseline_auc,
        "baseline_only_accuracy": baseline_acc,
        "dv_probe_auc": dv_auc,
        "n_original_thresholds": n_original_thresholds,
        "n_pareto_thresholds": n_pareto,
        "threshold_results": [{k: v for k, v in r.items() if k != "valid_alpha_indices"} for r in threshold_results],
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results_json, indent=2))
    logger.info(f"Results JSON saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    default_config = Path(__file__).parent / "configs" / "dv_sgt" / "strong_expert_acc.yaml"
    parser = argparse.ArgumentParser(description="DV SGT Cascade Experiment")
    parser.add_argument("--config", type=str, default=str(default_config))
    default_output = os.path.join(os.environ.get("RESULTS_DIR", "results"), "dv_sgt_cascade")
    parser.add_argument("--output-dir", type=str, default=default_output)
    parser.add_argument("--use-clearml", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / Path(args.config).name)

    config = load_config(args.config)

    # ClearML init
    clearml_logger = None
    if args.use_clearml:
        from clearml_logger import ClearMLLogger

        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="dv_sgt_cascade",
            enabled=True,
        )

    results = run_dv_sgt_cascade(config, output_dir)

    if results is None:
        logger.warning("Experiment failed: no reliable (threshold, alpha) pair found.")
    else:
        logger.info("Generating plots...")
        figs = {
            "rejection_heatmap": plot_rejection_heatmap(results, output_dir),
            "alpha_discovery": plot_alpha_discovery_curve(results, output_dir),
        }
        logger.info(f"Plots saved to {output_dir}")

        if clearml_logger is not None:
            clearml_logger.connect_configuration(results.config)
            tags = [
                "dv-sgt-cascade",
                f"safety_risk-{results.guaranteed_safety_risk_name}",
                f"budget_guarantee-{results.budget_guarantee_level:.2f}",
                f"achieved_alpha-{results.achieved_alpha:.3f}",
                f"rejected-{results.n_rejected}/{results.n_hypotheses}",
                f"sgt-{results.sgt_graph_type}-{results.sgt_row_dimension}",
                f"pareto_testing-{results.config.get('pareto_testing', False)}",
            ]
            clearml_logger.add_tags(tags)
            clearml_logger.log_scalars(
                {
                    "achieved_alpha": results.achieved_alpha,
                    "reliable_threshold": results.reliable_threshold,
                    "budget_guarantee_level": results.budget_guarantee_level,
                    "cascade_auc": results.cascade_auc,
                    "cascade_accuracy": results.cascade_accuracy,
                    "mean_budget_cost": results.mean_budget_cost,
                    "probe_only_auc": results.probe_only_auc,
                    "baseline_only_auc": results.baseline_only_auc,
                    "dv_probe_auc": results.dv_probe_auc,
                    "n_rejected": results.n_rejected,
                    "n_hypotheses": results.n_hypotheses,
                }
            )
            for name, fig in figs.items():
                clearml_logger.log_figure("DV-SGT Cascade", name, fig)
            clearml_logger.finalize()

        plt.close("all")

    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
