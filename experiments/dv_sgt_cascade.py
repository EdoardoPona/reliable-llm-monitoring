"""Experiment: SGT over DV cascade.

Applies Sequential Graphical Testing (Bretz et al. 2009) to a DV-scored
cascade so we can provide formal guarantees on a **safety risk**
(e.g., 1-AUROC or 1-accuracy) at the cascade level, testing multiple
alpha values simultaneously.

Key showcase: if the target alpha is unreachable, the procedure discovers
the tightest achievable alpha.

Combines:
- ``dv_cascade_comparison.py``: DV probe training + data loading
- ``sgt_cascade.py``: SGT over a (threshold × alpha) grid

Usage::

    uv run experiments/dv_sgt_cascade.py \\
        --config configs/dv_sgt/strong_expert_acc.yaml

    uv run experiments/dv_sgt_cascade.py \\
        --config configs/dv_sgt/strong_expert_acc.yaml --use-clearml
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
    guaranteed_risk_name: str
    guarantee_probability: float

    # SGT configuration
    sgt_graph_type: str
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
    budget_target: float | None

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
    """Heatmap of the (DV threshold × alpha) grid showing rejected hypotheses."""
    n_t = results.n_thresholds
    n_a = results.n_alphas
    grid = np.zeros((n_t, n_a))
    for t_idx, a_idx in results.rejected_pairs:
        grid[t_idx, a_idx] = 1.0

    fig, ax = plt.subplots(figsize=(max(8, n_a * 0.6), max(6, n_t * 0.4)))
    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")

    alpha_labels = [f"{a:.3f}" for a in results.ordered_alphas]
    threshold_labels = [f"{t:.3f}" for t in results.ordered_thresholds]
    step_a = max(1, n_a // 10)
    step_t = max(1, n_t // 10)
    ax.set_xticks(range(0, n_a, step_a))
    ax.set_xticklabels(alpha_labels[::step_a], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(0, n_t, step_t))
    ax.set_yticklabels(threshold_labels[::step_t], fontsize=8)

    ax.set_xlabel("Alpha (risk bound)", fontweight="bold")
    ax.set_ylabel("DV threshold (tau)", fontweight="bold")
    ax.set_title(
        f"DV-SGT Rejection Map ({results.n_rejected}/{results.n_hypotheses} rejected)\n"
        f"Graph: {results.sgt_graph_type}, Risk: {results.guaranteed_risk_name}",
        fontweight="bold",
    )

    # Mark selected pair
    best_t_idx = int(np.argmin(np.abs(results.ordered_thresholds - results.reliable_threshold)))
    best_a_idx = int(np.argmin(np.abs(results.ordered_alphas - results.achieved_alpha)))
    ax.plot(best_a_idx, best_t_idx, marker="*", markersize=18, color="gold", markeredgecolor="black", linewidth=1.5)
    ax.annotate(
        f"Best: tau={results.reliable_threshold:.3f}, alpha={results.achieved_alpha:.3f}",
        xy=(best_a_idx, best_t_idx),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    fig.tight_layout()
    fig.savefig(output_dir / "rejection_heatmap.pdf", bbox_inches="tight")
    return fig


def plot_alpha_discovery_curve(
    results: DVSGTCascadeResults,
    output_dir: Path,
) -> plt.Figure:
    """For each valid threshold, plot tightest achievable alpha vs budget cost."""
    if not results.threshold_results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid thresholds", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_dir / "alpha_discovery.pdf", bbox_inches="tight")
        return fig

    budgets = [r["mean_budget_cost"] for r in results.threshold_results]
    alphas = [r["best_alpha"] for r in results.threshold_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(budgets, alphas, "o-", color="steelblue", markersize=8, linewidth=2)

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
        f"Selected: alpha={results.achieved_alpha:.3f}, budget={results.mean_budget_cost:.1%}",
        xy=(results.mean_budget_cost, results.achieved_alpha),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    ax.set_xlabel("Budget cost (delegation rate)", fontweight="bold")
    ax.set_ylabel("Tightest achievable alpha", fontweight="bold")
    ax.set_title(f"Alpha Discovery Curve ({results.guaranteed_risk_name})", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower alpha = tighter = better

    fig.tight_layout()
    fig.savefig(output_dir / "alpha_discovery.pdf", bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_dv_sgt_cascade(config, output_dir: Path) -> DVSGTCascadeResults | None:
    """Run the DV SGT cascade experiment.

    Stages:
    1. Train probes & load data
    2. Evaluate empirical safety risks on calib using DV thresholds
    3. Build SGT grid and run graphical testing
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

    # ---- Stage 2: Evaluate empirical safety risks on calib ----
    logger.info("=== Stage 2: Evaluate empirical risks on calib ===")

    guaranteed_risk_name = config.guaranteed_risk
    GuaranteedRisk = RISK_RGISTRY.get(guaranteed_risk_name)
    if GuaranteedRisk is None:
        raise ValueError(f"Invalid guaranteed_risk: '{guaranteed_risk_name}'. Available: {list(RISK_RGISTRY.keys())}")

    merge_strategy = config.merge_strategy
    thresholds = data.dv_tau_grid

    calib_eval_result = evaluate_threshold_risks(
        calib_ps,
        calib_bs,
        thresholds,
        risks=GuaranteedRisk,
        labels=calib_labels if GuaranteedRisk is not BudgetCostRisk else None,
        merge_strategy=merge_strategy,
        delegation_scores=calib_dv,
    )

    logger.info(f"Empirical {GuaranteedRisk.description} risks computed on {n_calib} calib samples.")
    for thr, risk in zip(calib_eval_result.thresholds, calib_eval_result[guaranteed_risk_name], strict=True):
        logger.info(f"  tau={thr:.4f}, empirical risk={risk:.4f}")

    n_original_thresholds = len(thresholds)

    # ---- Optional: Pareto pre-filtering ----
    pareto_testing = config.pareto_testing
    n_pareto = None

    if pareto_testing:
        opt_risk_name = config.opt_risk
        OptRisk = RISK_RGISTRY.get(opt_risk_name)
        if OptRisk is None:
            raise ValueError(f"Invalid opt_risk: '{opt_risk_name}'. Available: {list(RISK_RGISTRY.keys())}")
        if OptRisk.name == GuaranteedRisk.name:
            raise ValueError(f"opt_risk and guaranteed_risk must differ, both are '{OptRisk.name}'")

        pareto_proportion = config.pareto_split_proportion
        pf, ht_idx, _opt_idx = pareto_filter_dv_thresholds(
            calib_ps,
            calib_bs,
            calib_labels,
            calib_dv,
            thresholds,
            risks=[GuaranteedRisk, OptRisk],
            merge_strategy=merge_strategy,
            pareto_proportion=pareto_proportion,
            seed=seed,
        )
        thresholds = pf.taus
        n_pareto = len(thresholds)

        # Re-evaluate guaranteed risk on ht split (independent of Pareto selection)
        calib_eval_result = evaluate_threshold_risks(
            calib_ps[ht_idx],
            calib_bs[ht_idx],
            thresholds,
            risks=GuaranteedRisk,
            labels=calib_labels[ht_idx] if GuaranteedRisk is not BudgetCostRisk else None,
            merge_strategy=merge_strategy,
            delegation_scores=calib_dv[ht_idx],
        )
        calib_empirical = calib_eval_result[guaranteed_risk_name]
    else:
        calib_empirical = calib_eval_result[guaranteed_risk_name]

    # ---- Optional: Deduplicate thresholds ----
    deduplicate_thresholds = config.deduplicate_thresholds
    if deduplicate_thresholds:
        _, unique_indices = np.unique(calib_empirical, return_index=True)
        unique_indices = np.sort(unique_indices)
        n_before = len(thresholds)
        thresholds = thresholds[unique_indices]
        calib_empirical = calib_empirical[unique_indices]
        logger.info(f"Deduplicated thresholds: {n_before} -> {len(thresholds)}")

    # ---- Stage 3: Build SGT grid and run graphical testing ----
    logger.info("=== Stage 3: SGT graphical testing ===")

    alpha_grid = np.linspace(config.alpha_start, config.alpha_end, config.alpha_steps)

    guarantee_probability = config.guarantee_probability
    delta = 1 - guarantee_probability

    n_t, n_a = len(thresholds), len(alpha_grid)
    logger.info(f"SGT grid: {n_t} thresholds x {n_a} alphas = {n_t * n_a} hypotheses")
    logger.info(f"Alpha range: [{alpha_grid.min():.3f}, {alpha_grid.max():.3f}], FWER delta={delta:.3f}")

    # Order thresholds: large tau = conservative = lowest risk -> safest first
    threshold_order = np.argsort(-thresholds)
    # Order alphas: most permissive (largest) first
    alpha_order = np.argsort(-alpha_grid)

    ordered_thresholds = thresholds[threshold_order]
    ordered_alphas = alpha_grid[alpha_order]
    ordered_empirical = calib_empirical[threshold_order]

    n_samples = calib_eval_result.n_samples
    bound_fn = GuaranteedRisk.p_value_bound_fn

    # Default: rows=thresholds, cols=alphas
    n_rows, n_cols = n_t, n_a
    hypotheses = [
        Hypothesis(
            p_value_fn=lambda r=risk, a=alpha: float(bound_fn(r, n_samples, a)),
            params={"threshold": float(ordered_thresholds[t_idx]), "alpha": float(alpha)},
        )
        for t_idx, risk in enumerate(ordered_empirical)
        for alpha in ordered_alphas
    ]

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
    rejected_pairs = [(idx // n_cols, idx % n_cols) for idx in sgt_result.rejected]
    logger.info(f"SGT rejected {len(rejected_pairs)}/{n_t * n_a} hypotheses")

    # Log achievable guarantees per alpha level
    for a_idx, alpha_val in enumerate(ordered_alphas):
        valid_t = [t for (t, a) in rejected_pairs if a == a_idx]
        if valid_t:
            t_range = f"{ordered_thresholds[min(valid_t)]:.4f}-{ordered_thresholds[max(valid_t)]:.4f}"
            logger.info(f"  alpha={alpha_val:.3f}: {len(valid_t)} valid thresholds ({t_range})")

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
            f"  tau={thr:.4f}: alpha<={best_alpha:.4f}, "
            f"budget={budget_cost:.4f}, auc={met['auc']:.4f}, acc={met['accuracy']:.4f}"
        )

    # ---- Select headline (threshold, alpha) pair ----
    selection_mode = config.selection_mode
    budget_target: float | None = None

    if selection_mode == "best_alpha":
        # Tightest alpha, break ties by lowest budget
        selected = min(threshold_results, key=lambda r: (r["best_alpha"], r["mean_budget_cost"]))
    elif selection_mode == "budget_target":
        budget_target = config.budget_target
        selected = min(
            threshold_results,
            key=lambda r: (abs(r["mean_budget_cost"] - budget_target), r["best_alpha"]),
        )
    else:
        raise ValueError(f"Unknown selection_mode: '{selection_mode}'")

    reliable_threshold = selected["threshold"]
    achieved_alpha = selected["best_alpha"]
    mean_budget = selected["mean_budget_cost"]

    logger.info(
        f"Selected ({selection_mode}): tau={reliable_threshold:.4f}, "
        f"alpha={achieved_alpha:.4f}, budget={mean_budget:.4f}"
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
    logger.info(f"Guaranteed:    risk({guaranteed_risk_name}) <= {achieved_alpha:.4f}")
    logger.info(f"DV probe AUC:  {dv_auc:.4f}")
    logger.info("===================================\n")

    results = DVSGTCascadeResults(
        config=vars(config) if hasattr(config, "__dict__") else dict(config),
        seed=seed,
        n_calib=n_calib,
        n_eval=n_eval,
        guaranteed_risk_name=guaranteed_risk_name,
        guarantee_probability=guarantee_probability,
        sgt_graph_type=graph_type,
        n_thresholds=n_t,
        n_alphas=n_a,
        n_hypotheses=n_t * n_a,
        n_rejected=len(rejected_pairs),
        rejected_pairs=rejected_pairs,
        ordered_thresholds=ordered_thresholds,
        ordered_alphas=ordered_alphas,
        threshold_results=threshold_results,
        selection_mode=selection_mode,
        budget_target=budget_target,
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

    # ---- Stage 5: Plots ----
    logger.info("=== Stage 5: Generate plots ===")
    figs: dict[str, plt.Figure] = {}
    figs["rejection_heatmap"] = plot_rejection_heatmap(results, output_dir)
    figs["alpha_discovery"] = plot_alpha_discovery_curve(results, output_dir)
    plt.close("all")
    logger.info(f"Plots saved to {output_dir}")

    # ---- Stage 6: Save results JSON ----
    results_json = {
        "guaranteed_risk": guaranteed_risk_name,
        "guarantee_probability": guarantee_probability,
        "sgt_graph_type": graph_type,
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
    elif clearml_logger is not None:
        clearml_logger.connect_configuration(results.config)
        tags = [
            "dv-sgt-cascade",
            f"guaranteed_risk-{results.guaranteed_risk_name}",
            f"achieved_alpha-{results.achieved_alpha:.3f}",
            f"rejected-{results.n_rejected}/{results.n_hypotheses}",
            f"sgt-{results.sgt_graph_type}",
            f"baseline:{results.config.get('baseline_model_name', 'unknown')}",
            f"dv_target:{results.config.get('dv_target', 'continuous')}",
            f"pareto_testing-{results.config.get('pareto_testing', False)}",
        ]
        clearml_logger.add_tags(tags)
        clearml_logger.log_scalars(
            {
                "achieved_alpha": results.achieved_alpha,
                "reliable_threshold": results.reliable_threshold,
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
        clearml_logger.finalize()

    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
