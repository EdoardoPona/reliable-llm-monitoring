"""Experiment: Batched top-k ranking comparison.

Compares the ranking quality of different delegation signals in the
batched cascade setting.  For each batch, examples are ranked by signal
and the top-k are delegated to the baseline.  This isolates ranking
quality by using the same mechanism (batched top-k) for all signals.

Signals compared:
  - Probe uncertainty: u(x) = min(p, 1-p)
  - DV probe score:    d(x) = P(v=1 | z)
  - Oracle:            v(x) = 1[probe wrong AND baseline correct]

Outputs:
  - ranking_comparison_B{batch_size}.png  (per batch size, for appendix)
  - ranking_comparison_main.png           (single batch size for main paper)
  - ranking_comparison_grid.png           (all batch sizes side by side)

Usage::

    uv run experiments/ranking_comparison.py --config configs/ranking_comparison.yaml
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from activation_registry import compute_or_fetch_activations
from config import load_config
from delegation_value_probe import compute_delegation_value, train_dv_probe
from dotenv import load_dotenv
from dv_ltt_cascade import _load_split
from sklearn.metrics import accuracy_score, roc_auc_score

from reliable_monitoring.cascade import offline_batch_cascade
from reliable_monitoring.dataset import ActivationConfig, load_dataset
from reliable_monitoring.probes import SequenceProbe

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
    ranking_scores: np.ndarray,
    batch_size: int,
    k_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep over k values and compute AUC and accuracy for each.

    Uses ``offline_batch_cascade`` with the ``topk`` selection strategy,
    which ranks by ``ranking_scores`` within each batch and delegates the
    top-k to the baseline (replace strategy).

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
            merge_strategy="replace",
            amount=int(k),
            ranking_scores=ranking_scores,
        )
        aucs[i] = roc_auc_score(labels, result.final_scores)
        accs[i] = accuracy_score(labels, (result.final_scores >= 0.5).astype(int))

    return aucs, accs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {"Probe uncertainty": "C1", "DV probe": "C0", "Oracle": "C2"}
MARKERS = {"Probe uncertainty": "s", "DV probe": "o", "Oracle": "^"}
STYLES = {"Probe uncertainty": "-", "DV probe": "-", "Oracle": "--"}


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
) -> None:
    """Plot AUC and accuracy vs budget for one batch size on given axes."""
    for name, (aucs, accs) in signal_results.items():
        me = max(1, len(budget_fractions) // 10)
        ax_auc.plot(
            budget_fractions,
            aucs,
            label=name,
            color=COLORS[name],
            marker=MARKERS[name],
            linestyle=STYLES[name],
            markersize=4,
            markevery=me,
        )
        ax_acc.plot(
            budget_fractions,
            accs,
            label=name,
            color=COLORS[name],
            marker=MARKERS[name],
            linestyle=STYLES[name],
            markersize=4,
            markevery=me,
        )

    for ax, ref_probe, ref_base, ylabel in [
        (ax_auc, probe_auc, baseline_auc, "Cascade ROC AUC"),
        (ax_acc, probe_acc, baseline_acc, "Cascade Accuracy"),
    ]:
        ax.axhline(ref_probe, color="gray", ls=":", alpha=0.5, label=f"Probe only ({ref_probe:.3f})")
        ax.axhline(ref_base, color="gray", ls="--", alpha=0.5, label=f"Baseline only ({ref_base:.3f})")
        ax.set_xlabel("Budget fraction ($k / B$)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Batch size $B = {batch_size}$")
        ax.grid(alpha=0.3)
        if show_legend:
            ax.legend(fontsize=7, loc="lower right")


def plot_single_batch_size(
    budget_fractions: np.ndarray,
    signal_results: dict[str, tuple[np.ndarray, np.ndarray]],
    probe_auc: float,
    probe_acc: float,
    baseline_auc: float,
    baseline_acc: float,
    batch_size: int,
    output_dir: Path,
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
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"ranking_comparison_B{batch_size}.png", dpi=150, bbox_inches="tight")
    return fig


def plot_grid(
    all_results: dict[int, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    probe_auc: float,
    probe_acc: float,
    baseline_auc: float,
    baseline_acc: float,
    output_dir: Path,
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
        )
        # Only put y-label on leftmost column
        if j > 0:
            axes[0, j].set_ylabel("")
            axes[1, j].set_ylabel("")

    fig.tight_layout()
    fig.savefig(output_dir / "ranking_comparison_grid.png", dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Batched top-k ranking comparison")
    parser.add_argument("--config", type=str, default="configs/ranking_comparison.yaml")
    parser.add_argument("--output-dir", type=str, default="results/ranking_comparison")
    parser.add_argument("--use-clearml", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)

    # --- ClearML init ---
    clearml_logger = None
    if args.use_clearml:
        import os

        from clearml_logger import ClearMLLogger

        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="ranking_comparison",
            enabled=True,
        )
        clearml_logger.add_tags(["ranking-comparison"])

    activation_config = ActivationConfig(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
    )

    # --- Train safety probe ---
    use_modal = getattr(config, "use_modal", False)
    modal_gpu = getattr(config, "modal_gpu", None)
    reduction = config.reduction_strategy

    logger.info("Loading training data and fitting safety probe...")
    train_dataset = load_dataset(Path(config.train_dataset_path), activation_config=None)
    train_acts = compute_or_fetch_activations(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
        reduction=reduction,
        dataset=train_dataset,
        dataset_path=config.train_dataset_path,
        local=not use_modal,
        gpu=modal_gpu,
    )
    train_dataset = train_dataset.assign(**{f"activations_{reduction}": train_acts})
    safety_probe = SequenceProbe(reduction_strategy=reduction)
    safety_probe.fit(train_dataset)
    del train_dataset, train_acts

    # --- Load dev split and train DV probe ---
    logger.info("Loading dev split for DV probe training...")
    dev_ps, dev_bs, dev_labels, X_dev, dev_groups = _load_split(
        "dev",
        config,
        activation_config,
        safety_probe,
    )
    v_dev = compute_delegation_value(dev_ps, dev_bs, dev_labels)
    logger.info(f"Dev delegation value: v=1 rate={v_dev.mean():.1%} (n={len(v_dev)})")

    logger.info("Training DV probe on dev split...")
    dv_clf = train_dv_probe(X_dev, v_dev)
    del X_dev

    # --- Load test split ---
    logger.info("Loading test split...")
    test_ps, test_bs, test_labels, X_test, test_groups = _load_split(
        "test",
        config,
        activation_config,
        safety_probe,
    )
    v_test = compute_delegation_value(test_ps, test_bs, test_labels)
    logger.info(f"Test delegation value: v=1 rate={v_test.mean():.1%} (n={len(v_test)})")

    # DV scores (out-of-sample)
    dv_scores = dv_clf.predict_proba(X_test)[:, 1]
    del X_test

    dv_auc = roc_auc_score(v_test, dv_scores)
    logger.info(f"DV probe AUC on test: {dv_auc:.4f}")

    # --- Compute ranking signals ---
    uncertainty = np.minimum(test_ps, 1 - test_ps)
    oracle = v_test.astype(float)

    signals = {
        "Probe uncertainty": uncertainty,
        "DV probe": dv_scores,
        "Oracle": oracle,
    }

    # --- Reference metrics ---
    probe_auc = float(roc_auc_score(test_labels, test_ps))
    probe_acc = float(accuracy_score(test_labels, (test_ps >= 0.5).astype(int)))
    baseline_auc = float(roc_auc_score(test_labels, test_bs))
    baseline_acc = float(accuracy_score(test_labels, (test_bs >= 0.5).astype(int)))
    logger.info(f"Probe only:    AUC={probe_auc:.4f}, Acc={probe_acc:.4f}")
    logger.info(f"Baseline only: AUC={baseline_auc:.4f}, Acc={baseline_acc:.4f}")

    # --- Run batched top-k sweep for each batch size ---
    batch_sizes = getattr(config, "batch_sizes", [32, 64, 128])
    # Number of k steps to sweep (between 0 and batch_size)
    n_k_steps = getattr(config, "n_k_steps", 20)

    all_results: dict[int, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}

    for bs in batch_sizes:
        logger.info(f"\n--- Batch size B={bs} ---")
        # k values from 0 to bs (inclusive), with n_k_steps points
        k_values = np.unique(np.linspace(0, bs, n_k_steps + 1).astype(int))
        budget_fractions = k_values / bs

        signal_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, ranking in signals.items():
            aucs, accs = batched_topk_sweep(
                test_ps,
                test_bs,
                test_labels,
                ranking,
                bs,
                k_values,
            )
            signal_results[name] = (aucs, accs)

            # Log summary at a few budget levels
            for frac in [0.1, 0.2, 0.3, 0.5]:
                idx = np.argmin(np.abs(budget_fractions - frac))
                logger.info(
                    f"  {name:>18s} @ k/B={budget_fractions[idx]:.2f}: AUC={aucs[idx]:.4f}, Acc={accs[idx]:.4f}"
                )

        all_results[bs] = (budget_fractions, signal_results)

    # --- Generate plots ---
    logger.info("\nGenerating plots...")
    figs: dict[str, plt.Figure] = {}

    # Per-batch-size plots (appendix candidates)
    for bs in batch_sizes:
        budget_fractions, signal_results = all_results[bs]
        fig = plot_single_batch_size(
            budget_fractions,
            signal_results,
            probe_auc,
            probe_acc,
            baseline_auc,
            baseline_acc,
            bs,
            output_dir,
        )
        figs[f"B{bs}"] = fig

    # Grid plot (all batch sizes)
    fig_grid = plot_grid(
        all_results,
        probe_auc,
        probe_acc,
        baseline_auc,
        baseline_acc,
        output_dir,
    )
    figs["grid"] = fig_grid

    plt.close("all")
    logger.info(f"Plots saved to {output_dir}")

    # --- ClearML logging ---
    if clearml_logger is not None:
        scalars = {
            "dv_probe_auc": dv_auc,
            "probe_only_auc": probe_auc,
            "baseline_only_auc": baseline_auc,
        }
        # Log AUC advantage at 20% budget for each batch size
        for bs in batch_sizes:
            budget_fractions, signal_results = all_results[bs]
            idx_20 = np.argmin(np.abs(budget_fractions - 0.2))
            dv_auc_20 = signal_results["DV probe"][0][idx_20]
            unc_auc_20 = signal_results["Probe uncertainty"][0][idx_20]
            scalars[f"dv_advantage_auc_B{bs}_at_20pct"] = dv_auc_20 - unc_auc_20

        clearml_logger.log_scalars(scalars)
        for name, fig in figs.items():
            clearml_logger.log_figure("Ranking Comparison", name, fig)
        clearml_logger.finalize()
        logger.info("Results logged to ClearML.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
