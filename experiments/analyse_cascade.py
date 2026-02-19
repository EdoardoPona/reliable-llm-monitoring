"""Analyse one or two cascade experiment results.

Usage::

    # Single experiment analysis
    python analyse_cascade.py --task-ids <id>

    # Compare two experiments
    python analyse_cascade.py --task-ids <id1> <id2> --labels "Adaptive (SGT)" "Fixed"

Figures are saved to ``--output-dir`` (default ``figures/``) and optionally
logged to a new ClearML task (``--use-clearml``).
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
from cascade_utils import load_results_from_clearml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse cascade experiment results")
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        required=True,
        help="One or two ClearML task IDs to analyse.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Display labels for the experiments (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save figures (default: figures/).",
    )
    parser.add_argument(
        "--use-clearml",
        action="store_true",
        help="Log figures to a new ClearML task.",
    )
    return parser.parse_args()


def _save_figure(fig, output_dir: Path, name: str) -> None:
    """Save a single figure as PDF."""
    path = output_dir / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    logger.info(f"Saved {path}")


def _save_figures(figures, output_dir: Path, clearml_logger=None, title_prefix: str = "") -> None:
    """Save a dict of figures (handles nested dicts)."""
    for name, fig in figures.items():
        if fig is None:
            continue
        if isinstance(fig, dict):
            for sub_name, sub_fig in fig.items():
                if sub_fig is not None:
                    _save_figure(sub_fig, output_dir, f"{name}_{sub_name}")
                    if clearml_logger is not None:
                        clearml_logger.log_figure(title=title_prefix + name, series=sub_name, figure=sub_fig)
        else:
            _save_figure(fig, output_dir, name)
            if clearml_logger is not None:
                clearml_logger.log_figure(title=title_prefix + "Analysis", series=name, figure=fig)


def _infer_label(results) -> str:
    """Infer a display label from a results object."""
    cls_name = type(results).__name__
    if "SGT" in cls_name:
        return "Adaptive (SGT)"
    if "GuaranteedRisk" in cls_name:
        return "Adaptive (GR)"
    if "Fixed" in cls_name:
        rate = getattr(results, "fixed_budget_rate", None)
        return f"Fixed ({rate:.2f})" if rate is not None else "Fixed"
    return cls_name


def analyse_single(results, output_dir: Path, label: str, clearml_logger=None) -> None:
    """Single-experiment analysis using guaranteed_risk_cascade_plotting functions."""
    from guaranteed_risk_cascade_plotting import (
        plot_batch_distributions,
        plot_batch_metric_boxplots,
        plot_batch_uncertainty_vs_metrics,
        plot_cascade_vs_probe,
        plot_overall_performance,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "overall_performance": plot_overall_performance(results),
        "batch_distributions": plot_batch_distributions(results),
        "uncertainty_vs_metrics": plot_batch_uncertainty_vs_metrics(results),
        "batch_boxplots": plot_batch_metric_boxplots(results),
        "cascade_vs_probe": plot_cascade_vs_probe(results),
    }

    _save_figures(figures, output_dir, clearml_logger, title_prefix=f"{label} / ")

    # Print summary
    logger.info(f"\n=== {label} Summary ===")
    logger.info(
        f"  Cascade:  Acc={results.cascade_accuracy:.4f}  F1={results.cascade_f1_score:.4f}  "
        f"ROC-AUC={results.cascade_roc_auc:.4f}"
    )
    logger.info(
        f"  Probe:    Acc={results.probe_only_accuracy:.4f}  F1={results.probe_only_f1_score:.4f}  "
        f"ROC-AUC={results.probe_only_roc_auc:.4f}"
    )
    logger.info(
        f"  Baseline: Acc={results.baseline_only_accuracy:.4f}  F1={results.baseline_only_f1_score:.4f}  "
        f"ROC-AUC={results.baseline_only_roc_auc:.4f}"
    )
    logger.info(f"  Budget:   mean={results.mean_budget_cost:.4f}")

    plt.close("all")


def analyse_comparison(results_a, results_b, output_dir: Path, label_a: str, label_b: str, clearml_logger=None) -> None:
    """Two-experiment comparison using comparison_plotting functions."""
    from comparison_plotting import (
        plot_comparison_batch_distributions,
        plot_comparison_boxplots,
        plot_comparison_overall,
        plot_comparison_paired_scatter,
        plot_comparison_uncertainty_vs_metrics,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "comparison_overall": plot_comparison_overall(results_a, results_b, label_a, label_b),
        "comparison_distributions": plot_comparison_batch_distributions(results_a, results_b, label_a, label_b),
        "comparison_boxplots": plot_comparison_boxplots(results_a, results_b, label_a, label_b),
        "comparison_uncertainty": plot_comparison_uncertainty_vs_metrics(results_a, results_b, label_a, label_b),
        "comparison_paired": plot_comparison_paired_scatter(results_a, results_b, label_a, label_b),
    }

    _save_figures(figures, output_dir, clearml_logger, title_prefix="Comparison / ")

    # Print comparison summary
    logger.info(f"\n=== Comparison: {label_a} vs {label_b} ===")
    logger.info(
        f"  {label_a:>20s}:  Acc={results_a.cascade_accuracy:.4f}  F1={results_a.cascade_f1_score:.4f}  "
        f"ROC-AUC={results_a.cascade_roc_auc:.4f}  Budget={results_a.mean_budget_cost:.4f}"
    )
    logger.info(
        f"  {label_b:>20s}:  Acc={results_b.cascade_accuracy:.4f}  F1={results_b.cascade_f1_score:.4f}  "
        f"ROC-AUC={results_b.cascade_roc_auc:.4f}  Budget={results_b.mean_budget_cost:.4f}"
    )

    plt.close("all")


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Load results
    task_ids = args.task_ids
    if len(task_ids) > 2:
        raise ValueError("At most two task IDs can be provided.")

    logger.info(f"Loading results from ClearML task(s): {task_ids}")
    results_list = [load_results_from_clearml(tid) for tid in task_ids]

    # Determine labels
    if args.labels is not None:
        labels = args.labels
        if len(labels) != len(results_list):
            raise ValueError(f"Expected {len(results_list)} labels, got {len(labels)}.")
    else:
        labels = [_infer_label(r) for r in results_list]

    # ClearML logger
    clearml_logger = None
    if args.use_clearml:
        from clearml_logger import ClearMLLogger

        task_name = "analyse_" + "_vs_".join(task_ids)
        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name=task_name,
            enabled=True,
        )

    if len(results_list) == 1:
        analyse_single(results_list[0], output_dir, labels[0], clearml_logger)
    else:
        analyse_comparison(results_list[0], results_list[1], output_dir, labels[0], labels[1], clearml_logger)

    if clearml_logger is not None:
        clearml_logger.finalize()

    logger.info("Analysis complete!")
