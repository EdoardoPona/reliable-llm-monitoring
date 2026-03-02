"""Run the full paper pipeline: SGT cascade → analysis → fixed cascade → analysis → comparison.

Each step logs to ClearML and task IDs are piped between steps automatically.
All tasks in a pipeline run share a common datetime prefix for easy identification.

All results, scalars, configs, and figures are saved locally to ``--output-dir``
(default: ``results/pipeline/{run_prefix}/``), regardless of whether ClearML is enabled.

Usage::

    # Default config
    uv run experiments/run_cascade_experiments_pipeline.py --config configs/sgt_cascade.yaml

    # Skip ClearML logging
    uv run experiments/run_cascade_experiments_pipeline.py --config configs/sgt_cascade.yaml --no-clearml

    # Override fixed-cascade budget rate
    uv run experiments/run_cascade_experiments_pipeline.py --config configs/sgt_cascade.yaml --budget-rate 0.25

    # Custom output directory
    uv run experiments/run_cascade_experiments_pipeline.py --config configs/sgt_cascade.yaml --output-dir results/run1
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from analyse_cascade import analyse_comparison, analyse_single
from analyse_grouped_cascade import run_grouped_analysis
from cascade_utils import save_results_to_clearml
from clearml_logger import ClearMLLogger
from clearml_serialization import ClearMLSerializer
from config import load_config
from fixed_cascade import run_fixed_cascade
from sgt_cascade import SGTCascadeResults, run_sgt_cascade_experiment
from sgt_cascade_plotting import log_sgt_figures_to_clearml, make_sgt_figures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full paper pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SGT cascade config YAML.",
    )
    parser.add_argument(
        "--budget-rate",
        type=float,
        default=None,
        help="Budget rate for fixed cascade (default: match adaptive mean).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/pipeline",
        help="Root directory for results (default: results/pipeline).",
    )
    parser.add_argument(
        "--no-clearml",
        action="store_true",
        help="Disable ClearML logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Local saving helpers
# ---------------------------------------------------------------------------


def _save_figures_locally(figures: dict[str, Any], output_dir: Path) -> None:
    """Save a dict of matplotlib figures as PNGs (handles nested dicts)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        if fig is None:
            continue
        if isinstance(fig, dict):
            for sub_name, sub_fig in fig.items():
                if sub_fig is not None:
                    path = output_dir / f"{name}_{sub_name}.png"
                    sub_fig.savefig(path, bbox_inches="tight", dpi=150)
                    logger.info(f"Saved {path}")
        else:
            path = output_dir / f"{name}.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            logger.info(f"Saved {path}")


def _save_experiment_locally(results: Any, output_dir: Path) -> None:
    """Save results pickle, scalars JSON, and config JSON to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results pickle
    pkl_path = output_dir / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved {pkl_path}")

    # Scalars
    serializer = ClearMLSerializer()
    scalars = serializer.to_clearml_scalars(results)
    scalars = {k: float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v for k, v in scalars.items()}
    (output_dir / "scalars.json").write_text(json.dumps(scalars, indent=2, default=str))
    logger.info(f"Saved {output_dir / 'scalars.json'}")

    # Config
    config = getattr(results, "config", None)
    if config is not None:
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        logger.info(f"Saved {output_dir / 'config.json'}")


# ---------------------------------------------------------------------------
# ClearML helper
# ---------------------------------------------------------------------------


def _make_clearml_logger(run_prefix: str, step_name: str) -> ClearMLLogger:
    return ClearMLLogger(
        project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
        task_name=f"{run_prefix}/{step_name}",
        enabled=True,
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_sgt_cascade(
    config: dict[str, Any], output_dir: Path, run_prefix: str, use_clearml: bool
) -> tuple[SGTCascadeResults, str | None]:
    """Step 1: Run the SGT cascade experiment.

    Returns (results, task_id) — task_id is None when ClearML is disabled.
    """
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: SGT Cascade Experiment")
    logger.info("=" * 60)

    clearml_logger = _make_clearml_logger(run_prefix, "sgt_cascade") if use_clearml else None

    results = run_sgt_cascade_experiment(config)
    if results is None:
        logger.error("SGT experiment failed: no reliable (threshold, alpha) pair found.")
        if clearml_logger is not None:
            clearml_logger.add_tags(["pipeline", "failed"])
            clearml_logger.finalize()
        sys.exit(1)
    assert results is not None  # for type checker (sys.exit is NoReturn)

    # Generate figures
    figures = make_sgt_figures(results)

    # Save locally (always)
    step_dir = output_dir / "sgt_cascade"
    _save_experiment_locally(results, step_dir)
    _save_figures_locally(figures, step_dir / "figures")

    # Log to ClearML
    task_id = None
    if clearml_logger is not None:
        clearml_logger.connect_configuration(results.config)
        tags = [
            f"sgt-{results.sgt_graph_type}",
            f"guaranteed_risk-{results.guaranteed_risk_name}",
            f"achieved_alpha-{results.achieved_alpha:.3f}",
            f"rejected-{results.n_rejected}/{results.n_hypotheses}",
            f"probe-{results.config['reduction_strategy']}",
            f"merge-{results.config['cascade_merge_strategy']}",
            f"probe-degraded-{results.config.get('probe_degradation_enabled', False)}",
        ]
        calibration_method = results.config.get("calibration_method")
        tags.append(f"calibration-{calibration_method}" if calibration_method else "not-calibrated")
        if results.debug_mode:
            tags.append("debug")
        tags.append("pipeline")
        clearml_logger.add_tags(tags)

        serializer = ClearMLSerializer()
        scalars = serializer.to_clearml_scalars(results)
        scalars = {
            k: float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v for k, v in scalars.items()
        }
        clearml_logger.log_scalars(scalars)
        clearml_logger.log_artifacts(serializer.to_clearml_artifacts(results))
        save_results_to_clearml(clearml_logger, results)

        log_sgt_figures_to_clearml(clearml_logger, figures)

        assert clearml_logger.task is not None
        task_id = clearml_logger.task.id
        clearml_logger.finalize()
        logger.info(f"SGT task ID: {task_id}")

    plt.close("all")
    return results, task_id


def step_analyse_sgt(sgt_results: Any, output_dir: Path, run_prefix: str, use_clearml: bool) -> None:
    """Step 2: Analyse the SGT cascade results."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Analyse SGT Cascade")
    logger.info("=" * 60)

    clearml_logger = _make_clearml_logger(run_prefix, "analyse_sgt") if use_clearml else None
    if clearml_logger is not None:
        clearml_logger.add_tags(["pipeline", "analysis", "sgt"])

    analyse_single(sgt_results, output_dir / "analyse_sgt", label="Adaptive (SGT)", clearml_logger=clearml_logger)

    if clearml_logger is not None:
        clearml_logger.finalize()


def step_fixed_cascade(
    sgt_results: Any, budget_rate: float | None, output_dir: Path, run_prefix: str, use_clearml: bool
) -> tuple[object, str | None]:
    """Step 3: Run the fixed-budget cascade using SGT test data.

    Returns (results, task_id).
    """
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Fixed-Budget Cascade")
    logger.info("=" * 60)

    clearml_logger = _make_clearml_logger(run_prefix, "fixed_cascade") if use_clearml else None

    results = run_fixed_cascade(sgt_results, budget_rate=budget_rate)

    # Save locally (always)
    _save_experiment_locally(results, output_dir / "fixed_cascade")

    # Log to ClearML
    task_id = None
    if clearml_logger is not None:
        clearml_logger.connect_configuration(results.config)
        clearml_logger.add_tags(
            [
                f"fixed_budget_rate-{results.fixed_budget_rate:.3f}",
                "pipeline",
            ]
        )

        serializer = ClearMLSerializer()
        clearml_logger.log_scalars(serializer.to_clearml_scalars(results))
        clearml_logger.log_artifacts(serializer.to_clearml_artifacts(results))
        save_results_to_clearml(clearml_logger, results)

        assert clearml_logger.task is not None
        task_id = clearml_logger.task.id
        clearml_logger.finalize()
        logger.info(f"Fixed cascade task ID: {task_id}")

    return results, task_id


def step_analyse_fixed(fixed_results: Any, output_dir: Path, run_prefix: str, use_clearml: bool) -> None:
    """Step 4: Analyse the fixed cascade results."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Analyse Fixed Cascade")
    logger.info("=" * 60)

    clearml_logger = _make_clearml_logger(run_prefix, "analyse_fixed") if use_clearml else None
    if clearml_logger is not None:
        clearml_logger.add_tags(["pipeline", "analysis", "fixed"])

    analyse_single(fixed_results, output_dir / "analyse_fixed", label="Fixed", clearml_logger=clearml_logger)

    if clearml_logger is not None:
        clearml_logger.finalize()


def step_grouped_analysis(sgt_results: Any, output_dir: Path, run_prefix: str, use_clearml: bool) -> None:
    """Step 5: Group-stratified batching analysis on SGT results (mixed data only)."""
    test_groups = getattr(sgt_results, "test_groups", None)
    if test_groups is None:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Grouped Analysis — SKIPPED (no group labels)")
        logger.info("=" * 60)
        return

    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Grouped Analysis")
    logger.info("=" * 60)

    clearml_logger = _make_clearml_logger(run_prefix, "grouped_analysis") if use_clearml else None
    if clearml_logger is not None:
        clearml_logger.add_tags(["pipeline", "analysis", "grouped"])

    run_grouped_analysis(sgt_results, output_dir / "grouped", clearml_logger)

    if clearml_logger is not None:
        clearml_logger.finalize()


def step_comparison(sgt_results: Any, fixed_results: Any, output_dir: Path, run_prefix: str, use_clearml: bool) -> None:
    """Step 6: Compare SGT vs fixed cascade."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Comparison (SGT vs Fixed)")
    logger.info("=" * 60)

    clearml_logger = _make_clearml_logger(run_prefix, "comparison") if use_clearml else None
    if clearml_logger is not None:
        clearml_logger.add_tags(["pipeline", "analysis", "comparison"])

    analyse_comparison(
        sgt_results,
        fixed_results,
        output_dir / "comparison",
        label_a="Adaptive (SGT)",
        label_b="Fixed",
        clearml_logger=clearml_logger,
    )

    if clearml_logger is not None:
        clearml_logger.finalize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    config = load_config(args.config)
    use_clearml = not args.no_clearml

    run_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Each run gets its own subdirectory
    output_dir = Path(args.output_dir) / run_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"ClearML: {'enabled' if use_clearml else 'disabled'}")
    logger.info(f"Run prefix: {run_prefix}")

    # Step 1: SGT cascade
    sgt_results, sgt_task_id = step_sgt_cascade(config, output_dir, run_prefix, use_clearml)

    # Step 2: Analyse SGT
    step_analyse_sgt(sgt_results, output_dir, run_prefix, use_clearml)

    # Step 3: Fixed cascade (reuses SGT test data in-memory)
    fixed_results, fixed_task_id = step_fixed_cascade(
        sgt_results, args.budget_rate, output_dir, run_prefix, use_clearml
    )

    # Step 4: Analyse fixed
    step_analyse_fixed(fixed_results, output_dir, run_prefix, use_clearml)

    # Step 5: Grouped analysis (only when mixed data provides group labels)
    step_grouped_analysis(sgt_results, output_dir, run_prefix, use_clearml)

    # Step 6: Comparison
    step_comparison(sgt_results, fixed_results, output_dir, run_prefix, use_clearml)

    # Save run metadata
    metadata: dict[str, Any] = {
        "run_prefix": run_prefix,
        "config_path": str(args.config),
        "clearml_enabled": use_clearml,
    }
    if use_clearml:
        metadata["sgt_task_id"] = sgt_task_id
        metadata["fixed_task_id"] = fixed_task_id
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Run prefix: {run_prefix}")
    logger.info(f"Results saved to: {output_dir}")
    if use_clearml:
        logger.info(f"SGT task ID:      {sgt_task_id}")
        logger.info(f"Fixed task ID:    {fixed_task_id}")


if __name__ == "__main__":
    main()
