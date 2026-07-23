"""Sweep over experimental dimensions for the DV cascade comparison.

Runs ``run_dv_cascade_experiment`` for each configuration in the cartesian
product of BASE_CONFIG and SWEEP_CONFIG, then collects a summary JSON.

Edit BASE_CONFIG and SWEEP_CONFIG to define fixed values and swept values.

Usage::

    uv run experiments/dv_cascade_comparison_sweep.py
    uv run experiments/dv_cascade_comparison_sweep.py --use-clearml
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from config import SweepRun, build_sweep_configs
from dv_cascade_comparison import DVCascadeExperimentResults, run_dv_cascade_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------

# Fixed values shared across all runs.  Must match the config used for the
# paper's main results (configs/dv_cascade/dv_cascade_comparison_continuous_strong_acc.yaml).
BASE_CONFIG: dict = {
    "dv_target": "continuous",
    "batch_sizes": [32, 64, 128],
    "n_k_steps": 20,
    "calib_fraction": 0.5,
    "guarantee_probability": 0.9,
    "tau_steps": 100,
    "n_alpha_steps": 20,
    "pareto_testing": True,
    "pareto_split_proportion": 0.3,
    "guaranteed_risk": "budget",
    "opt_risk": "accuracy_error",
    "merge_strategy": "replace",
    "reduction_strategy": "mean",
    "use_modal": True,
    "seed": 42,
    "train_dataset_path": "${DATA_DIR}/training/prompts_4x/train.jsonl",
    "dev_dataset_path": "${DATA_DIR}/evals/dev/anthropic_balanced_apr_23.jsonl",
    "test_dataset_path": "${DATA_DIR}/evals/test/anthropic_test_balanced_apr_23.jsonl",
    "mixed_datasets": {
        "balance_strategy": 500,
        "sources": [
            {
                "group": "anthropic",
                "dev": "${DATA_DIR}/evals/dev/anthropic_balanced_apr_23.jsonl",
                "test": "${DATA_DIR}/evals/test/anthropic_test_balanced_apr_23.jsonl",
            },
            {
                "group": "mt",
                "dev": "${DATA_DIR}/evals/dev/mt_balanced_apr_30.jsonl",
                "test": "${DATA_DIR}/evals/test/mt_test_balanced_apr_30.jsonl",
            },
            {
                "group": "mts",
                "dev": "${DATA_DIR}/evals/dev/mts_balanced_apr_22.jsonl",
                "test": "${DATA_DIR}/evals/test/mts_test_balanced_apr_22.jsonl",
            },
            {
                "group": "toolace",
                "dev": "${DATA_DIR}/evals/dev/toolace_balanced_apr_22.jsonl",
                "test": "${DATA_DIR}/evals/test/toolace_test_balanced_apr_22.jsonl",
            },
        ],
    },
}

# Values to sweep over (cartesian product).
# Tuple keys are paired (zipped, not crossed) — see build_sweep_configs.
SWEEP_CONFIG: dict = {
    ("activations_model_name", "activations_layer"): [
        ("meta-llama/Llama-3.3-70B-Instruct", 31),
    ],
    "baseline_model_name": [
        "google/gemma-3-27b-it",
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
}


# ---------------------------------------------------------------------------
# Summary collection
# ---------------------------------------------------------------------------

SUMMARY_BUDGET_FRACS = [0.1, 0.2, 0.3, 0.5]


def _extract_summary_row(label: str, config, results: DVCascadeExperimentResults) -> dict:
    """Extract a single summary row from experiment results."""
    row: dict = {
        "label": label,
        "reduction_strategy": getattr(config, "reduction_strategy", None),
        "baseline_model_name": getattr(config, "baseline_model_name", None),
        "activations_model_name": getattr(config, "activations_model_name", None),
        "activations_layer": getattr(config, "activations_layer", None),
        "probe_auc": results.probe_auc,
        "probe_acc": results.probe_acc,
        "baseline_auc": results.baseline_auc,
        "baseline_acc": results.baseline_acc,
        "dv_auc": results.dv_auc,
    }

    # FAIC per method
    for method, scores in results.faic.items():
        safe_name = method.replace(" ", "_").replace(".", "").lower()
        row[f"faic_auc_{safe_name}"] = scores["auc"]
        row[f"faic_acc_{safe_name}"] = scores["acc"]

    # CTD metrics at representative budget levels
    ctd_rows = results.ltt_sweep_results.get("CTD", [])
    for r in ctd_rows:
        alpha = r["alpha"]
        for target in SUMMARY_BUDGET_FRACS:
            if abs(alpha - target) < 0.02:
                row[f"ctd_auc_at_{int(target * 100)}pct"] = r["auc"]
                row[f"ctd_acc_at_{int(target * 100)}pct"] = r["accuracy"]

    return row


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def run_sweep(sweep_runs: list[SweepRun], output_dir: Path, use_clearml: bool) -> None:
    summary_rows: list[dict] = []

    for run in sweep_runs:
        logger.info("\n" + "=" * 60)
        logger.info(f"Sweep run {run.index}: {run.label}")
        logger.info("=" * 60)

        run_dir = output_dir / run.label.replace(",", "_").replace("/", "-").replace(" ", "")
        run_dir.mkdir(parents=True, exist_ok=True)

        clearml_logger = None
        if use_clearml:
            from clearml_logger import ClearMLLogger

            clearml_logger = ClearMLLogger(
                project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
                task_name=f"dv_cascade_sweep_{run.label}",
                enabled=True,
            )

        results = run_dv_cascade_experiment(run.config, run_dir)
        summary_rows.append(_extract_summary_row(run.label, run.config, results))

        if clearml_logger is not None:
            scalars: dict[str, float] = {
                "probe_auc": results.probe_auc,
                "dv_auc": results.dv_auc,
                "baseline_auc": results.baseline_auc,
            }
            for method, scores in results.faic.items():
                scalars[f"faic_auc/{method}"] = scores["auc"]
                scalars[f"faic_acc/{method}"] = scores["acc"]
            clearml_logger.log_scalars(scalars)
            for name, fig in results.figs.items():
                clearml_logger.log_figure("DV Cascade Sweep", name, fig)
            clearml_logger.finalize()

        # Free memory between runs
        plt.close("all")
        gc.collect()

    # --- Summary ---
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    logger.info(f"Summary saved to {summary_path}")

    logger.info("All sweep runs complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DV cascade comparison sweep")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.environ.get("RESULTS_DIR", "results"), "dv_cascade_sweep"),
    )
    parser.add_argument("--use-clearml", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = build_sweep_configs(BASE_CONFIG, SWEEP_CONFIG)
    logger.info(f"Sweep: {len(runs)} configurations")
    for r in runs:
        logger.info(f"  [{r.index}] {r.label}")

    run_sweep(runs, output_dir, use_clearml=args.use_clearml)
