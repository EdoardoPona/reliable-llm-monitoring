"""Probe difficulty check: evaluate one probe across multiple eval datasets.

Trains a single probe on synthetic training data and evaluates it on each
evaluation dataset (anthropic, mt, mts, toolace). Produces a cross-dataset
comparison plot showing how probe performance varies by dataset source.

This validates the assumption that different datasets have meaningfully
different difficulty levels for the probe, which is required for
group-stratified batching to benefit the adaptive cascade method.

Usage::

    python probe_difficulty_check.py [--use-clearml] [--debug]
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from clearml_serialization import artifact_field, scalar_field
from config import expand_env_vars
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from reliable_monitoring.cascade import probe_uncertainty
from reliable_monitoring.dataset import ActivationConfig, load_dataset, sample_from_dataset
from reliable_monitoring.probes import DegradedProbe, SequenceProbe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_SAMPLE_SIZE = 128

# Datasets to evaluate (balanced versions only)
EVAL_DATASETS = expand_env_vars(
    {
        "anthropic": {
            "dev": "${DATA_DIR}/evals/dev/anthropic_balanced_apr_23.jsonl",
            "test": "${DATA_DIR}/evals/test/anthropic_test_balanced_apr_23.jsonl",
        },
        "mt": {
            "dev": "${DATA_DIR}/evals/dev/mt_balanced_apr_30.jsonl",
            "test": "${DATA_DIR}/evals/test/mt_test_balanced_apr_30.jsonl",
        },
        "mts": {
            "dev": "${DATA_DIR}/evals/dev/mts_balanced_apr_22.jsonl",
            "test": "${DATA_DIR}/evals/test/mts_test_balanced_apr_22.jsonl",
        },
        "toolace": {
            "dev": "${DATA_DIR}/evals/dev/toolace_balanced_apr_22.jsonl",
            "test": "${DATA_DIR}/evals/test/toolace_test_balanced_apr_22.jsonl",
        },
    }
)

TRAIN_DATASET_PATH = expand_env_vars("${DATA_DIR}/training/prompts_4x/train.jsonl")


@dataclass
class DatasetMetrics:
    """Metrics for a single dataset evaluation."""

    name: str
    split: str
    size: int
    accuracy: float
    f1_score: float
    roc_auc: float
    mean_uncertainty: float
    scores: np.ndarray
    labels: np.ndarray


@dataclass
class DifficultyCheckResults:
    """Results from the cross-dataset difficulty check."""

    seed: int = scalar_field()
    debug_mode: bool = scalar_field()
    model_name: str = scalar_field()
    layer: int = scalar_field()
    reduction_strategy: str = scalar_field()
    train_size: int = scalar_field()
    dataset_names: list[str] = artifact_field()
    per_dataset: dict[str, dict[str, DatasetMetrics]] = artifact_field()


def run_difficulty_check(
    seed: int = 42,
    debug: bool = False,
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    layer: int = 11,
    reduction_strategy: str = "mean",
) -> DifficultyCheckResults:
    """Train one probe, evaluate on all datasets, return comparative results."""
    import torch

    np.random.seed(seed)

    activation_config = ActivationConfig(model_name=model_name, layer=layer)

    # Load model once for all datasets
    from models_under_pressure.model import LLMModel

    logger.info(f"Loading model {model_name}...")
    shared_model = LLMModel.load(model_name, batch_size=32)

    load_kwargs: dict[str, object] = {
        "auto_compute": True,
        "cleanup_after_load": True,
        "model": shared_model,
        "compute_batch_size": 32,
    }

    # Train probe
    logger.info("Loading training dataset...")
    train_dataset = load_dataset(Path(TRAIN_DATASET_PATH), activation_config, **load_kwargs)  # type: ignore[arg-type]
    if debug:
        train_dataset = sample_from_dataset(train_dataset, DEBUG_SAMPLE_SIZE, seed=seed)

    logger.info(f"Training probe on {len(train_dataset)} examples...")
    base_probe = SequenceProbe(reduction_strategy=reduction_strategy)
    probe = DegradedProbe(base_probe, enabled=False, seed=seed)
    probe.fit(train_dataset)

    # Evaluate on each dataset
    per_dataset: dict[str, dict[str, DatasetMetrics]] = {}

    for dataset_name, paths in EVAL_DATASETS.items():
        per_dataset[dataset_name] = {}

        for split, path in paths.items():
            logger.info(f"Evaluating on {dataset_name}/{split}...")
            ds = load_dataset(Path(path), activation_config, **load_kwargs)  # type: ignore[arg-type]
            if debug:
                ds = sample_from_dataset(ds, min(DEBUG_SAMPLE_SIZE, len(ds)), seed=seed)

            scores = probe.predict(ds)
            labels = ds.labels_numpy()
            labels_binary = (labels > 0).astype(int)

            predictions = (scores >= 0.5).astype(int)
            uncertainty = probe_uncertainty(scores)

            metrics = DatasetMetrics(
                name=dataset_name,
                split=split,
                size=len(ds),
                accuracy=float(accuracy_score(labels_binary, predictions)),
                f1_score=float(f1_score(labels_binary, predictions, pos_label=1)),
                roc_auc=float(roc_auc_score(labels_binary, scores)),
                mean_uncertainty=float(uncertainty.mean()),
                scores=scores,
                labels=labels_binary,
            )
            per_dataset[dataset_name][split] = metrics

            logger.info(
                f"  {dataset_name}/{split} (n={metrics.size}): "
                f"Acc={metrics.accuracy:.4f}, F1={metrics.f1_score:.4f}, "
                f"ROC-AUC={metrics.roc_auc:.4f}, mean_unc={metrics.mean_uncertainty:.4f}"
            )

    # Clean up
    del shared_model
    gc.collect()
    torch.cuda.empty_cache()

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-DATASET DIFFICULTY COMPARISON (test splits)")
    logger.info("=" * 80)
    logger.info(f"{'Dataset':<12} {'Size':>6} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10} {'Mean Unc':>10}")
    logger.info("-" * 60)
    for name in EVAL_DATASETS:
        m = per_dataset[name]["test"]
        logger.info(
            f"{name:<12} {m.size:>6} {m.accuracy:>10.4f} {m.f1_score:>10.4f} {m.roc_auc:>10.4f} {m.mean_uncertainty:>10.4f}"
        )
    logger.info("=" * 80)

    return DifficultyCheckResults(
        seed=seed,
        debug_mode=debug,
        model_name=model_name,
        layer=layer,
        reduction_strategy=reduction_strategy,
        train_size=len(train_dataset),
        dataset_names=list(EVAL_DATASETS.keys()),
        per_dataset=per_dataset,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_performance_comparison(results: DifficultyCheckResults) -> Figure:
    """Bar chart comparing accuracy, F1, and ROC-AUC across datasets."""
    names = results.dataset_names
    n = len(names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Probe Performance Across Datasets\n(trained on synthetic data, evaluated on each dataset)",
        fontsize=14,
        fontweight="bold",
    )

    metrics_info = [("accuracy", "Accuracy"), ("f1_score", "F1 Score"), ("roc_auc", "ROC-AUC")]
    cmap = plt.colormaps["Set2"]
    colors = cmap(np.linspace(0, 1, n))

    for ax, (metric, ylabel) in zip(axes, metrics_info, strict=False):
        dev_vals = [getattr(results.per_dataset[name]["dev"], metric) for name in names]
        test_vals = [getattr(results.per_dataset[name]["test"], metric) for name in names]

        x = np.arange(n)
        width = 0.35
        ax.bar(x - width / 2, dev_vals, width, label="Dev", color=colors, alpha=0.6, edgecolor="black")
        bars_test = ax.bar(x + width / 2, test_vals, width, label="Test", color=colors, alpha=1.0, edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Annotate test values
        for bar, val in zip(bars_test, test_vals, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        # Show range
        all_vals = dev_vals + test_vals
        spread = max(all_vals) - min(all_vals)
        ax.set_title(f"Spread: {spread:.3f}", fontsize=10)

    plt.tight_layout()
    return fig


def plot_uncertainty_distributions(results: DifficultyCheckResults) -> Figure:
    """Histogram of probe uncertainty per dataset (test split)."""
    names = results.dataset_names
    n = len(names)
    cmap = plt.colormaps["Set2"]
    colors = cmap(np.linspace(0, 1, n))

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    fig.suptitle("Probe Uncertainty Distribution by Dataset (test split)", fontsize=14, fontweight="bold")

    for ax, name, color in zip(axes, names, colors, strict=False):
        m = results.per_dataset[name]["test"]
        uncertainty = probe_uncertainty(m.scores)

        ax.hist(uncertainty, bins=30, alpha=0.8, color=color, edgecolor="black")
        ax.axvline(m.mean_uncertainty, color="red", linestyle="--", linewidth=2, label=f"mean={m.mean_uncertainty:.3f}")
        ax.set_xlabel("Uncertainty (min(p, 1-p))")
        if ax == axes[0]:
            ax.set_ylabel("Count")
        ax.set_title(f"{name}\nAcc={m.accuracy:.3f}, AUC={m.roc_auc:.3f}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_score_distributions(results: DifficultyCheckResults) -> Figure:
    """Score distribution by label for each dataset (test split)."""
    names = results.dataset_names
    n = len(names)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    fig.suptitle("Score Distribution by Label (test split)", fontsize=14, fontweight="bold")

    for ax, name in zip(axes, names, strict=False):
        m = results.per_dataset[name]["test"]
        pos_scores = m.scores[m.labels == 1]
        neg_scores = m.scores[m.labels == 0]

        ax.hist(neg_scores, bins=30, alpha=0.6, color="steelblue", label=f"Low-stakes (n={len(neg_scores)})")
        ax.hist(pos_scores, bins=30, alpha=0.6, color="coral", label=f"High-stakes (n={len(pos_scores)})")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Probe score")
        if ax == axes[0]:
            ax.set_ylabel("Count")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset probe difficulty check")
    parser.add_argument("--use-clearml", action="store_true", help="Log to ClearML")
    parser.add_argument("--debug", action="store_true", help="Use small sample sizes")
    parser.add_argument("--output-dir", type=str, default="figures/difficulty_check", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_difficulty_check(debug=args.debug)

    # Generate figures
    figures = {
        "performance_comparison": plot_performance_comparison(results),
        "uncertainty_distributions": plot_uncertainty_distributions(results),
        "score_distributions": plot_score_distributions(results),
    }

    for name, fig in figures.items():
        path = output_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved {path}")

    if args.use_clearml:
        from clearml_logger import ClearMLLogger
        from clearml_serialization import ClearMLSerializer

        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="probe_difficulty_check",
            enabled=True,
        )
        clearml_logger.add_tags(["difficulty-check", "cross-dataset"])

        serializer = ClearMLSerializer()
        clearml_logger.log_scalars(serializer.to_clearml_scalars(results))

        for name, fig in figures.items():
            clearml_logger.log_figure(title="Difficulty Check", series=name, figure=fig)

        clearml_logger.finalize()

    plt.close("all")
    logger.info("Difficulty check complete!")
