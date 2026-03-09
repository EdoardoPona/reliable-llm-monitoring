"""Compute probe and baseline scores for all data splits.

This script handles the expensive work: data loading, probe training, probe
inference, and LLM baseline inference.  The result is a ``ScoreArtifact``
pickle that downstream experiment scripts can consume cheaply.

Baseline scores are cached per ``(model, dataset_file)`` pair on ClearML via
the :mod:`baseline_registry`.  Subsequent runs with the same model and dataset
fetch baseline scores from the cache instantly.

Usage::

    uv run experiments/compute_scores.py --config configs/sgt_cascade.yaml
    uv run experiments/compute_scores.py --config configs/sgt_cascade.yaml --output scores/my_run.pkl
    uv run experiments/compute_scores.py --config configs/sgt_cascade.yaml --no-baseline
    uv run experiments/compute_scores.py --config configs/sgt_cascade.yaml --no-cache
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from baseline_registry import compute_or_fetch_baseline
from config import load_config
from dotenv import load_dotenv
from mixed_dataset import (
    fetch_per_source_baselines,
    get_mixed_splits,
    has_mixed_config,
    load_mixed_dataset,
    load_mixed_dataset_with_baselines,
)
from score_artifact import ScoreArtifact, make_score_artifact, save_score_artifact

from reliable_monitoring.dataset import ActivationConfig, load_dataset, sample_from_dataset
from reliable_monitoring.probes import DegradedProbe, SequenceProbe

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_SAMPLE_SIZE = 256


def _baseline_kwargs(config) -> dict:
    """Extract common baseline inference kwargs from config."""
    return {
        "baseline_batch_size": config.baseline_batch_size,
        "local": not getattr(config, "use_modal", False),
        "gpu": getattr(config, "modal_gpu", None),
    }


def _debug_subsample(dataset, baselines: np.ndarray | None, size: int, seed: int) -> tuple:
    """Subsample both dataset and baselines with the same indices."""
    if len(dataset) <= size:
        return dataset, baselines
    random.seed(seed)
    indices = random.sample(range(len(dataset)), size)
    ds_sub = dataset[indices]
    bl_sub = baselines[indices] if baselines is not None else None
    return ds_sub, bl_sub


def compute_scores(
    config,
    *,
    skip_baseline: bool = False,
    skip_cache: bool = False,
) -> ScoreArtifact:
    """Train a probe and compute scores on all splits.

    This is the expensive step of the pipeline: it loads datasets, trains
    a probe, and runs LLM baseline inference.  The returned
    ``ScoreArtifact`` contains raw (uncalibrated) probe scores and baseline
    scores for all splits.

    Baseline scores are cached per ``(model, dataset_file)`` pair on ClearML.
    Set ``skip_cache=True`` to bypass the cache.

    Args:
        config: Experiment configuration (from ``load_config``).
        skip_baseline: If True, skip baseline LLM inference (useful for
            fast iteration when only probe scores are needed).
        skip_cache: If True, skip the ClearML baseline cache (always
            compute from scratch, do not upload).

    Returns:
        ScoreArtifact with uncalibrated scores.
    """
    seed = config.seed
    np.random.seed(seed)

    activation_config = ActivationConfig(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
    )
    bl_kwargs = _baseline_kwargs(config)

    # --- Load data + baselines ---
    logger.info("Loading datasets...")
    train_dataset = load_dataset(Path(config.train_dataset_path), activation_config=activation_config)

    calib_baseline_scores: np.ndarray
    test_baseline_scores: np.ndarray

    if has_mixed_config(config):
        mixed_cfg = config.mixed_datasets
        sources = mixed_cfg["sources"]
        balance = mixed_cfg.get("balance_strategy", "min_size")
        mixed_splits = get_mixed_splits(config)

        # --- Mixed test split (always mixed when mixed config is present) ---
        if skip_baseline:
            test_dataset = load_mixed_dataset(sources, "test", activation_config, balance, seed)
            test_baseline_scores = np.full(len(test_dataset), np.nan)
        else:
            logger.info("Fetching per-source baselines for MIXED test split")
            test_baselines = fetch_per_source_baselines(
                sources,
                "test",
                config.baseline_model_name,
                skip_cache=skip_cache,
                **bl_kwargs,
            )
            test_dataset, test_baseline_scores = load_mixed_dataset_with_baselines(
                sources,
                "test",
                test_baselines,
                activation_config,
                balance,
                seed,
            )

        # --- Calib split (mixed or single-source) ---
        if "calib" in mixed_splits:
            if skip_baseline:
                calib_dataset = load_mixed_dataset(sources, "dev", activation_config, balance, seed)
                calib_baseline_scores = np.full(len(calib_dataset), np.nan)
            else:
                logger.info("Fetching per-source baselines for MIXED calib split")
                calib_baselines = fetch_per_source_baselines(
                    sources,
                    "dev",
                    config.baseline_model_name,
                    skip_cache=skip_cache,
                    **bl_kwargs,
                )
                calib_dataset, calib_baseline_scores = load_mixed_dataset_with_baselines(
                    sources,
                    "dev",
                    calib_baselines,
                    activation_config,
                    balance,
                    seed,
                )
        else:
            logger.info("Loading single-source calibration dataset")
            calib_dataset = load_dataset(Path(config.calib_dataset_path), activation_config=activation_config)
            if skip_baseline:
                calib_baseline_scores = np.full(len(calib_dataset), np.nan)
            else:
                calib_baseline_scores = compute_or_fetch_baseline(
                    model_name=config.baseline_model_name,
                    dataset=calib_dataset,
                    dataset_path=config.calib_dataset_path,
                    skip_cache=skip_cache,
                    **bl_kwargs,
                )
    else:
        # --- Single-source path ---
        calib_dataset = load_dataset(Path(config.calib_dataset_path), activation_config=activation_config)
        test_dataset = load_dataset(Path(config.test_dataset_path), activation_config=activation_config)

        if skip_baseline:
            logger.info("Skipping baseline inference (--no-baseline flag).")
            calib_baseline_scores = np.full(len(calib_dataset), np.nan)
            test_baseline_scores = np.full(len(test_dataset), np.nan)
        else:
            calib_baseline_scores = compute_or_fetch_baseline(
                model_name=config.baseline_model_name,
                dataset=calib_dataset,
                dataset_path=config.calib_dataset_path,
                skip_cache=skip_cache,
                **bl_kwargs,
            )
            test_baseline_scores = compute_or_fetch_baseline(
                model_name=config.baseline_model_name,
                dataset=test_dataset,
                dataset_path=config.test_dataset_path,
                skip_cache=skip_cache,
                **bl_kwargs,
            )

    # --- Debug subsampling ---
    debug_mode = getattr(config, "debug", False)
    if debug_mode:
        logger.warning("Running in debug mode with smaller datasets.")
        train_dataset = sample_from_dataset(train_dataset, DEBUG_SAMPLE_SIZE, seed=seed)
        calib_dataset, calib_baseline_scores = _debug_subsample(
            calib_dataset,
            calib_baseline_scores,
            DEBUG_SAMPLE_SIZE,
            seed,
        )
        test_dataset, test_baseline_scores = _debug_subsample(
            test_dataset,
            test_baseline_scores,
            DEBUG_SAMPLE_SIZE,
            seed,
        )

    # Extract group labels for mixed datasets
    train_groups = np.array(train_dataset.other_fields["group"]) if "group" in train_dataset.other_fields else None
    calib_groups = np.array(calib_dataset.other_fields["group"]) if "group" in calib_dataset.other_fields else None
    test_groups = np.array(test_dataset.other_fields["group"]) if "group" in test_dataset.other_fields else None

    if test_groups is not None:
        unique_groups, group_counts = np.unique(test_groups, return_counts=True)
        logger.info(f"Test dataset groups: {dict(zip(unique_groups, group_counts, strict=True))}")

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Calibration dataset size: {len(calib_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # --- Train probe ---
    degrade_enabled = getattr(config, "probe_degradation_enabled", False)
    if degrade_enabled:
        logger.warning("Probe degradation enabled (fixed settings).")

    logger.info("Fitting probe...")
    base_probe = SequenceProbe(reduction_strategy=config.reduction_strategy)
    probe = DegradedProbe(base_probe, enabled=degrade_enabled, seed=seed)
    probe.fit(train_dataset)

    # --- Compute probe scores (uncalibrated) ---
    logger.info("Computing probe scores on training dataset...")
    train_probe_scores = probe.predict(train_dataset)
    train_labels = train_dataset.labels_numpy()

    logger.info("Computing probe scores on calibration dataset...")
    calib_probe_scores = probe.predict(calib_dataset)
    calib_labels = calib_dataset.labels_numpy()

    logger.info("Computing probe scores on test dataset...")
    test_probe_scores = probe.predict(test_dataset)
    test_labels = test_dataset.labels_numpy()

    # --- Build artifact ---
    artifact = make_score_artifact(
        train_probe_scores=train_probe_scores,
        train_labels=train_labels,
        calib_probe_scores=calib_probe_scores,
        calib_baseline_scores=calib_baseline_scores,
        calib_labels=calib_labels,
        test_probe_scores=test_probe_scores,
        test_baseline_scores=test_baseline_scores,
        test_labels=test_labels,
        config=vars(config),
        seed=seed,
        train_groups=train_groups,
        calib_groups=calib_groups,
        test_groups=test_groups,
    )

    logger.info("Score computation complete.")
    logger.info(
        f"  Train: {len(train_labels)} examples, Calib: {len(calib_labels)} examples, Test: {len(test_labels)} examples"
    )

    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description="Compute probe and baseline scores")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for score artifact pickle. Default: results/scores/<timestamp>.pkl",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline LLM inference (scores will be NaN).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip the ClearML baseline cache (always compute, do not upload).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from datetime import datetime

    args = parse_args()
    config = load_config(args.config)

    artifact = compute_scores(config, skip_baseline=args.no_baseline, skip_cache=args.no_cache)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / "scores" / f"{timestamp}.pkl"

    save_score_artifact(artifact, output_path)
    logger.info(f"Score artifact saved to {output_path}")
