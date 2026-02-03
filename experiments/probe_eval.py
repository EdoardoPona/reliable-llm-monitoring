"""
Probe evaluation experiment.

This experiment:
1. Trains a probe on a training dataset
2. Evaluates on dev and test datasets
3. Computes performance and calibration metrics
4. Generates diagnostic plots
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from clearml_serialization import artifact_field, scalar_field
from config import load_config
from dotenv import load_dotenv
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from reliable_monitoring.dataset import ActivationConfig, load_dataset, sample_from_dataset, split_dataset
from reliable_monitoring.probes import SequenceProbe

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_SAMPLE_SIZE = 128


@dataclass
class ProbeEvalResults:
    """Results from probe evaluation experiment."""

    # Experiment metadata
    config: dict = artifact_field()
    seed: int = scalar_field()
    debug_mode: bool = scalar_field()

    # Dataset info
    train_dataset_path: str = scalar_field()
    dev_dataset_path: str = scalar_field()
    test_dataset_path: str = scalar_field()
    train_size: int = scalar_field()
    dev_size: int = scalar_field()
    test_size: int = scalar_field()

    # Model config
    reduction_strategy: str = scalar_field()
    model_name: str = scalar_field()
    layer: int = scalar_field()

    # Performance metrics - dev
    dev_accuracy: float = scalar_field()
    dev_f1_score: float = scalar_field()
    dev_roc_auc: float = scalar_field()

    # Performance metrics - test
    test_accuracy: float = scalar_field()
    test_f1_score: float = scalar_field()
    test_roc_auc: float = scalar_field()

    # Calibration metrics
    train_ece: float = scalar_field()
    dev_ece: float = scalar_field()
    test_ece: float = scalar_field()

    # Score distributions
    train_scores: np.ndarray = artifact_field()
    train_labels: np.ndarray = artifact_field()
    dev_scores: np.ndarray = artifact_field()
    dev_labels: np.ndarray = artifact_field()
    test_scores: np.ndarray = artifact_field()
    test_labels: np.ndarray = artifact_field()


def compute_ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Args:
        scores: Predicted probabilities
        labels: True binary labels
        n_bins: Number of bins for calibration

    Returns:
        Expected Calibration Error (lower is better calibrated)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(scores)

    for i in range(n_bins):
        bin_mask = (scores > bin_boundaries[i]) & (scores <= bin_boundaries[i + 1])
        bin_size = bin_mask.sum()

        if bin_size > 0:
            bin_accuracy = labels[bin_mask].mean()
            bin_confidence = scores[bin_mask].mean()
            ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def run_probe_eval(config) -> ProbeEvalResults:
    """Run the probe evaluation experiment.

    Args:
        config: Configuration object with dataset paths and model settings
    """
    import gc

    import torch

    seed = config.seed
    np.random.seed(seed)

    activation_config = ActivationConfig(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
    )

    auto_compute = getattr(config, "auto_compute_activations", True)
    cleanup_after_load = getattr(config, "cleanup_activations_after_load", True)
    debug_mode = getattr(config, "debug", False)

    # Batch sizes for various operations
    activation_batch_size = getattr(config, "activation_batch_size", 64)
    reduction_batch_size = getattr(config, "reduction_batch_size", 512)

    # Load model once if we'll be computing activations in-memory
    # This avoids loading the model multiple times across datasets
    shared_model = None
    if auto_compute and cleanup_after_load:
        from models_under_pressure.model import LLMModel

        logger.info("Loading model for activation computation...")
        shared_model = LLMModel.load(config.activations_model_name, batch_size=activation_batch_size)

    # Load datasets
    logger.info("Loading training dataset...")
    train_dataset = load_dataset(
        Path(config.train_dataset_path),
        activation_config=activation_config,
        auto_compute=auto_compute,
        cleanup_after_load=cleanup_after_load,
        model=shared_model,
        compute_batch_size=activation_batch_size,
        reduction_batch_size=reduction_batch_size,
    )

    logger.info("Loading dev dataset...")
    dev_dataset = load_dataset(
        Path(config.dev_dataset_path),
        activation_config=activation_config,
        auto_compute=auto_compute,
        cleanup_after_load=cleanup_after_load,
        model=shared_model,
        compute_batch_size=activation_batch_size,
        reduction_batch_size=reduction_batch_size,
    )

    logger.info("Loading test dataset...")
    test_dataset = load_dataset(
        Path(config.test_dataset_path),
        activation_config=activation_config,
        auto_compute=auto_compute,
        cleanup_after_load=cleanup_after_load,
        model=shared_model,
        compute_batch_size=activation_batch_size,
        reduction_batch_size=reduction_batch_size,
    )

    # Clean up model to free GPU memory
    if shared_model is not None:
        del shared_model
        gc.collect()
        torch.cuda.empty_cache()

    if debug_mode:
        logger.warning("Running in debug mode with smaller datasets.")
        train_dataset = sample_from_dataset(train_dataset, DEBUG_SAMPLE_SIZE, seed=seed)
        dev_dataset = sample_from_dataset(dev_dataset, min(DEBUG_SAMPLE_SIZE, len(dev_dataset)), seed=seed)
        test_dataset = sample_from_dataset(test_dataset, min(DEBUG_SAMPLE_SIZE, len(test_dataset)), seed=seed)

    calibration_method = getattr(config, "calibration_method", None)
    if calibration_method is not None:
        logger.info(f"Preparing dev dataset split for probe calibration ({calibration_method})")
        dev_dataset, probe_calib_dataset = split_dataset(
            dev_dataset,
            proportions=[0.7, 0.3],
            shuffle=True,
            seed=seed,
        )

    logger.info(f"Train size: {len(train_dataset)}, Dev size: {len(dev_dataset)}, Test size: {len(test_dataset)}")

    # Train probe
    logger.info(f"Fitting probe with reduction strategy: {config.reduction_strategy}")
    probe = SequenceProbe(reduction_strategy=config.reduction_strategy)
    probe.fit(train_dataset)

    if calibration_method:
        logger.info(f"Calibrating probe probabilities with {calibration_method}")
        probe.calibrate(probe_calib_dataset, method=calibration_method)

    # Get predictions
    train_scores = probe.predict(train_dataset)
    train_labels = train_dataset.labels_numpy()
    dev_scores = probe.predict(dev_dataset)
    dev_labels = dev_dataset.labels_numpy()
    test_scores = probe.predict(test_dataset)
    test_labels = test_dataset.labels_numpy()

    # Ensure labels are binary (0/1) for sklearn metrics
    train_labels = (train_labels > 0).astype(int)
    dev_labels = (dev_labels > 0).astype(int)
    test_labels = (test_labels > 0).astype(int)

    # Compute performance metrics
    dev_predictions = (dev_scores >= 0.5).astype(int)
    test_predictions = (test_scores >= 0.5).astype(int)

    dev_accuracy = float(accuracy_score(dev_labels, dev_predictions))
    dev_f1 = float(f1_score(dev_labels, dev_predictions, pos_label=1))
    dev_roc_auc = float(roc_auc_score(dev_labels, dev_scores))

    test_accuracy = float(accuracy_score(test_labels, test_predictions))
    test_f1 = float(f1_score(test_labels, test_predictions, pos_label=1))
    test_roc_auc = float(roc_auc_score(test_labels, test_scores))

    # Compute calibration metrics
    train_ece = compute_ece(train_scores, train_labels)
    dev_ece = compute_ece(dev_scores, dev_labels)
    test_ece = compute_ece(test_scores, test_labels)

    logger.info("\n=== RESULTS ===")
    logger.info(f"Dev:  Acc={dev_accuracy:.4f}, F1={dev_f1:.4f}, ROC-AUC={dev_roc_auc:.4f}, ECE={dev_ece:.4f}")
    logger.info(f"Test: Acc={test_accuracy:.4f}, F1={test_f1:.4f}, ROC-AUC={test_roc_auc:.4f}, ECE={test_ece:.4f}")
    logger.info("===============\n")

    return ProbeEvalResults(
        config=vars(config),
        seed=seed,
        debug_mode=debug_mode,
        train_dataset_path=config.train_dataset_path,
        dev_dataset_path=config.dev_dataset_path,
        test_dataset_path=config.test_dataset_path,
        train_size=len(train_dataset),
        dev_size=len(dev_dataset),
        test_size=len(test_dataset),
        reduction_strategy=config.reduction_strategy,
        model_name=config.activations_model_name,
        layer=config.activations_layer,
        dev_accuracy=dev_accuracy,
        dev_f1_score=dev_f1,
        dev_roc_auc=dev_roc_auc,
        test_accuracy=test_accuracy,
        test_f1_score=test_f1,
        test_roc_auc=test_roc_auc,
        train_ece=train_ece,
        dev_ece=dev_ece,
        test_ece=test_ece,
        train_scores=train_scores,
        train_labels=train_labels,
        dev_scores=dev_scores,
        dev_labels=dev_labels,
        test_scores=test_scores,
        test_labels=test_labels,
    )


def make_figures(results: ProbeEvalResults) -> dict[str, Figure]:
    """Generate all figures for probe evaluation results."""
    from probe_eval_plotting import (
        plot_calibration_summary,
        plot_performance_summary,
        plot_reliability_diagram,
        plot_roc_curve,
        plot_score_histogram,
    )

    figures: dict[str, Figure] = {}

    # Performance summary
    figures["performance"] = plot_performance_summary(results)

    # Calibration summary
    figures["calibration"] = plot_calibration_summary(results)

    # Reliability diagrams
    figures["reliability_train"] = plot_reliability_diagram(
        results.train_scores, results.train_labels, title="Reliability Diagram (Train)"
    )
    figures["reliability_dev"] = plot_reliability_diagram(
        results.dev_scores, results.dev_labels, title="Reliability Diagram (Dev)"
    )
    figures["reliability_test"] = plot_reliability_diagram(
        results.test_scores, results.test_labels, title="Reliability Diagram (Test)"
    )

    # Score histograms
    figures["histogram_train"] = plot_score_histogram(
        results.train_scores, results.train_labels, title="Score Distribution (Train)"
    )
    figures["histogram_dev"] = plot_score_histogram(
        results.dev_scores, results.dev_labels, title="Score Distribution (Dev)"
    )
    figures["histogram_test"] = plot_score_histogram(
        results.test_scores, results.test_labels, title="Score Distribution (Test)"
    )

    # ROC curves
    figures["roc_train"] = plot_roc_curve(results.train_scores, results.train_labels, title="ROC Curve (Train)")
    figures["roc_dev"] = plot_roc_curve(results.dev_scores, results.dev_labels, title="ROC Curve (Dev)")
    figures["roc_test"] = plot_roc_curve(results.test_scores, results.test_labels, title="ROC Curve (Test)")

    return figures


def log_to_clearml(
    clearml_logger,
    results: ProbeEvalResults,
    figures: dict[str, Figure],
):
    """Log experiment results and figures to ClearML."""
    from clearml_serialization import ClearMLSerializer

    # Log configuration
    clearml_logger.connect_configuration(results.config)

    # Add tags
    tags = ["probe-eval"]
    if results.debug_mode:
        tags.append("debug")

    # Model name (short form)
    model_short = results.model_name.split("/")[-1]
    tags.append(model_short)

    # Reduction strategy
    tags.append(f"reduction-{results.reduction_strategy}")

    # Dataset name (extract from path, e.g., "anthropic_balanced" from full path)
    dataset_name = (
        Path(results.dev_dataset_path).stem.replace("_apr_23", "").replace("_apr_22", "").replace("_apr_30", "")
    )
    tags.append(f"dataset-{dataset_name}")

    # Probe architecture
    tags.append("probe-logistic-regression")

    # Layer
    tags.append(f"layer-{results.layer}")

    # calibration
    calibration_method = results.config.get("calibration_method")
    tags.append(f"calibration-{calibration_method}" if calibration_method else "not-calibrated")

    clearml_logger.add_tags(tags)

    # Log scalars
    serializer = ClearMLSerializer()
    clearml_logger.log_scalars(serializer.to_clearml_scalars(results))

    # Log artifacts
    clearml_logger.log_artifacts(serializer.to_clearml_artifacts(results))

    # Log figures
    logger.info("Logging figures to ClearML...")

    clearml_logger.log_figure(title="Performance", series="Summary", figure=figures["performance"])
    clearml_logger.log_figure(title="Calibration", series="ECE Summary", figure=figures["calibration"])

    clearml_logger.log_figure(title="Calibration", series="Reliability - Train", figure=figures["reliability_train"])
    clearml_logger.log_figure(title="Calibration", series="Reliability - Dev", figure=figures["reliability_dev"])
    clearml_logger.log_figure(title="Calibration", series="Reliability - Test", figure=figures["reliability_test"])

    clearml_logger.log_figure(title="Score Distributions", series="Train", figure=figures["histogram_train"])
    clearml_logger.log_figure(title="Score Distributions", series="Dev", figure=figures["histogram_dev"])
    clearml_logger.log_figure(title="Score Distributions", series="Test", figure=figures["histogram_test"])

    clearml_logger.log_figure(title="ROC Curves", series="Train", figure=figures["roc_train"])
    clearml_logger.log_figure(title="ROC Curves", series="Dev", figure=figures["roc_dev"])
    clearml_logger.log_figure(title="ROC Curves", series="Test", figure=figures["roc_test"])

    # Close figures
    import matplotlib.pyplot as plt

    plt.close("all")

    logger.info("All figures logged to ClearML")
    clearml_logger.finalize()


def parse_args():
    """Parse command line arguments."""
    default_config_path = Path(__file__).parent / "configs" / "probe_eval.yaml"

    parser = argparse.ArgumentParser(description="Probe Evaluation Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--use-clearml",
        action="store_true",
        help="Enable ClearML experiment tracking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize ClearML logger if enabled
    clearml_logger = None
    if args.use_clearml:
        from clearml_logger import ClearMLLogger

        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="probe_eval",
            enabled=True,
        )

    # Run experiment
    results = run_probe_eval(config)

    # Generate figures
    logger.info("Generating figures...")
    figures = make_figures(results)

    if clearml_logger is not None:
        log_to_clearml(clearml_logger, results, figures)

    logger.info("Experiment complete!")
