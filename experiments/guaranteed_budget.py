"""
This experiment finds hyperparameters that guarantee control over the baseline model budget.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from clearml_serialization import (
    artifact_field,
    conditional_field,
    derived_field,
    scalar_field,
)
from config import load_config
from dotenv import load_dotenv

from reliable_monitoring.bounds import hb_p_value
from reliable_monitoring.cascade import run_llm_baseline, run_offline_cascade
from reliable_monitoring.dataset import ActivationConfig, load_dataset, sample_from_dataset
from reliable_monitoring.learn_then_test import fixed_sequence_testing
from reliable_monitoring.probes import SequenceProbe
from reliable_monitoring.risks import baseline_budget_cost, evaluate_threshold_risks

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEBUG_SAMPLE_SIZE = 32


@dataclass
class GuaranteedBudgetResults:
    """Pure data structure for guaranteed budget experiment results.

    This is a "structured manifesto" of experiment outputs. It defines the shape
    and contents of results without any serialization logic. All ClearML
    serialization is handled by ClearMLSerializer.
    """

    # Experiment metadata (scalars/artifacts)
    config: dict = artifact_field()
    seed: int = scalar_field()
    debug_mode: bool = scalar_field()

    # Dataset sizes (scalars)
    train_size: int = scalar_field()
    calib_size: int = scalar_field()
    test_size: int = scalar_field()

    # Probe information (scalar)
    probe_reduction_strategy: str = scalar_field()

    # Calibration phase - thresholds and metrics (artifacts)
    thresholds: np.ndarray = artifact_field()
    empirical_budget_risks: np.ndarray = artifact_field()
    p_values: np.ndarray = artifact_field()
    delta: float = scalar_field()
    reliable_hyperparameters: list[int] = artifact_field()

    # Calibration phase - raw scores (artifacts)
    calib_probe_scores: np.ndarray = artifact_field()
    calib_baseline_scores: np.ndarray = artifact_field()

    # Best threshold selection (scalars)
    success: bool = scalar_field()
    best_threshold: float | None = conditional_field(condition="success")
    best_index: int | None = conditional_field(condition="success")

    # Test phase results (conditional artifacts)
    test_budget_cost: float | None = conditional_field(condition="success")
    test_probe_scores: np.ndarray | None = conditional_field(condition="success")
    test_baseline_scores: np.ndarray | None = conditional_field(condition="success")
    test_cascade_scores: np.ndarray | None = conditional_field(condition="success")

    # Derived scalars (computed on-the-fly, not stored)
    mean_empirical_risk: float = derived_field(derive_fn=lambda r: float(r.empirical_budget_risks.mean()))
    min_empirical_risk: float = derived_field(derive_fn=lambda r: float(r.empirical_budget_risks.min()))
    max_empirical_risk: float = derived_field(derive_fn=lambda r: float(r.empirical_budget_risks.max()))
    num_reliable_hyperparameters: int = derived_field(derive_fn=lambda r: len(r.reliable_hyperparameters))


def parse_args():
    """Parse command line arguments to get config file path."""
    default_config_path = Path(__file__).parent / "configs" / "guaranteed_budget.yaml"

    parser = argparse.ArgumentParser(description="Guaranteed Budget Experiment")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with smaller datasets.",
    )
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


def run_guaranteed_budget_experiment(args: argparse.Namespace | None = None) -> GuaranteedBudgetResults:
    if args is None:
        args = parse_args()
    config = load_config(args.config)

    seed = config.seed
    np.random.seed(seed)

    activation_config = ActivationConfig(
        model_name=config.activations_model_name,
        layer=config.activations_layer,
    )
    train_dataset = load_dataset(
        Path(config.train_dataset_path),
        activation_config=activation_config,
    )
    calib_dataset = load_dataset(
        Path(config.calib_dataset_path),
        activation_config=activation_config,
    )
    test_dataset = load_dataset(
        Path(config.test_dataset_path),
        activation_config=activation_config,
    )

    if args.debug:
        logger.warning("Running in debug mode with smaller datasets.")
        train_dataset = sample_from_dataset(train_dataset, DEBUG_SAMPLE_SIZE, seed=seed)
        calib_dataset = sample_from_dataset(calib_dataset, DEBUG_SAMPLE_SIZE, seed=seed)
        test_dataset = sample_from_dataset(test_dataset, DEBUG_SAMPLE_SIZE, seed=seed)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Calibration dataset size: {len(calib_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    logger.info("Fitting probe...")
    probe = SequenceProbe(reduction_strategy=config.reduction_strategy)
    probe.fit(train_dataset)

    # Computing offline scores on calibration dataset
    logger.info("Computing probe scores on calibration dataset...")
    probe_scores = probe.predict(calib_dataset)

    logger.info("Computing baseline model costs on calibration dataset...")
    baseline_scores = run_llm_baseline(
        baseline_model_name=config.baseline_model_name,
        dataset=calib_dataset,
        baseline_batch_size=config.baseline_batch_size,
    )

    # Evaluate empirical risks across threshold grid
    logger.info("Computing empirical risks on calibration dataset...")
    thresholds = np.linspace(
        getattr(config, "threshold_start", 0.5),
        getattr(config, "threshold_end", 1.0),
        getattr(config, "threshold_steps", 10),
    )

    eval_result = evaluate_threshold_risks(
        probe_scores,
        baseline_scores,
        thresholds,
        merge_strategy=config.cascade_merge_strategy,
    )

    # Compute p-values using Hoeffding-Bentkus bound
    p_values = hb_p_value(
        r_hat=eval_result.empirical_risks,
        n=eval_result.n_samples,
        alpha=config.budget,
    )

    # run FST to find hyperparameters that guarantee budget control
    # the risks (and p-values) are already in the order in which we want them
    # that is, ascending in thresholds, which we expect to correspond to ascending in cost as well.
    delta = 1 - config.guarantee_probability
    reliable_hyperparameters = fixed_sequence_testing(
        p_values=p_values,
        delta=delta,
    )

    # now evaluate on the test set and build results
    if len(reliable_hyperparameters) == 0:
        logger.info("No hyperparameters found that guarantee the budget.")
        return GuaranteedBudgetResults(
            config=vars(config),
            seed=seed,
            debug_mode=args.debug,
            train_size=len(train_dataset),
            calib_size=len(calib_dataset),
            test_size=len(test_dataset),
            probe_reduction_strategy=config.reduction_strategy,
            thresholds=eval_result.thresholds,
            empirical_budget_risks=eval_result.empirical_risks,
            p_values=p_values,
            delta=delta,
            reliable_hyperparameters=list(reliable_hyperparameters),
            calib_probe_scores=probe_scores,
            calib_baseline_scores=baseline_scores,
            success=False,
            best_threshold=None,
            best_index=None,
            test_budget_cost=None,
            test_probe_scores=None,
            test_baseline_scores=None,
            test_cascade_scores=None,
        )
    else:
        # Pick the most aggressive hyperparameters that still guarantee the budget
        best_index = reliable_hyperparameters[-1]
        best_threshold = eval_result.thresholds[best_index]

        logger.info(
            f"Best threshold guaranteeing budget {config.budget} with probability {config.guarantee_probability} is {best_threshold:.4f}"
        )

        # compute test scores
        logger.info("Computing probe scores on test dataset...")
        probe_test_scores = probe.predict(test_dataset)

        logger.info("Computing baseline model costs on test dataset...")
        baseline_test_scores = run_llm_baseline(
            baseline_model_name=config.baseline_model_name,
            dataset=test_dataset,
            baseline_batch_size=config.baseline_batch_size,
        )

        logger.info("Computing empirical risks on test dataset...")
        test_scores = run_offline_cascade(
            probe_test_scores,
            baseline_test_scores,
            threshold=best_threshold,
            merge_strategy=config.cascade_merge_strategy,
        )
        test_budget_cost = baseline_budget_cost(test_scores)

        logger.info(f"Test budget cost with threshold {best_threshold:.4f}: {test_budget_cost:.4f}")

        return GuaranteedBudgetResults(
            config=vars(config),
            seed=seed,
            debug_mode=args.debug,
            train_size=len(train_dataset),
            calib_size=len(calib_dataset),
            test_size=len(test_dataset),
            probe_reduction_strategy=config.reduction_strategy,
            thresholds=eval_result.thresholds,
            empirical_budget_risks=eval_result.empirical_risks,
            p_values=p_values,
            delta=delta,
            reliable_hyperparameters=list(reliable_hyperparameters),
            calib_probe_scores=probe_scores,
            calib_baseline_scores=baseline_scores,
            success=True,
            best_threshold=float(best_threshold),
            best_index=int(best_index),
            test_budget_cost=float(test_budget_cost),
            test_probe_scores=probe_test_scores,
            test_baseline_scores=baseline_test_scores,
            test_cascade_scores=test_scores.final_scores,
        )


if __name__ == "__main__":
    import os

    from clearml_logger import ClearMLLogger
    from clearml_serialization import ClearMLSerializer

    args = parse_args()

    # Initialize ClearML logger if enabled
    clearml_logger = None
    if args.use_clearml:
        clearml_logger = ClearMLLogger(
            project_name=os.environ.get("CLEARML_PROJECT_NAME", "reliable-llm-monitoring"),
            task_name="guaranteed_budget_experiment",
            enabled=True,
        )

    # Run experiment
    results = run_guaranteed_budget_experiment(args)

    # Log to ClearML if enabled (using generic serializer)
    if clearml_logger:
        # Log configuration
        clearml_logger.connect_configuration(results.config)

        # Manually log model information since auto-detection is disabled
        # This explicitly tracks which models were used in the experiment
        if clearml_logger.task:
            clearml_logger.task.connect_configuration(
                {
                    "activation_model": results.config["activations_model_name"],
                    "activation_layer": results.config["activations_layer"],
                    "baseline_model": results.config["baseline_model_name"],
                    "reduction_strategy": results.config["reduction_strategy"],
                },
                name="Models",
            )

        # Add tags
        tags = []
        if results.debug_mode:
            tags.append("debug")
        tags.append(results.config["baseline_model_name"].split("/")[-1])
        tags.append("success" if results.success else "failure")
        clearml_logger.add_tags(tags)

        # Use serializer for clean data extraction
        serializer = ClearMLSerializer()
        clearml_logger.log_scalars(serializer.to_clearml_scalars(results))
        clearml_logger.log_artifacts(serializer.to_clearml_artifacts(results))
        clearml_logger.finalize()
