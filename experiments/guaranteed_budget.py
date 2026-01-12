"""
This experiment finds hyperparameters that guarantee control over the baseline model budget.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from config import load_config
from dotenv import load_dotenv

from reliable_monitoring.bounds import hb_p_value
from reliable_monitoring.cascade import run_llm_baseline, run_offline_cascade
from reliable_monitoring.dataset import ActivationConfig, load_dataset, sample_from_dataset
from reliable_monitoring.learn_then_test import fixed_sequence_testing
from reliable_monitoring.probes import SequenceProbe
from reliable_monitoring.risks import (
    baseline_budget_cost,
)

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEBUG_SAMPLE_SIZE = 32


@dataclass
class GuaranteedBudgetResults:
    """Results from a guaranteed budget experiment.

    Comprehensive results structure for experiment tracking and analysis.
    Includes configuration, calibration results, and test results.
    """

    # Experiment metadata
    config: dict  # Config as dict for easy serialization
    seed: int
    debug_mode: bool

    # Dataset information
    train_size: int
    calib_size: int
    test_size: int

    # Probe information
    probe_reduction_strategy: str  # Just the key parameter

    # Calibration phase - thresholds and metrics
    thresholds: np.ndarray  # Array of thresholds tested
    empirical_budget_risks: np.ndarray  # Risk for each threshold
    p_values: np.ndarray  # P-value for each threshold
    delta: float  # 1 - guarantee_probability
    rejected_hyperparams: list[int]  # Indices of thresholds that passed FST

    # Calibration phase - raw scores
    calib_probe_scores: np.ndarray  # Probe predictions on calibration set
    calib_baseline_scores: np.ndarray  # Baseline predictions on calibration set

    # Best threshold selection
    success: bool  # Whether any valid threshold was found
    best_threshold: float | None  # None if no valid threshold found
    best_index: int | None  # Index into thresholds array

    # Test phase results (only if success=True)
    test_budget_cost: float | None  # Actual budget cost on test set
    test_probe_scores: np.ndarray | None  # Probe predictions on test set
    test_baseline_scores: np.ndarray | None  # Baseline predictions on test set
    test_cascade_scores: np.ndarray | None  # Final cascade scores on test set


def parse_args():
    """Parse command line arguments to get config file path."""
    default_config_path = Path(__file__).parent / "configs" / "guaranteed_budget.yaml"

    parser = argparse.ArgumentParser(description="Guaranteed Budget Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def run_guaranteed_budget_experiment() -> GuaranteedBudgetResults:
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

    if config.debug:
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

    # computing offline scores on calibration dataset
    logger.info("Computing probe scores on calibration dataset...")
    probe_scores = probe.predict(calib_dataset)

    logger.info("Computing baseline model costs on calibration dataset...")
    baseline_scores = run_llm_baseline(
        baseline_model_name=config.baseline_model_name,
        dataset=calib_dataset,
        baseline_batch_size=config.baseline_batch_size,
    )

    # compute empirical risks on cascade
    logger.info("Computing empirical risks on calibration dataset...")
    thresholds = np.linspace(0.5, 1, 10)

    empirical_budget_risks = np.zeros_like(thresholds)
    # empirical_performance_scores = np.zeros_like(thresholds)

    for i, t in enumerate(thresholds):
        print(f"Evaluating threshold: {t:.2f}, {i} of {len(thresholds)}")
        scores = run_offline_cascade(
            probe_scores,
            baseline_scores,
            threshold=t,
            merge_strategy=config.cascade_merge_strategy,
        )
        empirical_budget_risks[i] = baseline_budget_cost(scores)
        # empirical_performance_scores[i] = empirical_accuracy(scores, calib_dataset)

    # compute p-values for corresponding empirical risks
    p_values = hb_p_value(
        r_hat=empirical_budget_risks,
        n=len(calib_dataset),
        alpha=config.budget,
    )

    # run FST to find hyperparameters that guarantee budget control
    # the risks (and p-values) are already in the order in which we want them
    # that is, ascending in thresholds, which we expect to correspond to ascending in cost as well.
    delta = 1 - config.guarantee_probability
    rejected_hyperparams = fixed_sequence_testing(
        p_values=p_values,
        delta=delta,
    )

    # now evaluate on the test set and build results
    if len(rejected_hyperparams) == 0:
        logger.info("No hyperparameters found that guarantee the budget.")
        return GuaranteedBudgetResults(
            config=vars(config),
            seed=seed,
            debug_mode=config.debug,
            train_size=len(train_dataset),
            calib_size=len(calib_dataset),
            test_size=len(test_dataset),
            probe_reduction_strategy=config.reduction_strategy,
            thresholds=thresholds,
            empirical_budget_risks=empirical_budget_risks,
            p_values=p_values,
            delta=delta,
            rejected_hyperparams=list(rejected_hyperparams),
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
        # pick the most aggressive hyperparameters that still guarantee the budget
        best_index = rejected_hyperparams[-1]
        best_threshold = thresholds[best_index]

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
            debug_mode=config.debug,
            train_size=len(train_dataset),
            calib_size=len(calib_dataset),
            test_size=len(test_dataset),
            probe_reduction_strategy=config.reduction_strategy,
            thresholds=thresholds,
            empirical_budget_risks=empirical_budget_risks,
            p_values=p_values,
            delta=delta,
            rejected_hyperparams=list(rejected_hyperparams),
            calib_probe_scores=probe_scores,
            calib_baseline_scores=baseline_scores,
            success=True,
            best_threshold=float(best_threshold),
            best_index=int(best_index),
            test_budget_cost=float(test_budget_cost),
            test_probe_scores=probe_test_scores,
            test_baseline_scores=baseline_test_scores,
            test_cascade_scores=test_scores,
        )


if __name__ == "__main__":
    results = run_guaranteed_budget_experiment()
    # Future: publish results to experiment tracking service
