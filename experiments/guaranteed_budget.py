"""
This experiment finds hyperparameters that guarantee control over the baseline model budget.
"""

import argparse
import logging
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


DEBUG_SAMPLE_SIZE = 100


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


def run_guaranteed_budget_experiment():
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

    # now evaluate on the test set and print the test (real) budget cost
    if len(rejected_hyperparams) == 0:
        logger.info("No hyperparameters found that guarantee the budget.")
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


if __name__ == "__main__":
    run_guaranteed_budget_experiment()
