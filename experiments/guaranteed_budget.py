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
    reliable_hyperparameters: list[int]  # Indices of thresholds that passed FST (null rejected)

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

    def to_clearml_scalars(self) -> dict[str, float | int]:
        """Extract metrics suitable for ClearML scalar logging."""
        scalars = {
            "success": float(self.success),
            "delta": self.delta,
            "num_reliable_hyperparameters": len(self.reliable_hyperparameters),
            "train_size": self.train_size,
            "calib_size": self.calib_size,
            "test_size": self.test_size,
            "mean_empirical_risk": float(self.empirical_budget_risks.mean()),
            "min_empirical_risk": float(self.empirical_budget_risks.min()),
            "max_empirical_risk": float(self.empirical_budget_risks.max()),
        }
        if self.success:
            scalars.update(
                {
                    "best_threshold": self.best_threshold,
                    "test_budget_cost": self.test_budget_cost,
                    "best_index": self.best_index,
                }
            )
        return scalars

    def to_clearml_artifacts(self) -> dict[str, object]:
        """Extract numpy arrays and data suitable for ClearML artifact logging."""
        # Create boolean mask: mask[i] = 1 if hyperparameter i is reliable (null rejected)
        reliable_mask = np.zeros(len(self.thresholds), dtype=np.uint8)
        reliable_mask[self.reliable_hyperparameters] = 1

        artifacts = {
            "thresholds": self.thresholds,
            "reliable_mask": reliable_mask,
            "empirical_budget_risks": self.empirical_budget_risks,
            "p_values": self.p_values,
            "calib_probe_scores": self.calib_probe_scores,
            "calib_baseline_scores": self.calib_baseline_scores,
            "config": self.config,
        }
        if self.success:
            artifacts.update(
                {
                    "test_probe_scores": self.test_probe_scores,
                    "test_baseline_scores": self.test_baseline_scores,
                    "test_cascade_scores": self.test_cascade_scores,
                }
            )
        return artifacts

    @staticmethod
    def from_clearml(task: object) -> "GuaranteedBudgetResults":
        """Reconstruct GuaranteedBudgetResults from a ClearML task.

        This allows you to load experiment results back from ClearML with full typing.

        Args:
            task: ClearML Task object

        Returns:
            GuaranteedBudgetResults reconstructed from the task

        Example:
            >>> from clearml import Task
            >>> task = Task.get_task(task_id="abc123")
            >>> results = GuaranteedBudgetResults.from_clearml(task)
            >>> print(results.best_threshold)
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required to use from_clearml()") from e

        # Fetch config.
        # In ClearML 2.x, connect_configuration() shows up under configuration objects,
        # and get_parameter("Configuration") may return None.
        config_dict = None

        if hasattr(task, "get_configuration_object_as_dict"):
            try:
                config_dict = task.get_configuration_object_as_dict("Configuration")
            except Exception:
                config_dict = None

        if config_dict is None and hasattr(task, "get_parameter"):
            try:
                config_dict = task.get_parameter("Configuration")
            except Exception:
                config_dict = None

        if isinstance(config_dict, str):
            config_dict = yaml.safe_load(config_dict)

        # Fetch artifacts
        artifacts = task.artifacts

        # Fallback: config may be logged as an artifact named "config".
        if config_dict is None and "config" in artifacts:
            config_path = artifacts["config"].get_local_copy()
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(
                "Could not load configuration from ClearML task; expected a 'Configuration' configuration object or 'config' artifact"
            )

        # Normalize to plain dict for downstream .get() usage
        if not isinstance(config_dict, dict):
            config_dict = dict(config_dict)

        thresholds = np.load(artifacts["thresholds"].get_local_copy())
        reliable_mask = np.load(artifacts["reliable_mask"].get_local_copy())
        empirical_budget_risks = np.load(artifacts["empirical_budget_risks"].get_local_copy())
        p_values = np.load(artifacts["p_values"].get_local_copy())
        calib_probe_scores = np.load(artifacts["calib_probe_scores"].get_local_copy())
        calib_baseline_scores = np.load(artifacts["calib_baseline_scores"].get_local_copy())

        # Reconstruct reliable_hyperparameters from boolean mask
        reliable_hyperparameters = list(np.where(reliable_mask == 1)[0])

        def _get_scalar(results_section: dict, name: str, default: float | int | None = None):
            series = results_section.get(name)
            if series is None:
                return default
            if isinstance(series, dict) and "y" in series:
                y = series.get("y")
                if isinstance(y, list) and len(y) > 0:
                    return y[-1]
                return default
            if isinstance(series, list) and len(series) > 0:
                return series[-1]
            return series

        # Fetch metrics (ClearML 2.x)
        if not hasattr(task, "get_reported_scalars"):
            raise AttributeError(
                "ClearML task object does not expose get_reported_scalars(); cannot reconstruct metrics"
            )
        scalars = task.get_reported_scalars()
        results_section = scalars.get("Results", {})

        success = bool(_get_scalar(results_section, "success", 0))
        delta = float(_get_scalar(results_section, "delta", 0.0))

        best_threshold = None
        best_index = None
        test_budget_cost = None
        test_probe_scores = None
        test_baseline_scores = None
        test_cascade_scores = None

        if success:
            best_threshold = float(_get_scalar(results_section, "best_threshold", 0.0))
            best_index = int(_get_scalar(results_section, "best_index", 0))
            test_budget_cost = float(_get_scalar(results_section, "test_budget_cost", 0.0))

            # Try to load test artifacts if they exist
            if "test_probe_scores" in artifacts:
                test_probe_scores = np.load(artifacts["test_probe_scores"].get_local_copy())
            if "test_baseline_scores" in artifacts:
                test_baseline_scores = np.load(artifacts["test_baseline_scores"].get_local_copy())
            if "test_cascade_scores" in artifacts:
                test_cascade_scores = np.load(artifacts["test_cascade_scores"].get_local_copy())

        return GuaranteedBudgetResults(
            config=config_dict,
            seed=int(config_dict.get("seed", 42)),
            debug_mode=bool(config_dict.get("debug", False)),
            train_size=int(_get_scalar(results_section, "train_size", 0)),
            calib_size=int(_get_scalar(results_section, "calib_size", 0)),
            test_size=int(_get_scalar(results_section, "test_size", 0)),
            probe_reduction_strategy=str(config_dict.get("reduction_strategy", "")),
            thresholds=thresholds,
            empirical_budget_risks=empirical_budget_risks,
            p_values=p_values,
            delta=delta,
            reliable_hyperparameters=reliable_hyperparameters,
            calib_probe_scores=calib_probe_scores,
            calib_baseline_scores=calib_baseline_scores,
            success=success,
            best_threshold=best_threshold,
            best_index=best_index,
            test_budget_cost=test_budget_cost,
            test_probe_scores=test_probe_scores,
            test_baseline_scores=test_baseline_scores,
            test_cascade_scores=test_cascade_scores,
        )


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
            debug_mode=config.debug,
            train_size=len(train_dataset),
            calib_size=len(calib_dataset),
            test_size=len(test_dataset),
            probe_reduction_strategy=config.reduction_strategy,
            thresholds=thresholds,
            empirical_budget_risks=empirical_budget_risks,
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
        # pick the most aggressive hyperparameters that still guarantee the budget
        best_index = reliable_hyperparameters[-1]
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
            reliable_hyperparameters=list(reliable_hyperparameters),
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
    import os

    from clearml_logger import ClearMLLogger

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

    # Log to ClearML if enabled (using methods from results dataclass)
    if clearml_logger:
        # Log configuration
        clearml_logger.connect_configuration(results.config)

        # Manually log model information since auto-detection is disabled
        # This explicitly tracks which models were used in the experiment
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

        # Use results methods for clean data extraction
        clearml_logger.log_scalars(results.to_clearml_scalars())
        clearml_logger.log_artifacts(results.to_clearml_artifacts())
        clearml_logger.finalize()
