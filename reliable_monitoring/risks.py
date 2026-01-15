from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from reliable_monitoring.cascade import CascadePredictionResults

if TYPE_CHECKING:
    from models_under_pressure.interfaces.dataset import LabelledDataset


def baseline_budget_cost(cascade_scores: CascadePredictionResults) -> float:
    """Rate at which you call the baseline"""
    return cascade_scores.used_baseline.mean()


def empirical_roc_auc(cascade_scores: CascadePredictionResults, dataset: "LabelledDataset") -> float:
    """Empirical performance of the cascade."""
    return roc_auc_score(
        dataset.labels_numpy(),
        cascade_scores.final_scores,
    )


def empirical_accuracy(cascade_scores: CascadePredictionResults, dataset: "LabelledDataset") -> float:
    """Empirical accuracy of the cascade."""
    predicted_labels = (cascade_scores.final_scores >= 0.5).astype(int)
    return accuracy_score(
        dataset.labels_numpy(),
        predicted_labels,
    )


@dataclass
class ThresholdEvaluationResult:
    """Results from evaluating cascade thresholds on calibration data.

    This contains the empirical risks for a grid of thresholds, which can
    then be passed to statistical bounds (e.g., Hoeffding-Bentkus) and
    multiple hypothesis testing procedures (e.g., fixed-sequence testing).
    """

    thresholds: np.ndarray  # Threshold grid evaluated
    empirical_risks: np.ndarray  # Empirical risk per threshold
    n_samples: int  # Number of calibration samples


def evaluate_threshold_risks(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    thresholds: np.ndarray,
    *,
    risk_function: Callable | None = None,
    merge_strategy: str = "avg",
) -> ThresholdEvaluationResult:
    """Evaluate empirical risks for a grid of cascade thresholds.

    This function sweeps a threshold grid and computes the empirical risk
    (e.g., budget cost) for each threshold on calibration data.

    This is the core duplicated logic from guaranteed_budget.py and
    cascade_comparison.py experiments, extracted for reusability.

    Args:
        probe_scores: Probe predictions on calibration set, shape (n,)
        baseline_scores: Baseline predictions on calibration set, shape (n,)
        thresholds: Threshold grid to evaluate, shape (k,)
        risk_function: Function to compute risk from cascade result.
            Defaults to baseline_budget_cost. Should have signature:
            risk_function(cascade_result) -> float
        merge_strategy: Cascade merge strategy ("avg", "probe", "baseline")

    Returns:
        ThresholdEvaluationResult with:
            - thresholds: The evaluated threshold grid
            - empirical_risks: Empirical risk per threshold
            - n_samples: Number of calibration samples

    Example:
        >>> # Compute empirical risks
        >>> result = evaluate_threshold_risks(
        ...     probe_scores,
        ...     baseline_scores,
        ...     thresholds=np.linspace(0.5, 1, 10),
        ... )
        >>> # Then compute p-values and apply testing procedure
        >>> from reliable_monitoring.bounds import hb_p_value
        >>> from reliable_monitoring.learn_then_test import fixed_sequence_testing
        >>> p_values = hb_p_value(result.empirical_risks, n=result.n_samples, alpha=0.3)
        >>> reliable_indices = fixed_sequence_testing(p_values, delta=0.05)
    """
    from reliable_monitoring.cascade import run_offline_cascade

    # Default risk function
    if risk_function is None:
        risk_function = baseline_budget_cost

    n_samples = len(probe_scores)
    n_thresholds = len(thresholds)

    # Sweep thresholds and compute empirical risks
    empirical_risks = np.zeros(n_thresholds)
    for i, threshold in enumerate(thresholds):
        cascade_result = run_offline_cascade(
            probe_scores,
            baseline_scores,
            threshold=threshold,
            merge_strategy=merge_strategy,
        )
        empirical_risks[i] = risk_function(cascade_result)

    return ThresholdEvaluationResult(
        thresholds=thresholds,
        empirical_risks=empirical_risks,
        n_samples=n_samples,
    )
