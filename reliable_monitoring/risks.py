from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from reliable_monitoring.bounds import binomial, hb_p_value
from reliable_monitoring.cascade import CascadePredictionResults

if TYPE_CHECKING:
    from models_under_pressure.interfaces.dataset import LabelledDataset


# Type alias for statistical bound functions
BoundFunction = Callable[[np.ndarray | float, int, float], np.ndarray]
# Signature: (empirical_risks, n_samples, alpha) -> p_values (always returns array)


@dataclass
class RiskEvaluationContext:
    """Context containing data for risk evaluation.

    Different risks need different data:
    - Budget cost: only needs cascade_scores
    - Accuracy/ROC-AUC: needs cascade_scores + dataset
    """

    cascade_scores: CascadePredictionResults
    dataset: "LabelledDataset | None" = None


# Standalone empirical computation functions
def budget_cost_computation(context: RiskEvaluationContext) -> float:
    """Compute budget cost: rate at which cascade calls baseline."""
    return float(context.cascade_scores.used_baseline.mean())


def accuracy_computation(context: RiskEvaluationContext) -> float:
    """Compute error rate (1 - accuracy) as a risk measure."""
    if context.dataset is None:
        raise ValueError("accuracy_computation requires dataset")
    predicted_labels = (context.cascade_scores.final_scores >= 0.5).astype(int)
    accuracy = accuracy_score(context.dataset.labels_numpy(), predicted_labels)
    return float(1.0 - accuracy)  # Error rate


def roc_auc_computation(context: RiskEvaluationContext) -> float:
    """Compute negative ROC AUC (1 - AUC) as a risk measure."""
    if context.dataset is None:
        raise ValueError("roc_auc_computation requires dataset")
    auc = roc_auc_score(context.dataset.labels_numpy(), context.cascade_scores.final_scores)
    return float(1.0 - auc)


@dataclass
class Risk:
    """Pairs an empirical risk computation with its statistical bound.

    Attributes:
        name: Human-readable name for this risk.
        empirical_computation: Function that computes empirical risk from context.
            Signature: (RiskEvaluationContext) -> float
        p_value_bound_fn: Statistical bound function for computing p-values.
            Signature: (empirical_risks, n_samples, alpha) -> p_values
    """

    name: str
    empirical_computation: Callable[[RiskEvaluationContext], float]
    p_value_bound_fn: BoundFunction


# Pre-made Risk instances with sensible defaults
BudgetCostRisk = Risk(
    name="Budget Cost",
    empirical_computation=budget_cost_computation,
    p_value_bound_fn=binomial,
)

AccuracyRisk = Risk(
    name="Error Rate (1 - Accuracy)",
    empirical_computation=accuracy_computation,
    p_value_bound_fn=hb_p_value,
)

RocAucRisk = Risk(
    name="Negative ROC AUC (1 - AUC)",
    empirical_computation=roc_auc_computation,
    p_value_bound_fn=hb_p_value,
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
    risk: Risk  # Risk used (name, empirical_computation, p_value_bound_fn)

    def compute_p_values(self, alpha: float) -> np.ndarray:
        """Compute p-values using the risk's appropriate bound.

        This couples the p-value computation with the risk,
        ensuring the correct bound is used.

        Args:
            alpha: Risk threshold to test against

        Returns:
            P-values for each threshold
        """
        return self.risk.p_value_bound_fn(self.empirical_risks, self.n_samples, alpha)


def evaluate_threshold_risks(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    thresholds: np.ndarray,
    *,
    risk: Risk,
    dataset: "LabelledDataset | None" = None,
    merge_strategy: str = "avg",
) -> ThresholdEvaluationResult:
    """Evaluate empirical risks for a grid of cascade thresholds.

    This function sweeps a threshold grid and computes the empirical risk
    (e.g., budget cost) for each threshold on calibration data.

    Args:
        probe_scores: Probe predictions on calibration set, shape (n,)
        baseline_scores: Baseline predictions on calibration set, shape (n,)
        thresholds: Threshold grid to evaluate, shape (k,)
        risk: Risk instance pairing empirical computation with p-value bound.
            Commonly: BudgetCostRisk, AccuracyRisk, RocAucRisk,
            or custom: Risk(name=..., empirical_computation=..., p_value_bound_fn=...)
        dataset: Optional dataset with labels (required for risks like AccuracyRisk, RocAucRisk)
        merge_strategy: Cascade merge strategy ("avg", "probe", "baseline")

    Returns:
        ThresholdEvaluationResult with:
            - thresholds: The evaluated threshold grid
            - empirical_risks: Empirical risk per threshold
            - n_samples: Number of calibration samples
            - risk: The Risk instance used

    Example:
        >>> from reliable_monitoring.risks import BudgetCostRisk
        >>> result = evaluate_threshold_risks(
        ...     probe_scores,
        ...     baseline_scores,
        ...     thresholds=np.linspace(0.5, 1, 10),
        ...     risk=BudgetCostRisk,
        ... )
        >>> # Compute p-values using the risk's appropriate bound
        >>> p_values = result.compute_p_values(alpha=0.3)
        >>> # Apply testing procedure
        >>> from reliable_monitoring.learn_then_test import fixed_sequence_testing
        >>> reliable_indices = fixed_sequence_testing(p_values, delta=0.05)
    """
    from reliable_monitoring.cascade import run_offline_cascade

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
        # Create context and compute risk
        context = RiskEvaluationContext(
            cascade_scores=cascade_result,
            dataset=dataset,
        )
        empirical_risks[i] = risk.empirical_computation(context)

    return ThresholdEvaluationResult(
        thresholds=thresholds,
        empirical_risks=empirical_risks,
        n_samples=n_samples,
        risk=risk,
    )
