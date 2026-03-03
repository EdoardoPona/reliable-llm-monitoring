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
    - Accuracy/ROC-AUC: needs cascade_scores + labels (or dataset)

    Labels can be provided directly via ``labels`` or indirectly via
    ``dataset.labels_numpy()``.  When both are provided, ``labels`` takes
    precedence.
    """

    cascade_scores: CascadePredictionResults
    dataset: "LabelledDataset | None" = None
    labels: np.ndarray | None = None

    def get_labels(self) -> np.ndarray:
        """Return labels from whichever source is available."""
        if self.labels is not None:
            return self.labels
        if self.dataset is not None:
            return self.dataset.labels_numpy()
        raise ValueError("RiskEvaluationContext has neither labels nor dataset")


# Standalone empirical computation functions
def budget_cost_computation(context: RiskEvaluationContext) -> float:
    """Compute budget cost: rate at which cascade calls baseline."""
    return float(context.cascade_scores.used_baseline.mean())


def accuracy_computation(context: RiskEvaluationContext) -> float:
    """Compute error rate (1 - accuracy) as a risk measure."""
    predicted_labels = (context.cascade_scores.final_scores >= 0.5).astype(int)
    accuracy = accuracy_score(context.get_labels(), predicted_labels)
    return float(1.0 - accuracy)  # Error rate


def roc_auc_computation(context: RiskEvaluationContext) -> float:
    """Compute negative ROC AUC (1 - AUC) as a risk measure."""
    auc = roc_auc_score(context.get_labels(), context.cascade_scores.final_scores)
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
    description: str
    empirical_computation: Callable[[RiskEvaluationContext], float]
    p_value_bound_fn: BoundFunction


RISK_RGISTRY: dict[str, Risk] = {}  # Global registry for Risk instances by name


def register_risk(risk: Risk) -> None:
    """Register a Risk instance in the global registry."""
    if risk.name in RISK_RGISTRY:
        raise ValueError(f"Risk with name '{risk.name}' already registered.")
    RISK_RGISTRY[risk.name] = risk


# Pre-made Risk instances with sensible defaults
BudgetCostRisk = Risk(
    name="budget",
    description="Budget Cost",
    empirical_computation=budget_cost_computation,
    p_value_bound_fn=binomial,
)
register_risk(BudgetCostRisk)

AccuracyRisk = Risk(
    name="accuracy_error",
    description="Error Rate (1 - Accuracy)",
    empirical_computation=accuracy_computation,
    p_value_bound_fn=hb_p_value,
)
register_risk(AccuracyRisk)

RocAucRisk = Risk(
    name="roc_auc_error",
    description="Negative ROC AUC (1 - AUC)",
    empirical_computation=roc_auc_computation,
    p_value_bound_fn=hb_p_value,
)
register_risk(RocAucRisk)


@dataclass
class ThresholdEvaluationResult:
    """Results from evaluating cascade thresholds on calibration data.

    Always stores results in dict format. For single risk, dict has one entry.
    This enables both single-risk and multi-risk evaluation with a unified interface.
    """

    thresholds: np.ndarray  # Threshold grid evaluated
    n_samples: int  # Number of calibration samples
    empirical_risks: dict[str, np.ndarray]  # Always dict: risk_name -> empirical_risk_array
    risks: list[Risk]  # List of risks evaluated (single risk = list of length 1)

    def __getitem__(self, risk_name: str) -> np.ndarray:
        """Dictionary-like access: result['Budget Cost'] returns empirical risk array."""
        if risk_name not in self.empirical_risks:
            raise KeyError(f"Risk '{risk_name}' not found. Available: {list(self.empirical_risks.keys())}")
        return self.empirical_risks[risk_name]

    def get_risk(self, risk_name: str) -> Risk:
        """Get the Risk object by name."""
        for risk in self.risks:
            if risk.name == risk_name:
                return risk
        raise KeyError(f"Risk '{risk_name}' not found")

    def get_empirical_risks_array(self) -> np.ndarray:
        """Stack empirical risks into 2D array (n_thresholds × n_risks).

        Risks ordered alphabetically by name. Used for Pareto frontier computation.
        """
        if len(self.risks) == 1:
            # For single risk, return as column vector
            return self.empirical_risks[self.risks[0].name].reshape(-1, 1)

        ordered_names = sorted(self.empirical_risks.keys())
        return np.column_stack([self.empirical_risks[name] for name in ordered_names])

    def compute_p_values(self, alpha: float | dict[str, float]) -> dict[str, np.ndarray]:
        """Compute p-values using appropriate bound(s).

        Args:
            alpha: Single float (applied to all risks) OR dict mapping risk name to alpha

        Returns:
            Dict mapping risk name to p-value array
        """
        if isinstance(alpha, dict):
            alphas = alpha
        else:
            # Single alpha value: apply to all risks
            alphas = {risk.name: alpha for risk in self.risks}

        return {
            risk.name: risk.p_value_bound_fn(self.empirical_risks[risk.name], self.n_samples, alphas[risk.name])
            for risk in self.risks
        }


def evaluate_threshold_risks(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    thresholds: np.ndarray,
    *,
    risks: Risk | list[Risk],
    dataset: "LabelledDataset | None" = None,
    labels: np.ndarray | None = None,
    merge_strategy: str = "avg",
) -> ThresholdEvaluationResult:
    """Evaluate empirical risks for a grid of cascade thresholds.

    This function sweeps a threshold grid and computes the empirical risk(s)
    for each threshold on calibration data. Supports both single and multiple risks.

    Args:
        probe_scores: Probe predictions on calibration set, shape (n,)
        baseline_scores: Baseline predictions on calibration set, shape (n,)
        thresholds: Threshold grid to evaluate, shape (k,)
        risks: Single Risk or list of Risks to evaluate.
            Commonly: BudgetCostRisk, AccuracyRisk, RocAucRisk,
            or custom: Risk(name=..., empirical_computation=..., p_value_bound_fn=...)
        dataset: Optional dataset with labels (required for risks like AccuracyRisk, RocAucRisk).
        labels: Optional label array, shape (n,).  Alternative to ``dataset`` for
            providing labels.  When both are given, ``labels`` takes precedence.
        merge_strategy: Cascade merge strategy ("avg", "probe", "baseline")

    Returns:
        ThresholdEvaluationResult with:
            - thresholds: The evaluated threshold grid
            - empirical_risks: Dict mapping risk names to empirical risk arrays
            - n_samples: Number of calibration samples
            - risks: List of Risk instances evaluated

    Example:
        >>> from reliable_monitoring.risks import BudgetCostRisk, AccuracyRisk
        >>> result = evaluate_threshold_risks(
        ...     probe_scores,
        ...     baseline_scores,
        ...     thresholds=np.linspace(0.5, 1, 10),
        ...     risks=[BudgetCostRisk, AccuracyRisk],
        ...     dataset=dataset,
        ... )
        >>> # Compute p-values for each risk
        >>> p_values_dict = result.compute_p_values(alpha=0.3)
        >>> # Access specific risk results
        >>> budget_risks = result['Budget Cost']
    """
    from reliable_monitoring.cascade import run_offline_cascade

    # Normalize to list
    risk_list = [risks] if isinstance(risks, Risk) else risks

    n_samples = len(probe_scores)
    n_thresholds = len(thresholds)

    # Initialize storage for all risks
    empirical_risks_dict = {r.name: np.zeros(n_thresholds) for r in risk_list}

    # Sweep thresholds ONCE, evaluate all risks per threshold
    for i, threshold in enumerate(thresholds):
        cascade_result = run_offline_cascade(
            probe_scores,
            baseline_scores,
            threshold=threshold,
            merge_strategy=merge_strategy,
        )
        context = RiskEvaluationContext(
            cascade_scores=cascade_result,
            dataset=dataset,
            labels=labels,
        )
        for risk_obj in risk_list:
            empirical_risks_dict[risk_obj.name][i] = risk_obj.empirical_computation(context)

    return ThresholdEvaluationResult(
        thresholds=thresholds,
        n_samples=n_samples,
        empirical_risks=empirical_risks_dict,
        risks=risk_list,
    )
