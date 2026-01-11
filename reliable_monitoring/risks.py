from models_under_pressure.interfaces.dataset import LabelledDataset
from sklearn.metrics import accuracy_score, roc_auc_score

from reliable_monitoring.cascade import CascadePredictionResults


def baseline_budget_cost(cascade_scores: CascadePredictionResults) -> float:
    """Rate at which you call the baseline"""
    return cascade_scores.used_baseline.mean()


def empirical_roc_auc(cascade_scores: CascadePredictionResults, dataset: LabelledDataset) -> float:
    """Empirical performance of the cascade."""
    return roc_auc_score(
        dataset.labels_numpy(),
        cascade_scores.final_scores,
    )


def empirical_accuracy(cascade_scores: CascadePredictionResults, dataset: LabelledDataset) -> float:
    """Empirical accuracy of the cascade."""
    predicted_labels = (cascade_scores.final_scores >= 0.5).astype(int)
    return accuracy_score(
        dataset.labels_numpy(),
        predicted_labels,
    )
