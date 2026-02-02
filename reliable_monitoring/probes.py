"""Probe implementations for classification based on model activations.

This module provides a Protocol-based interface for probes and two implementations:
- LinearProbe: Works with pre-reduced activations (most efficient)
- SequenceProbe: Handles raw sequence activations on-the-fly (flexible)
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from models_under_pressure.interfaces.dataset import LabelledDataset
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@runtime_checkable
class Probe(Protocol):
    """Protocol for probe models that predict from datasets.

    A probe takes a LabelledDataset and returns probability scores for the
    positive class. The probe handles extracting the appropriate features
    from the dataset's other_fields.

    The protocol supports both training and inference:
    - fit(dataset): Train the probe on a dataset
    - predict(dataset): Generate predictions on a dataset
    """

    def fit(self, dataset: LabelledDataset) -> None:
        """Train the probe on the given dataset.

        Args:
            dataset: Training dataset with labels and features in other_fields
        """
        ...

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        """Generate probability predictions for the positive class.

        Args:
            dataset: Dataset to predict on

        Returns:
            Array of probabilities for positive class, shape (n_samples,)
        """
        ...

    def calibrate(self, dataset: LabelledDataset, method: str = "platt-scaling") -> None:
        """Calibrate the probe's probability outputs.

        Args:
            dataset: Dataset with labels and features in other_fields
            method: Calibration method - "platt-scaling" or "isotonic-regression"
        """
        ...


class LinearProbe:
    """Linear probe using logistic regression on reduced activations.

    This is the most efficient probe implementation. It expects reduced activations
    (already aggregated over the sequence dimension) in the dataset's other_fields.

    Use this when:
    - You've pre-computed reductions using reduce_activations()
    - You want maximum inference speed
    - Memory is not a constraint (can store reduced activations)

    Attributes:
        activation_field: Name of field containing reduced activations
        clf: The underlying sklearn classifier
    """

    def __init__(
        self,
        activation_field: str = "activations_mean",
        max_iter: int = 1000,
        **sklearn_kwargs,
    ):
        """Initialize linear probe.

        Args:
            activation_field: Name of dataset field with reduced activations.
                Default is "activations_mean" (assumes mean reduction was applied).
            max_iter: Maximum iterations for logistic regression
            **sklearn_kwargs: Additional arguments passed to LogisticRegression
        """
        self.activation_field = activation_field
        self.clf = LogisticRegression(max_iter=max_iter, **sklearn_kwargs)
        self.calibration_clf: LogisticRegression | IsotonicRegression | None = None

    def fit(self, dataset: LabelledDataset) -> None:
        """Train the probe on reduced activations.

        Args:
            dataset: Dataset with self.activation_field in other_fields

        Raises:
            ValueError: If the required activation field is missing
        """
        if self.activation_field not in dataset.other_fields:
            raise ValueError(
                f"Dataset missing field '{self.activation_field}'. "
                f"Did you forget to call reduce_activations()? "
                f"Available fields: {list(dataset.other_fields.keys())}"
            )

        X = dataset.other_fields[self.activation_field]
        y = dataset.labels_numpy()

        # Convert to numpy if tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        self.clf.fit(X, y)

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        """Predict probabilities for positive class.

        Args:
            dataset: Dataset with self.activation_field in other_fields

        Returns:
            Probability scores for positive class, shape (n_samples,)

        Raises:
            ValueError: If the required activation field is missing
        """
        if self.activation_field not in dataset.other_fields:
            raise ValueError(
                f"Dataset missing field '{self.activation_field}'. "
                f"Did you forget to call reduce_activations()? "
                f"Available fields: {list(dataset.other_fields.keys())}"
            )

        X = dataset.other_fields[self.activation_field]

        # Convert to numpy if tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        scores = self.clf.predict_proba(X)[:, 1]

        # Apply calibration if fitted
        if self.calibration_clf is not None:
            if isinstance(self.calibration_clf, IsotonicRegression):
                scores = self.calibration_clf.predict(scores)
            else:
                scores = self.calibration_clf.predict_proba(scores.reshape(-1, 1))[:, 1]

        return scores

    def calibrate(self, dataset: LabelledDataset, method: str = "platt-scaling") -> None:
        """Calibrate the probe's probability outputs.

        Args:
            dataset: Dataset with self.activation_field in other_fields
            method: Calibration method - "platt-scaling" or "isotonic-regression"

        Raises:
            ValueError: If the required activation field is missing or method is invalid
        """
        if method not in ("platt-scaling", "isotonic-regression"):
            raise ValueError(f"Unknown calibration method: {method}")

        if self.activation_field not in dataset.other_fields:
            raise ValueError(
                f"Dataset missing field '{self.activation_field}'. "
                f"Did you forget to call reduce_activations()? "
                f"Available fields: {list(dataset.other_fields.keys())}"
            )

        X = dataset.other_fields[self.activation_field]
        y = dataset.labels_numpy()

        # Convert to numpy if tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # Get current probe scores
        scores = self.clf.predict_proba(X)[:, 1]

        # Fit calibration model
        if method == "platt-scaling":
            self.calibration_clf = LogisticRegression(max_iter=1000)
            self.calibration_clf.fit(scores.reshape(-1, 1), y)
        else:  # isotonic-regression
            self.calibration_clf = IsotonicRegression(out_of_bounds="clip")
            self.calibration_clf.fit(scores, y)


class SequenceProbe:
    """Linear probe that handles raw sequence activations.

    This probe computes reduction on-the-fly during inference. Use this when:
    - You want to keep raw activations in the dataset
    - You're experimenting with different reduction strategies
    - Memory isn't a constraint

    For production use with large datasets, prefer LinearProbe with pre-computed reductions.

    Attributes:
        reduction_strategy: Name of reduction strategy or custom function
        batch_size: Batch size for on-the-fly reduction
        clf: The underlying sklearn classifier
    """

    def __init__(
        self,
        reduction_strategy: str | Callable = "mean",
        max_iter: int = 1000,
        batch_size: int = 256,
        **sklearn_kwargs,
    ):
        """Initialize sequence probe.

        Args:
            reduction_strategy: How to reduce sequence dimension.
                Can be built-in name ("mean", "max", etc.) or custom function.
            max_iter: Maximum iterations for logistic regression
            batch_size: Batch size for on-the-fly reduction
            **sklearn_kwargs: Additional arguments passed to LogisticRegression
        """
        self.reduction_strategy = reduction_strategy
        self.batch_size = batch_size
        self.clf = LogisticRegression(max_iter=max_iter, **sklearn_kwargs)
        self.calibration_clf: LogisticRegression | IsotonicRegression | None = None

    def _get_reduced_activations(self, dataset: LabelledDataset) -> np.ndarray:
        """Compute reduced activations from raw sequence activations."""
        from reliable_monitoring.reductions import apply_reduction_batched, get_reduction_function

        activations = dataset.other_fields["activations"]
        attention_mask = dataset.other_fields["attention_mask"]

        # Convert to torch.Tensor if needed
        if not isinstance(activations, torch.Tensor):
            activations = torch.tensor(activations)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)

        # Get reduction function
        if isinstance(self.reduction_strategy, str):
            reduction_fn = get_reduction_function(self.reduction_strategy)
        else:
            reduction_fn = self.reduction_strategy

        # Apply reduction
        reduced = apply_reduction_batched(
            activations=activations,
            attention_mask=attention_mask,
            reduction_fn=reduction_fn,
            batch_size=self.batch_size,
            show_progress=True,
        )

        return reduced.numpy()

    def fit(self, dataset: LabelledDataset) -> None:
        """Train the probe on raw sequence activations.

        Args:
            dataset: Dataset with "activations" and "attention_mask" fields

        Raises:
            ValueError: If required fields are missing
        """
        if "activations" not in dataset.other_fields:
            raise ValueError("Dataset missing 'activations' field")
        if "attention_mask" not in dataset.other_fields:
            raise ValueError("Dataset missing 'attention_mask' field")

        X = self._get_reduced_activations(dataset)
        y = dataset.labels_numpy()
        self.clf.fit(X, y)

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        """Predict probabilities for positive class.

        Args:
            dataset: Dataset with "activations" and "attention_mask" fields

        Returns:
            Probability scores for positive class, shape (n_samples,)

        Raises:
            ValueError: If required fields are missing
        """
        if "activations" not in dataset.other_fields:
            raise ValueError("Dataset missing 'activations' field")
        if "attention_mask" not in dataset.other_fields:
            raise ValueError("Dataset missing 'attention_mask' field")

        X = self._get_reduced_activations(dataset)
        scores = self.clf.predict_proba(X)[:, 1]

        # Apply calibration if fitted
        if self.calibration_clf is not None:
            if isinstance(self.calibration_clf, IsotonicRegression):
                scores = self.calibration_clf.predict(scores)
            else:
                scores = self.calibration_clf.predict_proba(scores.reshape(-1, 1))[:, 1]

        return scores

    def calibrate(self, dataset: LabelledDataset, method: str = "platt-scaling") -> None:
        """Calibrate the probe's probability outputs.

        Args:
            dataset: Dataset with "activations" and "attention_mask" fields
            method: Calibration method - "platt-scaling" or "isotonic-regression"

        Raises:
            ValueError: If required fields are missing or method is invalid
        """
        if method not in ("platt-scaling", "isotonic-regression"):
            raise ValueError(f"Unknown calibration method: {method}")

        if "activations" not in dataset.other_fields:
            raise ValueError("Dataset missing 'activations' field")
        if "attention_mask" not in dataset.other_fields:
            raise ValueError("Dataset missing 'attention_mask' field")

        X = self._get_reduced_activations(dataset)
        y = dataset.labels_numpy()

        # Get current probe scores
        scores = self.clf.predict_proba(X)[:, 1]

        # Fit calibration model
        if method == "platt-scaling":
            self.calibration_clf = LogisticRegression(max_iter=1000)
            self.calibration_clf.fit(scores.reshape(-1, 1), y)
        else:  # isotonic-regression
            self.calibration_clf = IsotonicRegression(out_of_bounds="clip")
            self.calibration_clf.fit(scores, y)

