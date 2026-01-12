"""Unit tests for probe implementations."""

import numpy as np
import pytest
import torch
from models_under_pressure.interfaces.dataset import LabelledDataset

from reliable_monitoring.dataset import reduce_activations
from reliable_monitoring.probes import LinearProbe, SequenceProbe


class TestLinearProbe:
    """Test LinearProbe implementation."""

    def test_linear_probe_initialization(self):
        """Test LinearProbe initialization."""
        probe = LinearProbe(activation_field="activations_mean", max_iter=500)

        assert probe.activation_field == "activations_mean"
        assert probe.clf.max_iter == 500

    def test_linear_probe_missing_field_raises_error_fit(self):
        """Test that fit raises error when activation field is missing."""
        dataset = LabelledDataset(
            inputs=["test"] * 10,
            ids=[str(i) for i in range(10)],
            other_fields={"labels": [0, 1] * 5, "wrong_field": torch.randn(10, 64)},
        )

        probe = LinearProbe(activation_field="activations_mean")

        with pytest.raises(ValueError, match="Dataset missing field 'activations_mean'"):
            probe.fit(dataset)

    def test_linear_probe_missing_field_raises_error_predict(self):
        """Test that predict raises error when activation field is missing."""
        # Create a simple trained probe
        train_data = LabelledDataset(
            inputs=["test"] * 10,
            ids=[str(i) for i in range(10)],
            other_fields={"labels": [0, 1] * 5, "activations_mean": torch.randn(10, 64)},
        )
        probe = LinearProbe(activation_field="activations_mean")
        probe.fit(train_data)

        # Try to predict on dataset without the field
        test_dataset = LabelledDataset(
            inputs=["test"] * 10,
            ids=[str(i) for i in range(10)],
            other_fields={"labels": [0, 1] * 5, "wrong_field": torch.randn(10, 64)},
        )

        with pytest.raises(ValueError, match="Dataset missing field 'activations_mean'"):
            probe.predict(test_dataset)


class TestSequenceProbe:
    """Test SequenceProbe implementation."""

    def test_sequence_probe_initialization(self):
        """Test SequenceProbe initialization."""
        probe = SequenceProbe(reduction_strategy="mean", max_iter=500, batch_size=128)

        assert probe.reduction_strategy == "mean"
        assert probe.batch_size == 128
        assert probe.clf.max_iter == 500

    def test_sequence_probe_with_different_strategies(self):
        """Test SequenceProbe with different reduction strategies."""
        n_samples, seq_len, hidden_dim = 50, 10, 32
        activations = torch.randn(n_samples, seq_len, hidden_dim)
        attention_mask = torch.ones(n_samples, seq_len)
        labels = np.random.randint(0, 2, n_samples)

        dataset = LabelledDataset(
            inputs=[f"sample_{i}" for i in range(n_samples)],
            ids=[str(i) for i in range(n_samples)],
            other_fields={
                "labels": labels.tolist(),
                "activations": activations,
                "attention_mask": attention_mask,
            },
        )

        strategies = ["mean", "max", "first", "last"]

        for strategy in strategies:
            probe = SequenceProbe(reduction_strategy=strategy)
            probe.fit(dataset)
            scores = probe.predict(dataset)

            assert scores.shape == (n_samples,), f"Failed for strategy: {strategy}"
            assert np.all((scores >= 0) & (scores <= 1)), f"Invalid probabilities for strategy: {strategy}"

    def test_sequence_probe_with_custom_reduction(self):
        """Test SequenceProbe with custom reduction function."""

        def custom_reduction(activations, attention_mask):
            # Just take the mean, ignoring mask
            return activations.mean(dim=1)

        n_samples, seq_len, hidden_dim = 50, 10, 32
        activations = torch.randn(n_samples, seq_len, hidden_dim)
        attention_mask = torch.ones(n_samples, seq_len)
        labels = np.random.randint(0, 2, n_samples)

        dataset = LabelledDataset(
            inputs=[f"sample_{i}" for i in range(n_samples)],
            ids=[str(i) for i in range(n_samples)],
            other_fields={
                "labels": labels.tolist(),
                "activations": activations,
                "attention_mask": attention_mask,
            },
        )

        probe = SequenceProbe(reduction_strategy=custom_reduction)
        probe.fit(dataset)
        scores = probe.predict(dataset)

        assert scores.shape == (n_samples,)

    def test_sequence_probe_missing_activations_raises_error(self):
        """Test that missing activations raises error."""
        dataset = LabelledDataset(
            inputs=["test"] * 10,
            ids=[str(i) for i in range(10)],
            other_fields={"labels": [0, 1] * 5, "attention_mask": torch.ones(10, 5)},
        )

        probe = SequenceProbe()

        with pytest.raises(ValueError, match="Dataset missing 'activations' field"):
            probe.fit(dataset)

    def test_sequence_probe_missing_mask_raises_error(self):
        """Test that missing attention mask raises error."""
        dataset = LabelledDataset(
            inputs=["test"] * 10,
            ids=[str(i) for i in range(10)],
            other_fields={"labels": [0, 1] * 5, "activations": torch.randn(10, 5, 32)},
        )

        probe = SequenceProbe()

        with pytest.raises(ValueError, match="Dataset missing 'attention_mask' field"):
            probe.fit(dataset)


class TestProbeEquivalence:
    """Test equivalence between LinearProbe and SequenceProbe."""

    def test_linear_and_sequence_probe_equivalence(self):
        """Test that LinearProbe and SequenceProbe give same results when using same data."""
        # Create dataset with raw activations
        n_samples, seq_len, hidden_dim = 50, 10, 32
        activations = torch.randn(n_samples, seq_len, hidden_dim)
        attention_mask = torch.ones(n_samples, seq_len)
        labels = np.random.randint(0, 2, n_samples)

        dataset_raw = LabelledDataset(
            inputs=[f"sample_{i}" for i in range(n_samples)],
            ids=[str(i) for i in range(n_samples)],
            other_fields={"labels": labels.tolist(), "activations": activations, "attention_mask": attention_mask},
        )

        # Create dataset with pre-reduced activations
        dataset_reduced = reduce_activations(dataset_raw, "mean", inplace=False)

        # Train SequenceProbe on raw activations
        sequence_probe = SequenceProbe(reduction_strategy="mean")
        sequence_probe.fit(dataset_raw)
        sequence_scores = sequence_probe.predict(dataset_raw)

        # Train LinearProbe on reduced activations
        linear_probe = LinearProbe(activation_field="activations_mean")
        linear_probe.fit(dataset_reduced)
        linear_scores = linear_probe.predict(dataset_reduced)

        # Results should be very similar (sklearn might have minor numerical differences)
        correlation = np.corrcoef(sequence_scores, linear_scores)[0, 1]
        assert correlation > 0.95  # Should be highly correlated
