"""Unit tests for dataset reduction functionality."""

import pytest
import torch
from models_under_pressure.interfaces.dataset import LabelledDataset

from reliable_monitoring.dataset import reduce_activations


class TestReduceActivations:
    """Test reduce_activations function."""

    def create_mock_dataset(self, n_samples=10, seq_len=5, hidden_dim=64):
        """Helper to create a mock dataset with activations."""
        activations = torch.randn(n_samples, seq_len, hidden_dim)
        attention_mask = torch.ones(n_samples, seq_len)

        dataset = LabelledDataset(
            inputs=["test"] * n_samples,
            ids=[str(i) for i in range(n_samples)],
            other_fields={
                "labels": [0, 1] * (n_samples // 2),
                "activations": activations,
                "attention_mask": attention_mask,
            },
        )
        return dataset

    def test_reduce_single_strategy(self):
        """Test reducing activations with single strategy."""
        dataset = self.create_mock_dataset(n_samples=10, seq_len=5, hidden_dim=64)

        result = reduce_activations(dataset, "mean")

        # Check new field exists
        assert "activations_mean" in result.other_fields

        # Check shape is correct (sequence dimension removed)
        assert result.other_fields["activations_mean"].shape == (10, 64)

        # Original field should still exist
        assert "activations" in result.other_fields

    def test_reduce_multiple_strategies(self):
        """Test reducing with multiple strategies."""
        dataset = self.create_mock_dataset(n_samples=10, seq_len=5, hidden_dim=64)

        result = reduce_activations(dataset, strategies={"mean": "mean", "max": "max", "last": "last"})

        # Check all fields exist
        assert "activations_mean" in result.other_fields
        assert "activations_max" in result.other_fields
        assert "activations_last" in result.other_fields

        # All should have correct shape
        for field_name in ["activations_mean", "activations_max", "activations_last"]:
            assert result.other_fields[field_name].shape == (10, 64)

    def test_reduce_with_drop_raw(self):
        """Test dropping raw activations after reduction."""
        dataset = self.create_mock_dataset(n_samples=10, seq_len=5, hidden_dim=64)

        result = reduce_activations(dataset, "mean", drop_raw=True)

        # Raw field should be gone
        assert "activations" not in result.other_fields

        # Reduced field should exist
        assert "activations_mean" in result.other_fields

        # Other fields should remain
        assert "attention_mask" in result.other_fields

    def test_reduce_with_custom_function(self):
        """Test reduction with custom function."""

        def custom_reduction(activations, attention_mask):
            # Simple custom: sum over sequence
            return activations.sum(dim=1)

        dataset = self.create_mock_dataset(n_samples=10, seq_len=5, hidden_dim=64)

        result = reduce_activations(dataset, strategies={"custom": custom_reduction})

        # Check field exists
        assert "activations_custom" in result.other_fields

        # Check shape
        assert result.other_fields["activations_custom"].shape == (10, 64)

    def test_reduce_respects_inplace_false(self):
        """Test that inplace=False creates a copy."""
        dataset = self.create_mock_dataset(n_samples=10, seq_len=5, hidden_dim=64)
        original_fields = set(dataset.other_fields.keys())

        result = reduce_activations(dataset, "mean", inplace=False)

        # Original should be unchanged
        assert set(dataset.other_fields.keys()) == original_fields
        assert "activations_mean" not in dataset.other_fields

        # Result should have new field
        assert "activations_mean" in result.other_fields

    def test_reduce_with_custom_field_names(self):
        """Test reduction with custom activation field name."""
        n_samples, seq_len, hidden_dim = 10, 5, 64
        activations = torch.randn(n_samples, seq_len, hidden_dim)
        attention_mask = torch.ones(n_samples, seq_len)

        dataset = LabelledDataset(
            inputs=["test"] * n_samples,
            ids=[str(i) for i in range(n_samples)],
            other_fields={
                "labels": [0, 1] * (n_samples // 2),
                "custom_activations": activations,
                "custom_mask": attention_mask,
            },
        )

        result = reduce_activations(
            dataset,
            "mean",
            activation_field="custom_activations",
            mask_field="custom_mask",
        )

        # Should create field with custom prefix
        assert "custom_activations_mean" in result.other_fields
        assert result.other_fields["custom_activations_mean"].shape == (n_samples, hidden_dim)


class TestReduceActivationsErrors:
    """Test error handling in reduce_activations."""

    def test_missing_activation_field_raises_error(self):
        """Test that missing activation field raises ValueError."""
        dataset = LabelledDataset(
            inputs=["test"] * 10,
            ids=[str(i) for i in range(10)],
            other_fields={"labels": [0, 1] * 5, "attention_mask": torch.ones(10, 5)},
        )

        with pytest.raises(ValueError, match="Dataset missing field 'activations'"):
            reduce_activations(dataset, "mean")

    def test_missing_mask_field_raises_error(self):
        """Test that missing attention mask raises ValueError."""
        dataset = LabelledDataset(
            inputs=["test"] * 10,
            ids=[str(i) for i in range(10)],
            other_fields={"labels": [0, 1] * 5, "activations": torch.randn(10, 5, 64)},
        )

        with pytest.raises(ValueError, match="Dataset missing field 'attention_mask'"):
            reduce_activations(dataset, "mean")

    def test_invalid_activation_shape_raises_error(self):
        """Test that invalid activation shape raises ValueError."""
        # 2D activations (missing sequence dimension)
        dataset = LabelledDataset(
            inputs=["test"] * 10,
            ids=[str(i) for i in range(10)],
            other_fields={
                "labels": [0, 1] * 5,
                "activations": torch.randn(10, 64),  # Should be (10, seq_len, 64)
                "attention_mask": torch.ones(10, 5),
            },
        )

        with pytest.raises(ValueError, match="Expected activations with shape"):
            reduce_activations(dataset, "mean")

    def test_unknown_strategy_raises_error(self):
        """Test that unknown strategy name raises ValueError."""
        activations = torch.randn(10, 5, 64)
        attention_mask = torch.ones(10, 5)

        dataset = LabelledDataset(
            inputs=["test"] * 10,
            ids=[str(i) for i in range(10)],
            other_fields={"labels": [0, 1] * 5, "activations": activations, "attention_mask": attention_mask},
        )

        with pytest.raises(ValueError, match="Unknown reduction strategy"):
            reduce_activations(dataset, "nonexistent_strategy")

    def test_invalid_strategy_type_raises_error(self):
        """Test that invalid strategy type raises TypeError."""
        activations = torch.randn(10, 5, 64)
        attention_mask = torch.ones(10, 5)

        dataset = LabelledDataset(
            inputs=["test"] * 10,
            ids=[str(i) for i in range(10)],
            other_fields={"labels": [0, 1] * 5, "activations": activations, "attention_mask": attention_mask},
        )

        with pytest.raises(TypeError, match="Strategy must be string or callable"):
            reduce_activations(dataset, strategies={"bad": 12345})  # Integer instead of function
