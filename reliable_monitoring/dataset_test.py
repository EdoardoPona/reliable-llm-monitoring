"""Unit tests for dataset functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from models_under_pressure.interfaces.dataset import LabelledDataset

from reliable_monitoring.dataset import (
    ActivationConfig,
    compute_activations,
    load_dataset,
    reduce_activations,
)


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
        field = result.other_fields["activations_mean"]
        assert isinstance(field, (np.ndarray, torch.Tensor))
        assert field.shape == (10, 64)

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
            field = result.other_fields[field_name]
            assert isinstance(field, (np.ndarray, torch.Tensor))
            assert field.shape == (10, 64)

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
        field = result.other_fields["activations_custom"]
        assert isinstance(field, (np.ndarray, torch.Tensor))
        assert field.shape == (10, 64)

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
        field = result.other_fields["custom_activations_mean"]
        assert isinstance(field, (np.ndarray, torch.Tensor))
        assert field.shape == (n_samples, hidden_dim)


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
            reduce_activations(dataset, strategies={"bad": 12345})  # type: ignore[arg-type]


class TestLoadDatasetAutoCompute:
    """Test auto_compute and cleanup functionality in load_dataset."""

    @patch("reliable_monitoring.dataset.ActivationStore")
    @patch("reliable_monitoring.dataset.LabelledDataset.load_from")
    def test_raises_when_missing_and_auto_compute_false(self, mock_load_from, mock_store_class):
        """Missing activations raise FileNotFoundError when auto_compute=False."""
        mock_load_from.return_value = MagicMock()
        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_store_class.return_value = mock_store

        config = ActivationConfig(model_name="test-model", layer=5)
        with pytest.raises(FileNotFoundError, match="Set auto_compute=True"):
            load_dataset(Path("/fake/path.jsonl"), config, auto_compute=False)

    @patch("reliable_monitoring.dataset.compute_activations")
    @patch("reliable_monitoring.dataset.ActivationStore")
    @patch("reliable_monitoring.dataset.LabelledDataset.load_from")
    def test_computes_when_missing_and_auto_compute_true(self, mock_load_from, mock_store_class, mock_compute):
        """Activations are computed when missing and auto_compute=True."""
        mock_load_from.return_value = MagicMock()
        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_store_class.return_value = mock_store

        config = ActivationConfig(model_name="test-model", layer=5)
        load_dataset(Path("/fake/path.jsonl"), config, auto_compute=True)

        mock_compute.assert_called_once_with(
            dataset_path=Path("/fake/path.jsonl"),
            model="test-model",
            layer=5,
            batch_size=4,
        )

    @patch("reliable_monitoring.dataset.cleanup_activations")
    @patch("reliable_monitoring.dataset.ActivationStore")
    @patch("reliable_monitoring.dataset.LabelledDataset.load_from")
    def test_cleanup_called_when_requested(self, mock_load_from, mock_store_class, mock_cleanup):
        """cleanup_activations is called when cleanup_after_load=True."""
        mock_load_from.return_value = MagicMock()
        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store_class.return_value = mock_store

        config = ActivationConfig(model_name="test-model", layer=5)
        load_dataset(Path("/fake/path.jsonl"), config, cleanup_after_load=True)

        mock_cleanup.assert_called_once_with(
            dataset_path=Path("/fake/path.jsonl"),
            model_name="test-model",
            layer=5,
        )

    @patch("reliable_monitoring.dataset.cleanup_activations")
    @patch("reliable_monitoring.dataset.ActivationStore")
    @patch("reliable_monitoring.dataset.LabelledDataset.load_from")
    def test_cleanup_not_called_when_not_requested(self, mock_load_from, mock_store_class, mock_cleanup):
        """cleanup_activations is not called when cleanup_after_load=False."""
        mock_load_from.return_value = MagicMock()
        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store_class.return_value = mock_store

        config = ActivationConfig(model_name="test-model", layer=5)
        load_dataset(Path("/fake/path.jsonl"), config, cleanup_after_load=False)

        mock_cleanup.assert_not_called()


class TestComputeActivations:
    """Test compute_activations helper."""

    @patch("models_under_pressure.activation_store.ActivationsSpec")
    @patch("reliable_monitoring.dataset.ActivationStore")
    def test_skips_if_already_exists(self, mock_store_class, mock_spec_class):
        """Does nothing if activations already exist."""
        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store_class.return_value = mock_store

        compute_activations(Path("/fake/path.jsonl"), "test-model", layer=5)

        mock_store.save.assert_not_called()

    @patch("models_under_pressure.activation_store.ActivationsSpec")
    @patch("models_under_pressure.model.LLMModel.load")
    @patch("reliable_monitoring.dataset.LabelledDataset.load_from")
    @patch("reliable_monitoring.dataset.ActivationStore")
    def test_loads_model_when_given_string(self, mock_store_class, mock_load_from, mock_model_load, mock_spec_class):
        """Loads model when model parameter is a string."""
        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_store_class.return_value = mock_store
        mock_load_from.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.get_batched_activations_for_layers.return_value = (MagicMock(), MagicMock())
        mock_model_load.return_value = mock_model

        compute_activations(Path("/fake/path.jsonl"), "test-model", layer=5, batch_size=8)

        mock_model_load.assert_called_once_with("test-model", batch_size=8)

    @patch("models_under_pressure.activation_store.ActivationsSpec")
    @patch("models_under_pressure.model.LLMModel.load")
    @patch("reliable_monitoring.dataset.LabelledDataset.load_from")
    @patch("reliable_monitoring.dataset.ActivationStore")
    def test_uses_model_instance_directly(self, mock_store_class, mock_load_from, mock_model_load, mock_spec_class):
        """Uses model instance directly without loading when given LLMModel."""
        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_store_class.return_value = mock_store
        mock_load_from.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.get_batched_activations_for_layers.return_value = (MagicMock(), MagicMock())

        compute_activations(Path("/fake/path.jsonl"), mock_model, layer=5)

        # Should not call load since we passed an instance
        mock_model_load.assert_not_called()
        # Should use the instance directly
        mock_model.get_batched_activations_for_layers.assert_called_once()
