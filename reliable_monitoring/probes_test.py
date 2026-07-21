"""Unit tests for probe implementations."""

import numpy as np
import pytest
import torch
from models_under_pressure.interfaces.dataset import LabelledDataset

from reliable_monitoring.dataset import reduce_activations
from reliable_monitoring.probes import (
    LinearProbe,
    SequenceProbe,
    TorchSequenceProbe,
    _device_sequence_batches,
    _prepare_sequence_batch,
    _tensor_bytes,
    build_probe,
    default_torch_device,
    probe_requires_raw_activations,
)


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


@pytest.mark.parametrize("architecture", ["attention", "softmax", "mlp"])
def test_torch_probe_architectures(architecture):
    generator = torch.Generator().manual_seed(4)
    activations = torch.randn(40, 6, 12, generator=generator)
    mask = torch.ones(40, 6, dtype=torch.bool)
    mask[:10, -2:] = False
    activations[~mask] = 0
    labels = (activations[:, :, 0].sum(dim=1) > 0).int().tolist()
    dataset = LabelledDataset(
        inputs=[f"sample_{i}" for i in range(40)],
        ids=[str(i) for i in range(40)],
        other_fields={"labels": labels, "activations": activations, "attention_mask": mask},
    )
    hyperparams = {"epochs": 3, "patience": 2, "batch_size": 8, "validation_batch_size": 2, "seed": 3}
    if architecture == "softmax":
        hyperparams["temperature"] = 5.0
        hyperparams["gradient_accumulation_steps"] = 4
    probe = build_probe({"type": architecture, "hyperparams": hyperparams})
    probe.fit(dataset)
    scores = probe.predict(dataset)
    assert scores.shape == (40,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_sequence_probe_preserves_float16_activations():
    activations = np.zeros((3, 4, 5), dtype=np.float16)
    dataset = LabelledDataset(
        inputs=["a", "b", "c"],
        ids=["0", "1", "2"],
        other_fields={"labels": [0, 1, 0], "activations": activations},
    )
    x, _ = TorchSequenceProbe._arrays(dataset)
    assert x.dtype == torch.float16
    assert x.untyped_storage().data_ptr() == torch.as_tensor(activations).untyped_storage().data_ptr()


def test_prepare_sequence_batch_removes_batch_wide_padding():
    x = torch.randn(2, 7, 3)
    mask = torch.tensor(
        [
            [False, True, True, True, False, False, False],
            [False, True, True, True, True, False, False],
        ]
    )
    trimmed_x, trimmed_mask = _prepare_sequence_batch(x, mask, torch.device("cpu"))
    assert trimmed_x.shape == (2, 4, 3)
    assert trimmed_mask.shape == (2, 4)


def test_device_sequence_batches_preserve_requested_order():
    x = torch.arange(30).reshape(5, 3, 2).float()
    mask = torch.ones(5, 3, dtype=torch.bool)
    targets = torch.arange(5).float()
    indices = np.array([4, 1, 3])
    batches = list(_device_sequence_batches(x, mask, targets, indices, batch_size=2, shuffle=False, seed=0))
    assert torch.equal(torch.cat([batch[0] for batch in batches]), x[indices])
    assert torch.equal(torch.cat([batch[2] for batch in batches]), targets[indices])
    assert _tensor_bytes(x, mask, targets) == sum(
        tensor.numel() * tensor.element_size() for tensor in (x, mask, targets)
    )


def test_probe_factory_and_raw_requirement():
    assert isinstance(build_probe("mean_logreg"), SequenceProbe)
    assert probe_requires_raw_activations("attention")
    assert probe_requires_raw_activations("softmax")
    assert not probe_requires_raw_activations("mlp")


def test_default_device_uses_mps_before_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_torch_device().type == "mps"
