import numpy as np
import pytest
import torch
from models_under_pressure.interfaces.dataset import LabelledDataset

from reliable_monitoring.dv_probes import build_dv_probe


@pytest.fixture
def activation_dataset():
    generator = torch.Generator().manual_seed(7)
    activations = torch.randn(36, 5, 10, generator=generator)
    mask = torch.ones(36, 5, dtype=torch.bool)
    mean = activations.mean(dim=1)
    dataset = LabelledDataset(
        inputs=[f"sample_{i}" for i in range(36)],
        ids=[str(i) for i in range(36)],
        other_fields={
            "labels": [0, 1] * 18,
            "activations": activations,
            "attention_mask": mask,
            "activations_mean": mean,
        },
    )
    targets = mean[:, 0].numpy()
    return dataset, targets


def test_ridge_dv_probe(activation_dataset):
    dataset, targets = activation_dataset
    probe = build_dv_probe("ridge")
    probe.fit(dataset, targets)
    predictions = probe.predict(dataset)
    assert predictions.shape == targets.shape
    assert np.corrcoef(predictions, targets)[0, 1] > 0.9


@pytest.mark.parametrize("architecture", ["attention", "softmax", "mlp"])
def test_torch_dv_probes(architecture, activation_dataset):
    dataset, targets = activation_dataset
    hyperparams = {"epochs": 3, "patience": 2, "batch_size": 8, "validation_batch_size": 2, "seed": 5}
    if architecture == "softmax":
        hyperparams["temperature"] = 5.0
        hyperparams["gradient_accumulation_steps"] = 4
    probe = build_dv_probe({"type": architecture, "hyperparams": hyperparams})
    probe.fit(dataset, targets)
    predictions = probe.predict(dataset)
    assert predictions.shape == targets.shape
    assert np.isfinite(predictions).all()
