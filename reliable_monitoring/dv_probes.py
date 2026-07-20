"""Delegation-value probe implementations and configuration factory."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch
from models_under_pressure.interfaces.dataset import LabelledDataset
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reliable_monitoring.probes import (
    AttentionProbeModule,
    MeanMLPProbeModule,
    SoftmaxProbeModule,
    TorchSequenceProbe,
    default_torch_device,
)

logger = logging.getLogger(__name__)


class DVProbe(Protocol):
    def fit(self, dataset: LabelledDataset, targets: np.ndarray) -> None: ...

    def predict(self, dataset: LabelledDataset) -> np.ndarray: ...


DV_PROBE_REGISTRY: dict[str, Callable[..., DVProbe]] = {}


def register_dv_probe(name: str):
    def decorator(constructor: Callable[..., DVProbe]) -> Callable[..., DVProbe]:
        if name in DV_PROBE_REGISTRY:
            raise ValueError(f"DV probe {name!r} is already registered")
        DV_PROBE_REGISTRY[name] = constructor
        return constructor

    return decorator


def build_dv_probe(
    spec: str | dict[str, Any] | None = None,
    *,
    safety_probe=None,
    **overrides: Any,
) -> DVProbe:
    """Build a DV regressor from a name or ``{type, hyperparams}`` mapping."""
    if spec is None:
        name, hyperparams = "ridge", {}
    elif isinstance(spec, str):
        name, hyperparams = spec, {}
    else:
        name = spec.get("type", "ridge")
        hyperparams = dict(spec.get("hyperparams", {}))
    hyperparams.update(overrides)
    try:
        return DV_PROBE_REGISTRY[name](safety_probe=safety_probe, **hyperparams)
    except KeyError as exc:
        raise ValueError(f"Unknown DV probe {name!r}; available: {sorted(DV_PROBE_REGISTRY)}") from exc


class RidgeDVProbe:
    def __init__(self, *, alpha: float = 1.0, activation_field: str = "activations_mean", **_: Any):
        self.activation_field = activation_field
        self.model = Ridge(alpha=alpha)

    def _features(self, dataset: LabelledDataset) -> np.ndarray:
        if self.activation_field not in dataset.other_fields:
            raise ValueError(f"Dataset missing {self.activation_field!r}")
        return np.asarray(dataset.other_fields[self.activation_field])

    def fit(self, dataset: LabelledDataset, targets: np.ndarray) -> None:
        self.model.fit(self._features(dataset), targets)

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        return self.model.predict(self._features(dataset))


register_dv_probe("ridge")(RidgeDVProbe)


_DV_MODULES: dict[str, type[nn.Module]] = {
    "attention": AttentionProbeModule,
    "softmax": SoftmaxProbeModule,
    "mlp": MeanMLPProbeModule,
}


class TorchDVProbe:
    """Torch sequence regressor trained on continuous delegation value."""

    def __init__(
        self,
        architecture: str,
        *,
        safety_probe=None,
        reuse_attention: bool = False,
        seed: int = 42,
        batch_size: int = 16,
        epochs: int = 200,
        learning_rate: float = 5e-3,
        final_learning_rate: float = 1e-4,
        patience: int = 50,
        validation_fraction: float = 0.1,
        weight_decay: float = 0.0,
        gradient_accumulation_steps: int = 1,
        device: str | None = None,
        **module_kwargs: Any,
    ):
        self.architecture = architecture
        self.safety_probe = safety_probe
        self.reuse_attention = reuse_attention
        self.seed = seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.final_learning_rate = final_learning_rate
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.weight_decay = weight_decay
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device(device) if device else default_torch_device()
        self.module_kwargs = module_kwargs
        self.model: nn.Module | None = None

    def fit(self, dataset: LabelledDataset, targets: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        x, mask = TorchSequenceProbe._arrays(dataset)
        y = torch.as_tensor(targets, dtype=torch.float32)
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(y))
        n_val = max(1, int(len(y) * self.validation_fraction))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        self.model = _DV_MODULES[self.architecture](x.shape[-1], **self.module_kwargs).to(self.device)
        if self.reuse_attention:
            source = getattr(self.safety_probe, "model", None)
            if self.architecture != "attention" or not isinstance(source, AttentionProbeModule):
                raise ValueError("reuse_attention requires a fitted attention safety probe")
            self.model.context_query.load_state_dict(source.context_query.state_dict())
            for parameter in self.model.context_query.parameters():
                parameter.requires_grad = False
        optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.final_learning_rate
        )
        loader = DataLoader(
            TensorDataset(x[train_idx], mask[train_idx], y[train_idx]),
            batch_size=self.batch_size,
            shuffle=True,
        )
        criterion = nn.MSELoss()
        best_loss, stale, best_state = float("inf"), 0, None
        logger.info(
            "Training %s DV probe on %s for at most %d epochs (batch=%d, accumulation=%d)",
            self.architecture,
            self.device,
            self.epochs,
            self.batch_size,
            self.gradient_accumulation_steps,
        )
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            for batch_index, (xb, mb, yb) in enumerate(loader):
                loss = criterion(self.model(xb.to(self.device), mb.to(self.device)), yb.to(self.device))
                (loss / self.gradient_accumulation_steps).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                is_accumulation_boundary = (batch_index + 1) % self.gradient_accumulation_steps == 0
                is_last_batch = batch_index + 1 == len(loader)
                if is_accumulation_boundary or is_last_batch:
                    optimizer.step()
                    optimizer.zero_grad()
            scheduler.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = float(
                    criterion(
                        self.model(x[val_idx].to(self.device), mask[val_idx].to(self.device)),
                        y[val_idx].to(self.device),
                    )
                )
            if val_loss < best_loss - 1e-6:
                best_loss, stale = val_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    logger.info("Early stopping %s DV probe after epoch %d", self.architecture, epoch + 1)
                    break
            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.info(
                    "%s DV probe epoch %d/%d: validation loss %.5f",
                    self.architecture,
                    epoch + 1,
                    self.epochs,
                    val_loss,
                )
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        if self.model is None:
            raise ValueError("DV probe has not been fitted")
        x, mask = TorchSequenceProbe._arrays(dataset)
        loader = DataLoader(TensorDataset(x, mask), batch_size=max(self.batch_size, 64))
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for xb, mb in loader:
                outputs.append(self.model(xb.to(self.device), mb.to(self.device)).cpu())
        return torch.cat(outputs).numpy()


for _architecture in _DV_MODULES:
    register_dv_probe(_architecture)(
        lambda architecture=_architecture, **kwargs: TorchDVProbe(architecture=architecture, **kwargs)
    )
