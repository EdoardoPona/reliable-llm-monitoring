"""Probe implementations for classification based on model activations.

This module provides a Protocol-based interface for probes and two implementations:
- LinearProbe: Works with pre-reduced activations (most efficient)
- SequenceProbe: Handles raw sequence activations on-the-fly (flexible)
"""

import copy
import gc
import logging
import random
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from models_under_pressure.interfaces.dataset import LabelledDataset
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

logger = logging.getLogger(__name__)


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

    @property
    def _pre_reduced_field(self) -> str | None:
        """Field name for pre-reduced activations, or None if using a custom callable."""
        if isinstance(self.reduction_strategy, str):
            return f"activations_{self.reduction_strategy}"
        return None

    def _get_reduced_activations(self, dataset: LabelledDataset) -> np.ndarray:
        """Get reduced activations, using pre-computed ones if available.

        Checks for a pre-reduced field ``activations_{reduction_strategy}``
        first.  If found, returns it directly (avoids recomputing from raw
        activations).  Otherwise falls back to reducing raw activations
        on-the-fly.
        """
        pre_reduced_key = self._pre_reduced_field
        if pre_reduced_key is not None and pre_reduced_key in dataset.other_fields:
            reduced = dataset.other_fields[pre_reduced_key]
            if isinstance(reduced, torch.Tensor):
                reduced = reduced.numpy()
            return np.asarray(reduced)

        # Fall back to computing from raw activations
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

    def _validate_activation_fields(self, dataset: LabelledDataset) -> None:
        """Validate that dataset has required activation fields."""
        pre_reduced_key = self._pre_reduced_field
        if pre_reduced_key is not None and pre_reduced_key in dataset.other_fields:
            return  # Pre-reduced activations available, no raw needed
        if "activations" not in dataset.other_fields:
            raise ValueError("Dataset missing 'activations' field")
        if "attention_mask" not in dataset.other_fields:
            raise ValueError("Dataset missing 'attention_mask' field")

    def fit(self, dataset: LabelledDataset) -> None:
        """Train the probe on sequence activations.

        Accepts either raw activations (with attention_mask) or pre-reduced
        activations (``activations_{reduction_strategy}`` field).

        Args:
            dataset: Dataset with activation fields

        Raises:
            ValueError: If required fields are missing
        """
        self._validate_activation_fields(dataset)

        X = self._get_reduced_activations(dataset)
        y = dataset.labels_numpy()
        self.clf.fit(X, y)

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        """Predict probabilities for positive class.

        Accepts either raw activations (with attention_mask) or pre-reduced
        activations (``activations_{reduction_strategy}`` field).

        Args:
            dataset: Dataset with activation fields

        Returns:
            Probability scores for positive class, shape (n_samples,)

        Raises:
            ValueError: If required fields are missing
        """
        self._validate_activation_fields(dataset)

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

        self._validate_activation_fields(dataset)

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


class DegradedProbe:
    """Wrapper that optionally degrades probe predictions with fixed settings.

    This is intended for temporary delegation testing when baselines are weak.
        Degradation is applied only to outputs of predict():
            - Randomly flip scores (score -> 1 - score) with probability 0.2.
        This is uniform across confidence levels.
    """

    def __init__(self, probe: Probe, enabled: bool = False, seed: int | None = None):
        self.probe = probe
        self.enabled = enabled
        self.rng = np.random.default_rng(seed)

    def fit(self, dataset: LabelledDataset) -> None:
        self.probe.fit(dataset)

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        scores = self.probe.predict(dataset)
        if not self.enabled:
            return scores

        degraded = scores.astype(float, copy=True)
        flip_mask = self.rng.random(degraded.shape[0]) < 0.3
        degraded[flip_mask] = 1.0 - degraded[flip_mask]
        return np.clip(degraded, 0.0, 1.0)

    def calibrate(self, dataset: LabelledDataset, method: str = "platt-scaling") -> None:
        self.probe.calibrate(dataset, method=method)


PROBE_REGISTRY: dict[str, Callable[..., Probe]] = {}


def register_probe(name: str):
    """Register a probe constructor for configuration-driven experiments."""

    def decorator(constructor: Callable[..., Probe]) -> Callable[..., Probe]:
        if name in PROBE_REGISTRY:
            raise ValueError(f"Probe {name!r} is already registered")
        PROBE_REGISTRY[name] = constructor
        return constructor

    return decorator


def build_probe(spec: str | dict[str, Any] | None = None, **overrides: Any) -> Probe:
    """Build a probe from a name or ``{type, hyperparams}`` configuration."""
    if spec is None:
        name, hyperparams = "mean_logreg", {}
    elif isinstance(spec, str):
        name, hyperparams = spec, {}
    else:
        name = spec.get("type", "mean_logreg")
        hyperparams = dict(spec.get("hyperparams", {}))
    hyperparams.update(overrides)
    try:
        return PROBE_REGISTRY[name](**hyperparams)
    except KeyError as exc:
        raise ValueError(f"Unknown probe {name!r}; available: {sorted(PROBE_REGISTRY)}") from exc


def probe_requires_raw_activations(spec: str | dict[str, Any] | None) -> bool:
    """Return whether a configured probe consumes token-level activations."""
    name = spec if isinstance(spec, str) else (spec or {}).get("type", "mean_logreg")
    return name in {"attention", "softmax"}


@register_probe("mean_logreg")
def _build_mean_logreg(reduction_strategy: str = "mean", **kwargs: Any) -> Probe:
    if "seed" in kwargs:
        kwargs["random_state"] = kwargs.pop("seed")
    return SequenceProbe(reduction_strategy=reduction_strategy, **kwargs)


class AttentionProbeModule(nn.Module):
    """McKenzie's lightweight learned attention pooling probe."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.context_query = nn.Linear(embed_dim, 1)
        self.output = nn.Linear(embed_dim, 1)
        self.scale = embed_dim**0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.context_query(x).squeeze(-1) / self.scale
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
        context = torch.einsum("bs,bse->be", weights, x)
        return self.output(context).squeeze(-1)


class SoftmaxProbeModule(nn.Module):
    """Linear token scoring followed by softmax aggregation."""

    def __init__(self, embed_dim: int, temperature: float = 5.0):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        weights = torch.softmax(logits.masked_fill(~mask, float("-inf")) / self.temperature, dim=1)
        return (logits.masked_fill(~mask, 0.0) * weights).sum(dim=1)


class MeanMLPProbeModule(nn.Module):
    """Two-layer MLP applied after masked mean pooling."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.network(pooled).squeeze(-1)


_TORCH_MODULES: dict[str, type[nn.Module]] = {
    "attention": AttentionProbeModule,
    "softmax": SoftmaxProbeModule,
    "mlp": MeanMLPProbeModule,
}


def default_torch_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _activation_tensor(value: Any) -> torch.Tensor:
    """Wrap activation storage without expanding float16 caches to float32."""
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point() or tensor.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
        tensor = tensor.float()
    return tensor


def _sequence_loader(
    x: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    indices: np.ndarray | None = None,
    shuffle: bool = False,
) -> DataLoader:
    tensors = (x, mask) if targets is None else (x, mask, targets)
    dataset = TensorDataset(*tensors)
    if indices is not None:
        dataset = Subset(dataset, indices.tolist())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
    )


def _prepare_sequence_batch(
    x: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop batch-wide padding, then transfer inputs to the accelerator."""
    active_columns = mask.any(dim=0).nonzero(as_tuple=False).flatten()
    if len(active_columns):
        start, stop = int(active_columns[0]), int(active_columns[-1]) + 1
        x, mask = x[:, start:stop], mask[:, start:stop]
    non_blocking = device.type == "cuda"
    x = x.to(device, non_blocking=non_blocking)
    mask = mask.to(device, non_blocking=non_blocking)
    if device.type != "cuda" and x.dtype != torch.float32:
        x = x.float()
    return x, mask


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _try_gpu_resident_tensors(
    x: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    enabled: bool,
    reserve_gb: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Move a complete activation dataset to CUDA when it fits safely."""
    if not enabled or device.type != "cuda":
        return None
    # Prediction on the previous ablation cell can leave many GiB in PyTorch's
    # caching allocator even though no tensors still reference those blocks.
    gc.collect()
    torch.cuda.empty_cache()
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    free_bytes, _ = torch.cuda.mem_get_info(device_index)
    required_bytes = _tensor_bytes(x, mask, targets)
    reserve_bytes = int(reserve_gb * 1024**3)
    if required_bytes + reserve_bytes > free_bytes:
        logger.info(
            "Activation dataset needs %.1f GiB; keeping it on CPU (%.1f GiB free, %.1f GiB reserved)",
            required_bytes / 1024**3,
            free_bytes / 1024**3,
            reserve_gb,
        )
        return None
    logger.info(
        "Loading %.1f GiB activation dataset onto %s (%.1f GiB reserve)",
        required_bytes / 1024**3,
        device,
        reserve_gb,
    )
    try:
        return x.to(device), mask.to(device), targets.to(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("Full activation dataset did not fit on %s; falling back to streamed CPU batches", device)
        torch.cuda.empty_cache()
        return None


def _device_sequence_batches(
    x: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    index = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    if shuffle:
        generator = torch.Generator(device=x.device).manual_seed(seed)
        index = index[torch.randperm(len(index), generator=generator, device=x.device)]
    for start in range(0, len(index), batch_size):
        batch_index = index[start : start + batch_size]
        yield x[batch_index], mask[batch_index], targets[batch_index]


def _devices_match(actual: torch.device, requested: torch.device) -> bool:
    if actual.type != requested.type:
        return False
    return requested.index is None or actual.index == requested.index


def _batched_sequence_loss(
    model: nn.Module,
    criterion: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    mixed_precision: bool,
) -> float:
    if _devices_match(x.device, device):
        batches = _device_sequence_batches(
            x,
            mask,
            targets,
            indices,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        )
    else:
        batches = _sequence_loader(
            x,
            mask,
            targets,
            batch_size=batch_size,
            device=device,
            indices=indices,
        )
    total_loss = 0.0
    n_samples = 0
    model.eval()
    with torch.inference_mode():
        for xb, mb, yb in batches:
            xb, mb = _prepare_sequence_batch(xb, mb, device)
            yb = yb.to(device, non_blocking=device.type == "cuda")
            with _autocast(device, mixed_precision):
                loss = criterion(model(xb, mb), yb)
            total_loss += float(loss) * len(yb)
            n_samples += len(yb)
    return total_loss / n_samples


def _batched_sequence_outputs(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    mixed_precision: bool,
) -> np.ndarray:
    loader = _sequence_loader(x, mask, None, batch_size=batch_size, device=device)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for xb, mb in loader:
            xb, mb = _prepare_sequence_batch(xb, mb, device)
            with _autocast(device, mixed_precision):
                outputs.append(model(xb, mb).float().cpu())
    return torch.cat(outputs).numpy()


class TorchSequenceProbe:
    """Small torch probe with deterministic training and internal early stopping."""

    def __init__(
        self,
        architecture: str,
        *,
        seed: int = 42,
        batch_size: int = 16,
        epochs: int = 200,
        learning_rate: float = 5e-3,
        final_learning_rate: float = 1e-4,
        patience: int = 50,
        validation_fraction: float = 0.1,
        weight_decay: float = 0.0,
        gradient_accumulation_steps: int = 1,
        validation_batch_size: int | None = None,
        mixed_precision: bool = True,
        resident_on_device: bool = True,
        device_reserve_gb: float = 3.5,
        device: str | None = None,
        **module_kwargs: Any,
    ):
        if architecture not in _TORCH_MODULES:
            raise ValueError(f"Unknown torch architecture: {architecture}")
        self.architecture = architecture
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
        self.validation_batch_size = validation_batch_size or batch_size
        self.mixed_precision = mixed_precision
        self.resident_on_device = resident_on_device
        self.device_reserve_gb = device_reserve_gb
        self.device = torch.device(device) if device else default_torch_device()
        self.module_kwargs = module_kwargs
        self.model: nn.Module | None = None
        self.calibration_clf: LogisticRegression | IsotonicRegression | None = None

    @staticmethod
    def _arrays(dataset: LabelledDataset) -> tuple[torch.Tensor, torch.Tensor]:
        if "activations" in dataset.other_fields:
            x = _activation_tensor(dataset.other_fields["activations"])
            mask_value = dataset.other_fields.get("attention_mask")
            mask = torch.as_tensor(mask_value).bool() if mask_value is not None else x.abs().sum(dim=-1).ne(0)
        elif "activations_mean" in dataset.other_fields:
            x = _activation_tensor(dataset.other_fields["activations_mean"]).unsqueeze(1)
            mask = torch.ones(x.shape[:2], dtype=torch.bool)
        else:
            raise ValueError("Dataset requires 'activations' or 'activations_mean'")
        return x, mask

    def fit(self, dataset: LabelledDataset) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        x, mask = self._arrays(dataset)
        y = torch.as_tensor(dataset.labels_numpy(), dtype=torch.float32)
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(y))
        n_val = max(1, int(len(y) * self.validation_fraction))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        self.model = _TORCH_MODULES[self.architecture](x.shape[-1], **self.module_kwargs).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.final_learning_rate
        )
        loader = _sequence_loader(
            x,
            mask,
            y,
            batch_size=self.batch_size,
            device=self.device,
            indices=train_idx,
            shuffle=True,
        )
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision and self.device.type == "cuda")
        resident_tensors = _try_gpu_resident_tensors(
            x,
            mask,
            y,
            device=self.device,
            enabled=self.resident_on_device,
            reserve_gb=self.device_reserve_gb,
        )
        train_x, train_mask, train_y = resident_tensors or (x, mask, y)
        best_loss, stale, best_state = float("inf"), 0, None
        logger.info(
            "Training %s safety probe on %s for at most %d epochs (batch=%d, accumulation=%d)",
            self.architecture,
            self.device,
            self.epochs,
            self.batch_size,
            self.gradient_accumulation_steps,
        )
        for epoch in range(self.epochs):
            epoch_started = time.perf_counter()
            self.model.train()
            optimizer.zero_grad()
            if resident_tensors is None:
                batches = loader
            else:
                batches = _device_sequence_batches(
                    train_x,
                    train_mask,
                    train_y,
                    train_idx,
                    batch_size=self.batch_size,
                    shuffle=True,
                    seed=self.seed + epoch,
                )
            for batch_index, (xb, mb, yb) in enumerate(batches):
                xb, mb = _prepare_sequence_batch(xb, mb, self.device)
                yb = yb.to(self.device, non_blocking=self.device.type == "cuda")
                with _autocast(self.device, self.mixed_precision):
                    loss = criterion(self.model(xb, mb), yb)
                scaler.scale(loss / self.gradient_accumulation_steps).backward()
                is_accumulation_boundary = (batch_index + 1) % self.gradient_accumulation_steps == 0
                n_train_batches = (len(train_idx) + self.batch_size - 1) // self.batch_size
                is_last_batch = batch_index + 1 == n_train_batches
                if is_accumulation_boundary or is_last_batch:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            scheduler.step()
            val_loss = _batched_sequence_loss(
                self.model,
                criterion,
                train_x,
                train_mask,
                train_y,
                val_idx,
                batch_size=self.validation_batch_size,
                device=self.device,
                mixed_precision=self.mixed_precision,
            )
            if val_loss < best_loss - 1e-6:
                best_loss, stale = val_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    logger.info("Early stopping %s safety probe after epoch %d", self.architecture, epoch + 1)
                    break
            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.info(
                    "%s safety probe epoch %d/%d: validation loss %.5f",
                    self.architecture,
                    epoch + 1,
                    self.epochs,
                    val_loss,
                )
                logger.info("%s safety probe epoch time: %.2fs", self.architecture, time.perf_counter() - epoch_started)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if resident_tensors is not None:
            del batches, xb, mb, yb, resident_tensors, train_x, train_mask, train_y
            torch.cuda.empty_cache()

    def raw_predict(self, dataset: LabelledDataset) -> np.ndarray:
        if self.model is None:
            raise ValueError("Probe has not been fitted")
        x, mask = self._arrays(dataset)
        return _batched_sequence_outputs(
            self.model,
            x,
            mask,
            batch_size=max(self.batch_size, 64),
            device=self.device,
            mixed_precision=self.mixed_precision,
        )

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        scores = 1.0 / (1.0 + np.exp(-self.raw_predict(dataset)))
        if self.calibration_clf is None:
            return scores
        if isinstance(self.calibration_clf, IsotonicRegression):
            return self.calibration_clf.predict(scores)
        return self.calibration_clf.predict_proba(scores.reshape(-1, 1))[:, 1]

    def calibrate(self, dataset: LabelledDataset, method: str = "platt-scaling") -> None:
        scores, labels = self.predict(dataset), dataset.labels_numpy()
        if method == "platt-scaling":
            self.calibration_clf = LogisticRegression(max_iter=1000).fit(scores.reshape(-1, 1), labels)
        elif method == "isotonic-regression":
            self.calibration_clf = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
        else:
            raise ValueError(f"Unknown calibration method: {method}")


for _architecture in _TORCH_MODULES:
    register_probe(_architecture)(
        lambda architecture=_architecture, **kwargs: TorchSequenceProbe(architecture=architecture, **kwargs)
    )
