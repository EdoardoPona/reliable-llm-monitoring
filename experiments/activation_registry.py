"""ClearML-backed registry for cached reduced or raw activations.

Caches reduced activations per ``(model_name, layer, reduction, dataset_file)``
tuple both locally (pickle files) and on ClearML.  Lookup order:
local → ClearML → compute.

Usage::

    from activation_registry import compute_or_fetch_activations

    X = compute_or_fetch_activations(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        layer=11,
        reduction="mean",
        dataset_path="/data/evals/test/anthropic_test_balanced_apr_23.jsonl",
        dataset=my_dataset,
    )
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from baseline_registry import normalize_dataset_key
from models_under_pressure.interfaces.dataset import LabelledDataset

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "reliable-llm-monitoring/activation-cache"
DEFAULT_LOCAL_CACHE_DIR = Path("results") / "activation_cache"


def _make_tags(model_name: str, layer: int, reduction: str, dataset_key: str) -> list[str]:
    """Build ClearML tags for cache lookup/storage."""
    return [
        "activation-cache",
        f"model:{model_name}",
        f"layer:{layer}",
        f"reduction:{reduction}",
        f"dataset:{dataset_key}",
    ]


def _local_cache_path(model_name: str, layer: int, reduction: str, dataset_key: str, cache_dir: Path) -> Path:
    """Return the local cache file path."""
    safe_model = model_name.replace("/", "__")
    safe_dataset = dataset_key.replace("/", "__")
    return cache_dir / f"{safe_model}__L{layer}__{reduction}__{safe_dataset}.pkl"


# ---------------------------------------------------------------------------
# Local cache
# ---------------------------------------------------------------------------


def _load_local_cache(
    model_name: str, layer: int, reduction: str, dataset_key: str, cache_dir: Path
) -> np.ndarray | None:
    """Load reduced activations from local pickle cache."""
    path = _local_cache_path(model_name, layer, reduction, dataset_key, cache_dir)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        data = np.asarray(data)
        logger.info(f"Local cache hit: loaded activations {data.shape} from {path}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load local cache {path}: {e}")
        return None


def _save_local_cache(
    model_name: str,
    layer: int,
    reduction: str,
    dataset_key: str,
    data: np.ndarray,
    cache_dir: Path,
) -> None:
    """Save reduced activations to local pickle cache."""
    path = _local_cache_path(model_name, layer, reduction, dataset_key, cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved activations {data.shape} to local cache: {path}")
    except Exception as e:
        logger.warning(f"Failed to save local cache {path}: {e}")


# ---------------------------------------------------------------------------
# ClearML cache
# ---------------------------------------------------------------------------


def _fetch_clearml_cache(
    model_name: str,
    layer: int,
    reduction: str,
    dataset_key: str,
    project_name: str = DEFAULT_PROJECT,
) -> np.ndarray | None:
    """Query ClearML for cached reduced activations."""
    try:
        from clearml import Task
    except ImportError:
        logger.debug("ClearML not installed — cache lookup skipped.")
        return None

    tags = ["__$all", *_make_tags(model_name, layer, reduction, dataset_key)]
    try:
        tasks = Task.get_tasks(
            project_name=project_name,
            tags=tags,
            task_filter={"status": ["completed", "closed", "published"]},
        )
    except Exception as e:
        warnings.warn(f"ClearML cache lookup failed: {e}", UserWarning, stacklevel=2)
        return None

    if not tasks:
        logger.info(
            f"ClearML cache miss: model={model_name}, layer={layer}, reduction={reduction}, dataset={dataset_key}"
        )
        return None

    for task in tasks:
        artifact = task.artifacts.get("reduced_activations")
        if artifact is None:
            logger.warning(f"ClearML cache hit but no 'reduced_activations' artifact in task {task.id}")
            continue
        try:
            local_path = artifact.get_local_copy()
            if local_path is None:
                logger.warning(f"Artifact download returned None for task {task.id}")
                continue
            with open(local_path, "rb") as f:
                data = pickle.load(f)  # noqa: S301
            data = np.asarray(data)
            logger.info(
                f"ClearML cache hit: loaded activations {data.shape} "
                f"(model={model_name}, layer={layer}, reduction={reduction}, "
                f"dataset={dataset_key}, task={task.id})"
            )
            return data
        except Exception as e:
            logger.warning(f"Failed to load cached activations from task {task.id}: {e}")
            continue

    warnings.warn(
        f"ClearML found {len(tasks)} matching tasks but all artifact downloads failed "
        f"(model={model_name}, layer={layer}, reduction={reduction}, dataset={dataset_key})",
        UserWarning,
        stacklevel=2,
    )
    return None


def _upload_to_clearml(
    model_name: str,
    layer: int,
    reduction: str,
    dataset_key: str,
    data: np.ndarray,
    project_name: str = DEFAULT_PROJECT,
) -> str | None:
    """Upload reduced activations to ClearML cache."""
    try:
        from clearml import Task
    except ImportError:
        logger.debug("ClearML not installed — cache upload skipped.")
        return None

    try:
        task = Task.create(
            project_name=project_name,
            task_name=f"activations_{model_name}_L{layer}_{reduction}_{dataset_key}",
            task_type="data_processing",
        )
        task.add_tags(_make_tags(model_name, layer, reduction, dataset_key))

        task.upload_artifact(
            "reduced_activations",
            artifact_object=data,
            extension_name=".pkl",
            wait_on_upload=True,
        )

        task.mark_started()
        task.mark_completed()

        task_id = task.id
        logger.info(
            f"Uploaded activations {data.shape} to ClearML "
            f"(model={model_name}, layer={layer}, reduction={reduction}, "
            f"dataset={dataset_key}, task={task_id})"
        )
        return task_id
    except Exception as e:
        warnings.warn(f"Failed to upload activations to ClearML cache: {e}", UserWarning, stacklevel=2)
        return None


# ---------------------------------------------------------------------------
# Cross-cache sync
# ---------------------------------------------------------------------------


def _ensure_clearml_cache(
    model_name: str,
    layer: int,
    reduction: str,
    dataset_key: str,
    data: np.ndarray,
    project_name: str = DEFAULT_PROJECT,
) -> None:
    """Upload to ClearML if not already cached there."""
    existing = _fetch_clearml_cache(model_name, layer, reduction, dataset_key, project_name)
    if existing is not None:
        return
    logger.info(f"Syncing local cache → ClearML (model={model_name}, layer={layer}, reduction={reduction})")
    _upload_to_clearml(model_name, layer, reduction, dataset_key, data, project_name)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def _compute_activations(
    model_name: str,
    dataset: LabelledDataset,
    layer: int,
    reduction: str,
    *,
    local: bool = True,
    gpu: str | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Compute reduced or raw activations locally or on Modal."""
    if not local:
        from reliable_monitoring.modal_activations import compute_activations_modal

        return compute_activations_modal(
            model_name=model_name,
            dataset=dataset,
            layer=layer,
            reduction_strategy=reduction,
            batch_size=batch_size,
            gpu=gpu,
        )

    # Local computation
    from reliable_monitoring.dataset import (
        _apply_attention_mask_to_activations,
        _compute_raw_activations,
        _load_model,
        enrich_with_activations,
        reduce_activations,
    )

    model = _load_model(model_name, batch_size)
    activations, inputs = _compute_raw_activations(dataset, model, layer)
    dataset = enrich_with_activations(dataset, activations[0], inputs["input_ids"], inputs["attention_mask"])
    dataset = _apply_attention_mask_to_activations(dataset)
    if reduction == "raw":
        result = dataset.other_fields["activations"]
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        return np.asarray(result, dtype=np.float16)
    reduced = reduce_activations(dataset, reduction)
    result = reduced.other_fields[f"activations_{reduction}"]
    if isinstance(result, torch.Tensor):
        result = result.numpy()
    return np.asarray(result)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_or_fetch_activations(
    model_name: str,
    layer: int,
    reduction: str,
    dataset: LabelledDataset,
    dataset_path: str,
    *,
    local: bool = True,
    gpu: str | None = None,
    batch_size: int = 8,
    project_name: str = DEFAULT_PROJECT,
    local_cache_dir: Path = DEFAULT_LOCAL_CACHE_DIR,
    skip_cache: bool = False,
    local_only: bool = False,
    sync_clearml_on_local_hit: bool = True,
) -> np.ndarray:
    """Fetch cached reduced activations or compute and upload them.

    Lookup order: local file cache → ClearML → compute.
    On compute, saves to both local and ClearML caches.

    Args:
        model_name: Model used for activation extraction.
        layer: Layer number.
        reduction: Reduction strategy (e.g. ``"mean"``).
        dataset: Dataset to compute activations on (used on cache miss).
        dataset_path: Path to the dataset file (used for cache key).
        local: If True, compute locally; if False, use Modal.
        gpu: Override GPU type for Modal.
        batch_size: Batch size for model forward pass.
        project_name: ClearML project for cache storage.
        local_cache_dir: Directory for local pickle cache.
        skip_cache: If True, always compute (skip all cache lookups).
        local_only: If True, use only the local cache (no ClearML sync or
            fallback).  Avoids network calls when all data is cached locally.
        sync_clearml_on_local_hit: If False, return a valid local hit without
            checking that a duplicate exists on ClearML.

    Returns:
        Reduced activation array of shape ``(len(dataset), hidden_dim)``, or
        ``(len(dataset), sequence_length, hidden_dim)`` for ``reduction='raw'``.
    """
    dataset_key = normalize_dataset_key(dataset_path)
    n_expected = len(dataset)

    if not skip_cache:
        # 1. Try local cache
        local_hit = _load_local_cache(model_name, layer, reduction, dataset_key, local_cache_dir)
        if local_hit is not None:
            if len(local_hit) != n_expected:
                logger.warning(f"Local cache size mismatch: {len(local_hit)} != {n_expected}. Skipping.")
            else:
                if not local_only and sync_clearml_on_local_hit:
                    _ensure_clearml_cache(model_name, layer, reduction, dataset_key, local_hit, project_name)
                return local_hit

        # 2. Try ClearML cache
        if local_only:
            raise RuntimeError(
                f"local_only=True but no local cache found for model={model_name}, layer={layer}, "
                f"reduction={reduction}, dataset={dataset_key}"
            )
        clearml_hit = _fetch_clearml_cache(model_name, layer, reduction, dataset_key, project_name)
        if clearml_hit is not None:
            if len(clearml_hit) != n_expected:
                logger.warning(f"ClearML cache size mismatch: {len(clearml_hit)} != {n_expected}. Recomputing.")
            else:
                _save_local_cache(model_name, layer, reduction, dataset_key, clearml_hit, local_cache_dir)
                return clearml_hit

    # 3. Compute
    logger.info(
        f"Computing activations (model={model_name}, layer={layer}, reduction={reduction}, dataset={dataset_key})..."
    )
    data = _compute_activations(
        model_name=model_name,
        dataset=dataset,
        layer=layer,
        reduction=reduction,
        local=local,
        gpu=gpu,
        batch_size=batch_size,
    )

    # Save to both caches
    if not skip_cache:
        _save_local_cache(model_name, layer, reduction, dataset_key, data, local_cache_dir)
        if not local_only:
            _upload_to_clearml(model_name, layer, reduction, dataset_key, data, project_name)

    return data
