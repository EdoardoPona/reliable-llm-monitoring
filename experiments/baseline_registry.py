"""ClearML-backed registry for cached baseline LLM scores.

Caches baseline scores per ``(model_name, dataset_file)`` pair both locally
(pickle files) and on ClearML.  Lookup order: local → ClearML → compute.

Usage::

    from baseline_registry import compute_or_fetch_baseline

    scores = compute_or_fetch_baseline(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        dataset=my_dataset,
        dataset_path="/data/evals/test/anthropic_test_balanced_apr_23.jsonl",
        baseline_batch_size=8,
    )
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from models_under_pressure.interfaces.dataset import LabelledDataset

from reliable_monitoring.cascade import run_llm_baseline

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "reliable-llm-monitoring/baseline-cache"
DEFAULT_LOCAL_CACHE_DIR = Path("results") / "baseline_cache"


def normalize_dataset_key(resolved_path: str) -> str:
    """Normalize a resolved dataset path to a portable cache key.

    Strips the ``DATA_DIR`` prefix (if set) so that the same logical dataset
    produces the same key regardless of where it is mounted.  Falls back to
    the absolute path when ``DATA_DIR`` is not set.

    Examples:
        >>> os.environ["DATA_DIR"] = "/data"
        >>> normalize_dataset_key("/data/evals/test/foo.jsonl")
        'evals/test/foo.jsonl'
    """
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        data_dir = str(Path(data_dir).resolve())
        resolved = str(Path(resolved_path).resolve())
        if resolved.startswith(data_dir):
            return resolved[len(data_dir) :].lstrip("/")
    return str(Path(resolved_path).resolve())


def _make_tags(model_name: str, dataset_key: str) -> list[str]:
    """Build ClearML tags for cache lookup/storage."""
    return ["baseline-cache", f"model:{model_name}", f"dataset:{dataset_key}"]


def _local_cache_path(model_name: str, dataset_key: str, cache_dir: Path) -> Path:
    """Return the local cache file path for a (model, dataset) pair."""
    # Replace slashes in model name to create a flat filename
    safe_model = model_name.replace("/", "__")
    safe_dataset = dataset_key.replace("/", "__")
    return cache_dir / f"{safe_model}__{safe_dataset}.pkl"


# ---------------------------------------------------------------------------
# Local cache
# ---------------------------------------------------------------------------


def _load_local_cache(model_name: str, dataset_key: str, cache_dir: Path) -> np.ndarray | None:
    """Load baseline scores from local pickle cache."""
    path = _local_cache_path(model_name, dataset_key, cache_dir)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            scores = pickle.load(f)  # noqa: S301
        scores = np.asarray(scores)
        logger.info(f"Local cache hit: loaded {len(scores)} scores from {path}")
        return scores
    except Exception as e:
        logger.warning(f"Failed to load local cache {path}: {e}")
        return None


def _save_local_cache(model_name: str, dataset_key: str, scores: np.ndarray, cache_dir: Path) -> None:
    """Save baseline scores to local pickle cache."""
    path = _local_cache_path(model_name, dataset_key, cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(scores, f)
        logger.info(f"Saved {len(scores)} scores to local cache: {path}")
    except Exception as e:
        logger.warning(f"Failed to save local cache {path}: {e}")


# ---------------------------------------------------------------------------
# ClearML cache
# ---------------------------------------------------------------------------


def fetch_cached_baseline(
    model_name: str,
    dataset_key: str,
    project_name: str = DEFAULT_PROJECT,
) -> np.ndarray | None:
    """Query ClearML for cached baseline scores.

    Args:
        model_name: Baseline model name (e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``).
        dataset_key: Normalized dataset key (from :func:`normalize_dataset_key`).
        project_name: ClearML project to search in.

    Returns:
        Cached baseline score array, or ``None`` on cache miss.
    """
    try:
        from clearml import Task
    except ImportError:
        logger.debug("ClearML not installed — cache lookup skipped.")
        return None

    tags = ["__$all", *_make_tags(model_name, dataset_key)]
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
        logger.info(f"ClearML cache miss: model={model_name}, dataset={dataset_key}")
        return None

    # Try tasks in order — some may have broken artifacts (e.g. from
    # failed uploads).  Fall through to the next task on failure.
    for task in tasks:
        artifact = task.artifacts.get("baseline_scores")
        if artifact is None:
            logger.warning(f"ClearML cache hit but no 'baseline_scores' artifact in task {task.id}")
            continue

        try:
            local_path = artifact.get_local_copy()
            if local_path is None:
                logger.warning(f"Artifact download returned None for task {task.id}")
                continue
            with open(local_path, "rb") as f:
                scores = pickle.load(f)  # noqa: S301
            scores = np.asarray(scores)
            logger.info(
                f"ClearML cache hit: loaded {len(scores)} baseline scores "
                f"(model={model_name}, dataset={dataset_key}, task={task.id})"
            )
            return scores
        except Exception as e:
            logger.warning(f"Failed to load cached baseline from task {task.id}: {e}")
            continue

    warnings.warn(
        f"ClearML found {len(tasks)} matching tasks but all artifact downloads failed "
        f"(model={model_name}, dataset={dataset_key})",
        UserWarning,
        stacklevel=2,
    )
    return None


def upload_baseline_to_cache(
    model_name: str,
    dataset_key: str,
    scores: np.ndarray,
    project_name: str = DEFAULT_PROJECT,
) -> str | None:
    """Upload baseline scores to ClearML cache.

    Args:
        model_name: Baseline model name.
        dataset_key: Normalized dataset key.
        scores: Baseline score array to cache.
        project_name: ClearML project to upload to.

    Returns:
        ClearML task ID on success, ``None`` on failure.
    """
    try:
        from clearml import Task
    except ImportError:
        logger.debug("ClearML not installed — cache upload skipped.")
        return None

    try:
        # Use Task.create to avoid interfering with any existing Task.init session
        task = Task.create(
            project_name=project_name,
            task_name=f"baseline_{model_name}_{dataset_key}",
            task_type="data_processing",
        )
        task.add_tags(_make_tags(model_name, dataset_key))

        # Upload scores as pickle (extension_name=".pkl" forces pickle
        # serialization regardless of object type — without it ClearML
        # would store numpy arrays as .npz which complicates loading).
        # wait_on_upload=True ensures the upload completes before we
        # mark the task as completed.
        task.upload_artifact(
            "baseline_scores",
            artifact_object=scores,
            extension_name=".pkl",
            wait_on_upload=True,
        )

        # Task.create() produces tasks in "created" status.
        # mark_completed() only works after mark_started().
        task.mark_started()
        task.mark_completed()

        task_id = task.id
        logger.info(
            f"Uploaded {len(scores)} baseline scores to ClearML "
            f"(model={model_name}, dataset={dataset_key}, task={task_id})"
        )

        return task_id
    except Exception as e:
        warnings.warn(f"Failed to upload baseline to ClearML cache: {e}", UserWarning, stacklevel=2)
        return None


# ---------------------------------------------------------------------------
# Cross-cache sync
# ---------------------------------------------------------------------------


def _ensure_clearml_cache(
    model_name: str,
    dataset_key: str,
    scores: np.ndarray,
    project_name: str = DEFAULT_PROJECT,
) -> None:
    """Upload to ClearML if not already cached there."""
    existing = fetch_cached_baseline(model_name, dataset_key, project_name)
    if existing is not None:
        return
    logger.info(f"Syncing local cache → ClearML (model={model_name}, dataset={dataset_key})")
    upload_baseline_to_cache(model_name, dataset_key, scores, project_name)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_or_fetch_baseline(
    model_name: str,
    dataset: LabelledDataset,
    dataset_path: str,
    *,
    baseline_batch_size: int = 16,
    local: bool = True,
    gpu: str | None = None,
    project_name: str = DEFAULT_PROJECT,
    local_cache_dir: Path = DEFAULT_LOCAL_CACHE_DIR,
    skip_cache: bool = False,
) -> np.ndarray:
    """Fetch cached baseline scores or compute and upload them.

    Lookup order: local file cache → ClearML → compute.
    On compute, saves to both local and ClearML caches.

    Args:
        model_name: Baseline model name.
        dataset: Dataset to compute baseline on (used on cache miss).
        dataset_path: Path to the dataset file (used for cache key).
        baseline_batch_size: Batch size for LLM inference.
        local: If True, run inference locally; if False, use Modal.
        gpu: Override GPU type for Modal.
        project_name: ClearML project for cache storage.
        local_cache_dir: Directory for local pickle cache.
        skip_cache: If True, always compute (skip all cache lookups).

    Returns:
        Baseline score array of shape ``(len(dataset),)``.
    """
    dataset_key = normalize_dataset_key(dataset_path)
    n_expected = len(dataset)

    if not skip_cache:
        # 1. Try local cache
        local_hit = _load_local_cache(model_name, dataset_key, local_cache_dir)
        if local_hit is not None:
            if len(local_hit) != n_expected:
                logger.warning(f"Local cache size mismatch: {len(local_hit)} != {n_expected}. Skipping.")
            else:
                # Sync to ClearML if missing there
                _ensure_clearml_cache(model_name, dataset_key, local_hit, project_name)
                return local_hit

        # 2. Try ClearML cache
        clearml_hit = fetch_cached_baseline(model_name, dataset_key, project_name)
        if clearml_hit is not None:
            if len(clearml_hit) != n_expected:
                logger.warning(f"ClearML cache size mismatch: {len(clearml_hit)} != {n_expected}. Recomputing.")
            else:
                # Sync to local cache
                _save_local_cache(model_name, dataset_key, clearml_hit, local_cache_dir)
                return clearml_hit

    # 3. Compute
    logger.info(f"Computing baseline scores (model={model_name}, dataset={dataset_key})...")
    scores = run_llm_baseline(
        baseline_model_name=model_name,
        dataset=dataset,
        baseline_batch_size=baseline_batch_size,
        local=local,
        gpu=gpu,
    )

    # Save to both caches
    if not skip_cache:
        _save_local_cache(model_name, dataset_key, scores, local_cache_dir)
        upload_baseline_to_cache(model_name, dataset_key, scores, project_name)

    return scores
