"""Score artifact: data structures for pre-computed probe and baseline scores.

The ``ScoreArtifact`` is the output of ``compute_scores.py`` and the input to
all downstream experiment scripts (``sgt_cascade.py``, ``fixed_cascade.py``,
etc.).  It stores probe scores, baseline scores, labels, and optional
calibration information for train/calib/test splits.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SplitScores:
    """Scores and labels for a single data split (train, calib, or test)."""

    probe_scores: np.ndarray  # (n,) raw/uncalibrated probe predictions
    baseline_scores: np.ndarray | None  # (n,) baseline predictions, None for train
    labels: np.ndarray  # (n,) binary labels
    groups: np.ndarray | None  # (n,) group identifiers for mixed datasets, or None


@dataclass
class CalibrationInfo:
    """Calibration metadata and calibrated probe scores.

    Produced by ``calibrate_scores.py``.  Stores both the calibration
    configuration and the transformed scores so that downstream experiments
    can use calibrated scores without re-fitting.
    """

    method: str  # "isotonic-regression" or "platt-scaling"
    auxiliary_indices: np.ndarray  # indices in calib split used for fitting
    auxiliary_proportion: float  # proportion of calib data used for fitting
    train_probe_scores: np.ndarray  # (n_train,) calibrated
    calib_probe_scores: np.ndarray  # (n_calib,) calibrated (full calib split)
    test_probe_scores: np.ndarray  # (n_test,) calibrated


@dataclass
class ScoreArtifact:
    """Pre-computed scores for all data splits.

    Produced by ``compute_scores.py``, optionally enriched with calibration
    by ``calibrate_scores.py``.  Consumed by experiment scripts.
    """

    train: SplitScores
    calib: SplitScores
    test: SplitScores

    calibration: CalibrationInfo | None  # None until calibration step runs

    config: dict  # full experiment config for reproducibility
    seed: int
    created_at: str

    def get_probe_scores(self, split: str, *, calibrated: bool = True) -> np.ndarray:
        """Return probe scores for a split, preferring calibrated if available.

        Args:
            split: One of "train", "calib", "test".
            calibrated: If True and calibration info exists, return calibrated
                scores.  Otherwise return raw uncalibrated scores.

        Returns:
            Probe score array of shape (n,).
        """
        if calibrated and self.calibration is not None:
            return getattr(self.calibration, f"{split}_probe_scores")
        return getattr(self, split).probe_scores

    def get_calib_mask(self) -> np.ndarray:
        """Return a boolean mask over the calib split.

        Elements are True for samples that were *not* used for calibration
        fitting (i.e. usable for hypothesis testing).  If no calibration
        was performed, all elements are True.
        """
        n = len(self.calib.labels)
        mask = np.ones(n, dtype=bool)
        if self.calibration is not None:
            mask[self.calibration.auxiliary_indices] = False
        return mask


def save_score_artifact(artifact: ScoreArtifact, path: str | Path) -> Path:
    """Save a ScoreArtifact as a pickle file.

    Creates parent directories if needed.

    Returns:
        The resolved path the artifact was saved to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info(f"Saved score artifact to {path}")
    return path


def load_score_artifact(path: str | Path) -> ScoreArtifact:
    """Load a ScoreArtifact from a pickle file.

    Uses a custom unpickler to resolve classes pickled under ``__main__``.
    """
    path = Path(path)
    with open(path, "rb") as f:
        artifact = _ScoreArtifactUnpickler(f).load()  # noqa: S301
    if not isinstance(artifact, ScoreArtifact):
        raise TypeError(f"Expected ScoreArtifact, got {type(artifact).__name__}")
    logger.info(f"Loaded score artifact from {path}")
    return artifact


def make_score_artifact(
    *,
    train_probe_scores: np.ndarray,
    train_labels: np.ndarray,
    calib_probe_scores: np.ndarray,
    calib_baseline_scores: np.ndarray,
    calib_labels: np.ndarray,
    test_probe_scores: np.ndarray,
    test_baseline_scores: np.ndarray,
    test_labels: np.ndarray,
    config: dict,
    seed: int,
    train_groups: np.ndarray | None = None,
    calib_groups: np.ndarray | None = None,
    test_groups: np.ndarray | None = None,
) -> ScoreArtifact:
    """Convenience constructor for building a ScoreArtifact from arrays."""
    return ScoreArtifact(
        train=SplitScores(
            probe_scores=train_probe_scores,
            baseline_scores=None,
            labels=train_labels,
            groups=train_groups,
        ),
        calib=SplitScores(
            probe_scores=calib_probe_scores,
            baseline_scores=calib_baseline_scores,
            labels=calib_labels,
            groups=calib_groups,
        ),
        test=SplitScores(
            probe_scores=test_probe_scores,
            baseline_scores=test_baseline_scores,
            labels=test_labels,
            groups=test_groups,
        ),
        calibration=None,
        config=config,
        seed=seed,
        created_at=datetime.now().isoformat(),
    )


class _ScoreArtifactUnpickler(pickle.Unpickler):
    """Unpickler that resolves ``__main__`` classes from score_artifact."""

    def find_class(self, module: str, name: str):  # noqa: ANN001
        if module == "__main__":
            import importlib

            for mod_name in ("score_artifact",):
                try:
                    mod = importlib.import_module(mod_name)
                    if hasattr(mod, name):
                        return getattr(mod, name)
                except ImportError:
                    continue
        return super().find_class(module, name)
