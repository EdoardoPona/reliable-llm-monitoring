"""Calibrate probe scores in a pre-computed ScoreArtifact.

Takes a ``ScoreArtifact`` (from ``compute_scores.py``), fits a calibration
model on a held-out subset of the calibration split, and applies the
calibration transform to all splits.  The result is an updated
``ScoreArtifact`` with ``CalibrationInfo`` populated.

Usage::

    uv run experiments/calibrate_scores.py --scores scores.pkl --method isotonic-regression
    uv run experiments/calibrate_scores.py --scores scores.pkl --method platt-scaling --output scores_cal.pkl
"""

import argparse
import logging

import numpy as np
from score_artifact import CalibrationInfo, ScoreArtifact, load_score_artifact, save_score_artifact
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("isotonic-regression", "platt-scaling")


def calibrate_scores(
    artifact: ScoreArtifact,
    method: str = "isotonic-regression",
    auxiliary_proportion: float = 0.15,
    seed: int | None = None,
) -> ScoreArtifact:
    """Fit a calibration model on a calib subset and apply to all splits.

    The calibration operates purely in score space: it learns a monotonic
    mapping from raw probe scores to calibrated probabilities using a
    held-out auxiliary subset of the calibration data.

    Args:
        artifact: ScoreArtifact with uncalibrated probe scores.
        method: Calibration method — ``"isotonic-regression"`` or
            ``"platt-scaling"``.
        auxiliary_proportion: Fraction of calib data reserved for fitting
            the calibration model (the rest remains available for
            hypothesis testing).
        seed: Random seed for selecting auxiliary indices.  Defaults to
            the artifact's seed.

    Returns:
        A new ScoreArtifact with ``calibration`` field populated.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown calibration method: {method!r}. Supported: {SUPPORTED_METHODS}")

    if artifact.calibration is not None:
        logger.warning("Artifact already has calibration info — it will be replaced.")

    if seed is None:
        seed = artifact.seed

    rng = np.random.RandomState(seed)

    # --- Select auxiliary indices from calib split ---
    n_calib = len(artifact.calib.labels)
    n_auxiliary = max(1, int(n_calib * auxiliary_proportion))
    all_indices = np.arange(n_calib)
    rng.shuffle(all_indices)
    auxiliary_indices = np.sort(all_indices[:n_auxiliary])

    logger.info(
        f"Calibration: using {n_auxiliary}/{n_calib} calib examples ({auxiliary_proportion:.0%}) for {method} fitting"
    )

    # --- Fit calibration model ---
    aux_scores = artifact.calib.probe_scores[auxiliary_indices]
    aux_labels = artifact.calib.labels[auxiliary_indices]

    if method == "isotonic-regression":
        cal_model = IsotonicRegression(out_of_bounds="clip")
        cal_model.fit(aux_scores, aux_labels)
        transform = cal_model.predict
    else:  # platt-scaling
        cal_model = LogisticRegression(max_iter=1000)
        cal_model.fit(aux_scores.reshape(-1, 1), aux_labels)
        transform = lambda scores: cal_model.predict_proba(scores.reshape(-1, 1))[:, 1]  # noqa: E731

    # --- Apply calibration to all splits ---
    cal_train = transform(artifact.train.probe_scores)
    cal_calib = transform(artifact.calib.probe_scores)
    cal_test = transform(artifact.test.probe_scores)

    logger.info(
        f"Calibration applied. Score ranges: "
        f"train=[{cal_train.min():.3f}, {cal_train.max():.3f}], "
        f"calib=[{cal_calib.min():.3f}, {cal_calib.max():.3f}], "
        f"test=[{cal_test.min():.3f}, {cal_test.max():.3f}]"
    )

    # --- Build updated artifact ---
    calibration_info = CalibrationInfo(
        method=method,
        auxiliary_indices=auxiliary_indices,
        auxiliary_proportion=auxiliary_proportion,
        train_probe_scores=np.asarray(cal_train, dtype=np.float64),
        calib_probe_scores=np.asarray(cal_calib, dtype=np.float64),
        test_probe_scores=np.asarray(cal_test, dtype=np.float64),
    )

    return ScoreArtifact(
        train=artifact.train,
        calib=artifact.calib,
        test=artifact.test,
        calibration=calibration_info,
        config=artifact.config,
        seed=artifact.seed,
        created_at=artifact.created_at,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate probe scores in a ScoreArtifact")
    parser.add_argument(
        "--scores",
        type=str,
        required=True,
        help="Path to input ScoreArtifact pickle.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="isotonic-regression",
        choices=SUPPORTED_METHODS,
        help="Calibration method (default: isotonic-regression).",
    )
    parser.add_argument(
        "--auxiliary-proportion",
        type=float,
        default=0.15,
        help="Fraction of calib data for calibration fitting (default: 0.15).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for calibrated artifact. Default: overwrites input.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    artifact = load_score_artifact(args.scores)
    calibrated = calibrate_scores(
        artifact,
        method=args.method,
        auxiliary_proportion=args.auxiliary_proportion,
    )

    output_path = args.output or args.scores
    save_score_artifact(calibrated, output_path)
    logger.info(f"Calibrated artifact saved to {output_path}")
