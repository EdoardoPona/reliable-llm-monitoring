"""Pre-compute and cache baseline scores for all evaluation datasets.

Iterates over the 4 balanced evaluation datasets (anthropic, mt, mts, toolace)
× 2 splits (dev, test) and computes baseline LLM scores with gemma-3-27b-it,
uploading them to the ClearML cache via :mod:`baseline_registry`.

Usage::

    uv run experiments/precompute_baselines.py
    uv run experiments/precompute_baselines.py --splits test
    uv run experiments/precompute_baselines.py --no-cache
"""

import argparse
import logging
import os
from pathlib import Path

from baseline_registry import compute_or_fetch_baseline
from dotenv import load_dotenv

from reliable_monitoring.dataset import load_dataset

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "google/gemma-3-27b-it"

DATA_DIR = os.environ.get("DATA_DIR", "")

DATASETS = {
    "anthropic": {
        "dev": f"{DATA_DIR}/evals/dev/anthropic_balanced_apr_23.jsonl",
        "test": f"{DATA_DIR}/evals/test/anthropic_test_balanced_apr_23.jsonl",
    },
    "mt": {
        "dev": f"{DATA_DIR}/evals/dev/mt_balanced_apr_30.jsonl",
        "test": f"{DATA_DIR}/evals/test/mt_test_balanced_apr_30.jsonl",
    },
    "mts": {
        "dev": f"{DATA_DIR}/evals/dev/mts_balanced_apr_22.jsonl",
        "test": f"{DATA_DIR}/evals/test/mts_test_balanced_apr_22.jsonl",
    },
    "toolace": {
        "dev": f"{DATA_DIR}/evals/dev/toolace_balanced_apr_22.jsonl",
        "test": f"{DATA_DIR}/evals/test/toolace_test_balanced_apr_22.jsonl",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute baseline scores for evaluation datasets")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev", "test"],
        help="Which splits to compute (default: dev test).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputation (skip cache lookup, still uploads).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Model: {MODEL}")
    logger.info(f"Splits: {args.splits}")

    total = len(DATASETS) * len(args.splits)
    done = 0

    for group, paths in DATASETS.items():
        for split in args.splits:
            path = paths[split]
            done += 1
            logger.info(f"\n[{done}/{total}] {group} / {split}: {path}")

            ds = load_dataset(Path(path), activation_config=None)
            logger.info(f"  Dataset size: {len(ds)}")

            scores = compute_or_fetch_baseline(
                model_name=MODEL,
                dataset=ds,
                dataset_path=path,
                baseline_batch_size=4,
                local=False,
                skip_cache=args.no_cache,
            )
            logger.info(f"  Baseline scores: shape={scores.shape}, mean={scores.mean():.4f}")

    logger.info(f"\nDone. Computed/cached {total} baseline score arrays.")


if __name__ == "__main__":
    main()
