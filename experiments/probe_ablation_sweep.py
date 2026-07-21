"""Run the seven-cell probe ablation across experts and random seeds."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml
from config import expand_env_vars
from probe_ablation import compute_cell, save_cell

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_runs(config_path: Path) -> list[SimpleNamespace]:
    raw = expand_env_vars(yaml.safe_load(config_path.read_text()))
    shared = raw["shared"]
    runs = []
    for cell in raw["cells"]:
        for expert in raw["experts"]:
            # The seed also controls dataset subsampling and the calibration/evaluation
            # split, so deterministic probes still need every seed for paired comparisons.
            for seed in raw["seeds"]:
                config = deepcopy(shared)
                config.update(
                    {
                        "cell_name": cell["name"],
                        "probe": cell["probe"],
                        "dv_probe": cell["dv_probe"],
                        "expert_name": expert["name"],
                        "baseline_model_name": expert["model_name"],
                        "seed": seed,
                    }
                )
                runs.append(SimpleNamespace(**config))
    return runs


def run_sweep(config_path: Path, output_dir: Path, resume: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = build_runs(config_path)
    manifest = []
    for index, config in enumerate(runs, start=1):
        run_dir = output_dir / config.expert_name / config.cell_name / f"seed_{config.seed}"
        artifact = run_dir / "scores.npz"
        logger.info("[%d/%d] %s / %s / seed %d", index, len(runs), config.expert_name, config.cell_name, config.seed)
        if resume and artifact.exists():
            logger.info("Reusing %s", artifact)
        else:
            save_cell(config, compute_cell(config), run_dir)
        manifest.append(
            {
                "expert": config.expert_name,
                "cell": config.cell_name,
                "seed": config.seed,
                "artifact": str(artifact.relative_to(output_dir)),
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the probe architecture ablation")
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/probe_ablation.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    destination = args.output_dir or Path("results/probe_ablation") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_sweep(args.config, destination, resume=not args.no_resume)
