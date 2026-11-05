"""Calibrate and evaluate fixed-domain mixture shifts from saved CTD scores.

This entry point never trains models, generates activations, or edits datasets.
Use --inventory-only to check all inputs without performing risk control.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from config import load_config
from dv_ltt_cascade import pareto_ht_opt_split, split_calib_eval
from mixture_shift_inventory import add_row_inventory, counts, load_artifact
from saved_ablation_runs import artifact_path

from reliable_monitoring.learn_then_test import RiskParetoLTTResult, build_risk_pareto_ltt, fixed_sequence_testing
from reliable_monitoring.mixture_ltt import (
    MixtureLTTResult,
    build_mixture_pareto_ltt,
    evaluate_domain_risks,
    mixture_weights,
    simplex_grid,
)
from reliable_monitoring.risks import AccuracyRisk, BudgetCostRisk, evaluate_threshold_risks

METHODS = ("pooled", "pooled_matched", "mixture")


def json_safe(value):
    """Write standard JSON, with unavailable empirical metrics represented as null."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def score_inputs(arrays: dict, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        arrays["test_probe"][indices],
        arrays["test_expert"][indices],
        arrays["test_labels"][indices],
        arrays["test_dv"][indices],
    )


def selection_details(selector, alpha: float, delta: float) -> dict:
    if isinstance(selector, MixtureLTTResult):
        per_domain = selector.domain_p_values(alpha)
        p_values = selector.p_values(alpha)
    else:
        per_domain = None
        p_values = np.asarray(
            selector.guaranteed_risk.p_value_bound_fn(
                selector.ht_guaranteed_risks,
                selector.n_ht,
                alpha,
            ),
            dtype=float,
        )
    rejected = fixed_sequence_testing(p_values, delta)
    return {
        "threshold": selector.select_threshold(alpha, delta),
        "candidate_thresholds": selector.taus,
        "optimisation_risks": selector.opt_risks,
        "p_values": p_values,
        "domain_p_values": per_domain,
        "certified_indices": rejected,
    }


def evaluate_policy(arrays, indices, threshold, domains, reference, mixtures, merge_strategy):
    if threshold is None:
        return None
    evaluated = evaluate_domain_risks(
        *score_inputs(arrays, indices),
        arrays["test_domains"][indices],
        np.array([threshold]),
        domains=domains,
        risks=[AccuracyRisk, BudgetCostRisk],
        merge_strategy=merge_strategy,
    )
    result = {
        "domains": {
            g: {
                "n": v.n_samples,
                "accuracy_error": float(v[AccuracyRisk.name][0]),
                "budget": float(v[BudgetCostRisk.name][0]),
            }
            for g, v in evaluated.items()
        }
    }
    for risk in (AccuracyRisk, BudgetCostRisk):
        values = np.array([evaluated[g][risk.name][0] for g in domains])
        # Unknown risk in a domain with zero mixture mass does not make that
        # mixture unidentified; any positive mass on an unknown domain does.
        result[risk.name] = {
            "reference": np.where(reference == 0, 0, reference * values).sum(),
            "worst_domain": np.max(values),
            "mixtures": np.where(mixtures == 0, 0, mixtures * values).sum(axis=1),
        }
    # Retain medical subtype diagnostics without treating subtype shifts as
    # covered by the three-domain guarantee.
    sources = tuple(sorted(set(arrays["test_groups"].astype(str))))
    by_source = evaluate_domain_risks(
        *score_inputs(arrays, indices),
        arrays["test_groups"][indices],
        np.array([threshold]),
        domains=sources,
        risks=[AccuracyRisk, BudgetCostRisk],
        merge_strategy=merge_strategy,
    )
    result["sources"] = {
        g: {
            "n": v.n_samples,
            "accuracy_error": float(v[AccuracyRisk.name][0]),
            "budget": float(v[BudgetCostRisk.name][0]),
        }
        for g, v in by_source.items()
    }
    return result


def run_trial(
    arrays: dict,
    *,
    settings: dict,
    risk_name: str,
    targets: list[float],
    split_seed: int,
    domains: tuple[str, ...],
    reference_weights: dict[str, float],
    mixtures: np.ndarray,
) -> dict:
    """One paired split, with ordering on D_est and all tests on D_cal only."""
    if risk_name not in {AccuracyRisk.name, BudgetCostRisk.name}:
        raise ValueError("guaranteed_risk must be accuracy_error or budget")
    if not targets or any(not np.isfinite(a) or not 0 < a < 1 for a in targets):
        raise ValueError("Risk targets must lie strictly between zero and one")
    for name in ("calib_fraction", "pareto_split_proportion", "guarantee_probability"):
        if not 0 < settings[name] < 1:
            raise ValueError(f"{name} must lie strictly between zero and one")
    if settings["tau_steps"] < 2:
        raise ValueError("Need at least two candidate thresholds")
    merge = settings["merge_strategy"]
    if merge not in {"replace", "avg"}:
        raise ValueError("Unsupported merge strategy")
    guaranteed = AccuracyRisk if risk_name == AccuracyRisk.name else BudgetCostRisk
    objective = BudgetCostRisk if guaranteed is AccuracyRisk else AccuracyRisk
    reference = mixture_weights(reference_weights, domains)
    delta = 1 - settings["guarantee_probability"]
    # Same splitter as the official experiment. Keep global row indices for
    # reproducibility and to show explicitly that no dev/test rows were mixed.
    calib, evaluation = split_calib_eval(
        np.arange(len(arrays["test_labels"])),
        calib_fraction=settings["calib_fraction"],
        seed=split_seed,
    )
    calib_idx, eval_idx = calib[0], evaluation[0]
    assert calib_idx is not None and eval_idx is not None
    ht_local, opt_local = pareto_ht_opt_split(len(calib_idx), settings["pareto_split_proportion"], split_seed)
    ht_idx, opt_idx = calib_idx[ht_local], calib_idx[opt_local]
    if not len(ht_idx) or not len(opt_idx) or not len(eval_idx):
        raise ValueError("The split must provide nonempty D_est, D_cal and evaluation sets")
    dev_dv = arrays["dev_dv"]
    tau_grid = np.linspace(float(dev_dv.min()) - 0.01, float(dev_dv.max()) + 0.01, settings["tau_steps"])
    ht, opt = score_inputs(arrays, ht_idx), score_inputs(arrays, opt_idx)
    group_scores = arrays["test_domains"]
    mixture = build_mixture_pareto_ltt(
        *ht,
        group_scores[ht_idx],
        *opt,
        group_scores[opt_idx],
        tau_grid,
        guaranteed,
        objective,
        domains=domains,
        reference_weights=reference_weights,
        merge_strategy=merge,
    )
    pooled = build_risk_pareto_ltt(*ht, *opt, tau_grid, guaranteed, objective, merge_strategy=merge)
    matched_ht = evaluate_threshold_risks(
        ht[0],
        ht[1],
        mixture.taus,
        risks=guaranteed,
        labels=ht[2],
        delegation_scores=ht[3],
        merge_strategy=merge,
    )
    matched = RiskParetoLTTResult(
        mixture.taus,
        mixture.opt_risks,
        matched_ht[guaranteed.name],
        len(ht_idx),
        guaranteed,
    )
    selectors = dict(zip(METHODS, (pooled, matched, mixture), strict=True))
    partitions = {"ordering": opt_idx, "hypothesis_testing": ht_idx, "evaluation": eval_idx}
    result: dict = {
        "split_seed": split_seed,
        "split_indices": partitions,
        "domain_counts": {name: counts(group_scores[idx]) for name, idx in partitions.items()},
        "source_counts": {name: counts(arrays["test_groups"][idx]) for name, idx in partitions.items()},
        "guaranteed_risk": risk_name,
        "delta": delta,
        "results": [],
    }
    policy_cache = {}
    for alpha in targets:
        for method, selector in selectors.items():
            selection = selection_details(selector, alpha, delta)
            tau = selection["threshold"]
            if tau not in policy_cache:
                policy_cache[tau] = evaluate_policy(arrays, eval_idx, tau, domains, reference, mixtures, merge)
            metrics = policy_cache[tau]
            worst = None if metrics is None else metrics[risk_name]["worst_domain"]
            result["results"].append(
                {
                    "method": method,
                    "alpha": alpha,
                    "selection": selection,
                    "evaluation": metrics,
                    "observed_worst_domain_exceedance": None
                    if worst is None or not np.isfinite(worst)
                    else bool(worst > alpha),
                }
            )
    return result


def run_experiment(config, output_dir: Path, *, inventory_only=False, data_root=None, max_trials=None):
    root = Path(config.artifact_root).resolve()
    if output_dir.resolve().is_relative_to(root):
        raise ValueError("Write new results outside the original score-artifact directory")
    domains = tuple(dict.fromkeys(config.source_to_domain.values()))
    weights = mixture_weights(config.reference_weights, domains)
    grid = simplex_grid(len(domains), config.mixture_resolution)
    mixtures = np.unique(np.vstack([grid, weights, np.full(len(domains), 1 / len(domains))]), axis=0)
    n_trials = int(config.trials["n"])
    if max_trials is not None:
        n_trials = min(n_trials, max_trials)
    if n_trials < 1:
        raise ValueError("Need at least one trial")
    base_seed = config.trials.get("base_seed")
    if n_trials > 1 and base_seed is None:
        raise ValueError("Repeated splits require a base_seed")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", vars(config))
    write_json(output_dir / "mixtures.json", {"domains": domains, "weights": mixtures, "reference": weights})
    version = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    inventory: dict = {"code_commit": version, "artifacts": [], "inventory_only": inventory_only}
    trial_file = None
    try:
        if not inventory_only:
            trial_file = (output_dir / "trials.jsonl").open("w")
        for expert in config.experts:
            targets = list(config.targets[expert])
            # Configuration lists minimum accuracies for accuracy control;
            # convert once to error targets used by the existing Risk API.
            if config.guaranteed_risk == AccuracyRisk.name:
                targets = [1 - accuracy for accuracy in targets]
            for seed in config.seeds:
                path = artifact_path(root, expert, config.cell, seed)
                arrays, metadata, item = load_artifact(path, config.source_to_domain)
                cfg = metadata["config"]
                if cfg["seed"] != seed or cfg["expert_name"] != expert or cfg["cell_name"] != config.cell:
                    raise ValueError(f"Artifact metadata disagrees with its requested identity: {path}")
                item.update({"expert": expert, "seed": seed, "cell": config.cell})
                if data_root is not None:
                    add_row_inventory(arrays, metadata, item, Path(data_root))
                inventory["artifacts"].append(item)
                print(f"Checked {expert}/{config.cell}/seed_{seed}: {item['test_domains']}", flush=True)
                if inventory_only:
                    continue
                assert trial_file is not None
                for trial in range(n_trials):
                    split_seed = seed if base_seed is None else int(base_seed) + trial
                    result = run_trial(
                        arrays,
                        settings=config.shared,
                        risk_name=config.guaranteed_risk,
                        targets=targets,
                        split_seed=split_seed,
                        domains=domains,
                        reference_weights=config.reference_weights,
                        mixtures=mixtures,
                    )
                    result.update({"expert": expert, "model_seed": seed, "trial": trial, "cell": config.cell})
                    trial_file.write(json.dumps(json_safe(result), allow_nan=False) + "\n")
                    trial_file.flush()
        inventory["status"] = "complete"
    except Exception:
        inventory["status"] = "incomplete"
        raise
    finally:
        if trial_file is not None:
            trial_file.close()
        write_json(output_dir / "inventory.json", inventory)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, help="Override the saved-score root, e.g. on the lab server")
    parser.add_argument("--output-dir", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--data-root", type=Path, help="Optional original JSONL root for ID/overlap inventory")
    parser.add_argument(
        "--inventory-only", action="store_true", help="Validate scores/data without calibrating any thresholds"
    )
    parser.add_argument("--max-trials", type=int, help="Limit repeated splits for a smoke run")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.artifact_root is not None:
        config.artifact_root = str(args.artifact_root)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or Path("results/mixture_shift") / f"{args.config.stem}_{stamp}"
    run_experiment(
        config, output_dir, inventory_only=args.inventory_only, data_root=args.data_root, max_trials=args.max_trials
    )
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
