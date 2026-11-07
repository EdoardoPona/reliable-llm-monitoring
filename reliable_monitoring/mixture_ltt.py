"""LTT for arbitrary mixtures of fixed domains, using the existing risk tests.

Domain labels are used during calibration and evaluation, not for routing.
The max-p intersection--union test needs no correction for the domain count;
fixed-sequence testing handles the independently ordered threshold candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from reliable_monitoring.learn_then_test import fixed_sequence_testing, is_pareto
from reliable_monitoring.risks import (
    AccuracyRisk,
    BudgetCostRisk,
    Risk,
    ThresholdEvaluationResult,
    evaluate_threshold_risks,
)


def mixture_weights(weights: Mapping[str, float], domains: Sequence[str]) -> np.ndarray:
    """Validate and normalise nonnegative weights in an explicit domain order."""
    if not domains or len(set(domains)) != len(domains) or set(weights) != set(domains):
        raise ValueError("Weights must specify each unique domain exactly once")
    values = np.array([weights[g] for g in domains], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Mixture weights must be finite, nonnegative, and have positive sum")
    return values / values.sum()


def evaluate_domain_risks(
    probe_scores: np.ndarray,
    expert_scores: np.ndarray,
    labels: np.ndarray,
    dv_scores: np.ndarray,
    groups: np.ndarray,
    thresholds: np.ndarray,
    *,
    domains: Sequence[str],
    risks: Sequence[Risk],
    merge_strategy: str = "replace",
) -> dict[str, ThresholdEvaluationResult]:
    """Apply the existing cascade/risk evaluation separately to each domain.

    Missing domains are retained, with zero counts and unknown empirical risks.
    Only per-example accuracy error and call rate are supported: pooled AUROC is
    not linear in domain mixture weights.
    """
    domains = tuple(domains)
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("Specify a nonempty sequence of unique domains")
    arrays = [np.asarray(a) for a in (probe_scores, expert_scores, labels, dv_scores, groups)]
    if any(a.ndim != 1 or len(a) != len(arrays[0]) for a in arrays):
        raise ValueError("Scores, labels and groups must be aligned one-dimensional arrays")
    if set(arrays[-1]) - set(domains):
        raise ValueError("Observed groups are missing from domains")
    if not risks or any(r.name not in {AccuracyRisk.name, BudgetCostRisk.name} for r in risks):
        raise ValueError("Mixture calibration supports accuracy_error and budget only")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all():
        raise ValueError("Specify a nonempty finite threshold grid")
    result = {}
    for domain in domains:
        mask = arrays[-1] == domain
        if not mask.any():
            result[domain] = ThresholdEvaluationResult(
                thresholds=thresholds,
                n_samples=0,
                empirical_risks={r.name: np.full(len(thresholds), np.nan) for r in risks},
                risks=list(risks),
            )
        else:
            result[domain] = evaluate_threshold_risks(
                arrays[0][mask],
                arrays[1][mask],
                thresholds,
                risks=list(risks),
                labels=arrays[2][mask],
                delegation_scores=arrays[3][mask],
                merge_strategy=merge_strategy,
            )
    return result


@dataclass
class MixtureLTTResult:
    """Independent candidate objective plus domain-specific hypothesis tests."""

    taus: np.ndarray
    opt_risks: np.ndarray
    ht_by_domain: dict[str, ThresholdEvaluationResult]
    guaranteed_risk: Risk

    def domain_p_values(self, alpha: float) -> dict[str, np.ndarray]:
        if not 0 < alpha < 1:
            raise ValueError("Risk target must lie strictly between zero and one")
        values = {}
        for domain, evaluation in self.ht_by_domain.items():
            if evaluation.n_samples == 0:
                values[domain] = np.ones(len(self.taus))
            else:
                p = self.guaranteed_risk.p_value_bound_fn(
                    evaluation[self.guaranteed_risk.name],
                    evaluation.n_samples,
                    alpha,
                )
                # Existing concentration bounds can exceed 1; undefined values
                # must never be interpreted as evidence for certification.
                values[domain] = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 0, 1)
        return values

    def p_values(self, alpha: float) -> np.ndarray:
        return np.stack(list(self.domain_p_values(alpha).values())).max(axis=0)

    def select_threshold(self, alpha: float, delta: float) -> float | None:
        if not 0 < delta < 1:
            raise ValueError("Significance level must lie strictly between zero and one")
        rejected = fixed_sequence_testing(self.p_values(alpha), delta)
        if not rejected:
            return None
        best = rejected[int(np.argmin(self.opt_risks[rejected]))]
        return float(self.taus[best])


def build_mixture_pareto_ltt(
    ht_probe_scores: np.ndarray,
    ht_expert_scores: np.ndarray,
    ht_labels: np.ndarray,
    ht_dv_scores: np.ndarray,
    ht_groups: np.ndarray,
    opt_probe_scores: np.ndarray,
    opt_expert_scores: np.ndarray,
    opt_labels: np.ndarray,
    opt_dv_scores: np.ndarray,
    opt_groups: np.ndarray,
    tau_grid: np.ndarray,
    guaranteed_risk: Risk,
    opt_risk: Risk,
    *,
    domains: Sequence[str],
    reference_weights: Mapping[str, float],
    merge_strategy: str = "replace",
) -> MixtureLTTResult:
    """Pareto-filter/order on D_est; test every domain independently on D_cal.

    The frontier trades worst-domain guaranteed risk against reference-mixture
    optimisation risk. Budget candidates are ordered by descending threshold;
    accuracy candidates by estimated worst-domain error, then estimated cost.
    Neither candidate filtering nor ordering reads the hypothesis-testing data.
    """
    if guaranteed_risk.name == opt_risk.name:
        raise ValueError("guaranteed_risk and opt_risk must be different")
    domains = tuple(domains)
    weights = mixture_weights(reference_weights, domains)
    tau_grid = np.asarray(tau_grid, dtype=float)
    opt = evaluate_domain_risks(
        opt_probe_scores,
        opt_expert_scores,
        opt_labels,
        opt_dv_scores,
        opt_groups,
        tau_grid,
        domains=domains,
        risks=[guaranteed_risk, opt_risk],
        merge_strategy=merge_strategy,
    )
    # A missing optimisation domain contributes the conservative value 1.
    # This affects only candidate power/objectives, never the test's validity.
    guaranteed = np.nan_to_num(np.stack([opt[g][guaranteed_risk.name] for g in domains]), nan=1).max(axis=0)
    objective = weights @ np.nan_to_num(np.stack([opt[g][opt_risk.name] for g in domains]), nan=1)
    keep = np.flatnonzero(is_pareto(np.column_stack([guaranteed, objective])))
    if guaranteed_risk.name == BudgetCostRisk.name:
        order = np.argsort(-tau_grid[keep], kind="stable")
    else:
        order = np.lexsort((-tau_grid[keep], objective[keep], guaranteed[keep]))
    selected = keep[order]
    taus = tau_grid[selected]
    ht = evaluate_domain_risks(
        ht_probe_scores,
        ht_expert_scores,
        ht_labels,
        ht_dv_scores,
        ht_groups,
        taus,
        domains=domains,
        risks=[guaranteed_risk],
        merge_strategy=merge_strategy,
    )
    return MixtureLTTResult(taus, objective[selected], ht, guaranteed_risk)


def simplex_grid(n_domains: int, resolution: int) -> np.ndarray:
    """All simplex points with coordinates that are multiples of 1/resolution."""
    if n_domains < 1 or resolution < 1:
        raise ValueError("Domain count and grid resolution must be positive")

    def compositions(total, parts):
        if parts == 1:
            yield (total,)
        else:
            for first in range(total + 1):
                for rest in compositions(total - first, parts - 1):
                    yield (first, *rest)

    return np.array(list(compositions(resolution, n_domains)), dtype=float) / resolution
