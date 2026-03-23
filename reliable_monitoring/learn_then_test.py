from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from reliable_monitoring.risks import Risk


@dataclass
class Hypothesis:
    """A single hypothesis with a pre-filled p-value computation.

    Makes hypotheses first-class citizens: each carries its own callable
    that returns a p-value (with all parameters baked in), plus metadata
    describing what it represents.

    Attributes:
        p_value_fn: Zero-argument callable that returns the p-value for
            this hypothesis.  All parameters (empirical risk, sample size,
            alpha, bound function, …) are already captured.
        params: Arbitrary metadata describing the hypothesis, e.g.
            ``{"threshold": 0.7, "alpha": 0.3}``.
    """

    p_value_fn: Callable[[], float]
    params: dict[str, Any] = field(default_factory=dict)

    def p_value(self) -> float:
        """Compute and return the p-value for this hypothesis."""
        return float(self.p_value_fn())


def compute_p_values(hypotheses: list[Hypothesis]) -> np.ndarray:
    """Compute p-values for a list of hypotheses.

    Parameters
    ----------
    hypotheses : list[Hypothesis]
        Each hypothesis carries a pre-filled ``p_value_fn``.

    Returns
    -------
    np.ndarray, shape (m,)
        One p-value per hypothesis.
    """
    return np.array([h.p_value() for h in hypotheses])


@dataclass
class GraphicalTestResult:
    """Result of the graphical testing procedure (Bretz et al. 2009).

    Attributes:
        rejected: Indices of rejected hypotheses, in rejection order.
        final_weights: Alpha-weight vector after all rejections.
    """

    rejected: list[int]
    final_weights: np.ndarray


def graphical_testing(
    p_values: np.ndarray,
    weights: np.ndarray,
    transitions: np.ndarray,
    delta: float,
) -> GraphicalTestResult:
    """General graphical testing procedure (Bretz et al. 2009).

    Implements the sequential rejection algorithm on a weighted directed
    graph.  Controls the family-wise error rate (FWER) at level ``delta``.

    The algorithm is fully generic: it operates on flat arrays of p-values
    and an arbitrary graph structure.  It knows nothing about the semantics
    of the hypotheses (thresholds, risk levels, etc.).

    Fixed-sequence testing (FST) is recovered as a special case when
    ``weights = [1, 0, …, 0]`` and ``transitions`` is a chain
    (``g[i, i+1] = 1``).

    Parameters
    ----------
    p_values : np.ndarray, shape (m,)
        One p-value per hypothesis.
    weights : np.ndarray, shape (m,)
        Initial alpha-weights.  ``w[i] >= 0`` and ``sum(w) <= 1``.
    transitions : np.ndarray, shape (m, m)
        Transition matrix.  ``g[i, i] == 0`` and row sums ``<= 1``.
        ``g[i, j]`` is the fraction of ``w[i]`` propagated to hypothesis
        *j* when hypothesis *i* is rejected.
    delta : float
        Overall FWER level (significance budget).

    Returns
    -------
    GraphicalTestResult
        Rejected hypothesis indices (in order of rejection) and the
        final weight vector.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    w = np.array(weights, dtype=float)
    g = np.array(transitions, dtype=float)
    active = set(range(m))
    rejected: list[int] = []

    changed = True
    while changed:
        changed = False
        for j in sorted(active):
            if w[j] > 0 and p_values[j] <= w[j] * delta:
                rejected.append(j)
                active.remove(j)

                # Update weights for remaining hypotheses
                for i in active:
                    w[i] += w[j] * g[j, i]

                # Update transition matrix
                active_list = sorted(active)
                new_g = g.copy()
                for i in active_list:
                    for k in active_list:
                        if i != k:
                            denom = 1.0 - g[i, j] * g[j, i]
                            if denom > 1e-12:
                                new_g[i, k] = (g[i, k] + g[i, j] * g[j, k]) / denom
                            else:
                                new_g[i, k] = g[i, k]
                g = new_g

                # Zero out connections to/from rejected hypothesis
                g[j, :] = 0.0
                g[:, j] = 0.0
                w[j] = 0.0

                changed = True
                break  # restart scan

    return GraphicalTestResult(rejected=rejected, final_weights=w)


def is_pareto(costs, *, maximize: bool = False) -> np.ndarray:
    """
    Return a boolean mask indicating Pareto-efficient points.

    By default assumes `costs` are to be MINIMIZED.
    If `maximize=True`, assumes objectives are to be MAXIMIZED.

    A point i is dominated if there exists j != i such that:
      - minimization: costs[j] <= costs[i] for all dims AND costs[j] < costs[i] for at least one dim
      - maximization: costs[j] >= costs[i] for all dims AND costs[j] > costs[i] for at least one dim
    """
    costs = np.asarray(costs)
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        others = np.arange(n) != i
        if not np.any(others):
            continue

        if maximize:
            dominates_i = np.all(costs[others] >= costs[i], axis=1) & np.any(costs[others] > costs[i], axis=1)
        else:
            dominates_i = np.all(costs[others] <= costs[i], axis=1) & np.any(costs[others] < costs[i], axis=1)

        if np.any(dominates_i):
            is_efficient[i] = False

    return is_efficient


def fixed_sequence_testing(p_values: np.ndarray, delta: float) -> list[int]:
    """
    Fixed-sequence testing for an *ordered* 1D sequence of p-values.

    Parameters
    ----------
    p_values : sequence of float
        p_values[i] is the p-value of the i-th hypothesis in the testing order.
    delta : float
        Per-step significance threshold.

    Returns
    -------
    rejected_indices : list[int]
        The (0-based) indices rejected before the first non-rejection.
    """
    rejected_indices = []
    for i, p in enumerate(p_values):
        if p < delta:
            rejected_indices.append(i)
        else:
            break
    return rejected_indices


# ---------------------------------------------------------------------------
# Pareto filtering
# ---------------------------------------------------------------------------


@dataclass
class ParetoFilterResult:
    """Result of Pareto-filtering a threshold grid on two risks.

    Attributes:
        taus: Pareto-efficient thresholds, ordered safest-first (descending).
        pareto_mask: Boolean mask over the original ``tau_grid``.
        n_original: Number of thresholds before filtering.
    """

    taus: np.ndarray
    pareto_mask: np.ndarray
    n_original: int


def pareto_filter_thresholds(
    probe_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    delegation_scores: np.ndarray,
    tau_grid: np.ndarray,
    risks: list[Risk],
    merge_strategy: str = "replace",
) -> ParetoFilterResult:
    """Pareto-filter a threshold grid using two or more risks.

    Evaluates the given risks on the provided data for every threshold,
    computes the Pareto frontier (minimising all risks), and returns
    the efficient thresholds ordered safest-first (largest tau first).

    Args:
        probe_scores: Probe scores on the optimisation split.
        baseline_scores: Baseline scores on the optimisation split.
        labels: Ground-truth labels on the optimisation split.
        delegation_scores: Delegation signal on the optimisation split.
        tau_grid: Full grid of candidate thresholds.
        risks: List of Risk objects to Pareto-filter on.
        merge_strategy: Cascade merge strategy.

    Returns:
        :class:`ParetoFilterResult` with filtered thresholds and mask.
    """
    from reliable_monitoring.risks import evaluate_threshold_risks

    eval_result = evaluate_threshold_risks(
        probe_scores,
        baseline_scores,
        tau_grid,
        risks=risks,
        labels=labels,
        merge_strategy=merge_strategy,
        delegation_scores=delegation_scores,
    )

    risks_2d = eval_result.get_empirical_risks_array()
    pareto_mask = is_pareto(risks_2d, maximize=False)
    if not pareto_mask.any():
        pareto_mask = np.ones(len(tau_grid), dtype=bool)

    pareto_taus = tau_grid[pareto_mask]
    order = np.argsort(-pareto_taus)

    return ParetoFilterResult(
        taus=pareto_taus[order],
        pareto_mask=pareto_mask,
        n_original=len(tau_grid),
    )


# ---------------------------------------------------------------------------
# Pareto-filtered LTT threshold selection
# ---------------------------------------------------------------------------


@dataclass
class ParetoLTTResult:
    """Precomputed Pareto-filtered threshold grid for per-alpha LTT testing.

    Constructed once (expensive: risk evaluation + Pareto filtering), then
    queried cheaply per alpha via :meth:`select_threshold`.

    Attributes:
        taus: Pareto-filtered thresholds, ordered safest-first (descending).
        opt_risks: Optimisation-risk values for each tau (same order).
        ht_delegation_scores: Delegation scores on the hypothesis-testing split.
        budget_bound_fn: P-value bound function for the budget risk.
    """

    taus: np.ndarray
    opt_risks: np.ndarray
    ht_delegation_scores: np.ndarray
    budget_bound_fn: Callable

    def select_threshold(self, alpha_budget: float, delta: float) -> float | None:
        """Find the best reliable threshold for a given budget level.

        Runs fixed-sequence testing on the Pareto-filtered taus (safest first)
        using the hypothesis-testing split, then picks the reliable tau with
        the lowest optimisation risk.

        Returns None if no threshold passes the test at this alpha level.
        """
        n_ht = len(self.ht_delegation_scores)
        p_values = np.array(
            [
                float(
                    self.budget_bound_fn(
                        float((self.ht_delegation_scores > tau).mean()),
                        n_ht,
                        alpha_budget,
                    )
                )
                for tau in self.taus
            ]
        )
        rejected = fixed_sequence_testing(p_values, delta)
        if not rejected:
            return None
        # Among reliable taus, pick the one minimising the opt risk
        reliable_opt = self.opt_risks[rejected]
        best_idx = int(np.argmin(reliable_opt))
        return float(self.taus[rejected[best_idx]])


def build_pareto_ltt(
    ht_delegation_scores: np.ndarray,
    opt_probe_scores: np.ndarray,
    opt_baseline_scores: np.ndarray,
    opt_labels: np.ndarray,
    opt_delegation_scores: np.ndarray,
    tau_grid: np.ndarray,
    opt_risk: Risk,
    budget_risk: Risk,
    merge_strategy: str = "replace",
) -> ParetoLTTResult:
    """Build a Pareto-filtered LTT threshold selector.

    Pareto-filters thresholds on ``[budget_risk, opt_risk]`` using
    :func:`pareto_filter_thresholds`, then packages the result for
    per-alpha FST queries via :meth:`ParetoLTTResult.select_threshold`.

    Args:
        ht_delegation_scores: Delegation scores on the hypothesis-testing split
            (used only for per-alpha p-value computation later).
        opt_probe_scores: Probe scores on the optimisation split.
        opt_baseline_scores: Baseline scores on the optimisation split.
        opt_labels: Ground-truth labels on the optimisation split.
        opt_delegation_scores: Delegation scores on the optimisation split.
        tau_grid: Full grid of candidate thresholds.
        opt_risk: Risk object for the optimisation objective (e.g. AccuracyRisk).
        budget_risk: Risk object for the budget constraint (e.g. BudgetCostRisk).
        merge_strategy: Cascade merge strategy.

    Returns:
        ParetoLTTResult ready for per-alpha queries.
    """
    from reliable_monitoring.risks import evaluate_threshold_risks

    pf = pareto_filter_thresholds(
        opt_probe_scores,
        opt_baseline_scores,
        opt_labels,
        opt_delegation_scores,
        tau_grid,
        risks=[budget_risk, opt_risk],
        merge_strategy=merge_strategy,
    )

    # Evaluate opt_risk on the Pareto-filtered taus (for select_threshold tie-breaking)
    opt_eval = evaluate_threshold_risks(
        opt_probe_scores,
        opt_baseline_scores,
        pf.taus,
        risks=opt_risk,
        labels=opt_labels,
        merge_strategy=merge_strategy,
        delegation_scores=opt_delegation_scores,
    )

    return ParetoLTTResult(
        taus=pf.taus,
        opt_risks=opt_eval[opt_risk.name],
        ht_delegation_scores=ht_delegation_scores,
        budget_bound_fn=budget_risk.p_value_bound_fn,
    )
