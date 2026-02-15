from dataclasses import dataclass

import numpy as np


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
