import numpy as np


def is_pareto(costs, *, maximize: bool = False):
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


def fixed_sequence_testing(p_values, delta):
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
