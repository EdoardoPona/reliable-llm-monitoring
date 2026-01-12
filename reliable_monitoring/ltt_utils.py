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


def fixed_sequence_testing(h_sorted, p_vals, delta):
    list_rejected = []
    for b in range(len(h_sorted)):
        xx, yy, zz = np.unravel_index(h_sorted[b], p_vals.shape)
        if p_vals[xx, yy, zz] < delta:
            list_rejected.append((xx + 1, yy + 1, zz + 1))
        else:
            break

    return list_rejected
