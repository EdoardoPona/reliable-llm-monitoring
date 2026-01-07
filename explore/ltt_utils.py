import numpy as np


def is_pareto(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any((costs[:i]) >= c, axis=1)) and np.all(np.any((costs[i + 1 :]) >= c, axis=1))
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
