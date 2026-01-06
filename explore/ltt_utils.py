import numpy as np
import random
from typing import List, Sequence
from models_under_pressure.interfaces.dataset import LabelledDataset


def split_dataset(
    dataset: LabelledDataset,
    proportions: Sequence[float],
    shuffle: bool = True,
    random_seed: int | None = None,
) -> List[LabelledDataset]:
    """
    Split a LabelledDataset into multiple subsets according to specified proportions.
    
    Args:
        dataset: The LabelledDataset to split
        proportions: A sequence of floats (e.g. [0.6, 0.2, 0.2]) that sum to 1.0.
                    Determines the relative sizes of the returned subsets.
        shuffle: Whether to shuffle the dataset before splitting. Default: True
        random_seed: Seed for reproducibility. If None, uses random state.
    
    Returns:
        A list of LabelledDataset objects split according to proportions.
    
    Example:
        >>> train, val, test = split_dataset(dataset, [0.6, 0.2, 0.2])
    """
    # Validate proportions
    proportions = list(proportions)
    total = sum(proportions)
    if not np.isclose(total, 1.0):
        raise ValueError(f"Proportions must sum to 1.0, got {total}")
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Create indices
    indices = list(range(len(dataset)))
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(indices)
    
    # Split indices according to proportions
    splits = []
    start = 0
    for prop in proportions:
        end = start + int(np.round(prop * len(dataset)))
        splits.append(indices[start:end])
        start = end
    
    # Return split datasets
    return [dataset[split_indices] for split_indices in splits]


def is_pareto(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any((costs[:i])>=c, axis=1)) and np.all(np.any((costs[i+1:])>=c, axis=1))
    return is_efficient
    

def fixed_sequence_testing(h_sorted, p_vals, delta):
    list_rejected = []
    for b in range(len(h_sorted)):
        xx, yy, zz = np.unravel_index(h_sorted[b], p_vals.shape)  
        if p_vals[xx, yy, zz] < delta:
            list_rejected.append((xx+1,yy+1,zz+1))
        else:
            break    

    return list_rejected