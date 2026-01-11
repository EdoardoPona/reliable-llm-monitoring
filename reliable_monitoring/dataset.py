import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from models_under_pressure.activation_store import ActivationStore
from models_under_pressure.interfaces.dataset import LabelledDataset


@dataclass
class ActivationConfig:
    model_name: str
    layer: int
    aggregation_strategy = None  # TODO implement this


def load_dataset(
    dataset_path: Path,
    activation_config: ActivationConfig | None,
) -> LabelledDataset:
    dataset = LabelledDataset.load_from(dataset_path)
    if not activation_config:
        return dataset

    # Load and attach precomputed activations
    store = ActivationStore()  # uses DATA_DIR/activations via config
    dataset = store.enrich(
        dataset=dataset,
        path=dataset_path,
        model_name=activation_config.model_name,
        layer=activation_config.layer,
        mmap=True,  # Use memory-mapped files for large datasets
    )
    return dataset


def sample_from_dataset(dataset: LabelledDataset, num_samples: int) -> LabelledDataset:
    indices = list(range(len(dataset)))
    sample = random.sample(indices, num_samples)
    return dataset[sample]


def split_dataset(
    dataset: LabelledDataset,
    proportions: Sequence[float],
    shuffle: bool = True,
    random_seed: int | None = None,
) -> list[LabelledDataset]:
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
