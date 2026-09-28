"""Deterministic held-out split of a `SupervisedDataset` (for evaluation loss)."""

import copy
import json
import os

import numpy as np

from colm.data.get_training_dataset import SupervisedDataset

# Per-example parallel lists of SupervisedDataset; everything else is dataset-level metadata
# (source-name table, number of sources) and stays identical in both halves.
_PER_EXAMPLE_FIELDS = (
    "sources",
    "targets",
    "data_sources",
    "indices",
    "completion_lengths",
)
HOLDOUT_INDICES_FILENAME = "holdout_indices.json"


def select_examples(dataset: SupervisedDataset, positions) -> SupervisedDataset:
    """Copy of `dataset` restricted to the examples at `positions` (in that order)."""
    positions = list(positions)
    part = copy.copy(dataset)
    for name in _PER_EXAMPLE_FIELDS:
        values = getattr(dataset, name)
        setattr(part, name, [values[i] for i in positions])
    return part


def split_holdout(
    dataset: SupervisedDataset, size: int, seed: int
) -> tuple[SupervisedDataset, SupervisedDataset]:
    """Return (train, heldout); `heldout` is `size` examples drawn uniformly with `seed`.

    The draw depends only on (len(dataset), size, seed), so every run over the same data file
    holds out the same examples, whatever its training seed. Order of both halves is preserved.
    """
    total = len(dataset)
    if not 0 < size < total:
        raise ValueError(f"holdout size must be in (0, {total}), got {size}")
    chosen = set(np.random.default_rng(seed).permutation(total)[:size].tolist())
    heldout_pos = sorted(chosen)
    train_pos = [i for i in range(total) if i not in chosen]
    return select_examples(dataset, train_pos), select_examples(dataset, heldout_pos)


def save_holdout_indices(heldout: SupervisedDataset, output_dir: str) -> str:
    """Record the original MathInstruct indices of the held-out examples (provenance)."""
    path = os.path.join(output_dir, HOLDOUT_INDICES_FILENAME)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"original_index": [int(i) for i in heldout.indices]}, f)
    return path
