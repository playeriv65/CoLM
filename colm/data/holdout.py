"""Deterministic held-out split of a `SupervisedDataset` (for evaluation loss)."""

import copy
import json
import os
import re

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


_PROGRAM_HINT = re.compile(r"let'?s write a program\.?")
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


def question_key(prompt: str) -> str:
    """The question of a prompt, without the program hint and formatting: MathInstruct holds the
    same question several times (CoT and PoT solutions, several sources)."""
    return _NOT_ALNUM.sub(" ", _PROGRAM_HINT.sub("", prompt.lower())).strip()


def split_holdout(
    dataset: SupervisedDataset, size: int, seed: int
) -> tuple[SupervisedDataset, SupervisedDataset]:
    """Return (train, heldout); `heldout` is `size` examples, whole groups of the same question.

    Random groups are drawn with `seed` until `size` examples are held out, so no question of the
    held-out set (in any of its solutions) is trained on. The draw depends only on the data,
    `size` and `seed`; the order of both halves is preserved.
    """
    total = len(dataset)
    if not 0 < size < total:
        raise ValueError(f"holdout size must be in (0, {total}), got {size}")
    groups: dict[str, list[int]] = {}
    for i, prompt in enumerate(dataset.sources):
        groups.setdefault(question_key(prompt), []).append(i)
    members = list(groups.values())
    chosen, held = [], 0
    for g in np.random.default_rng(seed).permutation(len(members)):
        if held + len(members[g]) <= size:
            chosen.extend(members[g])
            held += len(members[g])
        if held == size:
            break
    if held != size:
        raise ValueError(f"cannot hold out exactly {size} examples in whole question groups")
    heldout_pos = sorted(chosen)
    is_held = set(heldout_pos)
    train_pos = [i for i in range(total) if i not in is_held]
    return select_examples(dataset, train_pos), select_examples(dataset, heldout_pos)


def save_holdout_indices(heldout: SupervisedDataset, output_dir: str) -> str:
    """Record the original MathInstruct indices of the held-out examples (provenance)."""
    path = os.path.join(output_dir, HOLDOUT_INDICES_FILENAME)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"original_index": [int(i) for i in heldout.indices]}, f)
    return path
