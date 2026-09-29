"""Padding-free (packed) batches: several examples concatenated in one row.

The layout is the one of transformers' `DataCollatorWithFlattening`: concatenated `input_ids`,
`position_ids` restarting at every example, no attention mask, and the label of the first token of
every example is -100, so no loss term crosses an example boundary. The models must be called with
`use_cache=False` (with a `DynamicCache` transformers no longer detects the packing and the
sequences attend to each other) and an attention implementation that reads `cu_seq_lens_q`
(`colm.train.attention`).
"""

from dataclasses import dataclass

import numpy as np
import torch

IGNORE_INDEX = -100
META = "colm_meta"  # key of the per-example bookkeeping tensors inside a batch


@dataclass
class Example:
    """One tokenised training example (CPU, numpy: it is pickled between ranks)."""

    input_ids: np.ndarray
    labels: np.ndarray  # token ids of the completion, -100 on the prompt
    source: int = 0
    index: int = -1  # position in the original data file
    completion_length: int = -1

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def num_labels(self) -> int:
        """Tokens of the completion (the first token of an example is never predicted)."""
        return int((self.labels[1:] != IGNORE_INDEX).sum())


def pack(examples: list[Example]) -> dict:
    """One row holding `examples`: the batch dict the models and losses take (CPU tensors)."""
    lengths = torch.tensor([len(e) for e in examples])
    cu = torch.zeros(len(examples) + 1, dtype=torch.int32)
    cu[1:] = lengths.cumsum(0)
    labels = torch.from_numpy(np.concatenate([e.labels for e in examples]))
    labels[cu[:-1].long()] = IGNORE_INDEX
    total = int(cu[-1])
    starts = cu[:-1].long()
    position_ids = torch.arange(total) - torch.repeat_interleave(starts, lengths)
    return {
        "input_ids": torch.from_numpy(np.concatenate([e.input_ids for e in examples]))[None],
        "position_ids": position_ids[None],
        "labels": labels[None],
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": int(lengths.max()),
        "max_length_k": int(lengths.max()),
        META: {
            "sources": torch.tensor([e.source for e in examples]),
            "indices": torch.tensor([e.index for e in examples]),
            "completion_lengths": torch.tensor([e.completion_length for e in examples]),
        },
    }


def model_inputs(batch: dict) -> dict:
    """Keyword arguments of a forward pass on a packed batch."""
    keys = (
        "input_ids",
        "position_ids",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
    )
    return {**{k: batch[k] for k in keys}, "use_cache": False}


def label_positions(batch: dict):
    """(positions, targets, segment): where the row predicts a label, what, and for which example."""
    labels = batch["labels"][0]
    targets = torch.cat([labels[1:], labels.new_full((1,), IGNORE_INDEX)])
    positions = (targets != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    segment = torch.bucketize(positions, batch["cu_seq_lens_q"][1:].long(), right=True)
    return positions, targets[positions], segment


def greedy_groups(lengths: list[int], budget: int) -> list[list[int]]:
    """Order-preserving split of items into groups of at most `budget` tokens (at least one item)."""
    groups, current, used = [], [], 0
    for i, n in enumerate(lengths):
        if current and used + n > budget:
            groups.append(current)
            current, used = [], 0
        current.append(i)
        used += n
    if current:
        groups.append(current)
    return groups
