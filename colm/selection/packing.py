"""Padding-free (packed) batches: several examples concatenated in one row.

The layout is the one of transformers' `DataCollatorWithFlattening(return_flash_attn_kwargs=True)`:
concatenated `input_ids`, `position_ids` restarting at every example, the cumulative lengths of the
flash kernels (`cu_seq_lens_*`, `max_length_*`), no attention mask, and the label of the first
token of every example is -100, so no loss term crosses an example boundary. The stock attention
implementations (sdpa / flex block masks, flash varlen) keep the sequences apart from these; the
models must be called with `use_cache=False` (with a `DynamicCache` transformers no longer detects
the packing and the sequences attend to each other).
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
    """One row holding `examples`: the batch dict the models and losses take (CPU tensors).

    Built with numpy only: the many tiny tensor operations of a torch version ran on the
    intra-op thread pool, whose barriers stretch to tens of milliseconds on a loaded host
    (p99.9 of 23 ms against 1.1 ms, `docs/system-audit.md`).
    """
    lengths = np.array([len(e) for e in examples], dtype=np.int64)
    cu = np.zeros(len(examples) + 1, dtype=np.int32)
    np.cumsum(lengths, out=cu[1:])
    labels = np.concatenate([e.labels for e in examples])
    starts = cu[:-1].astype(np.int64)
    labels[starts] = IGNORE_INDEX
    total = int(cu[-1])
    position_ids = np.arange(total) - np.repeat(starts, lengths)
    # Where the row predicts a label: computed here on the CPU, so that the losses need no
    # device-to-host synchronisation (`nonzero`, `bincount`) to find them.
    targets = np.append(labels[1:], IGNORE_INDEX)
    positions = np.flatnonzero(targets != IGNORE_INDEX)
    segment = np.searchsorted(cu[1:], positions, side="right")  # torch.bucketize(right=True)
    tensor = torch.from_numpy
    return {
        "input_ids": tensor(np.concatenate([e.input_ids for e in examples]))[None],
        "position_ids": tensor(position_ids)[None],
        "labels": tensor(labels)[None],
        "cu_seq_lens_q": tensor(cu),
        "cu_seq_lens_k": tensor(cu),
        "max_length_q": int(lengths.max()),
        "max_length_k": int(lengths.max()),
        META: {
            "sources": torch.tensor([e.source for e in examples]),
            "indices": torch.tensor([e.index for e in examples]),
            "completion_lengths": torch.tensor([e.completion_length for e in examples]),
            "label_positions": tensor(positions),
            "label_targets": tensor(targets[positions]),
            "label_segment": tensor(segment),
            "label_counts": tensor(np.bincount(segment, minlength=len(examples))),
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
    meta = batch[META]
    return meta["label_positions"], meta["label_targets"], meta["label_segment"]


def label_counts(batch: dict) -> torch.Tensor:
    """Label tokens of every example of the row."""
    return batch[META]["label_counts"]


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


def balanced_shares(lengths: list[int], wanted: np.ndarray, parts: int) -> list[list[int]]:
    """Split the `wanted` items over `parts` workers with (nearly) equal token counts.

    Longest first, each to the least loaded worker (ties: the lowest rank); every worker's items
    keep their original order. Deterministic, so all ranks compute the same split.
    """
    loads = [0] * parts
    shares: list[list[int]] = [[] for _ in range(parts)]
    for i in sorted(np.flatnonzero(wanted).tolist(), key=lambda i: (-lengths[i], i)):
        worker = loads.index(min(loads))
        shares[worker].append(i)
        loads[worker] += lengths[i]
    return [sorted(share) for share in shares]
