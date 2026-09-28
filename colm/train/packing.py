"""Padding-free (packed) inputs for the CoLM selection and training forwards.

Examples are concatenated along the time axis with restarting `position_ids` and no attention
mask, the format of transformers' `DataCollatorWithFlattening`: the label of the first token of
every example is set to -100, so no loss term crosses an example boundary. transformers only
detects packing when `past_key_values is None`: always call the model with `use_cache=False`
(Phi defaults to building a `DynamicCache`, and then sequences silently attend across
boundaries).

Rows may hold several examples. When a row is shorter than the longest one, its tail is filled
with pad tokens that form one more pseudo-sequence (positions restart at 0, labels -100), so
every kernel (dense block mask or varlen) treats it as an isolated, ignored sequence.

`cu_seq_lens` / `max_length` describe the flattened `[rows * width]` token axis (tails
included) in the layout transformers' flash-attention path takes as `cu_seq_lens_q/k` and
`max_length_q/k` keyword arguments.
"""

from dataclasses import dataclass

import torch

IGNORE_INDEX = -100


@dataclass
class Packed:
    input_ids: torch.Tensor  # [rows, width]
    position_ids: torch.Tensor  # [rows, width]
    labels: torch.Tensor  # [rows, width], -100 at every sequence start and in row tails
    segment_ids: torch.Tensor  # [rows, width], index of the example, -1 in row tails
    cu_seq_lens: torch.Tensor  # int32 [num_segments + 1] over the flattened token axis
    max_length: int
    num_real_tokens: int

    def attention_kwargs(self, device) -> dict:
        """Varlen keyword arguments (flash-attention names) for the model / decoder layers."""
        cu = self.cu_seq_lens.to(device, non_blocking=True)
        return {
            "cu_seq_lens_q": cu,
            "cu_seq_lens_k": cu,
            "max_length_q": self.max_length,
            "max_length_k": self.max_length,
        }


def real_length(attention_mask_row: torch.Tensor) -> int:
    """Number of real tokens of a right-padded row (checks that the padding is on the right)."""
    length = int(attention_mask_row.sum())
    if not bool(attention_mask_row[:length].all()):
        raise ValueError("packing expects right-padded inputs (attention mask 1...1 0...0)")
    return length


def unpad(batch: dict, row: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(input_ids, labels) of one example of a collated right-padded CPU batch."""
    length = real_length(batch["attention_mask"][row])
    return batch["input_ids"][row, :length], batch["labels"][row, :length]


def pack(
    sequences: list[tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int,
    rows: list[list[int]] | None = None,
) -> Packed:
    """Pack `(input_ids, labels)` 1-D CPU tensors into rows.

    `rows` lists the example indices of every row (default: all examples in one row).
    """
    if rows is None:
        rows = [list(range(len(sequences)))]
    widths = [sum(len(sequences[i][0]) for i in row) for row in rows]
    width = max(widths)
    shape = (len(rows), width)
    input_ids = torch.full(shape, pad_token_id, dtype=torch.long)
    position_ids = torch.zeros(shape, dtype=torch.long)
    labels = torch.full(shape, IGNORE_INDEX, dtype=torch.long)
    segment_ids = torch.full(shape, -1, dtype=torch.long)
    seg_lens = []
    for r, row in enumerate(rows):
        offset = 0
        for i in row:
            ids, lab = sequences[i]
            n = len(ids)
            if n == 0:
                raise ValueError(f"example {i} has no tokens")
            input_ids[r, offset : offset + n] = ids
            position_ids[r, offset : offset + n] = torch.arange(n)
            labels[r, offset + 1 : offset + n] = lab[1:]
            segment_ids[r, offset : offset + n] = i
            seg_lens.append(n)
            offset += n
        if offset < width:
            position_ids[r, offset:] = torch.arange(width - offset)
            seg_lens.append(width - offset)
    cu = torch.zeros(len(seg_lens) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(seg_lens, dtype=torch.int32).cumsum(0)
    return Packed(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        segment_ids=segment_ids,
        cu_seq_lens=cu,
        max_length=max(seg_lens),
        num_real_tokens=sum(len(s[0]) for s in sequences),
    )


def rows_by_token_budget(lengths: list[int], max_tokens: int) -> list[list[int]]:
    """Greedy, order-preserving split of items into rows of at most `max_tokens` tokens.

    `max_tokens <= 0` puts everything in one row; an item longer than the budget gets a row
    of its own.
    """
    if max_tokens <= 0:
        return [list(range(len(lengths)))]
    rows, current, used = [], [], 0
    for i, n in enumerate(lengths):
        if current and used + n > max_tokens:
            rows.append(current)
            current, used = [], 0
        current.append(i)
        used += n
    if current:
        rows.append(current)
    return rows


def shift_left(labels: torch.Tensor) -> torch.Tensor:
    """Targets aligned with the positions that predict them (`labels[:, 1:]`, -100 appended)."""
    pad = torch.full_like(labels[:, :1], IGNORE_INDEX)
    return torch.cat([labels[:, 1:], pad], dim=1)
