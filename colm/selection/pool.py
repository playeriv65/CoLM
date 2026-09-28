"""The selection pool: examples of one optimizer step, exchanged between ranks and re-batched."""

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100
META = "colm_meta"  # key of the per-example bookkeeping tensors inside a batch


# ----- collectives that degrade to no-ops in a single process -----------------------------
def distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def all_gather_object(obj) -> list:
    if not distributed():
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def gather_object(obj) -> list | None:
    """Objects of every rank on rank 0 (None elsewhere)."""
    if not distributed():
        return [obj]
    out = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(obj, object_gather_list=out, dst=0)
    return out


def broadcast_object(obj):
    if not distributed():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]


# ----- examples ---------------------------------------------------------------------------
def split_examples(batch: dict, legacy: bool = False) -> list[dict]:
    """One CPU dict per example of a micro-batch, `input_ids` / `labels` / `attention_mask` as 1-D rows.

    `legacy` (upstream error, extra padding in training): rows keep the padding they got in
    their micro-batch instead of being cut to their real length.
    """
    cpu = {k: v.cpu() for k, v in batch.items() if k != META}
    meta = {k: v.cpu() for k, v in batch[META].items()}
    examples = []
    for i in range(len(cpu["input_ids"])):
        row = {k: v[i] for k, v in cpu.items()}
        if not legacy:
            length = int(row["attention_mask"].sum())
            row = {k: v[:length] for k, v in row.items()}
        row[META] = {k: v[i] for k, v in meta.items()}
        examples.append(row)
    return examples


def collate_examples(examples: list[dict], pad_token_id: int) -> dict:
    """A batch of examples, right-padded to the longest."""
    pads = {"input_ids": pad_token_id, "labels": IGNORE_INDEX, "attention_mask": 0}
    batch = {
        k: pad_sequence([e[k] for e in examples], batch_first=True, padding_value=v)
        for k, v in pads.items()
    }
    batch[META] = {k: torch.stack([e[META][k] for e in examples]) for k in examples[0][META]}
    return batch
