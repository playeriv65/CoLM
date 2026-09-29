"""The selection pool: examples of one optimizer step, exchanged between ranks and re-batched."""

import torch.distributed as dist

from colm.selection.packing import META


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


def source_of(example) -> int:
    return int(example[META]["sources"]) if isinstance(example, dict) else example.source


def index_of(example) -> int:
    return int(example[META]["indices"]) if isinstance(example, dict) else example.index
