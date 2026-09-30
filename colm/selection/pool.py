"""The selection pool: examples of one optimizer step, exchanged between ranks and re-batched."""

import torch
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


def rank_and_world() -> tuple[int, int]:
    """(rank, world size) of this process; (0, 1) without a process group."""
    if not distributed():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def all_reduce_sum(tensor: torch.Tensor, device) -> torch.Tensor:
    """Sum of `tensor` over the ranks (a copy; the input is returned as it is in one process).

    NCCL reduces CUDA tensors only, so a CPU tensor is staged on `device` and comes back on CPU.
    """
    if not distributed():
        return tensor
    staged = tensor.to(device) if dist.get_backend() == "nccl" else tensor.clone()
    dist.all_reduce(staged)
    return staged.to(tensor.device)


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
