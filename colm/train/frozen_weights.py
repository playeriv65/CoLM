"""Frozen Linear weights stored in the autocast dtype, where nothing else needs them in fp32.

Under autocast a Linear casts its weight to the low-precision dtype at every forward. A frozen
weight is not one of the tensors autocast caches (only leaves that require grad are), so the whole
base model is cast again for every pack of the step, and the backward keeps the casts alive: for
Phi-2 15 GB of memory traffic per forward (~9 ms) and 5.6 GB of saved copies, 8 forwards per step.
Storing the weight once in the low dtype gives the same numbers, because autocast's cast is this
very cast (round to nearest), and removes both. Layers that run in fp32 (the fp32 tail and the
perturbed last layer of the selection), the head, the norms, the embeddings and every trainable
tensor keep their dtype.
"""

import torch
from torch import nn


def store_frozen_linears(model: nn.Module, dtype: torch.dtype, keep_last: int) -> int:
    """Convert the frozen fp32 `nn.Linear` modules of all but the last `keep_last` decoder layers
    to `dtype` (weight and bias); returns the bytes of memory that this frees."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = causal_lm.base_model.layers
    if not 0 <= keep_last <= len(layers):
        raise ValueError(f"keep_last must be in [0, {len(layers)}], got {keep_last}")
    freed = 0
    for layer in layers[: len(layers) - keep_last]:
        for module in layer.modules():
            if not isinstance(module, nn.Linear) or module.weight.dtype != torch.float32:
                continue
            if any(p.requires_grad for p in module.parameters()):
                continue  # LoRA A / B, or a layer being trained
            freed += sum(
                p.numel() * (p.element_size() - dtype.itemsize) for p in module.parameters()
            )
            module.to(dtype)
    return freed


def fp32_layers_needed(args) -> int | None:
    """How many trailing decoder layers must keep fp32 weights; None: convert nothing.

    The plain trainer runs everything under autocast. The efficient coreset trainer runs the last
    layer (perturbed) and its fp32 tail in fp32 during the selection, and the prefix under a
    float16 autocast, which only matches weights stored in float16. A selection prefix in fp32, or
    an extractor that runs the whole model in fp32, needs every weight in fp32.
    """
    if not args.frozen_base_low_precision or not (args.fp16 or args.bf16):
        return None
    if not args.coreset:
        return 0
    if args.efficient_mezo and args.selection_prefix_dtype == "float16" and args.fp16:
        return 1 + args.selection_prefix_fp32_tail
    return None
