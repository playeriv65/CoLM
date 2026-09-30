"""fp32 tail of the TRAINING forward: the last blocks run q_proj / k_proj and the attention in fp32.

Under fp16 autocast the gradient of the LoRA weights is far from the fp32 one (Phi-2: relative L2
error 1.0, cosine 0.65). The error comes from the last three blocks and needs BOTH halves of the
attention fixed: autocast rounds q and k to fp16 inside q_proj / k_proj, so a precise attention alone
leaves 0.6, and fp32 projections plus an fp32 attention in blocks 29-31 leave 0.04 (cosine 0.999;
`docs/training-precision.md`).

`TrainingPrecision` does that for the last `tail` blocks of one model, and only while `running()`
is active (the forward + backward of a training step; evaluation and the selection forward never
see it):

* the attention function of the training implementation is replaced by a dispatcher registered in
  transformers' public `AttentionInterface` under the same key, so `set_attn_implementation`
  switches between the training kernel and `selection_attn_implementation` as before. For a block
  of the tail (`module.layer_idx`) it casts q, k, v to fp32 and calls `sdpa` on the MATH backend
  with the block-diagonal causal mask of the packed row (`cu_seq_lens_q`); every other call, and
  any call without `cu_seq_lens_q` (padded batches), goes to the wrapped function unchanged;
* the `forward` of q_proj and k_proj (with their LoRA branch) of those blocks runs on fp32 inputs
  with autocast off. Their frozen weights must be fp32 (`frozen_weights.store_frozen_linears` keeps
  them).

`installed()` puts both in place and removes them again (also when the training fails).
"""

import contextlib
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

logger = logging.getLogger(__name__)

PROJECTIONS = ("q_proj", "k_proj")
UNSUPPORTED_KWARGS = ("sliding_window", "softcap", "s_aux")  # the fp32 path is plain causal


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The decoder blocks of a causal LM, with or without a PEFT wrapper."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    return causal_lm.base_model.layers


def check_tail(model: nn.Module, tail: int) -> None:
    """Fail fast: `tail` must fit in the model and the q / k weights of the tail must be fp32."""
    layers = decoder_layers(model)
    if not 0 <= tail <= len(layers):
        raise ValueError(f"train_fp32_tail must be in [0, {len(layers)}] layers, got {tail}")
    for index in range(len(layers) - tail, len(layers)):
        for name in PROJECTIONS:
            for param_name, param in getattr(layers[index].self_attn, name).named_parameters():
                if not param.requires_grad and param.dtype.itemsize < 4:
                    raise ValueError(
                        f"train_fp32_tail={tail} needs fp32 weights for q_proj / k_proj of the "
                        f"last {tail} layers, but layers.{index}.self_attn.{name}.{param_name} is "
                        f"{param.dtype}: use fp32 model weights (torch_dtype none / float32) "
                        "under fp16 autocast, or set train_fp32_tail=0"
                    )


def block_mask(cu_seq_lens: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    """[1, 1, L, L] boolean mask: causal inside every packed example, nothing across them."""
    ids = torch.bucketize(
        torch.arange(length, device=device), cu_seq_lens[1:].to(device).long(), right=True
    )
    causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    return ((ids[:, None] == ids[None, :]) & causal)[None, None]


def at_least_fp32(x: torch.Tensor) -> torch.Tensor:
    """`x` in fp32 (float64, which the tests use, stays)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def fp32_attention(query, key, value, mask, scaling):
    """Exact fp32 attention (sdpa MATH backend, autocast off): [B, H, L, D] in, [B, L, H, D] out."""
    with torch.autocast(query.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            at_least_fp32(query),
            at_least_fp32(key),
            at_least_fp32(value),
            attn_mask=mask,
            scale=scaling,
            enable_gqa=query.shape[1] != key.shape[1],
        )
    return out.transpose(1, 2).contiguous()


class TrainingPrecision:
    """The fp32 tail of one model's training forward (see the module docstring)."""

    def __init__(self, model: nn.Module, tail: int):
        check_tail(model, tail)
        self.tail = tail
        layers = decoder_layers(model)
        self.first = len(layers) - tail
        self.attention = [layer.self_attn for layer in layers[self.first :]]
        self.projections = [getattr(a, name) for a in self.attention for name in PROJECTIONS]
        self._active = False
        self._mask: tuple[torch.Tensor, torch.Tensor] | None = None

    @contextlib.contextmanager
    def running(self):
        """The forward + backward of a training step: the tail is on inside, off outside.
        (With `tail == 0` nothing changes.) Not reentrant."""
        if not self.tail:
            yield
            return
        self._active = True
        try:
            yield
        finally:
            self._active = False
            self._mask = None

    @contextlib.contextmanager
    def installed(self, implementation: str):
        """Register the dispatcher for attention implementation `implementation` and wrap the projections;
        both are undone on exit."""
        if not self.tail:
            yield
            return
        if implementation not in ALL_ATTENTION_FUNCTIONS:
            raise ValueError(
                f"train_fp32_tail needs a registered attention function to wrap, got '{implementation}': "
                "use flash_attention_2 or sdpa for the training forward"
            )
        wrapped = ALL_ATTENTION_FUNCTIONS[implementation]
        own = {id(module) for module in self.attention}

        def dispatch(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
            cu = kw.get("cu_seq_lens_q")
            if not self._active or id(module) not in own or cu is None:
                return wrapped(module, query, key, value, attention_mask, dropout, scaling, **kw)
            unsupported = [k for k in UNSUPPORTED_KWARGS if kw.get(k) is not None]
            if unsupported or query.shape[0] != 1:
                raise NotImplementedError(
                    f"the fp32 attention of train_fp32_tail is causal attention on one packed "
                    f"row; got {unsupported or 'a batch of ' + str(query.shape[0])}"
                )
            if self._mask is None or self._mask[0] is not cu:
                self._mask = (cu, block_mask(cu, query.shape[2], query.device))
            return fp32_attention(query, key, value, self._mask[1], scaling), None

        originals = [(m, m.__dict__.get("forward")) for m in self.projections]
        for module in self.projections:
            module.forward = self._fp32_forward(module, module.forward)
        AttentionInterface.register(implementation, dispatch)
        logger.info(
            f"Training forward: q/k projections and attention of the last {self.tail} blocks "
            f"in fp32 (dispatcher on attention implementation '{implementation}')"
        )
        try:
            yield
        finally:
            AttentionInterface.register(implementation, wrapped)
            for module, forward in originals:
                if forward is None:
                    del module.forward
                else:
                    module.forward = forward

    def _fp32_forward(self, module, forward):
        def run(x, *args, **kwargs):
            if not self._active:
                return forward(x, *args, **kwargs)
            with torch.autocast(x.device.type, enabled=False):
                return forward(at_least_fp32(x), *args, **kwargs)

        return run
