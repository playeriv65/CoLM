"""Attention over packed (padding-free) sequences: `colm_varlen`.

Packed batches hold several examples in one row, described by `cu_seq_lens_q` / `max_length_q`
(the names transformers' flash path uses); no attention mask exists, so the stock `sdpa` path
would build a dense `[T, T]` block mask and pay for every (query, key) pair of the row. This
implementation, registered through `AttentionInterface`, attends inside each sequence only:

* fp16 / bf16 on CUDA with gradients (training): `torch.nn.attention.varlen.varlen_attn` (flash);
  under `no_grad` it returned NaN on sm_120 (torch 2.13), so that case uses the kernel below;
* CUDA without gradients (the fp32 selection forward): the memory-efficient kernel that
  `F.scaled_dot_product_attention` uses, called with cumulative sequence lengths;
* elsewhere (CPU): one `scaled_dot_product_attention` call per sequence.

Inputs with a padding mask (or without `cu_seq_lens_q`) take the stock `sdpa` path.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

NAME = "colm_varlen"
_CAUSAL_FROM_TOP_LEFT = 1  # custom_mask_type of aten::_efficient_attention_forward


def _varlen_causal(q, k, v, cu_seq_lens, max_length, scale):
    """q / k / v `[T, H, D]` of the packed token axis -> `[T, H, D]`, causal inside each sequence."""
    differentiable = torch.is_grad_enabled() and q.requires_grad
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and differentiable:
        return varlen_attn(
            q,
            k,
            v,
            cu_seq_lens,
            cu_seq_lens,
            max_length,
            max_length,
            scale=scale,
            window_size=(-1, 0),
        )
    if q.is_cuda and not differentiable:
        out = torch.ops.aten._efficient_attention_forward(
            q[None],
            k[None],
            v[None],
            None,
            cu_seq_lens,
            cu_seq_lens,
            max_length,
            max_length,
            0.0,
            _CAUSAL_FROM_TOP_LEFT,
            False,
            scale=scale,
        )[0]
        return out[0]
    pieces = []
    bounds = cu_seq_lens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        qs, ks, vs = (t[start:end].transpose(0, 1)[None] for t in (q, k, v))
        pieces.append(
            F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=scale)[0].transpose(
                0, 1
            )
        )
    return torch.cat(pieces)


def varlen_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    cu_seq_lens_q=None,
    max_length_q=None,
    **kwargs,
):
    if attention_mask is not None or cu_seq_lens_q is None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )
    if dropout > 0.0 or key.shape[1] != query.shape[1]:
        raise NotImplementedError(f"{NAME}: no attention dropout and no grouped-query attention")
    batch, heads, length, head_dim = query.shape
    if batch != 1:
        raise ValueError(f"{NAME}: packed inputs are one row, got {batch}")
    # Under autocast the rotary embedding leaves q / k in fp32 and v in fp16: the kernels need one
    # dtype, and the autocast one is what F.scaled_dot_product_attention would use.
    dtype = (
        torch.get_autocast_dtype("cuda")
        if query.is_cuda and torch.is_autocast_enabled("cuda")
        else torch.promote_types(torch.promote_types(query.dtype, key.dtype), value.dtype)
    )
    # [1, H, T, D] -> [T, H, D]
    q, k, v = (t[0].transpose(0, 1).contiguous().to(dtype) for t in (query, key, value))
    out = _varlen_causal(q, k, v, cu_seq_lens_q, max_length_q, scaling)
    return out[None], None  # [1, T, H, D]


def varlen_mask(*args, attention_mask=None, **kwargs):
    """Packed inputs need no mask (the kernel reads cu_seq_lens); padded ones keep the sdpa mask."""
    return (
        None
        if attention_mask is None
        else sdpa_mask(*args, attention_mask=attention_mask, **kwargs)
    )


def register() -> str:
    AttentionInterface.register(NAME, varlen_attention_forward)
    AttentionMaskInterface.register(NAME, varlen_mask)
    return NAME
