"""Exact variable-length attention for the packed fp32 selection forward.

transformers ships varlen kernels only through flash attention, which needs fp16/bf16 (its
integration silently casts fp32 queries), and `sdpa` on packed inputs builds a dense
`[rows, 1, T, T]` block mask, so it pays for every (query, key) pair of the row instead of
only the pairs inside each sequence. The selection forward is fp32 by upstream design, so this
module registers one more implementation, `colm_varlen`, through transformers'
`AttentionInterface`:

* inputs with a padding mask, or without `cu_seq_lens_q`, go to the stock `sdpa` path
  (identical to `attn_implementation="sdpa"`);
* packed inputs (no mask, `cu_seq_lens_q` / `max_length_q` passed as keyword arguments, the
  names transformers' flash path uses) run causal attention per sequence: on CUDA the
  memory-efficient kernel that `F.scaled_dot_product_attention` itself uses for fp32, called
  with cumulative sequence lengths; elsewhere one `scaled_dot_product_attention` call per
  sequence.

Inference only (the selection forward runs under `torch.inference_mode`). The CUDA path calls
the private op `aten::_efficient_attention_forward` (what SDPA dispatches to); after a torch
upgrade re-run `tests/test_exact_opts.py` and `scripts/check_exact_opts.py` on a GPU.
"""

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

NAME = "colm_varlen"
# custom_mask_type of aten::_efficient_attention_forward: causal, aligned top-left.
_CAUSAL_FROM_TOP_LEFT = 1


def _varlen_causal(query, key, value, cu_seq_lens, max_length, scale):
    """query/key/value [1, T, H, D] -> [1, T, H, D], causal within each cu_seq_lens segment."""
    if query.is_cuda:
        return torch.ops.aten._efficient_attention_forward(
            query,
            key,
            value,
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
    out = torch.empty_like(query)
    bounds = cu_seq_lens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        q, k, v = (t[:, start:end].transpose(1, 2) for t in (query, key, value))
        out[:, start:end] = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale
        ).transpose(1, 2)
    return out


def varlen_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    max_length_q: int | None = None,
    **kwargs,
):
    if attention_mask is not None or cu_seq_lens_q is None:
        # Padded inputs, or one unpadded sequence per row: exactly the stock sdpa path.
        # (Packed rows must pass cu_seq_lens_q; colm.train.packing always does.)
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )
    if dropout > 0.0 or (torch.is_grad_enabled() and query.requires_grad):
        raise NotImplementedError(f"{NAME} is inference-only and has no dropout")
    if key.shape[1] != query.shape[1]:
        raise NotImplementedError(f"{NAME} does not implement grouped-query attention")
    batch, heads, length, head_dim = query.shape
    # [B, H, T, D] -> one token axis [1, B*T, H, D]; cu_seq_lens indexes that axis.
    q, k, v = (
        t.transpose(1, 2).reshape(1, batch * length, heads, head_dim) for t in (query, key, value)
    )
    out = _varlen_causal(q, k, v, cu_seq_lens_q, max_length_q, scaling)
    return out.view(batch, length, heads, head_dim), None


def varlen_mask(*args, attention_mask=None, **kwargs):
    """Padded inputs keep the sdpa mask; packed inputs need none (the kernel reads cu_seq_lens)."""
    if attention_mask is None:
        return None
    return sdpa_mask(*args, attention_mask=attention_mask, **kwargs)


def register() -> str:
    AttentionInterface.register(NAME, varlen_attention_forward)
    AttentionMaskInterface.register(NAME, varlen_mask)
    return NAME
