"""Which operation of the attention branch of blocks 29 / 30 makes fp16 costly? (Phi-2)

Diagnostic (kept because it documents a measurement, not part of the library). Result: docs/selection-precision.md.

`measure_layer_sensitivity.py` found that fp16 in the attention branch of blocks 29 and 30 alone
produces the fp16 error of g_i, the MLP branches and blocks 0-25 do not. This script keeps the
whole prefix in fp32 and emulates fp16 storage at ONE place of ONE block (values rounded to fp16
and cast back, the arithmetic stays fp32):

    qk / v / dense   outputs of q_proj + k_proj, v_proj, the output projection
    ln_out           the shared LayerNorm output (input of both branches)
    sdpa             q, k, v of the attention kernel (and its output) in fp16
    logits           statistics only: max |q k^T / sqrt(d)| of every block (fp32 run)

    python -u scripts/diagnostics/measure_attention_ops.py ... (arguments of measure_layer_sensitivity.py)
"""

import json
from contextlib import contextmanager

import torch
from measure_layer_sensitivity import arguments, block_stats, build_runner, precision_plan
from precision_arms import cast_state

from colm.selection.packing import model_inputs


def half_round(x):
    return x.half().float() if x.is_floating_point() else x


@contextmanager
def emulate(split, block, op):
    """fp16 rounding of `op` in `block` during a fp32 prefix."""
    layer, handles = split.layers[block], []
    attn = layer.self_attn
    state = {"active": False}
    if op == "qk":
        modules = [attn.q_proj, attn.k_proj]
    elif op == "v":
        modules = [attn.v_proj]
    elif op == "dense":
        modules = [attn.dense]
    elif op == "ln_out":
        modules = [layer.input_layernorm]
    else:
        modules = []
    for module in modules:
        handles.append(module.register_forward_hook(lambda m, a, o: half_round(o)))
    original = torch.nn.functional.scaled_dot_product_attention
    if op == "sdpa":
        handles.append(attn.register_forward_pre_hook(lambda m, a: state.update(active=True)))
        handles.append(attn.register_forward_hook(lambda m, a, o: state.update(active=False)))

        def patched(q, k, v, *args, **kwargs):
            if not state["active"]:
                return original(q, k, v, *args, **kwargs)
            return original(q.half(), k.half(), v.half(), *args, **kwargs).float()

        torch.nn.functional.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
        for handle in handles:
            handle.remove()


def state_emulated(split, batch, block, op):
    with emulate(split, block, op):
        with precision_plan(split, set(), set(), batch["input_ids"].device.type):
            state = split.prefix(**model_inputs(batch))
    return cast_state(state, torch.float32)


def logit_stats(split, batch):
    """max |logit| and the mean of the per-row maximum, per block (first pack, fp32)."""
    rows, current = {}, {"block": None}
    original = torch.nn.functional.scaled_dot_product_attention

    def patched(q, k, v, attn_mask=None, *args, scale=None, **kwargs):
        scale = scale if scale is not None else q.shape[-1] ** -0.5
        logits = (q.float() @ k.float().transpose(-1, -2)) * scale
        if attn_mask is None and kwargs.get("is_causal"):
            n = logits.shape[-1]
            attn_mask = torch.ones(n, n, dtype=torch.bool, device=logits.device).tril()
        if attn_mask is not None:
            logits = (
                logits.masked_fill(~attn_mask.bool(), float("nan"))
                if (attn_mask.dtype == torch.bool)
                else logits + attn_mask
            )
        finite = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        row_max = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e30)).amax(
            -1
        )
        rounded = (q.half().float() @ k.half().float().transpose(-1, -2)) * scale
        valid = torch.isfinite(logits)
        delta = torch.where(valid, rounded - logits, torch.zeros_like(logits))
        weights = torch.softmax(torch.where(valid, logits, torch.full_like(logits, -1e30)), -1)
        weights16 = torch.softmax(torch.where(valid, rounded, torch.full_like(logits, -1e30)), -1)
        rows[current["block"]] = {
            "max_abs_logit_error_fp16_qk": delta.abs().max().item(),
            "rms_logit_error_fp16_qk": (delta.pow(2).sum() / valid.sum()).sqrt().item(),
            "mean_attention_tv_change": (0.5 * (weights - weights16).abs().sum(-1)).mean().item(),
            "max_abs_logit": finite.abs().max().item(),
            "mean_row_max": row_max[row_max > -1e29].mean().item(),
            "q_max_abs": q.abs().max().item(),
            "k_max_abs": k.abs().max().item(),
        }
        return original(q, k, v, attn_mask, *args, scale=scale, **kwargs)

    handles = []
    for i in range(len(split.layers) - 1):
        attn = split.layers[i].self_attn
        handles.append(attn.register_forward_pre_hook(lambda m, a, i=i: current.update(block=i)))
    torch.nn.functional.scaled_dot_product_attention = patched
    try:
        with precision_plan(split, set(), set(), batch["input_ids"].device.type):
            split.prefix(**model_inputs(batch))
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
        for handle in handles:
            handle.remove()
    return rows


def determinism(split, batch):
    """Two identical fp32 prefix forwards: per block, how far apart are the residual streams?"""
    _, first = block_stats(split, batch, low_all=False)
    _, second = block_stats(split, batch, low_all=False)
    return {
        i: {
            "max_abs_diff": (first[i] - second[i]).abs().max().item(),
            "rel_l2": ((first[i] - second[i]).norm() / first[i].norm()).item(),
        }
        for i in first
    }


def main():
    args = arguments()
    split, runner, _ = build_runner(args)
    out = {"ops": {}}
    for block in () if args.logits_only else (30, 29, 28, 20):
        for op in ("qk", "v", "dense", "ln_out", "sdpa"):
            score = runner.score(lambda b, block=block, op=op: state_emulated(split, b, block, op))
            out["ops"][f"{block}_{op}"] = score
            print(f"block {block} {op} {json.dumps(score)}", flush=True)
    out["logits"] = logit_stats(split, runner.packs[0])
    out["determinism"] = determinism(split, runner.packs[0])
    print(json.dumps(out["determinism"]), flush=True)
    out["logits_repeat"] = logit_stats(split, runner.packs[0])  # fp32 forward, same pack
    print(json.dumps(out["logits"]), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / ("logits.json" if args.logits_only else "attention_ops.json")).write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
