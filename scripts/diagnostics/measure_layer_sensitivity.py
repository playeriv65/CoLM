"""Why do the last prefix blocks dominate the fp16 error of g_i? (Phi-2, `docs/selection-precision.md`)

Four measurements on the Phase 1 data (adapter, pools, directions; the exact reference R comes
from the Phase 1 npz):

1. `single_fp16`: fp32 prefix except ONE block j under fp16 autocast; `single_fp32`: fp16 prefix
   except ONE block j in fp32; `window3_fp32`: fp16 prefix with 3 consecutive fp32 blocks.
   g_i error against R per configuration (sign flips, median / p90 relative error).
2. `branch`: fp32 prefix except the attention branch OR the MLP branch of block j under fp16.
3. `stats`: per block, fp32 run: residual-stream norm / max / median |element| and the same for the
   attention and MLP branch outputs; the P run: dtypes of the block and branch outputs and the
   relative error of the residual stream against the fp32 run.
4. `noise`: g_i error when the fp32 hidden state at the input of block m (31, 30, 29) gets a
   relative Gaussian perturbation (1e-3, elementwise and per-token-norm scaled).

    python -u scripts/measure_layer_sensitivity.py --config ... --pool-file ... --adapter ... \
        --phase1 $OUT/measure/g.npz --out-dir $OUT/sensitivity
"""

import argparse
import json
import os
import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from measure_selection_precision import EPS, directions, load_model, pool_examples
from precision_arms import LOW_DTYPE, cast_state, fd_estimate, to_device
from transformers import AutoTokenizer

from colm.selection.packing import greedy_groups, model_inputs, pack
from colm.selection.zo import LastLayerSplit, zo_parameters
from colm.train.training_arguments import TrainingArguments


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pools", type=int, default=8)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--pack-tokens", type=int, default=1536)
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--logits-only", action="store_true", help="measure_attention_ops.py")
    return parser.parse_args()


@contextmanager
def precision_plan(split, low_layers, low_branches, device_type):
    """Run the prefix with autocast fp16 exactly on `low_layers` (whole blocks) and on the
    `(block, "attn"|"mlp")` branches of `low_branches`; everything else fp32."""
    handles, stack = [], {}

    def hooks(module, key, low):
        def pre(m, args, kwargs):
            context = torch.autocast(device_type, dtype=LOW_DTYPE, enabled=low)
            context.__enter__()
            stack[key] = context

        def post(m, args, kwargs, output):
            stack.pop(key).__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(module.register_forward_hook(post, with_kwargs=True))

    for i in range(len(split.layers) - 1):
        hooks(split.layers[i], (i, "layer"), i in low_layers)
    for i, part in low_branches:
        hooks(getattr(split.layers[i], "self_attn" if part == "attn" else "mlp"), (i, part), True)
    try:
        with torch.no_grad(), torch.autocast(device_type, enabled=False):
            yield
    finally:
        for handle in handles:
            handle.remove()
        for context in stack.values():
            context.__exit__(None, None, None)


def state_with(split, batch, low_layers=(), low_branches=()):
    with precision_plan(split, set(low_layers), set(low_branches), batch["input_ids"].device.type):
        state = split.prefix(**model_inputs(batch))
    return cast_state(state, torch.float32)


def errors(g, ref):
    rel = np.abs(g - ref) / np.abs(ref)
    return {
        "sign_flips": int(np.sum(np.sign(g) != np.sign(ref))),
        "median_rel": float(np.median(rel)),
        "p90_rel": float(np.percentile(rel, 90)),
        "rel_l2": float(np.linalg.norm(g - ref) / np.linalg.norm(ref)),
        "n": int(g.size),
    }


class Runner:
    def __init__(self, ctx, packs, reference, positions):
        self.ctx, self.packs, self.reference, self.positions = ctx, packs, reference, positions

    def g(self, make_state):
        """g_i of every example (directions x examples) for the prefix `make_state(batch)`."""
        c = self.ctx
        out = np.full(self.reference.shape, np.nan)
        for batch, position in zip(self.packs, self.positions, strict=True):
            state = make_state(batch)
            values = torch.stack(
                [
                    fd_estimate(c["split"], state, batch, c["names"], c["params"], z, EPS)
                    for z in c["zs"]
                ]
            )
            out[:, position] = values.cpu().double().numpy()
        return out

    def score(self, make_state):
        return errors(self.g(make_state), self.reference)


def block_stats(split, batch, low_all):
    """Per block (0..30): stats of the residual stream and branches, output dtypes."""
    rows, handles = {}, []

    def stat(x, first):
        x = x[0].detach().float()
        keep = torch.ones(len(x), dtype=torch.bool, device=x.device)
        keep[first] = False
        token = x.norm(dim=-1)
        absx = x.abs()
        return {
            "token_norm_mean": token[keep].mean().item(),
            "first_token_norm_mean": token[~keep].mean().item(),
            "max_abs": absx.max().item(),
            "median_abs": absx.median().item(),
            "rms": x.pow(2).mean().sqrt().item(),
        }

    first = (batch["cu_seq_lens_q"][:-1]).long()

    def add(i, key, out):
        tensor = out[0] if isinstance(out, tuple) else out
        rows.setdefault(i, {})[key] = {**stat(tensor, first), "dtype": str(tensor.dtype)}

    for i in range(len(split.layers) - 1):
        layer = split.layers[i]
        handles.append(layer.register_forward_hook(lambda m, a, o, i=i: add(i, "block", o)))
        handles.append(
            layer.self_attn.register_forward_hook(lambda m, a, o, i=i: add(i, "attn", o))
        )
        handles.append(layer.mlp.register_forward_hook(lambda m, a, o, i=i: add(i, "mlp", o)))
    outputs = {}
    for i in range(len(split.layers) - 1):
        handles.append(
            split.layers[i].register_forward_hook(
                lambda m, a, o, i=i: outputs.__setitem__(i, (o[0] if isinstance(o, tuple) else o))
            )
        )
    low = set(range(len(split.layers) - 1)) if low_all else set()
    try:
        with precision_plan(split, low, set(), batch["input_ids"].device.type):
            split.prefix(**model_inputs(batch))
    finally:
        for handle in handles:
            handle.remove()
    return rows, {i: v.detach().float().clone() for i, v in outputs.items()}


def noise_state(split, batch, layer, sigma, kind, seed):
    """fp32 prefix with a relative Gaussian perturbation of the input of block `layer`."""
    generator = torch.Generator(batch["input_ids"].device).manual_seed(seed)
    last = len(split.layers) - 1

    def perturb(h):
        n = torch.randn(h.shape, generator=generator, device=h.device, dtype=h.dtype)
        if kind == "elementwise":
            return h * (1 + sigma * n)
        scale = h.norm(dim=-1, keepdim=True) / h.shape[-1] ** 0.5
        return h + sigma * scale * n

    if layer == last:  # the input of the perturbed block is the prefix state itself
        state = state_with(split, batch)
        args = (perturb(state.args[0]), *state.args[1:]) if state.args else state.args
        if state.args:
            state.args = args
        else:
            state.kwargs["hidden_states"] = perturb(state.kwargs["hidden_states"])
        return state

    def pre(module, args, kwargs):
        if args:
            return (perturb(args[0]), *args[1:]), kwargs
        kwargs["hidden_states"] = perturb(kwargs["hidden_states"])
        return args, kwargs

    handle = split.layers[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        return state_with(split, batch)
    finally:
        handle.remove()


def build_runner(args):
    """(split, runner, seeds): the model, the packs of `args.pools` pools and the exact reference."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    recipe = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(recipe["model_name_or_path"], local_files_only=True)
    with args.pool_file.open("rb") as handle:
        pools = pickle.load(handle)
    phase1 = np.load(args.phase1)
    model = load_model(recipe, args.adapter, device)
    split = LastLayerSplit(model.get_base_model())
    zo_params = zo_parameters(model, ["v_proj"], -1)
    seeds, zs = directions(pools, zo_params, phase1["z"].shape[0], 734221)
    keep = args.directions
    pool_size = TrainingArguments(output_dir="unused").per_device_train_batch_size
    total = args.pools * pool_size
    reference = phase1["g_R"][:keep, :total]
    packs, positions = [], []
    for p in range(args.pools):
        examples = pool_examples(tokenizer, pools, p)
        for group in greedy_groups([len(e) for e in examples], args.pack_tokens):
            packs.append(to_device(pack([examples[i] for i in group]), device))
            positions.append([p * pool_size + i for i in group])
    ctx = {
        "split": split,
        "names": [n for n, _ in zo_params],
        "params": [p for _, p in zo_params],
        "zs": zs[:keep],
    }
    runner = Runner(ctx, packs, reference, positions)
    return split, runner, seeds


def main():
    args = arguments()
    split, runner, seeds = build_runner(args)
    packs, keep = runner.packs, args.directions
    blocks = len(split.layers) - 1
    out = {"blocks": blocks, "pools": args.pools, "directions": keep, "seeds": seeds[:keep]}
    split.prefix(**model_inputs(packs[0]))  # verifies the split in fp32 once

    print("reference checks", flush=True)
    out["fp32_all"] = runner.score(lambda b: state_with(split, b))
    out["fp16_all"] = runner.score(lambda b: state_with(split, b, range(blocks)))
    print(json.dumps({k: out[k] for k in ("fp32_all", "fp16_all")}), flush=True)

    out["single_fp16"], out["single_fp32"], out["window3_fp32"] = {}, {}, {}
    for j in range(blocks):
        out["single_fp16"][j] = runner.score(lambda b, j=j: state_with(split, b, [j]))
        out["single_fp32"][j] = runner.score(
            lambda b, j=j: state_with(split, b, [i for i in range(blocks) if i != j])
        )
        print(f"block {j} fp16-only {out['single_fp16'][j]} fp32-only {out['single_fp32'][j]}",
              flush=True)  # fmt: skip
    for j in (0, 5, 10, 15, 20, 25, 26, 27, 28):
        window = {j, j + 1, j + 2}
        out["window3_fp32"][j] = runner.score(
            lambda b, window=window: state_with(
                split, b, [i for i in range(blocks) if i not in window]
            )
        )
        print(f"window {j}-{j + 2} fp32 {out['window3_fp32'][j]}", flush=True)

    out["branch"] = {}
    for j in (0, 10, 20, 27, 28, 29, 30):
        out["branch"][j] = {
            part: runner.score(lambda b, j=j, part=part: state_with(split, b, (), [(j, part)]))
            for part in ("attn", "mlp")
        }
        print(f"branch block {j} {out['branch'][j]}", flush=True)

    print("stats", flush=True)
    accumulated = {}
    errors_rel = {i: [] for i in range(blocks)}
    for batch in packs[: args.pools]:
        rows32, outs32 = block_stats(split, batch, low_all=False)
        rows16, outs16 = block_stats(split, batch, low_all=True)
        for i in range(blocks):
            accumulated.setdefault(i, []).append(
                {"fp32": rows32[i], "fp16": {k: v["dtype"] for k, v in rows16[i].items()}}
            )
            diff = (outs16[i] - outs32[i])[0]
            reference_h = outs32[i][0]
            energy = diff.pow(2).sum(dim=0)
            top = torch.topk(reference_h.pow(2).mean(dim=0), 10).indices
            errors_rel[i].append(
                {
                    "rel_l2": (diff.norm() / reference_h.norm()).item(),
                    "top10_dim_error_share": (energy[top].sum() / energy.sum()).item(),
                    "max_abs_diff": diff.abs().max().item(),
                }
            )
    out["stats"] = {}
    for i in range(blocks):

        def mean(key, part, i=i):
            return float(np.mean([r["fp32"][part][key] for r in accumulated[i]]))

        out["stats"][i] = {
            part: {
                key: mean(key, part)
                for key in (
                    "token_norm_mean",
                    "first_token_norm_mean",
                    "max_abs",
                    "median_abs",
                    "rms",
                )
            }
            for part in ("block", "attn", "mlp")
        }
        out["stats"][i]["dtypes_fp16_run"] = accumulated[i][0]["fp16"]
        out["stats"][i]["state_error_fp16_vs_fp32"] = {
            k: float(np.mean([e[k] for e in errors_rel[i]])) for k in errors_rel[i][0]
        }
    print(json.dumps(out["stats"][30]), flush=True)

    print("noise", flush=True)
    out["noise"] = {}
    for kind in ("elementwise", "token_scaled"):
        for layer in (31, 30, 29):
            score = runner.score(
                lambda b, layer=layer, kind=kind: noise_state(
                    split, b, layer, args.noise, kind, seed=17
                )
            )
            score["amplification_median"] = score["median_rel"] / args.noise
            out["noise"][f"{kind}_block{layer}"] = score
            print(f"noise {kind} block {layer} {score}", flush=True)
    out["loadavg"] = os.getloadavg()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sensitivity.json").write_text(json.dumps(out, indent=1) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
