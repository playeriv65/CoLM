"""Per-example MeZO scalars g_i under different selection precisions, on saved pools.

Arms (see `scripts/precision_arms.py`): R exact float64 reference; F fp32; P fp16 prefix + fp32
suffix; H fp16 prefix and suffix; `L` = the library extractor (`MezoEfficient`) with `--library-tail K` fp32 tail blocks (only with that flag); suffix `r` = the pool packed in reverse order (packing-noise
floor / run-to-run noise); `_e2` = epsilon 1e-2. `Rfd3` / `Rfd2` are float64 finite differences at
eps 1e-3 / 1e-2 (the bias of the estimator itself). Output: one npz with g[arm] of shape
[directions, examples], the directions z, the example bookkeeping and the environment.

    python -u scripts/measure_selection_precision.py --config configs/prefix_precision_phi2.json \
        --pool-file $ROOT/inputs/pools.pkl --adapter $ROOT/inputs/adapter_model.safetensors \
        --out-dir $COLM_ARTIFACT_ROOT/artifacts/CoLM/precision-DATE
    ... --validate          # shortcut check: R against an all-float64 finite difference
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from precision_arms import (
    all_half_extract,
    cast_state,
    double_split,
    exact_estimate,
    fd_estimate,
    prefix_state,
    to_device,
)
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from colm.data.get_training_dataset import tokenize_examples
from colm.selection.features import MezoEfficient
from colm.selection.packing import greedy_groups, pack
from colm.selection.zo import LastLayerSplit, Perturbation, zo_parameters
from colm.train.training_arguments import TrainingArguments

EPS, EPS_LARGE = 1e-3, 1e-2
MAIN_ARMS = ["R", "Rfd3", "Rfd2", "F", "F_e2", "P", "P_e2", "H", "H_e2", "Fr", "Pr", "Hr"]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pools", type=int, default=16, help="pools of 32 examples")
    parser.add_argument("--directions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=734221, help="seed of directions 1..")
    parser.add_argument("--pack-tokens", type=int, default=1536)
    parser.add_argument(
        "--library-tail",
        type=int,
        default=None,
        help="add arm L: MezoEfficient with selection_prefix_fp32_tail=K (library path)",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-examples", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_model(recipe, adapter, device):
    name = recipe["model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    )
    lora = recipe["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
        ),
    ).to(device)
    loaded = set_peft_model_state_dict(model, load_file(adapter))
    if loaded.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {loaded.unexpected_keys[:5]}")
    return model.eval()


def directions(pools, params, count, seed):
    """z of every direction: direction 0 is the recipe's ZO seed, the others fixed seeds."""
    seeds = [pools["zo_random_seed"]] + [seed + i for i in range(count - 1)]
    return seeds, [Perturbation(params, EPS, s).z() for s in seeds]


def pool_examples(tokenizer, pools, index):
    micro = TrainingArguments(output_dir="unused").pool_micro_batches
    raw = [e for m in pools["instances"][index * micro : (index + 1) * micro] for e in m]
    return tokenize_examples(tokenizer, raw)


def measure_pack(ctx, batch, reverse):
    """{arm: [directions, examples of the pack]} for one packed batch."""
    split, split64, names, params, zs, zs64 = (
        ctx[k] for k in ("split", "split64", "names", "params", "zs", "zs64")
    )
    out = {}
    s32 = prefix_state(split, batch, low=False)
    slow = prefix_state(split, batch, low=True)
    sp = cast_state(slow, torch.float32)

    def run(arm, fn):
        out[arm] = torch.stack([fn(z, z64) for z, z64 in zip(zs, zs64, strict=True)]).cpu().double()

    tag = "r" if reverse else ""
    run("F" + tag, lambda z, z64: fd_estimate(split, s32, batch, names, params, z, EPS))
    run("P" + tag, lambda z, z64: fd_estimate(split, sp, batch, names, params, z, EPS))
    run("H" + tag, lambda z, z64: fd_estimate(split, slow, batch, names, params, z, EPS, True))
    if reverse:
        return out
    if ctx.get("library"):
        out["L"] = torch.stack([e.extract(batch) for e in ctx["library"]]).cpu().double()
    run("F_e2", lambda z, z64: fd_estimate(split, s32, batch, names, params, z, EPS_LARGE))
    run("P_e2", lambda z, z64: fd_estimate(split, sp, batch, names, params, z, EPS_LARGE))
    run("H_e2", lambda z, z64: fd_estimate(split, slow, batch, names, params, z, EPS_LARGE, True))
    s64 = cast_state(s32, torch.float64)
    p64 = [p.double() for p in params]
    run("R", lambda z, z64: exact_estimate(split64, s64, batch, names, z64))
    run("Rfd3", lambda z, z64: fd_estimate(split64, s64, batch, names, p64, z64, EPS))
    run("Rfd2", lambda z, z64: fd_estimate(split64, s64, batch, names, p64, z64, EPS_LARGE))
    return out


def check_extractor(ctx, batch, seed, z):
    """The script's F / P / H against the library extractor (H: the patched-in function)."""
    model, params = ctx["model"], ctx["zo_params"]
    reference = {}
    for arm, dtype in (("F", "float32"), ("P", "float16")):
        args = SimpleNamespace(
            mezo_eps=EPS,
            selection_prefix_dtype=dtype,
            selection_prefix_fp32_tail=0,
            mezo_selection="grad",
        )
        extractor = MezoEfficient(args, model, params, seed)
        reference[arm] = extractor.extract(batch).cpu().double()
    args = SimpleNamespace(
        mezo_eps=EPS,
        selection_prefix_dtype="float16",
        selection_prefix_fp32_tail=0,
        mezo_selection="grad",
    )
    reference["H"] = (
        all_half_extract(MezoEfficient(args, model, params, seed), batch).cpu().double()
    )
    mine = measure_pack(ctx, batch, reverse=False)
    result = {}
    for arm, value in reference.items():
        scale = value.abs().median().item()
        diff = (mine[arm][0] - value).abs()
        result[arm] = {"max_abs": diff.max().item(), "median_abs_g": scale}
    return result


def main():
    args = arguments()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    recipe = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(recipe["model_name_or_path"], local_files_only=True)
    with args.pool_file.open("rb") as handle:
        pools = pickle.load(handle)
    model = load_model(recipe, args.adapter, device)
    split = LastLayerSplit(model.get_base_model())
    zo_params = zo_parameters(model, ["v_proj"], -1)
    seeds, zs = directions(pools, zo_params, args.directions, args.seed)
    library = []
    if args.library_tail is not None:
        library_args = SimpleNamespace(
            mezo_eps=EPS,
            selection_prefix_dtype="float16",
            selection_prefix_fp32_tail=args.library_tail,
            mezo_selection="grad",
        )
        library = [MezoEfficient(library_args, model, zo_params, seed) for seed in seeds]
    ctx = {
        "library": library,
        "model": model,
        "split": split,
        "split64": double_split(split),
        "zo_params": zo_params,
        "names": [n for n, _ in zo_params],
        "params": [p for _, p in zo_params],
        "zs": zs,
        "zs64": [[z.double() for z in direction] for direction in zs],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    load_start = os.getloadavg()
    print(f"directions (zo seeds) {seeds}; load {load_start}", flush=True)

    if args.validate:
        validate(args, ctx, tokenizer, pools, device, recipe, seeds)
        return

    pool_size = TrainingArguments(output_dir="unused").per_device_train_batch_size
    total = args.pools * pool_size
    arms = MAIN_ARMS + (["L"] if library else [])
    g = {arm: np.full((args.directions, total), np.nan) for arm in arms}
    meta = {"source": [], "index": [], "length": [], "num_labels": []}
    extractor_check = None
    for p in range(args.pools):
        examples = pool_examples(tokenizer, pools, p)
        assert len(examples) == pool_size, len(examples)
        for e in examples:
            meta["source"].append(e.source)
            meta["index"].append(e.index)
            meta["length"].append(len(e))
            meta["num_labels"].append(e.num_labels)
        t0 = time.time()
        for reverse in (False, True):
            seq = list(range(pool_size))[::-1] if reverse else list(range(pool_size))
            ordered = [examples[i] for i in seq]
            for group in greedy_groups([len(e) for e in ordered], args.pack_tokens):
                batch = to_device(pack([ordered[i] for i in group]), device)
                positions = [p * pool_size + seq[i] for i in group]
                if extractor_check is None and not reverse:
                    extractor_check = check_extractor(ctx, batch, seeds[0], zs[0])
                    print("extractor check", json.dumps(extractor_check), flush=True)
                for arm, value in measure_pack(ctx, batch, reverse).items():
                    g[arm][:, positions] = value.numpy()
        print(f"pool {p + 1}/{args.pools} {time.time() - t0:.1f}s", flush=True)
        np.savez(
            args.out_dir / "g.npz",
            **{f"g_{k}": v for k, v in g.items()},
            **{k: np.asarray(v) for k, v in meta.items()},
            z=np.stack([torch.cat([t.flatten() for t in z]).cpu().numpy() for z in zs]),
            seeds=np.asarray(seeds),
            pools_done=p + 1,
        )
    info = {
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "seeds": seeds,
        "eps": [EPS, EPS_LARGE],
        "tf32": False,
        "extractor_check": extractor_check,
        "wall_s": time.time() - started,
        "loadavg_start": load_start,
        "loadavg_end": os.getloadavg(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
    }
    (args.out_dir / "measure_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print("DONE " + json.dumps(info), flush=True)


def validate(args, ctx, tokenizer, pools, device, recipe, seeds):
    """R (fp32 prefix state + float64 exact suffix) against an all-float64 finite difference."""
    model64 = load_model(recipe, args.adapter, device).double()
    split_full = LastLayerSplit(model64.get_base_model())
    params64 = zo_parameters(model64, ["v_proj"], -1)
    names, p64 = [n for n, _ in params64], [p for _, p in params64]
    examples = pool_examples(tokenizer, pools, 0)[: args.validate_examples]
    batch = to_device(pack(examples), device)
    rows = []
    for seed, z in list(zip(seeds, ctx["zs64"], strict=True))[:2]:
        start = time.time()
        fast = measure_pack(ctx, batch, reverse=False)
        s_full = prefix_state(split_full, batch, low=False)
        full_fd = {
            "fd3": fd_estimate(split_full, s_full, batch, names, p64, z, EPS),
            "fd4": fd_estimate(split_full, s_full, batch, names, p64, z, 1e-4),
        }
        direction = seeds.index(seed)
        row = {"seed": seed, "wall_s": time.time() - start}
        r = fast["R"][direction].to(device)
        rfd3 = fast["Rfd3"][direction].to(device)
        for name, value in full_fd.items():
            value = value.double()
            row[f"max_rel_R_vs_full_{name}"] = ((r - value).abs() / value.abs()).max().item()
            row[f"max_rel_fastfd3_vs_full_{name}"] = (
                ((rfd3 - value).abs() / value.abs()).max().item()
            )
        row["g_median_abs"] = r.abs().median().item()
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.out_dir / "validate.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
