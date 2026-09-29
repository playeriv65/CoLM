"""Hybrid prefix Pk: fp16 layers 0..30-k, fp32 layers 31-k..30, fp32 perturbed last layer.

k = 0 is P (fp16 prefix), k = 31 is F (fp32 prefix). Measures, on the data of the Phase 1 run
(`measure_selection_precision.py`, whose R and F arrays are reused): g_i for every k, and the time
of the selection forward (prefix, prefix + the two suffix replays of one direction) on the
examples the trainer actually forwards (`CoresetSelector.needed`), packed in 1536-token packs,
warm, CUDA-synchronised, one job on the GPU.

    python -u scripts/measure_hybrid_prefix.py --config configs/prefix_precision_phi2.json \
        --pool-file $ROOT/inputs/pools.pkl --adapter $ROOT/inputs/adapter_model.safetensors \
        --phase1 $OUT/measure/g.npz --out-dir $OUT/hybrid
    python scripts/analyze_selection_precision.py $OUT/hybrid/g.npz --arms P0,P1,... --pairs ...
    python scripts/measure_hybrid_prefix.py --table $OUT/hybrid/analysis.json $OUT/hybrid/timing.json
"""

import argparse
import json
import os
import pickle
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from measure_selection_precision import EPS, directions, load_model, pool_examples
from precision_arms import cast_state, fd_estimate, hybrid_prefix_state, prefix_state, to_device
from transformers import AutoTokenizer

from colm.selection.packing import greedy_groups, pack
from colm.selection.select import CoresetSelector
from colm.selection.zo import LastLayerSplit, zo_parameters
from colm.train.training_arguments import TrainingArguments

P_STEP_MS = 924.0  # measured P step (`docs/selection-precision.md`, F/P runs alone on GPU 2)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pool-file", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--phase1", type=Path, help="g.npz of measure_selection_precision.py")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 31])
    parser.add_argument("--pack-tokens", type=int, default=1536)
    parser.add_argument("--timing-pools", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--table", nargs=2, metavar=("ANALYSIS", "TIMING"), type=Path)
    return parser.parse_args()


def table(analysis: Path, timing: Path):
    a, t = json.loads(analysis.read_text()), json.loads(timing.read_text())
    base = t["k"]["0"]["total_step_ms"]
    print(
        "| k | sign flips /2560 | Pearson | median rel. err | p90 rel. err | FL picks shared "
        "(chain, vs R) | FL shared with P | prefix ms/step | selection fwd ms/step | est. step ms |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, row in t["k"].items():
        g = a["g"][f"P{k}"]
        chain = a["selection"][f"P{k}_vs_R"]["chain"]["fl_overlap_fraction"]
        shared = a["selection"].get(f"P{k}_vs_P0", {}).get("chain", {}).get("fl_overlap_fraction")
        est = P_STEP_MS + row["total_step_ms"] - base
        print(
            f"| {k} | {g['sign_differs']} | {g['pearson']:.4f} | {g['rel_err_median']:.4f} | "
            f"{g['rel_err_p90']:.3f} | {chain:.3f} | "
            f"{'-' if shared is None else f'{shared:.3f}'} | {row['prefix_step_ms']:.0f} | "
            f"{row['total_step_ms']:.0f} | {est:.0f} |"
        )


def main():
    args = arguments()
    if args.table:
        return table(*args.table)
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
    names, params = [n for n, _ in zo_params], [p for _, p in zo_params]
    count = phase1["z"].shape[0]
    seeds, zs = directions(pools, zo_params, count, 734221)
    assert list(seeds) == phase1["seeds"].tolist(), "directions differ from the Phase 1 run"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(output_dir="unused")
    pool_size = training.per_device_train_batch_size
    n_pools = int(phase1["pools_done"])
    load_start = os.getloadavg()

    def estimate(state, batch):
        return torch.stack([fd_estimate(split, state, batch, names, params, z, EPS) for z in zs])

    # ---- endpoints: k = 0 must be P and k = 31 must be F, on one pack -------------------------
    first = pool_examples(tokenizer, pools, 0)
    batch = to_device(pack(first[:8]), device)
    fp32 = prefix_state(split, batch, low=False)  # verifies the split
    low = cast_state(prefix_state(split, batch, low=True), torch.float32)
    verify = {}
    for k, ref, name in ((0, low, "P"), (31, fp32, "F")):
        state = hybrid_prefix_state(split, batch, k)
        verify[f"k{k}_vs_{name}"] = {
            "max_abs_hidden_diff": (state.args[0] - ref.args[0]).abs().max().item(),
            "max_abs_g_diff": (estimate(state, batch) - estimate(ref, batch)).abs().max().item(),
        }
    print("endpoint check", json.dumps(verify), flush=True)

    # ---- g_i for every k, all pools, all directions --------------------------------------------
    total = n_pools * pool_size
    g = {f"P{k}": np.full((count, total), np.nan) for k in args.ks}
    for p in range(n_pools):
        examples = pool_examples(tokenizer, pools, p)
        for group in greedy_groups([len(e) for e in examples], args.pack_tokens):
            batch = to_device(pack([examples[i] for i in group]), device)
            positions = [p * pool_size + i for i in group]
            for k in args.ks:
                state = hybrid_prefix_state(split, batch, k)
                g[f"P{k}"][:, positions] = estimate(state, batch).cpu().double().numpy()
        print(f"pool {p + 1}/{n_pools}", flush=True)
    carry = {k: phase1[k] for k in ("source", "index", "length", "num_labels", "z", "seeds")}
    np.savez(
        args.out_dir / "g.npz",
        g_R=phase1["g_R"],
        g_F=phase1["g_F"],
        **{f"g_{k}": v for k, v in g.items()},
        **carry,
        pools_done=n_pools,
    )

    # ---- timing on the examples the trainer forwards -------------------------------------------
    selector = CoresetSelector(training, model.config.num_hidden_layers)
    step_packs = []
    for p in range(args.timing_pools):
        examples = pool_examples(tokenizer, pools, p)
        wanted = selector.needed([e.source for e in examples], pool_size // 2)
        chosen = [e for e, w in zip(examples, wanted, strict=True) if w]
        step_packs.append(
            [
                to_device(pack([chosen[i] for i in group]), device)
                for group in greedy_groups([len(e) for e in chosen], args.pack_tokens)
            ]
        )
    tokens = [sum(int(b["cu_seq_lens_q"][-1]) for b in step) for step in step_packs]

    def timed(k, batch):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = hybrid_prefix_state(split, batch, k)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fd_estimate(split, state, batch, names, params, zs[0], EPS)
        torch.cuda.synchronize()
        return (t1 - t0) * 1000, (time.perf_counter() - t0) * 1000

    timing = {}
    for k in args.ks:
        for batch in step_packs[0]:  # warm-up (kernel selection, allocator)
            timed(k, batch)
        rows = [[timed(k, b) for b in step] for step in step_packs]
        flat = [r for step in rows for r in step]
        timing[str(k)] = {
            "prefix_pack_ms_median": statistics.median(r[0] for r in flat),
            "total_pack_ms_median": statistics.median(r[1] for r in flat),
            "prefix_step_ms": statistics.mean(sum(r[0] for r in step) for step in rows),
            "total_step_ms": statistics.mean(sum(r[1] for r in step) for step in rows),
        }
        print(f"k={k} {json.dumps(timing[str(k)])}", flush=True)
    info = {
        "k": timing,
        "forwarded_tokens_per_step_mean": statistics.mean(tokens),
        "packs_per_step_mean": statistics.mean(len(s) for s in step_packs),
        "timing_pools": args.timing_pools,
        "verify": verify,
        "loadavg_start": load_start,
        "loadavg_end": os.getloadavg(),
        "device": torch.cuda.get_device_name(device),
    }
    (args.out_dir / "timing.json").write_text(json.dumps(info, indent=1) + "\n")
    print("DONE " + json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
