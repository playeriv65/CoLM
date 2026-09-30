"""True per-example gradients of the MeZO parameter (last-layer v_proj LoRA-B) on real pools.

Diagnostic for docs/paper-vs-code.md (T1/T2/T3), not part of the library. GPU part of the audit; the
analysis of the saved gradients is `analyze_z_geometry.py` (CPU only).

For every example of the pools it stores, in fp32,

* `g_i = dL_i/dB` (327,680 values for Phi-2 r=128), by autograd through the last decoder layer,
  final norm, LM head and the own-length token-mean loss (the loss the library estimates), on the
  prefix state of the library (`selection_prefix_dtype=float16`, `selection_prefix_fp32_tail=2`,
  promoted to fp32 before the perturbed suffix), attention backward with the exact MATH kernel
  (the fp32 memory-efficient sdpa backward is ~0.3 off on this GPU, docs/errors.md);
* `s_i = g_i . z` for the recipe's fixed direction z (the ZO seed of the pool file), i.e. the
  limit eps -> 0 of the estimate the code computes, and the library's own finite-difference
  estimate `g_code` (`MezoEfficient.extract`, eps 1e-3) with the pool packed in order
  (`g_code`) and in reverse order (`g_code_rev`, the packing-noise floor of the fixed-z selection);
* the loss, the gradient norm, source, token length and label count.

Inputs: the 16 pools x 32 examples of the fixed pool file (`pools.pkl`; in a 4-GPU run four such
pools form one selection pool of 128) and `--extra-pools` more pools of 128 sampled uniformly
from MathInstruct (seeded), for the many-step tests. Output (out-dir): `G_pools.npy`,
`G_extra.npy` (float32 [n, 327680], pool order), `meta_pools.npz`, `meta_extra.npz`, `z0.npy` (the recipe's fixed direction), `info.json`.

    python -u scripts/diagnostics/measure_true_gradients.py \
        --config configs/diagnostics/prefix_precision_phi2.json \
        --pool-file $ROOT/inputs/pools.pkl --adapter $ROOT/inputs/adapter_model.safetensors \
        --out-dir $COLM_ARTIFACT_ROOT/artifacts/CoLM/paper-review-DATE --extra-pools 24 --device cuda:0

Result: docs/paper-vs-code.md.
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
from measure_selection_precision import EPS, load_model, pool_examples
from precision_arms import cast_state, to_device
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer

from colm.data.get_training_dataset import get_training_dataset, tokenize_examples
from colm.selection.features import MezoEfficient, example_means
from colm.selection.packing import greedy_groups, label_counts, label_positions, model_inputs, pack
from colm.selection.zo import LastLayerSplit, zo_parameters

POOL = 32  # examples of one rank's pool (4 x 8)
BIG_POOL = 128  # a 4-GPU selection pool


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-file", default="data/MathInstruct.jsonl")
    parser.add_argument("--token-cache-dir", default="cache/tokens")
    parser.add_argument(
        "--extra-pools", type=int, default=24, help="pools of 128 sampled at random"
    )
    parser.add_argument("--extra-seed", type=int, default=20260929)
    parser.add_argument("--pack-tokens", type=int, default=1536)
    parser.add_argument("--tail", type=int, default=2, help="fp32 tail blocks of the fp16 prefix")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def library_extractor(model, zo_params, seed, tail):
    args = SimpleNamespace(
        mezo_eps=EPS,
        selection_prefix_dtype="float16",
        selection_prefix_fp32_tail=tail,
        mezo_selection="grad",
    )
    return MezoEfficient(args, model, zo_params, seed)


def prefix_fp32(split, batch, tail):
    """Library prefix (fp16 autocast, `tail` fp32 blocks), promoted to fp32, with autograd allowed."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        state = split.prefix(tail=tail, device_type="cuda", **model_inputs(batch))
    return cast_state(state, torch.float32)


def true_gradients(split, state, batch, name, base):
    """(gradients [n, numel], losses [n]) of the examples of one pack w.r.t. the parameter `name`."""
    positions, targets, segment = label_positions(batch)
    counts = label_counts(batch)
    leaf = base.detach().clone().requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        hidden = split.hidden(state, {name: leaf})[0, positions]
        logits = split.head(hidden)
        losses = example_means(logits, targets, segment, counts)
        grads = []
        for i in range(len(counts)):
            (grad,) = torch.autograd.grad(losses[i], leaf, retain_graph=i < len(counts) - 1)
            grads.append(grad.flatten())
    return torch.stack(grads), losses.detach()


def measure_group(ctx, examples, store, offset, pack_tokens, tail):
    """Fill rows `offset..offset+len(examples)` of `store` for one pool of examples."""
    split, extractor = ctx["split"], ctx["extractor"]
    device, name, base = ctx["device"], ctx["name"], ctx["base"]
    z = torch.cat([t.flatten() for t in extractor.perturbation.z()])
    lengths = [len(e) for e in examples]
    # the pool in order (true gradients and g_code) and in reverse order (g_code_rev)
    for reverse in (False, True):
        seq = list(range(len(examples)))[::-1] if reverse else list(range(len(examples)))
        ordered = [examples[i] for i in seq]
        for group in greedy_groups([lengths[i] for i in seq], pack_tokens):
            batch = to_device(pack([ordered[i] for i in group]), device)
            rows = [offset + seq[i] for i in group]
            g_code = extractor.extract(batch).double().cpu().numpy()
            store["g_code_rev" if reverse else "g_code"][rows] = g_code
            if reverse:
                continue
            state = prefix_fp32(split, batch, tail)
            grads, losses = true_gradients(split, state, batch, name, base)
            store["G"][rows] = grads.cpu().numpy()
            store["s_exact"][rows] = (grads.double() @ z.double()).cpu().numpy()
            store["loss"][rows] = losses.double().cpu().numpy()
            store["gnorm"][rows] = grads.double().norm(dim=1).cpu().numpy()
            del state, grads, losses


def new_store(path, n, d):
    return {
        "G": np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, d)),
        **{k: np.full(n, np.nan) for k in ("g_code", "g_code_rev", "s_exact", "loss", "gnorm")},
    }


def save_meta(path, store, examples, pool_id):
    np.savez(
        path,
        **{k: v for k, v in store.items() if k != "G"},
        source=np.array([e.source for e in examples]),
        index=np.array([e.index for e in examples]),
        length=np.array([len(e) for e in examples]),
        num_labels=np.array([e.num_labels for e in examples]),
        pool_id=np.array(pool_id),
    )
    store["G"].flush()


def run_set(ctx, examples, pool_size, tag, out_dir, args):
    d = ctx["numel"]
    store = new_store(out_dir / f"G_{tag}.npy", len(examples), d)
    pool_id = [i // pool_size for i in range(len(examples))]
    started = time.time()
    for p in range(len(examples) // pool_size):
        t0 = time.time()
        chunk = examples[p * pool_size : (p + 1) * pool_size]
        measure_group(ctx, chunk, store, p * pool_size, args.pack_tokens, args.tail)
        print(
            f"{tag} pool {p + 1}/{len(examples) // pool_size} {time.time() - t0:.1f}s", flush=True
        )
        save_meta(out_dir / f"meta_{tag}.npz", store, examples, pool_id)
    return time.time() - started


def main():
    args = arguments()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    recipe = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(recipe["model_name_or_path"], local_files_only=True)
    with args.pool_file.open("rb") as handle:
        pools = pickle.load(handle)
    model = load_model(recipe, args.adapter, device)
    split = LastLayerSplit(model.get_base_model())
    zo_params = zo_parameters(model, ["v_proj"], -1)
    assert len(zo_params) == 1
    seed = pools["zo_random_seed"]
    extractor = library_extractor(model, zo_params, seed, args.tail)
    full_name, param = zo_params[0]
    ctx = {
        "model": model,
        "split": split,
        "extractor": extractor,
        "device": device,
        "name": split.relative_name(full_name),
        "base": param.detach(),
        "numel": param.numel(),
    }
    print(
        f"parameter {full_name} {tuple(param.shape)}; ZO seed {seed}; load {os.getloadavg()}",
        flush=True,
    )

    pool_examples_list = []
    for i in range(len(pools["instances"]) * 4 // POOL):
        pool_examples_list += pool_examples(tokenizer, pools, i)
    seconds = {"pools": run_set(ctx, pool_examples_list, POOL, "pools", args.out_dir, args)}

    if args.extra_pools:
        dataset = get_training_dataset(
            [args.train_file],
            tokenizer,
            model.config.max_position_embeddings,
            token_cache_dir=args.token_cache_dir,
        )
        rng = np.random.default_rng(args.extra_seed)
        picks = rng.choice(len(dataset), size=args.extra_pools * BIG_POOL, replace=False)
        extra = tokenize_examples(tokenizer, [dataset[int(i)] for i in picks])
        seconds["extra"] = run_set(ctx, extra, BIG_POOL, "extra", args.out_dir, args)

    info = {
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "zo_seed": seed,
        "parameter": full_name,
        "numel": ctx["numel"],
        "eps": EPS,
        "seconds": seconds,
        "loadavg_end": os.getloadavg(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    (args.out_dir / "info.json").write_text(json.dumps(info, indent=1) + "\n")
    print("DONE " + json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
