"""Where does the fp16 error of the TRAINING gradients come from? Per-layer attention precision.

Diagnostic (kept because it documents a measurement, not part of the library). Result: docs/training-precision.md.

Protocol (phi-2 fp32 weights + the saved LoRA r=128 / alpha=512 adapter, packs of at most 1536 tokens of
real examples, the loss of the trainer: token cross-entropy summed and divided by the labels of the
whole step, times the fp16 loss scale of the GradScaler, gradient of ALL LoRA parameters). The
reference is the fp32 gradient without autocast and with the exact `sdpa_kernel(MATH)` attention
(the fp32 memory-efficient backward is not valid on sm_120, docs/errors.md). Variants, all fp16
autocast with the `flash_attention_2` hub kernel unless said otherwise; per-layer dispatch is a
script-local replacement of the `flash_attention_2` entry of transformers' `AttentionInterface`:

    V0    current training
    V1    fp32 MATH attention in blocks 29, 30 (q, k, v cast to fp32 at the attention call)
    V1q   V1 + q_proj / k_proj (with their LoRA) in fp32 there: under autocast the projections are
          already rounded to fp16 BEFORE the attention, which V1 does not touch
    V2 / V2q   the same for blocks 26-30
    V3 / V3q   the same for all blocks (upper bound of what fixing attention can buy)
    V4    blocks 29, 30 entirely in fp32 (autocast off for the whole block: attention, MLP, dense)
    V5 / V5q  V1 / V1q + block 31 (blocks 29-31);  V5k  only q_proj / k_proj fp32 in 29-31 (flash stays fp16);
    V8q  blocks 28-31;  V6q  blocks 26-31;  V7  blocks 29-31 entirely fp32
    S<k>  (--scan) V1q for a single block k: which block carries the error

The gradient is compared per pack (loss scaled as inside a step) and per step (sum of the packs of a
step), relative L2 and cosine, in total and per parameter group (q/k/v/fc1/fc2 x layer range).
Run-to-run noise of V0: the same input twice (`V0b`), the examples of every pack reversed (`V0r`).
`V0s1` / `V0s1024` are V0 with loss scale 1 / 1024 instead of the GradScaler's 65536 (fp16 backward underflow). Timing: warm, cuda-synchronised median
of forward + backward of one full pack, peak allocated memory.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python -u scripts/diagnostics/measure_training_attention.py \
        --pool-file $ROOT/inputs/pools.pkl --adapter $ROOT/inputs/adapter_model.safetensors --out-dir OUT
"""

import argparse
import json
import os
import pickle
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import AttentionInterface

from colm.data.get_training_dataset import tokenize_examples
from colm.selection.packing import greedy_groups, label_positions, model_inputs, pack
from colm.train.training_arguments import TrainingArguments

SWEEP = "configs/rank_sweep/sweep.json"
RECIPE = "configs/diagnostics/prefix_precision_phi2.json"
LAYERS = 32
GROUP_MODULES = ("q_proj", "k_proj", "v_proj", "fc1", "fc2")
RANGES = {"0-25": range(0, 26), "26-28": range(26, 29), "29-30": range(29, 31), "31": range(31, 32)}
ALL = frozenset(range(LAYERS))


class Config:
    """The active variant: which blocks get fp32 attention / fp32 q,k projections / fp32 everything."""

    autocast = True
    attn: frozenset = frozenset()
    qk: frozenset = frozenset()
    block: frozenset = frozenset()
    mask_cache: tuple | None = None
    checkpoint_attn = False
    # learning check: evaluation (padded batches, eval mode) keeps the stock path
    training_only = False
    fp32_calls = 0  # fp32 attention calls: proves that the dispatch was active


CFG = Config()


def variants(scan: list[int]) -> dict[str, dict]:
    v = {
        "V0": {},
        "V0b": {},
        "V0r": {"reverse": True},
        "V0s1": {"scale": 1.0},
        "V0s1024": {"scale": 1024.0},
        "V1": {"attn": {29, 30}},
        "V1q": {"attn": {29, 30}, "qk": {29, 30}},
        "V2": {"attn": set(range(26, 31))},
        "V2q": {"attn": set(range(26, 31)), "qk": set(range(26, 31))},
        "V3": {"attn": set(ALL)},
        "V3q": {"attn": set(ALL), "qk": set(ALL)},
        "V4": {"block": {29, 30}},
        "V5": {"attn": {29, 30, 31}},
        "V5q": {"attn": {29, 30, 31}, "qk": {29, 30, 31}},
        "V5k": {"qk": {29, 30, 31}},
        "V8q": {"attn": set(range(28, 32)), "qk": set(range(28, 32))},
        "V6q": {"attn": set(range(26, 32)), "qk": set(range(26, 32))},
        "V7": {"block": {29, 30, 31}},
    }
    for k in scan:
        v[f"S{k}"] = {"attn": {k}, "qk": {k}}
    return v


# ---- per-layer dispatch ----------------------------------------------------------------------
def block_mask(cu: torch.Tensor, length: int, device) -> torch.Tensor:
    """[1, 1, L, L] boolean mask: causal inside every packed example, nothing across them."""
    if CFG.mask_cache is not None and CFG.mask_cache[0] is cu:
        return CFG.mask_cache[1]
    ids = torch.bucketize(torch.arange(length, device=device), cu[1:].to(device).long(), right=True)
    mask = (ids[:, None] == ids[None, :]) & torch.ones(length, length, device=device).tril().bool()
    CFG.mask_cache = (cu, mask[None, None])
    return CFG.mask_cache[1]


def math_attention(q, k, v, mask, scale):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def install_dispatch(key: str):
    """Replace the attention function `key` by a dispatcher on `module.layer_idx` (fp32 MATH or the original)."""
    original = AttentionInterface._global_mapping[key]

    def dispatch(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        if (CFG.training_only and not module.training) or module.layer_idx not in (
            CFG.attn | CFG.block
        ):
            return original(module, query, key, value, attention_mask, dropout, scaling, **kwargs)
        CFG.fp32_calls += 1
        mask = block_mask(kwargs["cu_seq_lens_q"], query.shape[2], query.device)
        args = (query.float(), key.float(), value.float(), mask, scaling)
        with torch.autocast("cuda", enabled=False):
            if CFG.checkpoint_attn:
                out = checkpoint(math_attention, *args, use_reentrant=False)
            else:
                out = math_attention(*args)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface._global_mapping[key] = dispatch


def fp32_forward(module, member, layer):
    """Run `module.forward` with autocast off and fp32 inputs when the variant asks for it."""
    original = module.forward

    def forward(*args, **kwargs):
        if layer not in getattr(CFG, member) or (CFG.training_only and not module.training):
            return original(*args, **kwargs)
        args = tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
        with torch.autocast("cuda", enabled=False):
            return original(*args, **kwargs)

    module.forward = forward


def install_precision_hooks(base):
    for i, layer in enumerate(base.model.layers):
        for name in ("q_proj", "k_proj"):
            fp32_forward(getattr(layer.self_attn, name), "qk", i)
        fp32_forward(layer, "block", i)


def apply(spec: dict, checkpoint_min: int):
    CFG.autocast = spec.get("autocast", True)
    CFG.block = frozenset(spec.get("block", ()))
    CFG.attn = frozenset(spec.get("attn", ()))
    CFG.qk = frozenset(spec.get("qk", ()))
    CFG.checkpoint_attn = len(CFG.attn | CFG.block) >= checkpoint_min


# ---- model, data -------------------------------------------------------------------------------
def load_model(device, adapter: Path):
    recipe = json.loads(Path(RECIPE).read_text())
    model = AutoModelForCausalLM.from_pretrained(
        recipe["model_name_or_path"],
        dtype=torch.float32,
        attn_implementation="flash_attention_2",
        local_files_only=True,
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
    return model.eval(), recipe


def make_packs(tokenizer, pools, n_packs: int, budget: int, per_step: int):
    micro = TrainingArguments(output_dir="unused").pool_micro_batches
    examples, index = [], 0
    while True:
        raw = [e for m in pools["instances"][index * micro : (index + 1) * micro] for e in m]
        examples += tokenize_examples(tokenizer, raw)
        groups = greedy_groups([len(e) for e in examples], budget)
        # the last group is still open: only closed groups are full packs
        if len(groups) > n_packs:
            break
        index += 1
    packs = [[examples[i] for i in g] for g in groups[:n_packs]]
    return [packs[i : i + per_step] for i in range(0, n_packs, per_step)]


def group_of(name: str):
    match = re.search(r"layers\.(\d+)\..*\b(q_proj|k_proj|v_proj|fc1|fc2)\b", name)
    if not match:
        return None
    layer, module = int(match.group(1)), match.group(2)
    return module, next(r for r, span in RANGES.items() if layer in span)


def gradient(model, params, batch, total_labels, scale, spec, checkpoint_min, device):
    """Unscaled gradient (list of tensors) of this pack's share of the step loss."""
    apply(spec, checkpoint_min)
    for p in params:
        p.grad = None
    positions, targets, segment = label_positions(batch)
    with torch.autocast("cuda", dtype=torch.float16, enabled=CFG.autocast):
        logits = model(**model_inputs(batch), logits_to_keep=positions).logits[0]
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    loss = F.cross_entropy(logits, targets, reduction="sum") / total_labels
    (loss * scale).backward()
    grads = [p.grad.detach() / scale for p in params]
    for p in params:
        p.grad = None
    return grads, float(loss.detach())


class Stats:
    """Sums for the relative error and the cosine, in total and per group."""

    def __init__(self, groups):
        self.acc = {g: np.zeros(4) for g in [*set(groups), "all"]}

    def add(self, group_ids, grads, ref):
        for gid, g, r in zip(group_ids, grads, ref, strict=True):
            d = g.double() - r.double()
            row = np.array(
                [
                    float(d.pow(2).sum()),
                    float(r.double().pow(2).sum()),
                    float(g.double().pow(2).sum()),
                    float((g.double() * r.double()).sum()),
                ]
            )
            self.acc[gid] += row
            self.acc["all"] += row

    def result(self):
        out = {}
        for key, (dd, rr, gg, gr) in self.acc.items():
            name = "all" if key == "all" else f"{key[0]}:{key[1]}"
            out[name] = {"rel": float(np.sqrt(dd / rr)), "cos": float(gr / np.sqrt(rr * gg))}
        return out


def finite(grads):
    return all(bool(torch.isfinite(g).all()) for g in grads)


def measure(args, model, params, group_ids, steps, specs, device):
    """rows[variant] = {'pack': [per pack stats], 'step': [per step stats]}."""
    ref_spec = {"autocast": False, "attn": set(ALL)}
    rows = {name: {"pack": [], "step": [], "loss": [], "finite": True} for name in specs}
    groups = sorted(set(group_ids), key=str)
    for s, step in enumerate(steps):
        batches = [pack_to_device(p, device) for p in step]
        total = sum(int(b["colm_meta"]["label_counts"].sum()) for b in batches)
        ref_packs, ref_sum, ref_loss = [], None, 0.0
        for b in batches:
            g, loss = gradient(model, params, b, total, 1.0, ref_spec, args.ckpt_layers, device)
            ref_packs.append(g)
            ref_sum = g if ref_sum is None else [a + c for a, c in zip(ref_sum, g, strict=True)]
            ref_loss += loss
        print(f"step {s}: reference done, loss {ref_loss:.6f}, labels {total}", flush=True)
        for name, spec in specs.items():
            scale = spec.get("scale", args.loss_scale)
            step_sum, step_loss = None, 0.0
            for b, ref, raw in zip(batches, ref_packs, step, strict=True):
                if spec.get("reverse"):
                    b = pack_to_device(raw[::-1], device)
                g, loss = gradient(model, params, b, total, scale, spec, args.ckpt_layers, device)
                rows[name]["finite"] &= finite(g)
                st = Stats(groups)
                st.add(group_ids, g, ref)
                rows[name]["pack"].append(st.result())
                step_sum = (
                    g if step_sum is None else [a + c for a, c in zip(step_sum, g, strict=True)]
                )
                step_loss += loss
                del g
            st = Stats(groups)
            st.add(group_ids, step_sum, ref_sum)
            rows[name]["step"].append(st.result())
            rows[name]["loss"].append(abs(step_loss - ref_loss) / abs(ref_loss))
            r = rows[name]["step"][-1]["all"]
            print(f"step {s} {name}: rel {r['rel']:.4f} cos {r['cos']:.4f}", flush=True)
            del step_sum
            torch.cuda.empty_cache()
    return rows


def validate_reference(args, model, params, step, device, key):
    """Reference of this script (block mask, MATH) against stock sdpa MATH on packs of one example."""
    batches = [pack_to_device(p, device) for p in step]
    total = sum(int(b["colm_meta"]["label_counts"].sum()) for b in batches)
    ref = [torch.zeros_like(p) for p in params]
    for b in batches:
        g, _ = gradient(
            model, params, b, total, 1.0, {"autocast": False, "attn": set(ALL)}, 10, device
        )
        ref = [a + c for a, c in zip(ref, g, strict=True)]
    model.set_attn_implementation("sdpa")
    stock = [torch.zeros_like(p) for p in params]
    with sdpa_kernel(SDPBackend.MATH):
        for raw in step:
            for example in raw:
                b = pack_to_device([example], device)
                g, _ = gradient(model, params, b, total, 1.0, {"autocast": False}, 10**9, device)
                stock = [a + c for a, c in zip(stock, g, strict=True)]
    model.set_attn_implementation(key)
    d = sum(float((a.double() - c.double()).pow(2).sum()) for a, c in zip(stock, ref, strict=True))
    n = sum(float(c.double().pow(2).sum()) for c in ref)
    print(
        f"reference check: script MATH vs stock sdpa MATH per example, rel {np.sqrt(d / n):.2e}",
        flush=True,
    )
    return float(np.sqrt(d / n))


def pack_to_device(examples, device):
    from precision_arms import to_device

    return to_device(pack(examples), device)


def time_variants(args, model, params, batch, specs, device):
    """Median warm forward + backward of one full pack, peak allocated memory."""
    total = int(batch["colm_meta"]["label_counts"].sum())
    out = {}
    for name, spec in specs.items():
        if name in ("V0b", "V0r", "V0s1", "V0s1024"):
            continue
        for use_ckpt in (False, True):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for i in range(args.warmup + args.repeats):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    gradient(
                        model, params, batch, total, args.loss_scale, spec,
                        1 if use_ckpt else 10**9, device,
                    )  # fmt: skip
                    torch.cuda.synchronize()
                    if i >= args.warmup:
                        times.append((time.perf_counter() - start) * 1000)
                out[name] = {
                    "median_ms": statistics.median(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "attn_checkpoint": use_ckpt,
                }
                break
            except torch.OutOfMemoryError:
                print(f"{name}: OOM without attention checkpointing, retry with it", flush=True)
                torch.cuda.empty_cache()
        print(f"timing {name}: {json.dumps(out.get(name))}", flush=True)
    return out


def install_for_training(spec: dict):
    """Make `colm.train.train.build_model` return a model whose blocks follow `spec` (learning check)."""
    import colm.train.train as train

    build = train.build_model

    def build_with_dispatch(*args, **kwargs):
        model = build(*args, **kwargs)
        base = model.get_base_model()
        install_dispatch(base.config._attn_implementation)
        install_precision_hooks(base)
        apply(spec, 10**9)
        CFG.training_only = True
        return model

    train.build_model = build_with_dispatch


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--pool-file", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--packs", type=int, default=6)
    p.add_argument("--packs-per-step", type=int, default=3)
    p.add_argument("--pack-tokens", type=int, default=1536)
    p.add_argument("--loss-scale", type=float, default=65536.0)
    p.add_argument("--variants", nargs="*", help="subset of variants (default: all)")
    p.add_argument("--scan", type=int, nargs="*", default=[], help="single-block V1q variants")
    p.add_argument(
        "--ckpt-layers", type=int, default=10, help="attention checkpointing from N layers"
    )
    p.add_argument("--validate-ref", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--no-gradients", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--memory-gib", type=float, default=35.0, help="soft cap of this process")
    return p.parse_args()


def main():
    args = arguments()
    os.environ.update(json.loads(Path(SWEEP).read_text())["env"])
    from colm.jobs.worker import local_kernel_env

    os.environ.update(local_kernel_env(json.loads(Path(SWEEP).read_text())))
    device = torch.device("cuda:0")
    bus = torch.cuda.get_device_properties(0).pci_bus_id
    print(f"device pci bus {bus} {torch.cuda.get_device_name(0)}", flush=True)
    torch.cuda.set_per_process_memory_fraction(
        args.memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory
    )
    assert not torch.backends.cuda.matmul.allow_tf32, "fp32 matmul must not use TF32"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, recipe = load_model(device, args.adapter)
    key = model.get_base_model().config._attn_implementation
    print(f"attention implementation of the model: {key}", flush=True)
    install_dispatch(key)
    install_precision_hooks(model.get_base_model())
    tokenizer = AutoTokenizer.from_pretrained(recipe["model_name_or_path"], local_files_only=True)
    with args.pool_file.open("rb") as handle:
        pools = pickle.load(handle)
    steps = make_packs(tokenizer, pools, args.packs, args.pack_tokens, args.packs_per_step)
    sizes = [[sum(len(e) for e in p) for p in s] for s in steps]
    print(f"packs (tokens): {sizes}", flush=True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    group_ids = [group_of(n) for n, _ in named]
    assert None not in group_ids, "a trainable parameter outside q/k/v/fc1/fc2"
    print(f"trainable parameters {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)
    specs = variants(args.scan)
    if args.variants:
        specs = {k: v for k, v in specs.items() if k in args.variants}
    result = {
        "packs": sizes,
        "loss_scale": args.loss_scale,
        "pci_bus": bus,
        "load": os.getloadavg(),
    }
    if args.validate_ref:
        result["reference_check_rel"] = validate_reference(
            args, model, params, steps[0], device, key
        )
    if not args.no_gradients:
        result["gradients"] = measure(args, model, params, group_ids, steps, specs, device)
        result["peak_gib_gradients"] = torch.cuda.max_memory_allocated() / 2**30
        (args.out_dir / "gradients.json").write_text(json.dumps(result, indent=1) + "\n")
    if args.timing:
        biggest = max((p for s in steps for p in s), key=lambda p: sum(len(e) for e in p))
        batch = pack_to_device(biggest, device)
        result["timing_tokens"] = sum(len(e) for e in biggest)
        result["timing"] = time_variants(args, model, params, batch, specs, device)
        result["load_after"] = os.getloadavg()
        (args.out_dir / "timing.json").write_text(json.dumps(result, indent=1) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
