"""Compare fp16-prefix/fp32-suffix MeZO differences with an fp32 prefix.

The inputs and adapter come from the saved layer-signal diagnostic. Each arm
uses the same packed examples, weights, directions, and fp32 suffix/loss.
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from colm.data.get_training_dataset import tokenize_examples
from colm.selection.features import MezoEfficient, example_means
from colm.selection.packing import label_counts, label_positions, model_inputs, pack
from colm.selection.select import CoresetSelector
from colm.selection.zo import LastLayerSplit, Perturbation, zo_parameters
from colm.train.training_arguments import TrainingArguments


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--pack-size", type=int, default=4)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=734221)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--selection-pools", type=int, default=0)
    parser.add_argument("--check-extractor", action="store_true")
    return parser.parse_args()


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


@torch.inference_mode()
def measure(split, perturbation, batch, prefix_dtype):
    positions, targets, segment = label_positions(batch)
    counts = label_counts(batch)
    context = (
        torch.autocast("cuda", dtype=torch.float16)
        if prefix_dtype == "float16"
        else torch.autocast("cuda", enabled=False)
    )
    start = time.perf_counter()
    with context:
        state = split.prefix(**model_inputs(batch))
    if prefix_dtype == "float16":
        state = state.float()
    torch.cuda.synchronize()
    prefix_ms = (time.perf_counter() - start) * 1000

    shifts = [z * perturbation.eps for z in perturbation.z()]
    values = []
    for sign in (1, -1):
        overrides = {
            split.relative_name(name): parameter + sign * shift
            for name, parameter, shift in zip(
                perturbation.names, perturbation.params, shifts, strict=True
            )
        }
        hidden = split.hidden(state, overrides)[0, positions]
        logits = split.head(hidden)
        values.append(example_means(logits, targets, segment, counts).cpu().double().numpy())
    return {
        "plus": values[0].tolist(),
        "minus": values[1].tolist(),
        "delta": (values[0] - values[1]).tolist(),
        "prefix_ms": prefix_ms,
    }


def main():
    args = arguments()
    recipe = json.loads(args.config.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    name = recipe["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    with args.pool_file.open("rb") as handle:
        pools = pickle.load(handle)
    raw = [example for pool in pools["instances"] for example in pool]
    examples = [
        example
        for example in tokenize_examples(tokenizer, raw)
        if example.num_labels and len(example) <= 512
    ][: args.examples]
    if len(examples) != args.examples:
        raise ValueError(f"Need {args.examples} examples; found {len(examples)}")
    lora = recipe["lora"]
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    )
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
    loaded = set_peft_model_state_dict(model, load_file(args.adapter))
    if loaded.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {loaded.unexpected_keys[:5]}")
    model.eval()
    split = LastLayerSplit(model.get_base_model())
    params = zo_parameters(model, ["v_proj"], -1)
    batches = [
        to_device(pack(examples[start : start + args.pack_size]), device)
        for start in range(0, len(examples), args.pack_size)
    ]
    records = []
    for direction in range(args.directions):
        perturbation = Perturbation(params, args.eps, args.seed + direction)
        for batch_index, batch in enumerate(batches):
            arms = {
                dtype: measure(split, perturbation, batch, dtype)
                for dtype in ("float32", "float16")
            }
            record = {"direction": direction, "pack": batch_index, **arms}
            records.append(record)
            print(
                json.dumps(
                    {
                        "direction": direction,
                        "pack": batch_index,
                        "prefix_ms": {key: value["prefix_ms"] for key, value in arms.items()},
                        "fp32_delta": arms["float32"]["delta"],
                        "mixed_delta": arms["float16"]["delta"],
                    }
                ),
                flush=True,
            )
    ref = np.concatenate([r["float32"]["delta"] for r in records])
    mixed = np.concatenate([r["float16"]["delta"] for r in records])
    summary = {
        "relative_l2": float(np.linalg.norm(mixed - ref) / np.linalg.norm(ref)),
        "sign_agreement": int(np.sum(np.sign(mixed) == np.sign(ref))),
        "count": int(len(ref)),
        "median_abs_delta_fp32": float(np.median(np.abs(ref))),
        "median_abs_delta_mixed": float(np.median(np.abs(mixed))),
        "prefix_ms_fp32_mean": float(np.mean([r["float32"]["prefix_ms"] for r in records])),
        "prefix_ms_mixed_mean": float(np.mean([r["float16"]["prefix_ms"] for r in records])),
    }
    if args.check_extractor:
        for dtype in ("float32", "float16"):
            extractor_args = SimpleNamespace(
                mezo_eps=args.eps,
                selection_prefix_dtype=dtype,
                mezo_selection="grad",
            )
            extractor = MezoEfficient(extractor_args, model, params, args.seed)
            actual = extractor.extract(batches[0]).cpu().double().numpy()
            expected = np.asarray(records[0][dtype]["delta"]) / (2 * args.eps)
            summary[f"extractor_max_abs_{dtype}"] = float(np.max(np.abs(actual - expected)))
            summary[f"extractor_resolved_{dtype}"] = extractor.prefix_dtype
    selection = []
    if args.selection_pools:
        training = TrainingArguments(output_dir=str(args.out.parent / "selection-check"))
        arms = ("float32", "floor", "float16")
        selectors = {arm: CoresetSelector(training, model.config.num_hidden_layers) for arm in arms}
        perturbation = Perturbation(params, args.eps, pools["zo_random_seed"])
        micro_batches = training.pool_micro_batches
        for step in range(args.selection_pools):
            first = step * micro_batches
            raw_pool = [
                example
                for micro in pools["instances"][first : first + micro_batches]
                for example in micro
            ]
            if len(raw_pool) != training.per_device_train_batch_size:
                raise ValueError(f"Incomplete saved pool at step {step}: {len(raw_pool)}")
            pool = tokenize_examples(tokenizer, raw_pool)
            sources = [example.source for example in pool]

            def gradients(examples, dtype):
                parts = []
                for start in range(0, len(examples), args.pack_size):
                    batch = to_device(pack(examples[start : start + args.pack_size]), device)
                    result = measure(split, perturbation, batch, dtype)
                    parts.extend(result["delta"])
                return torch.tensor(parts, device=device, dtype=torch.float32) / (2 * args.eps)

            values = {
                "float32": gradients(pool, "float32"),
                "floor": gradients(pool[::-1], "float32").flip(0),
                "float16": gradients(pool, "float16"),
            }
            selected = {}
            for arm, scalar in values.items():
                features = perturbation.features(scalar)
                choice = selectors[arm](features, sources, len(pool) // 2, step)
                selected[arm] = sorted(pool[index].index for index in choice.indices)
            reference = set(selected["float32"])
            overlap = {arm: len(reference & set(selected[arm])) for arm in arms if arm != "float32"}
            entry = {"step": step, "selected": selected, "overlap": overlap}
            selection.append(entry)
            print("SELECTION " + json.dumps(entry), flush=True)
        summary["selection_pools"] = len(selection)
        summary["overlap_floor_mean"] = float(np.mean([s["overlap"]["floor"] for s in selection]))
        summary["overlap_mixed_mean"] = float(np.mean([s["overlap"]["float16"] for s in selection]))
        summary["selected_per_pool"] = training.per_device_train_batch_size // 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "summary": summary,
                "records": records,
                "selection": selection,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
