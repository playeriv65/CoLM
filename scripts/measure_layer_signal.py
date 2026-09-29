"""Measure per-example ZO loss differences while perturbing more Phi-2 LoRA layers.

This is a diagnostic, not a training run. The same examples and random direction in
each layer are reused for every cumulative layer count. Results are written after
each plus/minus pair so an interrupted run remains inspectable.
"""

import argparse
import contextlib
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from colm.data.get_training_dataset import tokenize_examples


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/layer_signal_phi2.json"))
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--max-sample-tokens", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--autocast", choices=["off", "fp16"], default="off")
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--direction-seed", type=int, default=734221)
    parser.add_argument("--layer-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    return parser.parse_args()


def make_batch(tokenizer, pool_file, count, max_tokens, device):
    with pool_file.open("rb") as handle:
        pools = pickle.load(handle)
    raw = [example for pool in pools["instances"] for example in pool]
    examples = tokenize_examples(tokenizer, raw)
    chosen = [example for example in examples if example.num_labels and len(example) <= max_tokens][
        :count
    ]
    if len(chosen) != count:
        raise ValueError(
            f"Need {count} examples of at most {max_tokens} tokens; found {len(chosen)}"
        )
    width = max(len(example) for example in chosen)
    ids = torch.full((count, width), tokenizer.eos_token_id, dtype=torch.long)
    labels = torch.full((count, width), -100, dtype=torch.long)
    attention = torch.zeros((count, width), dtype=torch.long)
    for row, example in enumerate(chosen):
        length = len(example)
        ids[row, :length] = torch.from_numpy(example.input_ids)
        labels[row, :length] = torch.from_numpy(example.labels)
        attention[row, :length] = 1
    return (
        {"input_ids": ids.to(device), "attention_mask": attention.to(device)},
        labels.to(device),
        chosen,
    )


@torch.inference_mode()
def losses(model, inputs, labels, autocast):
    context = (
        torch.autocast("cuda", dtype=torch.float16)
        if autocast == "fp16"
        else contextlib.nullcontext()
    )
    with context:
        logits = model(**inputs, use_cache=False).logits.float()
        token = F.cross_entropy(
            logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none", ignore_index=-100
        )
        counts = (labels[:, 1:] != -100).sum(dim=1)
        return (token.sum(dim=1) / counts).cpu().double().numpy()


def main():
    args = arguments()
    recipe = json.loads(args.config.read_text())
    model_name = recipe["model_name_or_path"]
    lora = recipe["lora"]
    perturbed_module = recipe["perturbed_module"]
    if args.directions < 1 or args.epsilon <= 0:
        raise ValueError("directions and epsilon must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    inputs, labels, chosen = make_batch(
        tokenizer, args.pool_file, args.sample_count, args.max_sample_tokens, device
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
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
    pattern = re.compile(
        rf"\.layers\.(\d+)\.{re.escape(perturbed_module)}\.lora_B\.default\.weight$"
    )
    by_layer = {
        int(match.group(1)): parameter
        for name, parameter in model.named_parameters()
        if (match := pattern.search(name))
    }
    num_layers = model.base_model.model.config.num_hidden_layers
    if sorted(by_layer) != list(range(num_layers)):
        raise RuntimeError(f"Expected LoRA B in all {num_layers} layers, found {sorted(by_layer)}")
    state = load_file(args.adapter)
    loaded = set_peft_model_state_dict(model, state)
    if loaded.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {loaded.unexpected_keys[:5]}")
    model.eval()
    base = {layer: parameter.detach().clone() for layer, parameter in by_layer.items()}
    t0 = time.perf_counter()
    baseline = losses(model, inputs, labels, args.autocast)
    metadata = {
        "model": model_name,
        "config_file": str(args.config),
        "adapter": str(args.adapter),
        "pool_file": str(args.pool_file),
        "epsilon": args.epsilon,
        "weight_dtype": "fp32",
        "autocast": args.autocast,
        "attention": "sdpa",
        "tf32": False,
        "lora_r": lora["r"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "lora_target_modules": lora["target_modules"],
        "perturbed_parameter": perturbed_module + ".lora_B.default.weight",
        "layer_counts": args.layer_counts,
        "directions": args.directions,
        "direction_seed": args.direction_seed,
        "sample_lengths": [len(example) for example in chosen],
        "sample_label_counts": [example.num_labels for example in chosen],
        "sample_indices": [example.index for example in chosen],
        "baseline_losses": baseline.tolist(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    counts = sorted(set(args.layer_counts))
    if counts[0] < 1 or counts[-1] > num_layers:
        raise ValueError(f"layer counts must be in 1..{num_layers}")
    records = []
    with (args.output_dir / "pairs.jsonl").open("w", buffering=1) as handle:
        for direction in range(args.directions):
            generator = torch.Generator(device="cpu").manual_seed(args.direction_seed + direction)
            z = {
                layer: torch.randn(base[layer].shape, generator=generator).to(device)
                for layer in range(num_layers)
            }
            for count in counts:
                active = list(range(num_layers - count, num_layers))
                try:
                    for sign in (1, -1):
                        for layer in active:
                            by_layer[layer].data.copy_(base[layer] + sign * args.epsilon * z[layer])
                        measured = losses(model, inputs, labels, args.autocast)
                        if sign == 1:
                            plus = measured
                        else:
                            minus = measured
                finally:
                    for layer in active:
                        by_layer[layer].data.copy_(base[layer])
                delta = plus - minus
                record = {
                    "direction": direction,
                    "layers": count,
                    "layer_ids": active,
                    "plus": plus.tolist(),
                    "minus": minus.tolist(),
                    "delta": delta.tolist(),
                    "projected_grad": (delta / (2 * args.epsilon)).tolist(),
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                }
                records.append(record)
                handle.write(json.dumps(record) + "\n")
                print(
                    f"direction={direction} layers={count} "
                    f"median_abs_delta={np.median(np.abs(delta)):.6g} "
                    f"rms_delta={np.sqrt(np.mean(delta**2)):.6g} "
                    f"elapsed_s={record['elapsed_s']}",
                    flush=True,
                )
    summary = {}
    for count in counts:
        delta = np.concatenate(
            [np.asarray(record["delta"]) for record in records if record["layers"] == count]
        )
        summary[str(count)] = {
            "n": len(delta),
            "median_abs_delta": float(np.median(np.abs(delta))),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "rms_delta": float(np.sqrt(np.mean(delta**2))),
        }
    (args.output_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
