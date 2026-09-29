"""Command line / JSON configuration of a run.

`python -m colm.train.train config.json [--flag value ...]`: the JSON file gives the defaults,
flags override them, unknown keys are an error. The recipe of the model (LoRA targets, precision)
comes from `configs/model_profiles.json` and is resolved before the training arguments are built.
"""

import json
import os
import sys
from dataclasses import asdict, fields

from transformers import AutoConfig, HfArgumentParser

from colm.eval.arguments import HeldoutEvalArguments
from colm.train.data_arguments import DataArguments
from colm.train.model_arguments import ModelArguments
from colm.train.training_arguments import TrainingArguments

PROFILES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "model_profiles.json"
)
DATACLASSES = (ModelArguments, DataArguments, TrainingArguments, HeldoutEvalArguments)


def context_length(model_args: ModelArguments) -> int:
    """Tokens a training example may have: the context window of the model.

    There is no option to set another limit. Examples above it are dropped (counted and logged
    per source), never truncated: nothing in the code cuts a sequence.
    """
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )
    return config.max_position_embeddings


def model_profile(model_args: ModelArguments) -> dict:
    with open(PROFILES_FILE) as f:
        profiles = json.load(f)
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )
    return profiles.get(config.model_type, profiles["default"])


def eval_dtype(model_name_or_path: str) -> str:
    """Generation dtype of a model or LoRA checkpoint: its recipe's precision (see model_profiles.json)."""
    from peft import PeftConfig

    adapter = os.path.join(model_name_or_path, "adapter_config.json")
    base = (
        PeftConfig.from_pretrained(model_name_or_path).base_model_name_or_path
        if os.path.exists(adapter)
        else model_name_or_path
    )
    profile = model_profile(ModelArguments(model_name_or_path=base))
    return "float16" if profile["precision"] == "fp16_amp" else "bfloat16"


def flag_value(argv: list[str], name: str):
    """Last value of `--name value` / `--name=value` in `argv` (None when absent)."""
    value = None
    for i, arg in enumerate(argv):
        if arg == f"--{name}" and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith(f"--{name}="):
            value = arg.split("=", 1)[1]
    return value


def parse_args(argv: list[str] | None = None):
    """(model, data, training, eval) arguments from a JSON file and / or flags."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        HfArgumentParser(DATACLASSES, prog="colm-train").print_help()
        raise SystemExit(0)
    defaults = {}
    if argv and argv[0].endswith(".json"):
        with open(argv.pop(0)) as f:
            defaults = {k: v for k, v in json.load(f).items() if not k.startswith("_")}  # "_doc"
        known = {f.name for dc in DATACLASSES for f in fields(dc)}
        if unknown := set(defaults) - known:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")

    def parse(classes, extra_defaults):
        parser = HfArgumentParser(classes)
        parser.set_defaults(**{**defaults, **extra_defaults})
        return parser.parse_args_into_dataclasses(
            args=argv, look_for_args_file=False, return_remaining_strings=len(classes) == 1
        )

    # The model's recipe decides the precision, which TrainingArguments resolves when built.
    model_args = parse((ModelArguments,), {})[
        0
    ]  # (model_args, leftover namespace, remaining flags)
    profile = model_profile(model_args)
    extra = {}
    if not model_args.lora_target_modules:
        extra["lora_target_modules"] = profile["lora_target_modules"]
    if "selection_prefix_dtype" not in defaults:
        extra["selection_prefix_dtype"] = profile["selection_prefix_dtype"]
    # The fp32 tail belongs to the profile's fp16 prefix: another explicit dtype gets no tail.
    prefix_dtype = (
        flag_value(argv, "selection_prefix_dtype")
        or defaults.get("selection_prefix_dtype")
        or profile["selection_prefix_dtype"]
    )
    if "selection_prefix_fp32_tail" not in defaults:
        extra["selection_prefix_fp32_tail"] = (
            profile["selection_prefix_fp32_tail"] if prefix_dtype == "float16" else 0
        )
    if "pack_tokens" not in defaults:
        extra["pack_tokens"] = profile["pack_tokens"]
    if not model_args.attn_implementation and model_args.precision != "fp32":
        extra["attn_implementation"] = profile["attn_implementation"]
    explicit = defaults.get("fp16") or defaults.get("bf16") or "--fp16" in argv or "--bf16" in argv
    if model_args.precision == "fp32":
        extra.update(fp16=False, bf16=False, torch_dtype="float32")
    elif model_args.precision == "auto" and not explicit:
        if profile["precision"] == "fp16_amp":
            extra.update(fp16=True, torch_dtype="none")
        else:
            extra.update(bf16=True, torch_dtype="bfloat16")
    return parse(DATACLASSES, extra)


def resolved_config(model_args, data_args, training_args, eval_args, derived: dict) -> dict:
    """Every option of the run with its value (defaults included) and the values derived from the model and data."""
    return {
        "model": asdict(model_args),
        "data": asdict(data_args),
        "eval": asdict(eval_args),
        "training": training_args.to_dict(),
        "derived": derived,
    }


def save_resolved_config(config: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "resolved_config.json"), "w") as f:
        json.dump(config, f, indent=1, default=str)
