"""Command line / JSON configuration of a run.

`python -m colm.train.train config.json [--flag value ...]`: the JSON file gives the defaults,
flags override them, unknown keys are an error. The recipe of the model (LoRA targets, precision)
comes from `configs/model_profiles.json` and is resolved before the training arguments are built.
"""

import json
import os
import sys
from dataclasses import fields

from transformers import AutoConfig, HfArgumentParser

from colm.eval.arguments import HeldoutEvalArguments
from colm.train.data_arguments import DataArguments
from colm.train.model_arguments import ModelArguments
from colm.train.training_arguments import TrainingArguments

PROFILES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "model_profiles.json"
)
DATACLASSES = (ModelArguments, DataArguments, TrainingArguments, HeldoutEvalArguments)


LEGACY_MAX_LENGTH = 512  # where the upstream code cut every example


def sequence_limit(model_args: ModelArguments, data_args: DataArguments, legacy: bool) -> int:
    """Tokens a training example may have: the context window of the model.

    Examples above it are dropped, never truncated. `legacy` (upstream error E4b): cut at 512.
    """
    if data_args.max_seq_length:
        return data_args.max_seq_length
    if legacy:
        return LEGACY_MAX_LENGTH
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


def parse_args(argv: list[str] | None = None):
    """(model, data, training, eval) arguments from a JSON file and / or flags."""
    argv = list(sys.argv[1:] if argv is None else argv)
    defaults = {}
    if argv and argv[0].endswith(".json"):
        with open(argv.pop(0)) as f:
            defaults = json.load(f)
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
    explicit = defaults.get("fp16") or defaults.get("bf16") or "--fp16" in argv or "--bf16" in argv
    if model_args.precision == "fp32":
        extra.update(fp16=False, bf16=False, torch_dtype="float32")
    elif model_args.precision == "auto" and not explicit:
        if profile["precision"] == "fp16_amp":
            extra.update(fp16=True, torch_dtype="none")
        else:
            extra.update(bf16=True, torch_dtype="bfloat16")
    return parse(DATACLASSES, extra)
