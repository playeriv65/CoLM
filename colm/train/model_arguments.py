from dataclasses import dataclass, field
from typing import Literal

from colm.train.literals import check_literals


@dataclass
class ModelArguments:
    """Which model/config/tokenizer to fine-tune."""

    model_name_or_path: str | None = field(
        default=None, metadata={"help": "Model checkpoint (hub id or local path)."}
    )
    checkpoint_path: str | None = field(
        default=None, metadata={"help": "Trainer checkpoint to resume training from."}
    )
    config_name: str | None = field(
        default=None, metadata={"help": "Config name or path if not the same as model_name."}
    )
    tokenizer_name: str | None = field(
        default=None, metadata={"help": "Tokenizer name or path if not the same as model_name."}
    )
    cache_dir: str | None = field(
        default=None, metadata={"help": "HF cache override (default: $HF_HOME)."}
    )
    model_revision: str = field(default="main", metadata={"help": "Model revision."})
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Allow custom modeling code from the hub."}
    )
    attn_implementation: str | None = field(
        default=None,
        metadata={
            "help": "Attention kernel. Default: `colm_varlen` (packed inputs, colm/train/attention.py); "
            "`sdpa` with `legacy` (padded batches)."
        },
    )
    torch_dtype: Literal["auto", "bfloat16", "float16", "float32", "none"] = field(
        default="none",
        metadata={"help": "Load dtype of the base weights; 'none' means float32."},
    )
    precision: Literal["auto", "fp32", "explicit"] = field(
        default="auto",
        metadata={
            "help": "auto: the precision of the recipe of the model (configs/model_profiles.json: "
            "fp16 AMP over fp32 weights or bf16); fp32: no mixed precision; explicit: keep the "
            "fp16/bf16/torch_dtype flags as given."
        },
    )
    lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    lora_r: int = field(default=128, metadata={"help": "LoRA rank."})
    lora_alpha: float = field(default=512, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: list[str] = field(
        default_factory=list, metadata={"help": "LoRA target modules (inferred if empty)."}
    )
    enable_dropout: bool = field(default=True, metadata={"help": "Keep the model's dropout."})

    def __post_init__(self):
        check_literals(self)


def add_padding_to_tokenizer(tokenizer):
    """Add a padding token to the tokenizer if it has none."""
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
