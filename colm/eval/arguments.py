from dataclasses import dataclass, field

HELDOUT_SET = "heldout"
GSM8K_SET = "gsm8k"
EVAL_LOSS_SETS = (HELDOUT_SET, GSM8K_SET)


@dataclass
class HeldoutEvalArguments:
    """Held-out split and teacher-forced evaluation loss (all off by default = paper recipe)."""

    holdout_size: int = field(
        default=0,
        metadata={
            "help": "Examples removed from the training data for evaluation loss. 0 keeps the "
            "paper recipe (all data is trained on); > 0 is a deviation and must be identical "
            "across the runs that are compared."
        },
    )
    holdout_seed: int = field(
        default=0,
        metadata={"help": "Seed of the held-out draw (independent of the training seed)."},
    )
    eval_loss_steps: list[int] = field(
        default_factory=list,
        metadata={"help": "Optimizer steps after which the evaluation loss is computed."},
    )
    eval_loss_sets: list[str] = field(
        default_factory=lambda: [HELDOUT_SET, GSM8K_SET],
        metadata={
            "help": "Sets to evaluate: 'heldout' (needs holdout_size > 0) and/or 'gsm8k' "
            "(GSM8K test reference solutions).",
            "choices": list(EVAL_LOSS_SETS),
        },
    )
    eval_loss_batch_size: int = field(
        default=8, metadata={"help": "Sequences per forward pass of the loss evaluation."}
    )
    eval_loss_gsm8k_file: str = field(
        default="math_eval/dataset/gsm8k/gsm8k.jsonl",
        metadata={"help": "GSM8K test jsonl (question / answer with '#### n')."},
    )
