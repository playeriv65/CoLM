from dataclasses import dataclass, field

from transformers import TrainingArguments as HFTrainingArguments

# Names accepted by --last_layers and the per-layer modules each one expands to.
LAST_LAYER_GROUPS = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "qkvo_proj": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "fc": ["fc1", "fc2"],
}
ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}


@dataclass
class TrainingArguments(HFTrainingArguments):
    """HF TrainingArguments with CoLM selection options.

    The HF fields overridden below restore the paper's training recipe
    (the former colm/scripts/train/base_training_args.sh).
    """

    # --- HF fields with CoLM defaults ---
    output_dir: str | None = field(
        default=None,
        metadata={"help": "Output directory. Auto-generated from the key parameters if unset."},
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    per_device_train_batch_size: int = field(default=1)
    num_train_epochs: float = field(default=4.0)
    learning_rate: float = field(default=2e-5)
    warmup_steps: float = field(default=0.03, metadata={"help": "Warmup steps, or ratio if < 1."})
    optim: str = field(default="adamw_torch")
    logging_steps: float = field(default=1)
    save_strategy: str = field(default="steps")
    save_steps: float = field(default=256)
    seed: int = field(default=0)
    remove_unused_columns: bool = field(default=False)
    report_to: None | str | list[str] = field(
        default="none",
        metadata={"help": "Integrations to report to. W&B is opt-in: pass 'wandb' explicitly."},
    )

    # --- W&B (only read when report_to includes wandb) ---
    wandb_project: str | None = field(default=None, metadata={"help": "W&B project."})
    wandb_entity: str | None = field(default=None, metadata={"help": "W&B entity."})
    wandb_notes: str | None = field(default=None, metadata={"help": "W&B notes."})

    # --- Analysis ---
    analysis_mode: bool = field(default=False, metadata={"help": "Build an analysis eval set."})
    analysis_dataset: str = field(default="bbh", metadata={"help": "Dataset for analysis mode."})
    train_dataset_names: str | None = field(default=None, metadata={"help": "Space separated."})

    # --- Mini-batch coreset selection ---
    small_batch_ratio: float = field(
        default=0.5, metadata={"help": "Fraction of the large mini-batch that is trained on."}
    )
    data_selection_method: str = field(
        default="submodlib",
        metadata={
            "help": "How to select the small batch from the large batch.",
            "choices": ["submodlib", "weightedsubmodlib", "none"],
        },
    )
    efficient_mezo: bool = field(
        default=False,
        metadata={"help": "Batched last-layer MeZO estimate (SubsetTrainerEfficient)."},
    )
    data_selection_unit: str = field(
        default="mezo",
        metadata={
            "help": "Per-example feature used for selection.",
            "choices": ["rep", "mezo", "masked_grad", "completion_length", "length_loss_weighted"],
        },
    )
    facility_similarity: str = field(
        default="l1",
        metadata={
            "help": "Facility-location similarity.",
            "choices": ["cosine", "euclidean", "l1"],
        },
    )
    source_wise_selection: str = field(
        default="proportional",
        metadata={
            "help": "How many examples to select per data source.",
            "choices": ["none", "proportional", "balanced"],
        },
    )
    keep_sources: str = field(
        default="0_1_3_5_7_8_9_10_11_13",
        metadata={"help": "Source indices kept in full (not selected), separated by '_'."},
    )
    num_per_class_start: str = field(
        default="floor",
        metadata={"help": "Rounding of the per-source budget.", "choices": ["floor", "ceil"]},
    )
    save_indices: bool = field(
        default=False, metadata={"help": "Save the large-batch and selected example indices."}
    )

    # --- Zeroth-order (MeZO) last-layer gradient estimate ---
    mezo_eps: float = field(default=1e-3, metadata={"help": "MeZO perturbation scale."})
    mezo_transform: str = field(
        default="none",
        metadata={
            "help": "Transform of the gradient estimates (SubsetTrainer only).",
            "choices": ["none", "self_normalize", "normalize", "clip_full", "clip_last"],
        },
    )
    mezo_selection: str = field(
        default="grad",
        metadata={
            "help": "Feature built from the estimate.",
            "choices": ["weight_grad", "weight", "grad"],
        },
    )
    mezo_topk: str = field(
        default="largest",
        metadata={
            "help": "Which coordinates of the estimate are kept.",
            "choices": ["largest", "smallest", "random", "sampling", "largest_smallest"],
        },
    )
    mezo_optim: str = field(
        default="adam",
        metadata={"help": "Optimizer whose update is used as feature.", "choices": ["sgd", "adam"]},
    )
    zo_dim: int = field(default=2560, metadata={"help": "Number of kept coordinates."})
    last_layer_index: int = field(
        default=31, metadata={"help": "Index of the last decoder layer (31 for 32-layer models)."}
    )
    last_layers: str | list[str] = field(
        default="v_proj",
        metadata={
            "help": "Module(s) of the last layer whose LoRA B is perturbed.",
            "choices": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "fc1",
                "fc2",
                "qkv_proj",
                "qkvo_proj",
                "fc",
            ],
        },
    )

    # --- Stability ---
    assert_finite_grad_norm: bool = field(
        default=False,
        metadata={
            "help": "Fail fast when the pre-clip gradient norm is NaN/Inf (baseline trainer)."
        },
    )

    # --- Step timing (colm/train/step_timing.py) ---
    profile_timing: str = field(
        default="off",
        metadata={
            "help": "Per-phase wall-clock breakdown of every optimizer step. 'off' adds no "
            "synchronize; 'coarse' times the phases with a CUDA synchronize at each boundary; "
            "'fine' also times per-layer / per-op sub-phases (more syncs, slight inflation).",
            "choices": ["off", "coarse", "fine"],
        },
    )
    profile_timing_dir: str | None = field(
        default=None,
        metadata={"help": "Directory of the step-timing JSONL (default: output_dir)."},
    )
    profile_census_steps: int = field(
        default=0,
        metadata={
            "help": "Count host<->device copies and synchronizing calls during the first N "
            "steps (slow; keep N <= the summary warmup so these steps are not timed)."
        },
    )

    # --- SuperGLUE ---
    max_new_tokens: int = field(
        default=50, metadata={"help": "Maximum number of generated tokens."}
    )
    non_diff: bool = field(
        default=False, metadata={"help": "Non-differentiable objective (SQuAD F1)."}
    )
    only_train_option: bool = field(default=True, metadata={"help": "Only train the option part."})
    modify_forward: bool = field(
        default=False, metadata={"help": "Set when the forward is wrapped."}
    )

    def __post_init__(self):
        # train.py fills in a descriptive name once the model name is known.
        self.output_dir_is_auto = self.output_dir is None
        if isinstance(self.train_dataset_names, str):
            self.train_dataset_names = self.train_dataset_names.split(" ")
        if isinstance(self.last_layers, str):
            modules = LAST_LAYER_GROUPS.get(self.last_layers, [self.last_layers])
            self.last_layers = [
                f"layers.{self.last_layer_index}.{'self_attn' if m in ATTENTION_MODULES else 'mlp'}.{m}"
                for m in modules
            ]
        super().__post_init__()
