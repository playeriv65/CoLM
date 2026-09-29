from dataclasses import dataclass, field
from typing import Literal

from transformers import TrainingArguments as HFTrainingArguments

from colm.train.literals import check_literals

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

    # --- Recipe of the paper: the defaults of a plain run ---
    output_dir: str | None = field(
        default=None,
        metadata={"help": "Output directory. Auto-generated from the key parameters if unset."},
    )
    max_steps: int = field(default=1024, metadata={"help": "Optimizer steps."})
    per_device_train_batch_size: int = field(
        default=4,
        metadata={
            "help": "Examples per forward pass (micro-batch) of one rank. The selection pool of a "
            "rank is this x gradient_accumulation_steps examples."
        },
    )
    gradient_accumulation_steps: int = field(
        default=8, metadata={"help": "Micro-batches per selection pool (and per optimizer step)."}
    )
    learning_rate: float = field(default=2e-5)
    warmup_steps: float = field(default=0.03, metadata={"help": "Warmup steps, or ratio if < 1."})
    logging_steps: float = field(default=1)
    save_strategy: str = field(default="steps")
    save_steps: float = field(default=256)
    save_only_model: bool = field(
        default=True, metadata={"help": "Checkpoints hold the adapter only (no optimizer state)."}
    )
    seed: int = field(default=0)
    dataloader_num_workers: int = field(
        default=1,
        metadata={
            "help": "Worker processes of the data loader: one tokenises the pool of the next step "
            "(11 ms of 32 examples) while the GPU works on this one."
        },
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Keep the per-example bookkeeping the batches carry besides the model inputs."
        },
    )
    report_to: None | str | list[str] = field(
        default="none",
        metadata={"help": "Integrations to report to. W&B is opt-in: pass 'wandb' explicitly."},
    )

    # --- W&B (only read when report_to includes wandb) ---
    wandb_project: str | None = field(default=None, metadata={"help": "W&B project."})
    wandb_entity: str | None = field(default=None, metadata={"help": "W&B entity."})
    wandb_notes: str | None = field(default=None, metadata={"help": "W&B notes."})

    # --- Mini-batch coreset selection ---
    small_batch_ratio: float = field(
        default=0.5, metadata={"help": "Fraction of the large mini-batch that is trained on."}
    )
    micro_batch_size: int = field(
        default=0,
        metadata={
            "help": "Set by __post_init__: examples per forward of a coreset run. There "
            "per_device_train_batch_size is the whole selection pool (micro batch x "
            "gradient_accumulation_steps) and gradient_accumulation_steps is 1."
        },
    )
    pack_tokens: int = field(
        default=0,
        metadata={
            "help": "Tokens per packed selection forward (a forward always holds at least one "
            "example). 0: the tokens of one micro-batch of the padded recipe (micro batch size x "
            "mean example length of the data)."
        },
    )
    selection_attn_implementation: str = field(
        default="sdpa",
        metadata={
            "help": "Attention of the selection forward (no gradient); "
            "the training forward uses --attn_implementation. Packed rows "
            "carry `position_ids` and the cumulative lengths of the flash kernels; stock "
            "transformers builds the block mask (sdpa, flex_attention) or the varlen call "
            "(flash_attention_2) from them."
        },
    )
    selection_prefix_dtype: Literal["float32", "float16"] = field(
        default="float32",
        metadata={
            "help": "Precision of the unperturbed MeZO decoder prefix. The model profile "
            "sets the CLI default (Phi-2: float16); an explicit flag or JSON value overrides it. "
            "float16 requires fp32 model weights; the perturbed last layer, head, and loss stay fp32."
        },
    )
    train_max_tokens: int = field(
        default=1536,
        metadata={
            "help": "Packed tokens per training forward + backward. N > 0 (memory mode, default "
            "1536): the selected examples of a step are packed greedily into forwards of at most "
            "N tokens (an example longer than N goes alone; none is split or truncated) and the "
            "gradients are accumulated; phi-2: 1536 gives 32.3 GB peak and 1382 ms per step. "
            "0 (speed mode, unlimited): all selected examples of the step in ONE forward + "
            "backward; phi-2: 1383 ms per step (no faster) but 57 GB peak (94 GB reserved), "
            "so use it only with memory to spare. The step loss and gradient are the same "
            "in both modes."
        },
    )
    data_selection_method: Literal["submodlib", "weightedsubmodlib", "none"] = field(
        default="submodlib",
        metadata={"help": "How to select the small batch from the large batch."},
    )
    efficient_mezo: bool = field(
        default=True,
        metadata={
            "help": "Batched last-layer MeZO estimate (SubsetTrainerEfficient, the paper's method). "
            "Off: SubsetTrainer, one example per micro-batch (per_device_train_batch_size=1) and "
            "any data_selection_unit."
        },
    )
    data_selection_unit: Literal[
        "rep", "mezo", "masked_grad", "completion_length", "length_loss_weighted"
    ] = field(
        default="mezo",
        metadata={"help": "Per-example feature used for selection."},
    )
    facility_similarity: Literal["cosine", "euclidean", "l1"] = field(
        default="l1",
        metadata={"help": "Facility-location similarity."},
    )
    source_wise_selection: Literal["none", "proportional", "balanced"] = field(
        default="proportional",
        metadata={"help": "How many examples to select per data source."},
    )
    keep_sources: str = field(
        default="0_1_3_5_7_8_9_10_11_13",
        metadata={"help": "Source indices kept in full (not selected), separated by '_'."},
    )
    num_per_class_start: Literal["floor", "ceil"] = field(
        default="floor",
        metadata={"help": "Rounding of the per-source budget."},
    )
    save_indices: bool = field(
        default=False, metadata={"help": "Save the large-batch and selected example indices."}
    )

    # --- Zeroth-order (MeZO) last-layer gradient estimate ---
    mezo_eps: float = field(default=1e-3, metadata={"help": "MeZO perturbation scale."})
    mezo_transform: Literal["none", "self_normalize", "normalize", "clip_full", "clip_last"] = (
        field(
            default="none",
            metadata={"help": "Transform of the gradient estimates (SubsetTrainer only)."},
        )
    )
    mezo_selection: Literal["weight_grad", "weight", "grad"] = field(
        default="grad",
        metadata={"help": "Feature built from the estimate."},
    )
    mezo_topk: Literal["largest", "smallest", "random", "sampling", "largest_smallest"] = field(
        default="largest",
        metadata={"help": "Which coordinates of the estimate are kept."},
    )
    mezo_optim: Literal["sgd", "adam"] = field(
        default="adam",
        metadata={"help": "Optimizer whose update is used as feature."},
    )
    zo_dim: int = field(default=2560, metadata={"help": "Number of kept coordinates."})
    last_layer_index: int = field(
        default=-1, metadata={"help": "Decoder layer whose LoRA B is perturbed; -1 = the last."}
    )
    last_layers: str = field(
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
    profile_timing: Literal["off", "coarse", "fine"] = field(
        default="off",
        metadata={
            "help": "Per-phase wall-clock breakdown of every optimizer step. 'off' adds no "
            "synchronize; 'coarse' times the phases with a CUDA synchronize at each boundary; "
            "'fine' also times per-layer / per-op sub-phases (more syncs, slight inflation)."
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
    only_train_option: bool = field(default=True, metadata={"help": "Only train the option part."})

    def _validate(self) -> None:
        """Fail fast on inconsistent selection settings."""
        if not self.coreset:
            return
        if not 0 < self.small_batch_ratio <= 1:
            raise ValueError(f"small_batch_ratio must be in (0, 1], got {self.small_batch_ratio}")
        if self.efficient_mezo:
            if self.data_selection_unit != "mezo":
                raise ValueError(
                    "efficient_mezo estimates MeZO features: set data_selection_unit=mezo"
                )
            if self.mezo_transform != "none" or "weighted" in self.data_selection_method:
                raise ValueError("efficient_mezo applies no mezo_transform and trains unweighted")
            if int(self.micro_batch_size * self.small_batch_ratio) < 1:
                raise ValueError("per_device_train_batch_size * small_batch_ratio must be >= 1")
        else:
            if self.micro_batch_size != 1:
                raise ValueError(
                    "without efficient_mezo every example is its own micro-batch: "
                    "per_device_train_batch_size=1 and gradient_accumulation_steps = the pool size"
                )
            if int(self.pool_micro_batches * self.small_batch_ratio) < 1:
                raise ValueError("gradient_accumulation_steps * small_batch_ratio must be >= 1")

    @property
    def pool_micro_batches(self) -> int:
        return self.per_device_train_batch_size // (self.micro_batch_size or 1)

    @property
    def coreset(self) -> bool:
        return self.data_selection_method != "none"

    @property
    def keep_source_ids(self) -> list[int]:
        """`keep_sources` ("0_1_3", or a list) as integer source ids."""
        if isinstance(self.keep_sources, str):
            return [int(i) for i in self.keep_sources.split("_") if i]
        return [int(i) for i in self.keep_sources]

    def __post_init__(self):
        # train.py fills in a descriptive name once the model name is known.
        self.output_dir_is_auto = self.output_dir is None
        if isinstance(self.last_layers, str):
            self.last_layers = LAST_LAYER_GROUPS.get(self.last_layers, [self.last_layers])
        if self.coreset and not self.micro_batch_size:
            # The selection pool is one HF batch: one optimizer step = one training_step call.
            self.micro_batch_size = self.per_device_train_batch_size
            self.per_device_train_batch_size *= self.gradient_accumulation_steps
            self.gradient_accumulation_steps = 1
        super().__post_init__()
        check_literals(self)
        self._validate()
