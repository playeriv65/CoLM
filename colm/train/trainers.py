"""CoLM trainers as thin subclasses of the transformers 5.x `Trainer`.

The stock training loop (`Trainer._run_epoch`) fetches `gradient_accumulation_steps`
micro-batches per optimizer step through `get_batch_samples` and then calls
`training_step` on each of them. CoLM hooks into exactly these two places:

* `get_batch_samples` receives the large mini-batch (all micro-batches of one
  update), computes a per-example feature (a zeroth-order estimate of the
  last-layer gradient by default), gathers features and examples on rank 0, runs
  source-wise facility-location selection there, broadcasts the selected indices
  and returns the micro-batches that are actually trained on.
* `training_step` backpropagates those micro-batches with CoLM's loss scaling.

Everything else (optimizer, scheduler, clipping, logging, checkpointing, DDP) is the
unmodified HF implementation.
"""

import logging
import math
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from transformers import Trainer, TrainerCallback
from transformers.utils import is_flash_attn_2_available

from colm.train import attention, packing
from colm.train.custom_phi import DecomposedPhiCausalLM
from colm.train.facility_location import features_needed, get_orders_and_weights
from colm.train.step_timing import StepTimer, StepTimingCallback
from colm.train.utils import collate_fn

logger = logging.getLogger(__name__)

# Key under which the regular SubsetTrainer attaches the per-example loss weight.
SAMPLE_WEIGHT_KEY = "colm_sample_weight"
# Marks a packed training micro-batch (colm.train.packing) with its own loss.
PACKED_KEY = "colm_packed"
INDICES_DIRNAME = "indices"


# ---------------------------------------------------------------------------
# Collectives that degrade to no-ops in a single process.
# ---------------------------------------------------------------------------
def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _all_gather_object(obj) -> list:
    if not _distributed():
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def _gather_object(obj, dst: int = 0) -> list | None:
    if not _distributed():
        return [obj]
    is_dst = dist.get_rank() == dst
    out = [None] * dist.get_world_size() if is_dst else None
    dist.gather_object(obj, object_gather_list=out, dst=dst)
    return out


def _broadcast_object(obj, src: int = 0):
    if not _distributed():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def _barrier():
    if _distributed():
        dist.barrier()


def _to_cpu(inputs: dict) -> dict:
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _nbytes(inputs: dict) -> int:
    return sum(v.nbytes for v in inputs.values() if isinstance(v, torch.Tensor))


def _token_counts(timer: StepTimer, prefix: str, batch: dict) -> None:
    """Samples, padded / real (attention mask) / label tokens of one (CPU) micro-batch."""
    timer.count(f"{prefix}_samples", batch["input_ids"].shape[0])
    timer.count(f"{prefix}_tokens_padded", batch["input_ids"].numel())
    timer.count(f"{prefix}_tokens_real", int(batch["attention_mask"].sum()))
    timer.count(f"{prefix}_label_tokens", int((batch["labels"] != -100).sum()))


def _train_token_counts(timer: StepTimer, batch: dict) -> None:
    """`_token_counts` of a training micro-batch, padded or packed (placeholders: nothing)."""
    if not batch:
        return
    if not batch.get(PACKED_KEY):
        _token_counts(timer, "train", batch)
        return
    timer.count("train_samples", int((batch["position_ids"][0] == 0).sum()))
    timer.count("train_tokens_padded", batch["input_ids"].numel())
    timer.count("train_tokens_real", batch["num_real_tokens"])
    timer.count("train_label_tokens", len(batch["targets"]))


def _split_examples(inputs: dict) -> list[dict]:
    """Slice a collated batch into single-example batches (tensors keep dim 0)."""
    batch_size = len(inputs["input_ids"])
    return [{k: v[i : i + 1] for k, v in inputs.items()} for i in range(batch_size)]


class _RestoreAttention(TrainerCallback):
    """Set the model's attention implementation back at the end of every optimizer step."""

    def __init__(self, trainer, name: str):
        self.trainer, self.name = trainer, name

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer._set_attention(self.name)


class _CoLMTrainerBase(Trainer):
    """Shared behaviour of all CoLM trainers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-micro-batch mean loss divided by the number of accumulation steps, as in
        # transformers 4.43 (no cross-micro-batch token normalisation).
        self.model_accepts_loss_kwargs = False
        self.last_layers = self.args.last_layers
        logger.info(f"Last layers are {self.last_layers}")
        if self.args.fp16:
            self.dtype = torch.float16
        elif self.args.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self._micro_step = 0
        self._flops_per_token = None
        self._select_seconds = 0.0
        self._last_log = None  # (time, global_step) of the previous loss log
        # Opt-in per-phase step timing; a no-op (no synchronize) at level 'off'.
        self._timer = StepTimer(self.args.profile_timing)
        if self._timer.enabled:
            out_dir = self.args.profile_timing_dir or self.args.output_dir
            out_file = os.path.join(
                out_dir,
                f"step_timing-{self.args.run_name}-{self.args.profile_timing}"
                f"-rank{self.args.process_index}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl",
            )
            self.add_callback(
                StepTimingCallback(
                    self._timer,
                    out_file,
                    census_steps=self.args.profile_census_steps,
                    meta={
                        "trainer": type(self).__name__,
                        "model": getattr(self.model.config, "_name_or_path", None),
                        "per_device_train_batch_size": self.args.per_device_train_batch_size,
                        "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                        "small_batch_ratio": self.args.small_batch_ratio,
                        "max_steps": self.args.max_steps,
                        "fp16": self.args.fp16,
                        "bf16": self.args.bf16,
                    },
                )
            )
            logger.info(f"Step timing ({self.args.profile_timing}) -> {out_file}")
        if self.args.save_indices:
            self.indices_path = os.path.join(self.args.output_dir, INDICES_DIRNAME)
            os.makedirs(self.indices_path, exist_ok=True)
            logger.info(f"Save indices to {self.indices_path}")

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Add wall-clock step time, selection time and peak memory to every loss log."""
        if "loss" in logs:
            now, step = time.perf_counter(), self.state.global_step
            if self._last_log is not None and step > self._last_log[1]:
                num_steps = step - self._last_log[1]
                logs["step_time_s"] = round((now - self._last_log[0]) / num_steps, 4)
                logs["select_time_s"] = round(self._select_seconds / num_steps, 4)
            self._last_log = (now, step)
            self._select_seconds = 0.0
            if torch.cuda.is_available():
                logs["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        super().log(logs, start_time)

    # ----- step timing: HF loop pieces around the CoLM hooks ------------------
    def _maybe_log_save_evaluate(self, *args, **kwargs):
        with self._timer.section("log_save_eval"):
            return super()._maybe_log_save_evaluate(*args, **kwargs)

    def _clip_grad_norm(self, model):
        # "optimizer" closes at on_step_end (StepTimingCallback).
        self._timer.start("optimizer")
        with self._timer.section("clip_grad"):
            return super()._clip_grad_norm(model)

    def floating_point_ops(self, inputs):
        with self._timer.section("hf_floating_point_ops"):
            if not self.args.cache_flops:
                return super().floating_point_ops(inputs)
            # HF's formula; num_parameters() walks every parameter, so count them once.
            main_input = getattr(self.model, "main_input_name", "input_ids")
            if main_input not in inputs or not hasattr(self.model, "num_parameters"):
                return 0
            if self._flops_per_token is None:
                self._flops_per_token = 6 * self.model.num_parameters(exclude_embeddings=True)
            return inputs[main_input].numel() * self._flops_per_token

    def _track_num_input_tokens(self, inputs):
        with self._timer.section("hf_track_input_tokens"):
            return super()._track_num_input_tokens(inputs)

    def _model_inputs(self, inputs: dict) -> dict:
        """Drop collator fields (sources, indices, ...) the model forward does not take."""
        if not self.args.remove_unused_columns and not self.args.modify_forward:
            self._set_signature_columns_if_needed()
            return {k: v for k, v in inputs.items() if k in self._signature_columns}
        return {k: v for k, v in inputs.items() if k != "sources"}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(
            model, self._model_inputs(inputs), return_outputs=return_outputs
        )


class CustomTrainer(_CoLMTrainerBase):
    """Full-batch baseline (`data_selection_method=none`)."""

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples, num_items_in_batch = super().get_batch_samples(
            epoch_iterator, num_batches, device
        )
        if self.args.save_indices:
            for inputs in batch_samples:
                current_step = self._micro_step * self.args.world_size + self.args.process_index
                torch.save(
                    inputs["indices"],
                    os.path.join(self.indices_path, f"iter{current_step}_full_indices.pt"),
                )
                self._micro_step += 1
        return batch_samples, num_items_in_batch

    def _clip_grad_norm(self, model):
        grad_norm = super()._clip_grad_norm(model)
        if self.args.assert_finite_grad_norm:
            value = float(grad_norm)
            assert math.isfinite(value), (
                f"Non-finite grad norm {value} at step {self.state.global_step}"
            )
        return grad_norm


class SubsetTrainer(_CoLMTrainerBase):
    """CoLM with one example per micro-batch (`per_device_train_batch_size=1`).

    Each update draws `gradient_accumulation_steps` examples per rank, selects
    `gradient_accumulation_steps * small_batch_ratio` of them per rank and trains on
    each selected example as its own micro-batch, optionally weighted by its
    facility-location cluster size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_args()
        self._set_budgets()
        self.prev_m_t = None
        self.prev_v_t = None
        self._num_train_microbatches = 1
        # Attention implementation per phase (None: never switched); see _eval_mode.
        self._attn_config = self.model.config
        self._train_attn = self._zo_attn = None
        # Same seed on every rank: np is seeded by set_seed before the trainer is built.
        self.zo_random_seed = np.random.randint(1000000000)
        self.named_parameters_to_optim = []
        if self.args.data_selection_unit in ["mezo", "masked_grad"]:
            self.named_parameters_to_optim = [
                (name, param)
                for name, param in self.model.named_parameters()
                if any(substring in name for substring in self.last_layers)
            ]
            assert self.named_parameters_to_optim, (
                f"No layer found for last_layers={self.last_layers}"
            )
            self.param_dim = sum(p.numel() for _, p in self.named_parameters_to_optim)
            logger.info(
                f"Initialized named_parameters_to_optim: {len(self.named_parameters_to_optim)} "
                f"tensors, param_dim={self.param_dim}"
            )

    def _check_args(self):
        assert self.args.per_device_train_batch_size == 1, (
            "SubsetTrainer only supports per_device_train_batch_size = 1 "
            "(use efficient_mezo for larger device batches)."
        )

    def _set_budgets(self):
        args = self.args
        self.new_accumulation_steps = int(
            args.gradient_accumulation_steps
            * args.per_device_train_batch_size
            * args.small_batch_ratio
        )
        assert self.new_accumulation_steps > 0, "small_batch_ratio selects zero examples per rank"
        logger.info(
            f"Large batch per rank = {args.gradient_accumulation_steps * args.per_device_train_batch_size}, "
            f"selected per rank = {self.new_accumulation_steps}"
        )

    def _num_select_per_rank(self, num_batches: int) -> int:
        """Selected examples per rank for an update made of `num_batches` micro-batches."""
        if num_batches == self.args.gradient_accumulation_steps:
            return self.new_accumulation_steps
        return max(
            1,
            int(num_batches * self.args.per_device_train_batch_size * self.args.small_batch_ratio),
        )

    # ----- training ---------------------------------------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        with self._timer.section("train"):
            return self._training_step(model, inputs)

    def _training_step(self, model, inputs):
        t = self._timer
        if not inputs:
            # Placeholder that keeps the HF loop's accumulation count aligned.
            return torch.zeros((), device=self.args.device)
        weight = inputs.pop(SAMPLE_WEIGHT_KEY, None)
        packed = inputs.pop(PACKED_KEY, False)
        with t.fine("model_train"):
            self._train_mode(model)
        with t.section("prepare_inputs"):
            if t.enabled:
                t.count("train_h2d_bytes", _nbytes(inputs))
            inputs = self._prepare_inputs(inputs)
        with t.section("forward"):
            with self.compute_loss_context_manager():
                if packed:
                    loss = self._packed_loss(model, inputs)
                else:
                    loss = self.compute_loss(model, inputs)
        if weight is not None:
            loss = loss * weight
        if self.args.n_gpu > 1:
            loss = loss.mean()
        loss = loss / self._num_train_microbatches
        with t.section("backward"):
            self.accelerator.backward(loss)
        # CoLM logs the loss divided by small_batch_ratio (kept from the original
        # implementation so train/loss curves stay comparable).
        return loss.detach() / self.args.small_batch_ratio

    @staticmethod
    def _packed_loss(model, inputs):
        """Loss of a packed training micro-batch (`_train_microbatches`).

        Logits are computed only at the positions that predict a label (`logits_to_keep`).
        A row holding several sub-batches returns the sum of their token means (sum of the
        token losses of each sub-batch / its label count): exactly what the padded
        sub-batches return one by one.
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            use_cache=False,
            logits_to_keep=inputs["logit_positions"],
            **inputs["attention_kwargs"],
        )
        token_losses = torch.nn.functional.cross_entropy(
            outputs.logits[0].float(), inputs["targets"], reduction="none"
        )
        sums = token_losses.new_zeros(len(inputs["sub_batch_labels"]))
        sums.index_add_(0, inputs["target_sub_batch"], token_losses)
        return (sums / inputs["sub_batch_labels"]).sum()

    # ----- train / eval mode and attention per phase -------------------------
    def _set_attention(self, name: str | None) -> None:
        if name is not None and self._attn_config._attn_implementation != name:
            self._attn_config._attn_implementation = name

    def _train_mode(self, model) -> None:
        # The unwrapped model's flag is authoritative (a DDP wrapper keeps its own).
        if not self.args.lazy_mode_switch or not self.model.training:
            model.train()
        self._set_attention(self._train_attn)
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

    def _eval_mode(self) -> None:
        if not self.args.lazy_mode_switch or self.model.training:
            self.model.eval()
        self._set_attention(self._zo_attn)

    # ----- selection --------------------------------------------------------
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        with self._timer.section("data"):
            batch_samples, _ = super().get_batch_samples(epoch_iterator, num_batches, device)
        if not batch_samples:
            return batch_samples, None
        self._micro_step += len(batch_samples)
        if self._last_log is None:
            # First step: time it from the start of its selection.
            self._last_log = (time.perf_counter(), self.state.global_step)
        start = time.perf_counter()
        with self._timer.section("selection"):
            microbatches = self._select_microbatches(batch_samples)
        self._select_seconds += time.perf_counter() - start
        return microbatches, None

    def _select_microbatches(self, batch_samples: list[dict]) -> list[dict]:
        reps, examples = [], []
        for inputs in batch_samples:
            with self._timer.section("features"):
                rep = self.save_select(inputs)
            # Drop empty / NaN / all-zero features (and zero-valued scalar features).
            if isinstance(rep, (int, float)):
                if rep == 0:
                    continue
            elif rep.nelement() == 0 or torch.isnan(rep).any() or torch.norm(rep).item() == 0:
                continue
            reps.append(rep)
            examples.append(_to_cpu(inputs))

        if all(isinstance(r, int) for r in reps):
            local_reps = torch.tensor(reps, dtype=torch.long)
        elif all(isinstance(r, float) for r in reps):
            local_reps = torch.tensor(reps, dtype=self.dtype)
        else:
            local_reps = torch.stack(reps).cpu()

        num_per_rank = self._num_select_per_rank(len(batch_samples))
        selected_examples, selected_weights = self._select_across_ranks(
            local_reps, examples, num_per_rank
        )
        for example, weight in zip(selected_examples, selected_weights, strict=True):
            example[SAMPLE_WEIGHT_KEY] = weight
        self._num_train_microbatches = len(selected_examples)
        # Placeholders first: the HF loop syncs DDP gradients on the last micro-batch.
        padding = [{} for _ in range(len(batch_samples) - len(selected_examples))]
        return padding + selected_examples

    def _select_across_ranks(self, local_reps, local_examples, num_per_rank):
        """Gather features on rank 0, select there, return this rank's share.

        Returns `(examples, weights)` for this rank, `num_per_rank` of each.
        """
        t = self._timer
        with t.section("gather"):
            with t.section("barrier"):
                _barrier()
            with t.section("all_gather_examples"):
                complete_examples = [ex for exs in _all_gather_object(local_examples) for ex in exs]
            with t.section("gather_reps"):
                gathered_reps = _gather_object(local_reps)
        total = num_per_rank * self.args.world_size

        selection = None
        if self.args.process_index == 0:
            with t.section("main"):
                with t.section("reps_h2d"):
                    all_reps = torch.cat(gathered_reps, dim=0)
                    t.count("sel_main_h2d_bytes", all_reps.nbytes)
                    all_reps = all_reps.to(self.args.device)
                selection = self._select_on_main(all_reps, complete_examples, total)
        with t.section("scatter"):
            with t.section("broadcast"):
                selected_idx, selected_weights = _broadcast_object(selection)
            with t.section("slice"):
                rank = self.args.process_index
                mine = slice(num_per_rank * rank, num_per_rank * (rank + 1))
                examples = [dict(complete_examples[i]) for i in selected_idx[mine]]
                weights = torch.tensor(selected_weights[mine], dtype=torch.float32).to(self.dtype)
        return examples, weights

    def _select_on_main(self, all_reps, complete_examples, total):
        """Coreset selection over the gathered large batch (rank 0 only).

        Returns `(selected_idx, weights)` as python lists of length `total`, indices
        into `complete_examples`.
        """
        args, t = self.args, self._timer
        sampling_indices = np.arange(len(complete_examples))
        max_samples = total

        # Sources listed in keep_sources are always trained on and excluded from selection.
        list_idx_keep = []
        if len(args.keep_sources) > 0:
            with t.section("keep_sources"):
                with t.fine("scan"):
                    include_in_selection = []
                    for idx in sampling_indices:
                        if complete_examples[idx]["sources"][0] in args.keep_sources:
                            include_in_selection.append(False)
                            list_idx_keep.append(int(idx))
                        else:
                            include_in_selection.append(True)
                    max_samples -= len(list_idx_keep)
                with t.fine("log"):
                    logger.info(
                        f"Exclude {len(list_idx_keep)} examples from selection. Select "
                        f"{max_samples} from the remaining {sum(include_in_selection)} examples."
                    )
                with t.fine("filter_rows"):
                    all_reps = all_reps[include_in_selection]
                    sampling_indices = sampling_indices[include_in_selection]
            t.count("sel_kept_by_source", len(list_idx_keep))
            t.count("sel_fl_candidates", len(sampling_indices))
            t.count("sel_fl_budget", max_samples)

        with t.section("transform"):
            if all_reps.dim() == 1:
                # Scalar features (completion_length, length_loss_weighted) as 1-D points.
                all_reps = all_reps.unsqueeze(1)
            with t.fine("square"):
                all_reps_squared = torch.square(all_reps)
            all_reps = self._transform_reps(all_reps)

        m_t = v_t = None
        if args.mezo_optim == "adam":
            with t.section("adam"):
                all_reps, m_t, v_t = self._adam_update(all_reps, all_reps_squared)

        if args.source_wise_selection != "none":
            with t.section("source_list"):
                with t.fine("scan"):
                    source_list = []
                    for idx in sampling_indices:
                        source = complete_examples[idx]["sources"][0]
                        if isinstance(source, torch.Tensor):
                            source = source.item()
                        source_list.append(source)
                with t.fine("log"):
                    logger.info(f"Source count: {sorted(Counter(source_list).items())}")
        else:
            source_list = None

        if args.data_selection_unit not in ["completion_length", "length_loss_weighted"]:
            with t.section("topk_mask"):
                if args.mezo_topk == "random":
                    ranked_indices = torch.randperm(len(all_reps[0]))[: args.zo_dim]
                    all_reps = all_reps[:, ranked_indices]
                else:
                    all_reps = self.select_masking(all_reps, source_list)

        if max_samples > 0:
            with t.section("facility_location"):
                fl_idx, fl_weights = self.select_data(
                    all_reps, max_samples=max_samples, source_list=source_list
                )
            with t.section("index_weights"):
                # MeZO keeps its own Adam moments: the mean over the selected subset.
                if args.mezo_optim == "adam" and "grad" not in args.data_selection_unit:
                    with t.fine("adam_state"):
                        self.prev_m_t = m_t[fl_idx].mean(dim=0).detach()
                        self.prev_v_t = v_t[fl_idx].mean(dim=0).detach()
                with t.fine("index_map"):
                    selected_idx = list_idx_keep + sampling_indices[fl_idx].tolist()
                    selected_weights = [1.0] * len(list_idx_keep) + fl_weights.tolist()
        else:
            # More kept examples than the budget: train on the first `total` of them.
            selected_idx = list_idx_keep[:total]
            selected_weights = [1.0] * len(selected_idx)
        assert len(selected_idx) == total, f"selected {len(selected_idx)} != budget {total}"

        if args.save_indices:
            current_step = self._micro_step - 1
            for name, idx_list in [
                ("full", range(len(complete_examples))),
                ("sampling", sampling_indices),
                ("selected", selected_idx),
            ]:
                self.extract_and_save_original_indices(
                    complete_examples,
                    idx_list,
                    os.path.join(self.indices_path, f"iter{current_step}_{name}_indices.pt"),
                )

        if "weighted" in args.data_selection_method:
            weights = [w * args.small_batch_ratio for w in selected_weights]
        else:
            weights = [1.0] * len(selected_idx)
        return [int(i) for i in selected_idx], weights

    def _transform_reps(self, all_reps):
        if all_reps.dtype == torch.long:
            return all_reps
        args = self.args
        with self._timer.fine("mean_norm"):
            all_reps_norm = torch.norm(torch.mean(all_reps, dim=0), p=2)
        if args.mezo_transform == "self_normalize":
            all_reps = all_reps / torch.norm(all_reps, p=2, dim=1, keepdim=True)
        elif args.mezo_transform == "normalize":
            all_reps = all_reps / all_reps_norm
        elif args.mezo_transform == "clip_full":
            clip_coef = args.max_grad_norm / all_reps_norm
            if clip_coef < 1:
                all_reps = all_reps * clip_coef
        elif args.mezo_transform == "clip_last":
            # Approximate the last-layer share of the norm by dividing by the number of layers.
            clip_coef = args.max_grad_norm / (all_reps_norm / 32)
            if clip_coef < 1:
                all_reps = all_reps * clip_coef
        return all_reps

    def _adam_update(self, all_reps, all_reps_squared):
        """Replace per-example gradients by the Adam update they would produce."""
        args = self.args
        if "grad" in args.data_selection_unit:
            # Back-propagated gradients: take the moments from the real optimizer.
            state = self.optimizer.state[self.named_parameters_to_optim[0][1]]
            if "exp_avg" in state:
                prev_m_t = torch.cat(
                    [
                        self.optimizer.state[p]["exp_avg"].flatten()
                        for _, p in self.named_parameters_to_optim
                    ]
                )
                prev_v_t = torch.cat(
                    [
                        self.optimizer.state[p]["exp_avg_sq"].flatten()
                        for _, p in self.named_parameters_to_optim
                    ]
                )
            else:
                prev_m_t = torch.zeros_like(all_reps[0])
                prev_v_t = torch.zeros_like(all_reps_squared[0])
        else:
            if self.prev_m_t is None or self.prev_v_t is None:
                self.prev_m_t = torch.zeros_like(all_reps[1])
                self.prev_v_t = torch.zeros_like(all_reps_squared[1])
            prev_m_t, prev_v_t = self.prev_m_t, self.prev_v_t
        t = self._timer
        with t.fine("m"):
            m_t = args.adam_beta1 * prev_m_t + (1 - args.adam_beta1) * all_reps
        with t.fine("v"):
            v_t = args.adam_beta2 * prev_v_t + (1 - args.adam_beta2) * all_reps_squared
        with t.fine("bias_correction"):
            m_hat = m_t / (1 - args.adam_beta1 ** (self.state.global_step + 1))
            v_hat = v_t / (1 - args.adam_beta2 ** (self.state.global_step + 1))
        with t.fine("division"):
            update = m_hat / (torch.sqrt(v_hat) + args.adam_epsilon)
        return update, m_t, v_t

    @staticmethod
    def extract_and_save_original_indices(list_inputs, list_idx, out_file):
        list_idx = set(int(i) for i in list_idx)
        extracted = []
        for idx, inputs in enumerate(list_inputs):
            if idx in list_idx:
                extracted.extend(inputs["indices"])
        torch.save(extracted, out_file)

    def select_masking(self, all_reps, source_list, per_source=True):
        """Keep `zo_dim` coordinates of the features, chosen per source."""
        if (source_list is None) or (not per_source):
            source_list = np.zeros(all_reps.shape[0], dtype=np.int32)
        elif isinstance(source_list, list):
            source_list = np.array(source_list)

        t = self._timer
        with t.fine("alloc"):
            masked_reps = torch.zeros(
                (all_reps.shape[0], self.args.zo_dim), dtype=all_reps.dtype
            ).to(all_reps.device)
        for source in np.unique(source_list):
            with t.fine("source_indices"):
                source_indices = np.where(source_list == source)[0]
            with t.fine("gather_rows"):
                source_all_reps = all_reps[source_indices]

            if self.args.mezo_selection == "weight":
                weights = torch.cat(
                    [p.flatten() for _, p in self.named_parameters_to_optim]
                ).flatten()
                if not torch.all(weights == 0):
                    mean_reps = torch.abs(weights.flatten())
                else:
                    mean_reps = torch.abs(torch.mean(source_all_reps, dim=0))
            else:
                with t.fine("mean_abs"):
                    mean_reps = torch.abs(torch.mean(source_all_reps, dim=0))

            with t.fine("argsort"):
                ranked_indices = self._rank_coordinates(mean_reps)
            with t.fine("gather_columns"):
                masked_reps[source_indices] = source_all_reps[:, ranked_indices]
        return masked_reps

    def _rank_coordinates(self, mean_reps):
        """Indices of the `zo_dim` kept coordinates by `mezo_topk`."""
        if self.args.mezo_topk == "smallest":
            return torch.argsort(mean_reps)[: self.args.zo_dim]
        elif self.args.mezo_topk == "largest":
            return torch.argsort(mean_reps, descending=True)[: self.args.zo_dim]
        elif self.args.mezo_topk == "sampling":
            index_probs = mean_reps.cpu().numpy().astype("float64")
            index_probs = index_probs / index_probs.sum()
            return np.random.choice(
                len(mean_reps), size=self.args.zo_dim, replace=False, p=index_probs
            )
        elif self.args.mezo_topk == "largest_smallest":
            return torch.cat(
                (
                    torch.argsort(mean_reps)[: (self.args.zo_dim // 2)],
                    torch.argsort(mean_reps, descending=True)[: (self.args.zo_dim // 2)],
                )
            )
        raise ValueError(f"Unknown mezo_topk {self.args.mezo_topk}")

    def select_data(self, reps, max_samples, source_list=None):
        """Facility-location selection; returns (indices, cluster-size weights)."""
        return get_orders_and_weights(
            max_samples,
            reps,
            metric=self.args.facility_similarity,
            y=source_list,
            per_class_start=self.args.num_per_class_start,
            strategy=self.args.source_wise_selection,
            timer=self._timer,
        )

    # ----- per-example features --------------------------------------------
    def save_select(self, inputs):
        """Feature of one single-example micro-batch."""
        unit = self.args.data_selection_unit
        model = self.model
        if unit == "rep":
            prepared = self._prepare_inputs(inputs)
            input_ids = prepared["input_ids"]
            attention_mask = prepared["attention_mask"]
            with torch.inference_mode():
                hidden_states = model(
                    input_ids,
                    labels=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                ).hidden_states
            ids = torch.arange(len(input_ids), device=input_ids.device)
            pos = attention_mask.sum(dim=1) - 1
            # Last-token hidden state of the single example, as a flat feature vector.
            return hidden_states[-1][ids, pos].flatten()
        if unit == "mezo":
            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(inputs)
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = self.zo_forward(inputs)
            projected_grad = ((loss1 - loss2) / (2 * self.args.mezo_eps)).item()
            self.zo_perturb_parameters(scaling_factor=1)
            torch.manual_seed(self.zo_random_seed)
            res_list = []
            for _, param in self.named_parameters_to_optim:
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
                grad_update = projected_grad * z
                if self.args.mezo_selection == "weight_grad" and not torch.all(param.data == 0):
                    grad_update = grad_update * param.data
                res_list.append(grad_update.flatten())
            return torch.cat(res_list, dim=0).flatten()
        if unit == "masked_grad":
            # Exact per-example gradient of the last-layer parameters.
            model.train()
            prepared = self._prepare_inputs(dict(inputs))
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, prepared)
            params = [p for _, p in self.named_parameters_to_optim]
            grads = torch.autograd.grad(loss / self.new_accumulation_steps, params)
            res_list = []
            for grad, param in zip(grads, params, strict=True):
                grad_update = grad.detach()
                if self.args.mezo_selection == "weight_grad" and not torch.all(param.data == 0):
                    grad_update = grad_update * param.data
                res_list.append(grad_update.flatten())
            return torch.cat(res_list, dim=0).flatten()
        if unit == "completion_length":
            return inputs["completion_lengths"][0]
        if unit == "length_loss_weighted":
            completion_lengths = max(inputs["attention_mask"].sum(dim=1).max().item(), 1)
            with torch.no_grad():
                prepared = self._prepare_inputs(self._model_inputs(inputs))
                losses = model(**prepared).loss.view(-1)
                nan_mask = torch.isnan(losses)
                if nan_mask.any():
                    logger.warning(
                        f"NaN losses detected for {nan_mask.sum().item()} of {len(losses)} samples."
                    )
                    losses = torch.where(nan_mask, torch.zeros_like(losses), losses)
                losses = losses.tolist()
                assert len(losses) == 1
            res = completion_lengths * losses[0] / 10
            return 0.0 if (math.isnan(res) or math.isinf(res)) else max(res, 1e-8)
        raise ValueError(f"Unknown data_selection_unit {unit}")

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """theta <- theta + scaling_factor * eps * z with z ~ N(0, I) regenerated from the seed."""
        t = self._timer
        with t.fine("seed"):
            torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for _, param in self.named_parameters_to_optim:
            with t.fine("normal"):
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
            with t.fine("add"):
                param.data = param.data + scaling_factor * z * self.args.mezo_eps

    def zo_forward(self, inputs):
        """Loss without gradient and without dropout."""
        self._eval_mode()
        with torch.inference_mode():
            prepared = self._prepare_inputs(dict(inputs))
            with self.compute_loss_context_manager():
                loss = self.compute_loss(self.model, prepared)
        return loss.detach()


class SubsetTrainerEfficient(SubsetTrainer):
    """CoLM with batched last-layer MeZO (the paper's efficient implementation).

    Per update each rank draws `gradient_accumulation_steps` micro-batches of
    `per_device_train_batch_size` examples. The first L-1 layers run once per
    micro-batch; only the last layer is re-run for the +eps/-eps perturbations of
    its LoRA-B matrix. Each rank then trains on `gradient_accumulation_steps`
    micro-batches of `per_device_train_batch_size * small_batch_ratio` selected
    examples, re-collated from the gathered large batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.named_parameters_to_optim) == 1, (
            "Efficient MeZO perturbs exactly one tensor; got "
            f"{[n for n, _ in self.named_parameters_to_optim]}"
        )
        base_model = (
            self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        )
        self.decomposer = DecomposedPhiCausalLM(base_model, timer=self._timer)
        self.pad_token_id = self.processing_class.pad_token_id
        args = self.args
        self._zo_new_path = (
            args.skip_unused_features or args.zo_packing or args.zo_label_positions_only
        )
        self._resolve_attention(base_model)

    def _resolve_attention(self, base_model) -> None:
        """Attention implementation of the selection and of the training forward.

        The model config is shared by every layer, so the implementation is switched with the
        train/eval mode at each phase boundary (`_train_mode` / `_eval_mode`).
        """
        args = self.args
        self._attn_config = base_model.config
        original = self._attn_config._attn_implementation
        self._zo_attn = original
        if args.zo_packing and args.zo_attn_implementation == attention.NAME:
            self._zo_attn = attention.register()
        choice = args.train_attn_implementation
        if choice == "auto":
            capable = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)
            choice = "flash_attention_2" if capable and is_flash_attn_2_available() else "model"
        self._train_attn = original if choice == "model" else choice
        if self._train_attn != original:
            # Validates availability (raises if flash-attn is missing), then restores.
            base_model.set_attn_implementation(self._train_attn)
            base_model.set_attn_implementation(original)
        if self._zo_attn == self._train_attn == original:
            self._zo_attn = self._train_attn = None  # never switched
        else:
            # Outside selection / training (callbacks such as the eval loss, saving) the model
            # has its own attention again; runs before callbacks added after the trainer.
            self.add_callback(_RestoreAttention(self, original))
        logger.info(
            f"Attention: model {original}, selection {self._zo_attn or original}, "
            f"training {self._train_attn or original}"
        )

    def _check_args(self):
        args = self.args
        assert args.data_selection_unit == "mezo", "The efficient trainer only supports MeZO."
        assert args.mezo_transform == "none", "The efficient trainer does not apply mezo_transform."
        assert "weighted" not in args.data_selection_method, (
            "The efficient trainer trains unweighted."
        )

    def _set_budgets(self):
        args = self.args
        self.new_accumulation_steps = args.gradient_accumulation_steps
        self.new_bs = int(args.per_device_train_batch_size * args.small_batch_ratio)
        assert self.new_bs > 0, "per_device_train_batch_size * small_batch_ratio must be >= 1"
        self.num_orig = args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.num_select = self.new_accumulation_steps * self.new_bs
        logger.info(
            f"Large batch per rank = {self.num_orig}, selected per rank = {self.num_select} "
            f"in {self.new_accumulation_steps} micro-batches of {self.new_bs}"
        )

    def _num_select_per_rank(self, num_batches: int) -> int:
        return num_batches * self.new_bs

    def _select_microbatches(self, batch_samples: list[dict]) -> list[dict]:
        t = self._timer
        if self._zo_new_path:
            with t.section("examples_to_cpu"):
                with t.fine("to_cpu"):
                    cpu_batches = [_to_cpu(inputs) for inputs in batch_samples]
                with t.fine("split"):
                    examples = [ex for inputs in cpu_batches for ex in _split_examples(inputs)]
            num_per_rank = self._num_select_per_rank(len(batch_samples))
            with t.section("features"):
                reps = self.zo_features(cpu_batches, num_per_rank)
        else:
            reps = []
            for inputs in batch_samples:
                with t.section("features"):
                    reps.append(self.save_select(inputs))
            with t.section("reps_cat"):
                reps = torch.cat(reps, dim=0).float()
        with t.section("reps_d2h"):
            t.count("sel_reps_d2h_bytes", reps.nbytes)
            reps = reps.cpu()
        if not self._zo_new_path:
            with t.section("examples_to_cpu"):
                with t.fine("to_cpu"):
                    cpu_batches = [_to_cpu(inputs) for inputs in batch_samples]
                with t.fine("split"):
                    examples = [ex for inputs in cpu_batches for ex in _split_examples(inputs)]
            num_per_rank = self._num_select_per_rank(len(batch_samples))
        if t.enabled:
            with t.section("token_stats"):
                for batch in cpu_batches:
                    _token_counts(t, "sel", batch)
                    t.count("sel_examples_d2h_bytes", _nbytes(batch))
        selected_examples, _ = self._select_across_ranks(reps, examples, num_per_rank)
        with t.section("recollate"):
            microbatches = self._train_microbatches(selected_examples, len(batch_samples))
        if t.enabled:
            with t.section("token_stats"):
                for batch in microbatches:
                    _train_token_counts(t, batch)
        return microbatches

    # ----- training micro-batches ---------------------------------------------
    def _train_microbatches(self, selected_examples: list[dict], num_batches: int) -> list[dict]:
        """The HF loop's `num_batches` micro-batches: selected sub-batches of `new_bs`.

        `train_packing=none` re-collates every sub-batch with padding (the original);
        `sub_batch` packs each sub-batch into one row; `merged` packs whole sub-batches into
        as few forwards as `train_pack_max_tokens` allows, preceded by empty placeholders so
        the HF loop still counts `num_batches` (and syncs DDP on the last one).
        """
        sub_batches = [
            selected_examples[i : i + self.new_bs]
            for i in range(0, len(selected_examples), self.new_bs)
        ]
        self._num_train_microbatches = len(sub_batches)
        mode = self.args.train_packing
        if mode == "none":
            return [collate_fn(sub, self.pad_token_id) for sub in sub_batches]
        if mode == "sub_batch":
            groups = [[i] for i in range(len(sub_batches))]
        else:
            sizes = [
                sum(packing.real_length(ex["attention_mask"][0]) for ex in sub)
                for sub in sub_batches
            ]
            groups = packing.rows_by_token_budget(sizes, self.args.train_pack_max_tokens)
        packed = [self._pack_sub_batches([sub_batches[i] for i in g]) for g in groups]
        placeholders = [{} for _ in range(num_batches - len(packed))]
        return placeholders + packed

    def _pack_sub_batches(self, sub_batches: list[list[dict]]) -> dict:
        """One packed row of whole sub-batches, with the sub-batch of every target."""
        sequences, sub_batch_of, num_labels, indices = [], [], [], []
        for b, sub in enumerate(sub_batches):
            indices += [ex["indices"][0] for ex in sub]
            seqs = [packing.unpad(ex, 0) for ex in sub]
            # 0 labels gives 0 / 0 = NaN, the token mean of such a padded sub-batch too.
            num_labels.append(sum(int((lab[1:] != packing.IGNORE_INDEX).sum()) for _, lab in seqs))
            sequences += seqs
            sub_batch_of += [b] * len(seqs)
        row = packing.pack(sequences, self.pad_token_id)
        targets = packing.shift_left(row.labels)[0]
        logit_positions = (targets != packing.IGNORE_INDEX).nonzero(as_tuple=True)[0]
        segment = row.segment_ids[0, logit_positions]
        return {
            PACKED_KEY: True,
            "input_ids": row.input_ids,
            "position_ids": row.position_ids,
            "labels": row.labels,
            "logit_positions": logit_positions,
            "targets": targets[logit_positions],
            "target_sub_batch": torch.tensor(sub_batch_of, dtype=torch.long)[segment],
            "sub_batch_labels": torch.tensor(num_labels, dtype=torch.float32),
            "attention_kwargs": row.attention_kwargs("cpu"),
            "num_real_tokens": row.num_real_tokens,
            "indices": indices,
        }

    # ----- MeZO features, planned per step -------------------------------------
    def zo_features(self, cpu_batches: list[dict], num_per_rank: int) -> torch.Tensor:
        """Per-example MeZO features `g_i * z` of the large batch, shape [N, numel].

        Only examples whose feature can influence the selection are forwarded
        (`skip_unused_features`); the others get g_i = 0. Each example's loss keeps the
        original divisor (padded length of its micro-batch - 1, F2), however it is batched.
        """
        t, args = self._timer, self.args
        param = self.named_parameters_to_optim[0][1]
        with t.section("plan"):
            sources = [
                int(s.item() if isinstance(s, torch.Tensor) else s)
                for batch in cpu_batches
                for s in batch["sources"]
            ]
            needed = self._features_needed(sources, num_per_rank)
            groups = self._zo_groups(cpu_batches, needed)
        if t.enabled:
            t.count("sel_zo_examples", int(needed.sum()))
            t.count("sel_zo_tokens", sum(g["input_ids"].numel() for g in groups))
            t.count("sel_zo_logit_positions", sum(g["targets"].numel() for g in groups))
        projected_grads = torch.zeros(len(sources), dtype=torch.float32, device=param.device)
        for group in groups:
            group_grads = self._zo_projected_grads(group)
            projected_grads = projected_grads.to(group_grads.dtype)  # the loss dtype (fp32)
            projected_grads[group["examples"]] = group_grads
        with t.section("feature"):
            # Also the RNG epilogue of the per-micro-batch path (seed, then one z draw): the
            # training dropout stream starts from here even if nothing was forwarded.
            with t.fine("z_seed"):
                torch.manual_seed(self.zo_random_seed)
            with t.fine("z_normal"):
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
            with t.fine("outer_product"):
                grad_updates = projected_grads.view(-1, *([1] * param.dim())) * z.unsqueeze(0)
            if args.mezo_selection == "weight_grad" and not torch.all(param.data == 0):
                grad_updates = grad_updates * param.data.unsqueeze(0)
            return grad_updates.view(len(sources), -1).float()

    def _features_needed(self, sources: list[int], num_per_rank: int) -> np.ndarray:
        args = self.args
        if not args.skip_unused_features:
            return np.ones(len(sources), dtype=bool)
        per_rank = _all_gather_object(sources)
        offset = sum(len(r) for r in per_rank[: args.process_index])
        needed = features_needed(
            [s for r in per_rank for s in r],
            total=num_per_rank * args.world_size,
            keep_sources=args.keep_sources,
            strategy=args.source_wise_selection,
            per_class_start=args.num_per_class_start,
            need_selected=args.mezo_optim == "adam",
            per_source_rng=args.mezo_topk == "sampling",
        )
        return needed[offset : offset + len(sources)]

    def _zo_groups(self, cpu_batches: list[dict], needed: np.ndarray) -> list[dict]:
        """Forward groups of the needed examples: one packed group, or one per micro-batch."""
        locations, divisors = [], []
        for b, batch in enumerate(cpu_batches):
            batch_size, width = batch["input_ids"].shape
            locations += [(b, r) for r in range(batch_size)]
            divisors += [width - 1] * batch_size
        divisors = torch.tensor(divisors, dtype=torch.float32)
        index = np.where(needed)[0]
        if len(index) == 0:
            return []
        groups = []
        if self.args.zo_packing:
            sequences = [packing.unpad(cpu_batches[b], r) for b, r in (locations[i] for i in index)]
            rows = packing.rows_by_token_budget(
                [len(ids) for ids, _ in sequences], self.args.zo_pack_max_tokens
            )
            row = packing.pack(sequences, self.pad_token_id, rows)
            kwargs = row.attention_kwargs("cpu") if self._zo_attn == attention.NAME else {}
            groups.append(
                self._zo_group(
                    index,
                    row.input_ids,
                    None,
                    row.position_ids,
                    row.labels,
                    row.segment_ids,
                    divisors,
                    kwargs,
                )
            )
        else:
            offset = 0
            for batch in cpu_batches:
                batch_size = len(batch["input_ids"])
                rows = [r for r in range(batch_size) if needed[offset + r]]
                if rows:
                    sel = torch.tensor(rows)
                    labels = batch["labels"][sel]
                    segment = torch.arange(len(rows)).unsqueeze(1).expand_as(labels)
                    groups.append(
                        self._zo_group(
                            offset + np.array(rows),
                            batch["input_ids"][sel],
                            batch["attention_mask"][sel],
                            None,
                            labels,
                            segment,
                            divisors,
                            {},
                        )
                    )
                offset += batch_size
        return groups

    def _zo_group(
        self, examples, input_ids, attention_mask, position_ids, labels, segment, divisors, kwargs
    ) -> dict:
        """Device tensors of one forward group; segment ids index `examples`, -1 = no example."""
        targets = packing.shift_left(labels).reshape(-1)
        segment = segment.reshape(-1)
        if self.args.zo_label_positions_only:
            positions = (targets != packing.IGNORE_INDEX).nonzero(as_tuple=True)[0]
            targets, segment = targets[positions], segment[positions]
        else:
            positions = None
            targets = targets.view(labels.shape)
            # Positions without an example (row tails) go to a dropped extra slot.
            segment = torch.where(segment < 0, len(examples), segment)
        group = {
            "examples": torch.as_tensor(examples, dtype=torch.long),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "targets": targets,
            "positions": positions,
            "segment": segment,
            "divisor": divisors[torch.as_tensor(examples, dtype=torch.long)],
            "attention_kwargs": kwargs,
        }
        return self._prepare_input(group)

    def _zo_projected_grads(self, group: dict) -> torch.Tensor:
        """(L(theta + eps z) - L(theta - eps z)) / 2 eps for the examples of one group."""
        t = self._timer
        with t.section("forward_till_penultimate"):
            with t.fine("model_eval"):
                self._eval_mode()
            with torch.inference_mode():
                intermediate = self.decomposer.forward_till_penultimate(
                    input_ids=group["input_ids"],
                    attention_mask=group["attention_mask"],
                    position_ids=group["position_ids"],
                    attention_kwargs=group["attention_kwargs"],
                )
        with t.section("perturb_plus"):
            self.zo_perturb_parameters(scaling_factor=1)
        with t.section("final_layer_plus"):
            loss1 = self._zo_group_losses(group, intermediate)
        with t.section("perturb_minus"):
            self.zo_perturb_parameters(scaling_factor=-2)
        with t.section("final_layer_minus"):
            loss2 = self._zo_group_losses(group, intermediate)
        with t.section("feature"):
            with t.fine("projected_grad"):
                projected_grads = (loss1 - loss2) / (2 * self.args.mezo_eps)
        with t.section("restore"):
            self.zo_perturb_parameters(scaling_factor=1)
        return projected_grads

    def _zo_group_losses(self, group: dict, intermediate: dict) -> torch.Tensor:
        """Per-example loss: sum of its token losses / original divisor."""
        num = len(group["examples"])
        with torch.inference_mode():
            token_losses = self.decomposer.final_layer_token_losses(
                intermediate, group["targets"], group["positions"]
            )
            with self._timer.fine("per_example_sum"):
                sums = token_losses.new_zeros(num + 1)
                sums.index_add_(0, group["segment"], token_losses.reshape(-1))
                return sums[:num] / group["divisor"]

    def save_select(self, inputs):
        """Per-example MeZO estimates of the last-layer LoRA-B gradient, shape [B, numel]."""
        t = self._timer
        param = self.named_parameters_to_optim[0][1]
        with t.section("forward_till_penultimate"):
            intermediate = self.zo_forward_till_penultimate(inputs)
        with t.section("perturb_plus"):
            self.zo_perturb_parameters(scaling_factor=1)
        with t.section("final_layer_plus"):
            loss1 = self.zo_forward_final_layer(inputs["labels"], intermediate)
        with t.section("perturb_minus"):
            self.zo_perturb_parameters(scaling_factor=-2)
        with t.section("final_layer_minus"):
            loss2 = self.zo_forward_final_layer(inputs["labels"], intermediate)
        with t.section("feature"):
            with t.fine("projected_grad"):
                projected_grads = (loss1 - loss2) / (2 * self.args.mezo_eps)
        with t.section("restore"):
            self.zo_perturb_parameters(scaling_factor=1)

        with t.section("feature"):
            with t.fine("z_seed"):
                torch.manual_seed(self.zo_random_seed)
            with t.fine("z_normal"):
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
            with t.fine("outer_product"):
                grad_updates = projected_grads.view(-1, *([1] * param.dim())) * z.unsqueeze(0)
            if self.args.mezo_selection == "weight_grad" and not torch.all(param.data == 0):
                grad_updates = grad_updates * param.data.unsqueeze(0)
            return grad_updates.view(len(projected_grads), -1)

    def zo_forward_till_penultimate(self, inputs):
        t = self._timer
        with t.fine("model_eval"):
            self._eval_mode()
        with torch.inference_mode():
            with t.fine("prepare_inputs"):
                prepared = self._prepare_inputs(dict(inputs))
            return self.decomposer.forward_till_penultimate(
                input_ids=prepared["input_ids"], attention_mask=prepared["attention_mask"]
            )

    def zo_forward_final_layer(self, labels, intermediate):
        with self._timer.fine("model_eval"):
            self._eval_mode()
        with torch.inference_mode():
            loss, _ = self.decomposer.forward_final_layer(
                intermediate, labels=labels, per_sample_loss=True
            )
        return loss.detach()
