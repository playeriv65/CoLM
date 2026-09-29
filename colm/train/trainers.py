"""CoLM trainers on the stock transformers 5.x `Trainer`.

The baseline (`CustomTrainer`) is the plain Trainer. The coreset trainers treat the selection
pool of one optimizer step (per-device batch x gradient accumulation examples) as ONE
Hugging Face batch (`gradient_accumulation_steps=1`, see `TrainingArguments`) and override only
`training_step`: extract a feature per example, select on rank 0, broadcast, then forward and
backward the selected sub-batches. Optimizer step, clipping, scheduler, logging and checkpointing
are the unmodified Trainer.
"""

import contextlib
import json
import logging
import os
import time

import numpy as np
import torch
from transformers import Trainer

from colm.data.superglue import classification_loss
from colm.selection.batching import PackedBatching
from colm.selection.features import build_extractor
from colm.selection.packing import META
from colm.selection.pool import (
    all_gather_object,
    broadcast_object,
    gather_object,
    index_of,
    source_of,
)
from colm.selection.select import CoresetSelector
from colm.selection.zo import zo_parameters
from colm.train import attention
from colm.train.memory import MemoryMeter
from colm.train.step_timing import StepTimer, StepTimingCallback

logger = logging.getLogger(__name__)

INDICES_DIRNAME = "indices"


class _Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.n_gpu > 1:
            raise ValueError("one process per GPU is required: launch with torchrun")
        if self.args.coreset:
            # The trainer scales the loss of every batch itself (the share of the step's token
            # mean, see batching.py); the baseline keeps the transformers loss normalisation.
            self.model_accepts_loss_kwargs = False
        self.memory = MemoryMeter()
        self._select_seconds = 0.0
        self._last_log = None  # (time, global_step) of the previous loss log
        self._timer = StepTimer(self.args.profile_timing)
        if self._timer.enabled:
            out_dir = self.args.profile_timing_dir or self.args.output_dir
            out_file = os.path.join(
                out_dir,
                f"step_timing-{self.args.run_name}-{self.args.profile_timing}"
                f"-rank{self.args.process_index}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl",
            )
            meta = {
                "trainer": type(self).__name__,
                "model": getattr(self.model.config, "_name_or_path", None),
                "per_device_train_batch_size": self.args.per_device_train_batch_size,
                "micro_batch_size": self.args.micro_batch_size,
                "small_batch_ratio": self.args.small_batch_ratio,
                "max_steps": self.args.max_steps,
                "fp16": self.args.fp16,
                "bf16": self.args.bf16,
            }
            self.add_callback(
                StepTimingCallback(self._timer, out_file, self.args.profile_census_steps, meta=meta)
            )
            logger.info(f"Step timing ({self.args.profile_timing}) -> {out_file}")

    def describe(self) -> dict:
        return {"attn_implementation": self.model.config._attn_implementation}

    def _require_varlen(self) -> None:
        """Packed sequences would attend to each other without a kernel that reads cu_seq_lens."""
        implementation = self.model.config._attn_implementation
        if implementation != attention.NAME:
            raise ValueError(
                f"packed inputs need attn_implementation={attention.NAME}, not {implementation}"
            )

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
            logs.update(MemoryMeter.summary(self.memory.gather()))
        super().log(logs, start_time)

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.memory.start()
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.memory.stop("train")
        return loss

    def check_replicas(self) -> list[float]:
        """Checksum of the trainable parameters on every rank (a collective); raises if they differ."""
        total = sum(
            float(p.detach().double().abs().sum())
            for p in self.model.parameters()
            if p.requires_grad
        )
        sums = all_gather_object(total)
        if len(set(sums)) != 1:
            raise RuntimeError(f"the ranks trained different weights: {sums}")
        logger.info(f"Trainable weights identical on all {len(sums)} ranks (checksum {total!r})")
        return sums

    def save_memory_report(self) -> dict[str, float]:
        """Peaks of every rank over the whole run -> `memory.json` (a collective: call on all ranks)."""
        ranks = self.memory.gather(run=True)
        summary = MemoryMeter.summary(ranks)
        if self.is_world_process_zero() and summary:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, "memory.json"), "w") as f:
                json.dump({"ranks": ranks, **summary}, f, indent=1)
        return summary

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v for k, v in inputs.items() if k != META}
        if "cu_seq_lens_q" in inputs:
            self._require_varlen()
            inputs["use_cache"] = False  # a cache would end the packed-batch detection
        if "num_options" in inputs:  # classification-style SuperGLUE tasks
            logits = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).logits
            return classification_loss(logits, inputs)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


class CustomTrainer(_Trainer):
    """Full-batch baseline (`data_selection_method=none`)."""

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
        if self.args.save_indices:
            os.makedirs(os.path.join(self.args.output_dir, INDICES_DIRNAME), exist_ok=True)
            for i, batch in enumerate(b for b in batch_samples if META in b):
                name = f"iter{self.state.global_step}_{i}_rank{self.args.process_index}_full_indices.pt"
                torch.save(
                    batch[META]["indices"].tolist(),
                    os.path.join(self.args.output_dir, INDICES_DIRNAME, name),
                )
        return batch_samples, num_items


class CoresetTrainer(_Trainer):
    """CoLM: train on the part of each selection pool that a coreset of the features picks."""

    drop_invalid = False  # drop examples whose feature is empty / NaN / zero before the gather

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = self.args
        if not args.coreset:
            raise ValueError("coreset trainers need data_selection_method != none")
        self._check_args()
        self._require_varlen()
        # The same seed on every rank: numpy is seeded by set_seed before the trainer is built.
        self.zo_seed = int(np.random.randint(1000000000))
        needs_params = args.efficient_mezo or args.data_selection_unit in ("mezo", "masked_grad")
        self.zo_params = (
            zo_parameters(self.model, args.last_layers, args.last_layer_index)
            if needs_params
            else []
        )
        self.extractor = build_extractor(args, self.model, self.zo_params, self.zo_seed)
        dims = sum(p.numel() for _, p in self.zo_params) or self.model.config.hidden_size
        if not self.extractor.scalar and args.zo_dim > dims:
            raise ValueError(
                f"zo_dim={args.zo_dim} exceeds the {dims} features of {args.data_selection_unit}"
            )
        self.selector = CoresetSelector(
            args,
            self.model.config.num_hidden_layers,
            moments=self._optimizer_moments if args.data_selection_unit == "masked_grad" else None,
            weight_prior=self.extractor.weight_prior if self.zo_params else None,
        )
        self.batching = PackedBatching(
            args, getattr(self.train_dataset, "mean_tokens", None), self.extractor.batched
        )
        logger.info(
            f"ZO seed {self.zo_seed}; pool {args.per_device_train_batch_size} per rank "
            f"in micro-batches of {args.micro_batch_size}, {self._per_rank(args.pool_micro_batches)} "
            f"selected per rank"
        )

    def describe(self) -> dict:
        """What the trainer derived from the options, the model and the data."""
        args = self.args
        return {
            "zo_seed": self.zo_seed,
            "zo_parameters": [n for n, _ in self.zo_params],
            "pool_per_rank": args.per_device_train_batch_size,
            "micro_batch_size": args.micro_batch_size,
            "selected_per_rank": self._per_rank(args.pool_micro_batches),
            "pack_tokens": {
                "selection": self.batching.select_tokens,
                "training": self.batching.train_tokens,
            },
            "attn_implementation": self.model.config._attn_implementation,
        }

    # ----- budgets and sub-batches (differ between the two coreset trainers) --------------
    def _check_args(self):
        pass

    def _per_rank(self, num_micro_batches: int) -> int:
        raise NotImplementedError

    # ----- one optimizer step ---------------------------------------------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        if self._last_log is None:
            self._last_log = (time.perf_counter(), self.state.global_step)
        start = time.perf_counter()
        self.memory.start()
        with self._timer.section("selection"):
            sub_batches, total = self._select(inputs)
        self._select_seconds += time.perf_counter() - start
        self.memory.stop("selection")

        self.memory.start()
        with self._timer.section("train"):
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()  # schedule-free optimizers
            done = torch.zeros((), device=self.args.device)
            for i, (batch, weight) in enumerate(sub_batches):
                batch = self._prepare_inputs(batch)
                # One gradient all-reduce per optimizer step: on the last sub-batch only.
                last = i == len(sub_batches) - 1
                with contextlib.nullcontext() if last else self.accelerator.no_sync(model):
                    with self._timer.section("forward"), self.compute_loss_context_manager():
                        loss = self.batching.loss(self, model, batch, weight, total)
                    with self._timer.section("backward"):
                        self.accelerator.backward(loss)
                done += loss.detach()
        self.memory.stop("train")
        return done

    def _select(self, inputs: dict):
        args, t = self.args, self._timer
        if self.extractor.mode is not None:
            self.model.train(self.extractor.mode == "train")
        with t.section("features"):
            batches = [self._prepare_inputs(b) for b in self.batching.feature_batches(inputs)]
            values = torch.cat([self._features(b) for b in batches])
            examples = self.batching.examples(inputs)
            if self.drop_invalid:
                valid = (
                    values != 0
                    if values.dim() == 1
                    else (~torch.isnan(values).any(dim=1) & (torch.norm(values, dim=1) != 0))
                )
                examples = [e for e, ok in zip(examples, valid.tolist(), strict=True) if ok]
                values = values[valid]
            values = values.cpu()
        per_rank = self._per_rank(args.pool_micro_batches)
        world, rank = args.world_size, args.process_index
        with t.section("gather"):
            pool = [e for chunk in all_gather_object(examples) for e in chunk]
            gathered = gather_object(values)
        selection = None
        if rank == 0:
            with t.section("select"):
                feats = torch.cat(gathered).to(args.device)
                chosen = self.selector(
                    feats, [source_of(e) for e in pool], per_rank * world, self.state.global_step
                )
                if args.save_indices:
                    self._save_indices(pool, chosen)
                selection = (chosen.indices, chosen.weights)
        with t.section("scatter"):
            indices, weights = broadcast_object(selection)
        # Round-robin, so that every rank gets the same mixture (the list starts with the examples
        # of the kept sources).
        mine = slice(rank, None, world)
        total = self.batching.total_labels([pool[i] for i in indices])
        chosen_examples = [pool[i] for i in indices[mine]]
        return self.batching.train_batches(chosen_examples, weights[mine]), total

    def _features(self, batch: dict) -> torch.Tensor:
        values = self.extractor.extract(batch)
        if self.extractor.batched:
            values = values.float()
        return values

    def _optimizer_moments(self):
        """Adam moments of the real optimizer for the ZO parameters (None before its first step)."""
        params = [p for _, p in self.zo_params]
        if "exp_avg" not in self.optimizer.state[params[0]]:
            return None
        state = self.optimizer.state
        m = torch.cat([state[p]["exp_avg"].flatten() for p in params])
        v = torch.cat([state[p]["exp_avg_sq"].flatten() for p in params])
        return m, v

    def _save_indices(self, pool: list, chosen) -> None:
        directory = os.path.join(self.args.output_dir, INDICES_DIRNAME)
        os.makedirs(directory, exist_ok=True)
        for name, positions in [
            ("full", range(len(pool))),
            ("sampling", chosen.candidates),
            ("selected", chosen.indices),
        ]:
            wanted = set(int(i) for i in positions)
            original = [index_of(e) for i, e in enumerate(pool) if i in wanted]
            torch.save(
                original, os.path.join(directory, f"iter{self.state.global_step}_{name}_indices.pt")
            )


class SubsetTrainer(CoresetTrainer):
    """One example per micro-batch (`per_device_train_batch_size=1`), all selection units.

    Selected examples are trained one by one, optionally weighted by their cluster size.
    """

    drop_invalid = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.micro_batch_size != 1:
            raise ValueError(
                "SubsetTrainer needs per_device_train_batch_size = 1 (use efficient_mezo)"
            )

    def _per_rank(self, num_micro_batches: int) -> int:
        return max(1, int(num_micro_batches * self.args.small_batch_ratio))


class SubsetTrainerEfficient(CoresetTrainer):
    """Batched last-layer MeZO (the paper's method): several examples per forward."""

    def _check_args(self):
        args = self.args
        if args.data_selection_unit != "mezo" or not args.efficient_mezo:
            raise ValueError("the efficient trainer only supports efficient MeZO")
        if args.mezo_transform != "none" or "weighted" in args.data_selection_method:
            raise ValueError(
                "the efficient trainer applies no mezo_transform and trains unweighted"
            )
        if int(args.micro_batch_size * args.small_batch_ratio) < 1:
            raise ValueError("micro_batch_size * small_batch_ratio must be >= 1")

    def _per_rank(self, num_micro_batches: int) -> int:
        return num_micro_batches * int(self.args.micro_batch_size * self.args.small_batch_ratio)
