"""A few real optimizer steps of each CoLM trainer on a tiny random Phi (CPU)."""

import math
import sys

import pytest
import torch
from conftest import NUM_LAYERS, add_lora, make_phi

from colm.data.get_training_dataset import (
    DataCollatorForSupervisedDatasetWithSource,
    get_training_dataset,
)
from colm.train.trainers import (
    SAMPLE_WEIGHT_KEY,
    CustomTrainer,
    SubsetTrainer,
    SubsetTrainerEfficient,
)
from colm.train.training_arguments import TrainingArguments

MAX_STEPS = 2


def _args(tmp_path, **overrides):
    kwargs = dict(
        output_dir=str(tmp_path / "out"),
        use_cpu=not torch.cuda.is_available(),
        max_steps=MAX_STEPS,
        learning_rate=1e-3,
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        last_layer_index=NUM_LAYERS - 1,
        zo_dim=16,
        dataloader_num_workers=0,
    )
    kwargs.update(overrides)
    args = TrainingArguments(**kwargs)
    # train.py: parsed keep_sources and LoRA-B names of the last layer.
    args.keep_sources = [int(s) for s in args.keep_sources.split("_")] if args.keep_sources else []
    args.last_layers = [name + ".lora_B" for name in args.last_layers]
    return args


def _build(trainer_cls, args, tokenizer, mixture_file):
    model = add_lora(make_phi(tokenizer))
    dataset = get_training_dataset([mixture_file], tokenizer=tokenizer, max_seq_length=512)
    collator = DataCollatorForSupervisedDatasetWithSource(tokenizer=tokenizer)
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )
    return trainer, model


def _record(trainer, monkeypatch):
    """Record, per optimizer step, the example count of every trained micro-batch."""
    calls = []
    original_training_step = trainer.training_step
    original_select = trainer._select_on_main
    selections = []

    def training_step(model, inputs, num_items_in_batch=None):
        calls.append(
            (trainer.state.global_step, len(inputs["input_ids"]) if inputs else 0, dict(inputs))
        )
        return original_training_step(model, inputs, num_items_in_batch)

    def select_on_main(all_reps, complete_examples, total):
        idx, weights = original_select(all_reps, complete_examples, total)
        selections.append((len(complete_examples), idx, weights, complete_examples))
        return idx, weights

    monkeypatch.setattr(trainer, "training_step", training_step)
    monkeypatch.setattr(trainer, "_select_on_main", select_on_main)
    return calls, selections


def _losses(trainer):
    return [h["loss"] for h in trainer.state.log_history if "loss" in h]


def _lora_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def test_efficient_trainer_selects_and_trains(tmp_path, tokenizer, mixture_file, monkeypatch):
    wandb_loaded_before = "wandb" in sys.modules
    bs, gas, ratio = 4, 2, 0.5
    args = _args(
        tmp_path,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=gas,
        small_batch_ratio=ratio,
        efficient_mezo=True,
        keep_sources="0",
    )
    trainer, model = _build(SubsetTrainerEfficient, args, tokenizer, mixture_file)
    calls, selections = _record(trainer, monkeypatch)
    last_b = trainer.named_parameters_to_optim[0][1]
    before = _lora_snapshot(model)

    trainer.train()

    assert trainer.state.global_step == MAX_STEPS
    losses = _losses(trainer)
    assert len(losses) == MAX_STEPS and all(math.isfinite(v) for v in losses)

    new_bs = int(bs * ratio)
    for step in range(MAX_STEPS):
        step_calls = [c for c in calls if c[0] == step]
        # gas micro-batches of bs*ratio selected examples per optimizer step
        assert [c[1] for c in step_calls] == [new_bs] * gas
    assert len(selections) == MAX_STEPS
    for n_large, idx, _, examples in selections:
        assert n_large == bs * gas
        assert len(idx) == len(set(idx)) == gas * new_bs
        # Every example of a kept source is trained on.
        kept = [i for i, ex in enumerate(examples) if ex["sources"][0] in args.keep_sources]
        assert set(kept) <= set(idx)

    # Trained micro-batches are re-collated selected examples, padded with the pad id.
    for _, _, inputs in calls:
        pad = inputs["attention_mask"] == 0
        assert (inputs["input_ids"][pad] == tokenizer.pad_token_id).all()
        assert (inputs["labels"][pad] == -100).all()

    after = _lora_snapshot(model)
    assert any(not torch.equal(before[n], after[n]) for n in before)
    assert torch.isfinite(last_b).all()
    if not wandb_loaded_before:
        assert "wandb" not in sys.modules


@pytest.mark.parametrize("unit", ["mezo", "rep", "masked_grad", "length_loss_weighted"])
def test_subset_trainer_units(tmp_path, tokenizer, mixture_file, monkeypatch, unit):
    gas, ratio = 4, 0.5
    args = _args(
        tmp_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gas,
        small_batch_ratio=ratio,
        data_selection_unit=unit,
        data_selection_method="weightedsubmodlib",
        keep_sources="",
    )
    trainer, _ = _build(SubsetTrainer, args, tokenizer, mixture_file)
    calls, selections = _record(trainer, monkeypatch)

    trainer.train()

    assert trainer.state.global_step == MAX_STEPS
    assert all(math.isfinite(v) for v in _losses(trainer))
    n_select = int(gas * ratio)
    for step in range(MAX_STEPS):
        step_calls = [c for c in calls if c[0] == step]
        assert len(step_calls) == gas  # HF loop always sees gas micro-batches
        real = [c for c in step_calls if c[1] > 0]
        assert len(real) == n_select and all(c[1] == 1 for c in real)
        # Placeholders come first so the last micro-batch carries the DDP gradient sync.
        assert [c[1] for c in step_calls] == [0] * (gas - n_select) + [1] * n_select
        assert all(SAMPLE_WEIGHT_KEY in c[2] for c in real)
    for n_large, idx, weights, _ in selections:
        assert n_large <= gas and len(idx) == n_select
        # weightedsubmodlib: cluster sizes scaled by the ratio. Sources without budget
        # have no medoid, so the sum is at most ratio * large batch.
        assert all(w > 0 for w in weights)
        assert sum(weights) <= ratio * n_large + 1e-6


def test_custom_trainer_full_batch(tmp_path, tokenizer, mixture_file, monkeypatch):
    args = _args(
        tmp_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        data_selection_method="none",
        assert_finite_grad_norm=True,
        save_indices=True,
    )
    trainer, model = _build(CustomTrainer, args, tokenizer, mixture_file)
    before = _lora_snapshot(model)
    trainer.train()
    assert trainer.state.global_step == MAX_STEPS
    assert all(math.isfinite(v) for v in _losses(trainer))
    after = _lora_snapshot(model)
    assert any(not torch.equal(before[n], after[n]) for n in before)
    saved = sorted(p.name for p in (tmp_path / "out" / "indices").iterdir())
    assert len(saved) == MAX_STEPS * 2


def test_mezo_perturbation_is_restored(tmp_path, tokenizer, mixture_file):
    args = _args(
        tmp_path, per_device_train_batch_size=4, gradient_accumulation_steps=2, efficient_mezo=True
    )
    trainer, _ = _build(SubsetTrainerEfficient, args, tokenizer, mixture_file)
    param = trainer.named_parameters_to_optim[0][1]
    with torch.no_grad():
        param.normal_(std=0.1)
    reference = param.detach().clone()
    batch = next(iter(trainer.get_train_dataloader()))
    reps = trainer.save_select(batch)
    assert reps.shape == (4, param.numel())
    torch.testing.assert_close(param.detach(), reference, rtol=0, atol=1e-6)
