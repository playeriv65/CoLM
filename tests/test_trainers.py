"""A few real optimizer steps of each CoLM trainer on a tiny random Phi (CPU)."""

import math
import sys

import pytest
import torch
from equivalence.helpers import build, lora_state, make_args, model_fp64

from colm.selection.zo import per_sample_loss

MAX_STEPS = 2


def _record(trainer, monkeypatch):
    """The sub-batches trained in every step: (step, example indices, weight)."""
    trained = []
    make = trainer._sub_batches

    def sub_batches(examples, weights):
        out = make(examples, weights)
        for batch, weight in out:
            trained.append(
                (trainer.state.global_step, batch["colm_meta"]["indices"].tolist(), weight)
            )
        return out

    monkeypatch.setattr(trainer, "_sub_batches", sub_batches)
    return trained


def _record_pools(monkeypatch):
    """Original indices of the gathered pool of every step."""
    from colm.selection import pool as pool_module
    from colm.train import trainers

    pools, gather = [], pool_module.all_gather_object

    def record(obj):
        out = gather(obj)
        pools.append([int(e["colm_meta"]["indices"]) for chunk in out for e in chunk])
        return out

    monkeypatch.setattr(trainers, "all_gather_object", record)
    return pools


def _losses(trainer):
    return [h["loss"] for h in trainer.state.log_history if "loss" in h]


def test_efficient_trainer_selects_and_trains(tmp_path, tokenizer, mixture_file, monkeypatch):
    wandb_loaded_before = "wandb" in sys.modules
    bs, gas, ratio = 4, 2, 0.5
    args = make_args(
        tmp_path,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=gas,
        small_batch_ratio=ratio,
        efficient_mezo=True,
        keep_sources="0",
        max_steps=MAX_STEPS,
    )
    # The pool (bs * gas examples) is one Hugging Face batch.
    assert (args.per_device_train_batch_size, args.gradient_accumulation_steps) == (bs * gas, 1)
    trainer, model = build(args, tokenizer, mixture_file)
    trained = _record(trainer, monkeypatch)
    pools = _record_pools(monkeypatch)
    before = lora_state(model)
    trainer.train()

    assert trainer.state.global_step == MAX_STEPS
    assert len(_losses(trainer)) == MAX_STEPS and all(math.isfinite(v) for v in _losses(trainer))
    new_bs = int(bs * ratio)
    for step in range(MAX_STEPS):
        step_batches = [t for t in trained if t[0] == step]
        assert [len(t[1]) for t in step_batches] == [new_bs] * gas
        picked = [i for t in step_batches for i in t[1]]
        assert len(picked) == len(set(picked)) == gas * new_bs
        # Every example of a kept source (id 0 = every 4th example) is trained on.
        assert len(pools[step]) == bs * gas and set(picked) <= set(pools[step])
        assert {i for i in pools[step] if i % 4 == 0} <= set(picked)
    after = lora_state(model)
    assert any(not torch.equal(before[n], after[n]) for n in before)
    if not wandb_loaded_before:
        assert "wandb" not in sys.modules


@pytest.mark.parametrize(
    "unit", ["mezo", "rep", "masked_grad", "length_loss_weighted", "completion_length"]
)
def test_subset_trainer_units(tmp_path, tokenizer, mixture_file, monkeypatch, unit):
    gas, ratio = 4, 0.5
    args = make_args(
        tmp_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gas,
        small_batch_ratio=ratio,
        data_selection_unit=unit,
        data_selection_method="weightedsubmodlib",
        keep_sources="",
        max_steps=MAX_STEPS,
    )
    trainer, _ = build(args, tokenizer, mixture_file)
    trained = _record(trainer, monkeypatch)
    trainer.train()
    assert trainer.state.global_step == MAX_STEPS and all(
        math.isfinite(v) for v in _losses(trainer)
    )
    for step in range(MAX_STEPS):
        step_batches = [t for t in trained if t[0] == step]
        assert len(step_batches) == int(gas * ratio) and all(len(t[1]) == 1 for t in step_batches)
        assert all(w > 0 for _, _, w in step_batches)


def test_custom_trainer_full_batch(tmp_path, tokenizer, mixture_file):
    args = make_args(
        tmp_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        data_selection_method="none",
        assert_finite_grad_norm=True,
        save_indices=True,
        max_steps=MAX_STEPS,
    )
    trainer, model = build(args, tokenizer, mixture_file)
    before = lora_state(model)
    trainer.train()
    assert trainer.state.global_step == MAX_STEPS and all(
        math.isfinite(v) for v in _losses(trainer)
    )
    assert any(not torch.equal(before[n], p) for n, p in lora_state(model).items())
    assert len(list((tmp_path / "indices").iterdir())) == MAX_STEPS * 2


def _extract(tmp_path, tokenizer, mixture_file, legacy):
    args = make_args(
        tmp_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        efficient_mezo=True,
        keep_sources="",
        legacy=legacy,
    )
    trainer, model = build(args, tokenizer, mixture_file, model=model_fp64(tokenizer))
    batch = next(iter(trainer.get_train_dataloader()))["micro_batches"][0]
    model.eval()
    return trainer, model, batch


@pytest.mark.parametrize("legacy", [False, True])
def test_estimate_and_the_two_rng_streams(tmp_path, tokenizer, mixture_file, legacy):
    """The estimate does not move the parameters (E-drift) nor the training RNG (E3)."""
    trainer, model, batch = _extract(tmp_path, tokenizer, mixture_file, legacy)
    before = lora_state(model)
    torch.manual_seed(0)
    state = torch.get_rng_state()
    features = trainer.extractor.extract(batch)
    assert features.shape[0] == len(batch["input_ids"]) and torch.isfinite(features).all()
    after = lora_state(model)
    drift = max(float((before[n] - after[n]).abs().max()) for n in before)
    same_rng = torch.equal(torch.get_rng_state(), state)
    if legacy:  # in-place shifts leave rounding drift; the global RNG is reseeded
        assert not same_rng and drift < 1e-12
    else:
        assert same_rng and drift == 0


def test_loss_of_an_example_does_not_depend_on_its_batch(tmp_path, tokenizer, mixture_file):
    """Per-sample loss: mean over the example's label tokens (E2), not over the padded width."""
    trainer, model, batch = _extract(tmp_path, tokenizer, mixture_file, legacy=False)
    split = trainer.extractor.split
    with torch.no_grad():
        state = split.prefix(batch["input_ids"], batch["attention_mask"])
        logits = split.logits(state)
        fixed = per_sample_loss(logits, batch["labels"])
        padded = per_sample_loss(logits, batch["labels"], legacy=True)
    labels = batch["labels"][:, 1:]
    token_mean = torch.stack(
        [
            torch.nn.functional.cross_entropy(
                logits[i, :-1][labels[i] != -100], labels[i][labels[i] != -100]
            )
            for i in range(len(labels))
        ]
    )
    torch.testing.assert_close(fixed, token_mean)
    counts = (labels != -100).sum(1)
    torch.testing.assert_close(padded, fixed * counts / labels.shape[1])


@pytest.mark.parametrize("legacy", [False, True])
def test_logged_loss(tmp_path, tokenizer, mixture_file, monkeypatch, legacy):
    """The logged loss is the loss being minimised; `legacy` divides it by small_batch_ratio (E5)."""
    args = make_args(
        tmp_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        efficient_mezo=True,
        keep_sources="",
        max_steps=1,
        legacy=legacy,
    )
    trainer, _ = build(args, tokenizer, mixture_file)
    micro_losses = []
    compute_loss = trainer.compute_loss

    def record(model, inputs, **kw):
        loss = compute_loss(model, inputs, **kw)
        if model.training:
            micro_losses.append(float(loss.detach()))
        return loss

    monkeypatch.setattr(trainer, "compute_loss", record)
    trainer.train()
    mean = sum(micro_losses) / len(micro_losses)
    assert _losses(trainer)[0] == pytest.approx(
        mean / (args.small_batch_ratio if legacy else 1.0), rel=1e-6
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_trainers_run_in_both_modes(tmp_path, tokenizer, mixture_file, legacy):
    args = make_args(
        tmp_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        data_selection_unit="length_loss_weighted",
        keep_sources="",
        legacy=legacy,
        max_steps=1,
    )
    trainer, _ = build(args, tokenizer, mixture_file)
    trainer.train()
    assert math.isfinite(_losses(trainer)[0])
