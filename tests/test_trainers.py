"""A few real optimizer steps of each CoLM trainer on a tiny random Phi (CPU)."""

import math
import sys

import pytest
import torch
from equivalence.helpers import build, lora_state, make_args

MAX_STEPS = 2


def _record(trainer, monkeypatch):
    """The sub-batches trained in every step: (step, original indices of the examples, weights)."""
    trained = []
    train_batches = trainer.batching.train_batches

    def record(examples, weights):
        out = train_batches(examples, weights)
        step = trainer.state.global_step
        for batch, weight in out:
            trained.append((step, batch["colm_meta"]["indices"].tolist(), weight))
        return out

    monkeypatch.setattr(trainer.batching, "train_batches", record)
    return trained


def _record_pools(monkeypatch):
    """Original indices of the gathered pool of every step."""
    from colm.selection import pool as pool_module
    from colm.selection.pool import index_of
    from colm.train import trainers

    pools, gather = [], pool_module.all_gather_object

    def record(obj):
        out = gather(obj)
        pools.append([index_of(e) for chunk in out for e in chunk])
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
        assert 1 <= len(step_batches)  # packs of a token budget
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
        assert all(float(w) > 0 for _, _, w in step_batches)


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


def _extract(tmp_path, tokenizer, mixture_file):
    args = make_args(
        tmp_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        efficient_mezo=True,
        keep_sources="",
    )
    trainer, model = build(args, tokenizer, mixture_file)
    inputs = next(iter(trainer.get_train_dataloader()))
    model.eval()
    return trainer, model, trainer._prepare_inputs(trainer.batching.feature_batches(inputs)[0])


def test_estimate_and_the_two_rng_streams(tmp_path, tokenizer, mixture_file):
    """The estimate moves neither the parameters (E-drift) nor the training RNG (E3)."""
    trainer, model, batch = _extract(tmp_path, tokenizer, mixture_file)
    before = lora_state(model)
    torch.manual_seed(0)
    state = torch.get_rng_state()
    features = trainer.extractor.extract(batch)
    assert torch.isfinite(features).all()
    after = lora_state(model)
    assert all(torch.equal(before[n], after[n]) for n in before)
    assert torch.equal(torch.get_rng_state(), state)


def test_logged_loss_is_the_token_mean_of_the_step(tmp_path, tokenizer, mixture_file, monkeypatch):
    """Default: the loss of a step is the mean over all label tokens of the trained examples."""
    args = make_args(
        tmp_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        efficient_mezo=True,
        keep_sources="",
        max_steps=1,
    )
    trainer, model = build(args, tokenizer, mixture_file)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    trained = []
    train_batches = trainer.batching.train_batches

    def record(examples, weights):
        trained.extend(examples)
        return train_batches(examples, weights)

    monkeypatch.setattr(trainer.batching, "train_batches", record)
    trainer.train()

    model.load_state_dict(initial)
    model.train()
    with torch.no_grad():
        sums = [
            float(
                model(
                    input_ids=torch.tensor(e.input_ids)[None], labels=torch.tensor(e.labels)[None]
                ).loss
            )
            * e.num_labels
            for e in trained
        ]
    expected = sum(sums) / sum(e.num_labels for e in trained)
    assert _losses(trainer)[0] == pytest.approx(expected, rel=1e-6)


def test_trainer_runs_a_scalar_unit(tmp_path, tokenizer, mixture_file):
    args = make_args(
        tmp_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        data_selection_unit="length_loss_weighted",
        keep_sources="",
        max_steps=1,
    )
    trainer, _ = build(args, tokenizer, mixture_file)
    trainer.train()
    assert math.isfinite(_losses(trainer)[0])
