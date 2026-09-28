"""The exact step optimisations reproduce the original selection and training (tiny Phi, CPU).

Each optimisation is checked against the original code path (`ORIGINAL_PATH`) on identical
weights and batches: MeZO features g_i (float-close; skipped examples are exactly 0), selected
indices (identical), MeZO Adam state, RNG state after selection, and training gradients.

The MeZO feature is a difference quotient of two fp32 losses (the loss is fp32 in both paths),
so a one-ulp change of a loss moves g_i by ulp(loss) / (2 eps): ~1e-3 relative at eps = 1e-3
on this tiny model, enough to reorder near-tied facility-location gains. The model therefore
runs in float64 and `mezo_eps` is 0.05, which leaves summation-order noise at ~1e-5 relative;
the tests check the algorithm, the GPU check (`scripts/check_exact_opts.py`) the fp32 numerics.
"""

import copy

import numpy as np
import pytest
import torch
from conftest import NUM_LAYERS, ORIGINAL_PATH, add_lora, make_phi

from colm.data.get_training_dataset import (
    DataCollatorForSupervisedDatasetWithSource,
    get_training_dataset,
)
from colm.train import attention, packing
from colm.train.custom_phi import DecomposedPhiCausalLM
from colm.train.facility_location import _per_class_budget, features_needed
from colm.train.trainers import SubsetTrainerEfficient
from colm.train.training_arguments import TrainingArguments

BS, GAS = 4, 4
ALL_OFF = {k: v for k, v in ORIGINAL_PATH.items()}


def _args(tmp_path, **overrides):
    kwargs = dict(
        output_dir=str(tmp_path / "out"),
        use_cpu=True,
        max_steps=4,
        learning_rate=1e-3,
        warmup_steps=0,
        save_strategy="no",
        last_layer_index=NUM_LAYERS - 1,
        zo_dim=16,
        dataloader_num_workers=0,
        per_device_train_batch_size=BS,
        gradient_accumulation_steps=GAS,
        small_batch_ratio=0.5,
        efficient_mezo=True,
        keep_sources="0",
        mezo_eps=0.05,
    )
    kwargs.update(overrides)
    args = TrainingArguments(**kwargs)
    args.keep_sources = [int(s) for s in args.keep_sources.split("_")] if args.keep_sources else []
    args.last_layers = [name + ".lora_B" for name in args.last_layers]
    return args


@pytest.fixture()
def lora_phi(tokenizer):
    model = add_lora(make_phi(tokenizer))
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.05)
    return model.double()


def _trainer(tmp_path, tokenizer, mixture_file, model, **flags):
    dataset = get_training_dataset([mixture_file], tokenizer=tokenizer, max_seq_length=512)
    trainer = SubsetTrainerEfficient(
        model=copy.deepcopy(model),
        args=_args(tmp_path, **flags),
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSupervisedDatasetWithSource(tokenizer=tokenizer),
    )
    trainer.zo_random_seed = 1234
    return trainer


def _large_batches(trainer, num_steps):
    loader = iter(trainer.get_train_dataloader())
    return [[next(loader) for _ in range(GAS)] for _ in range(num_steps)]


def _run_selection(trainer, batch_samples):
    """Features, selected indices and the state selection leaves behind."""
    captured = {}
    original = trainer._select_on_main

    def select_on_main(all_reps, complete_examples, total):
        captured["reps"] = all_reps.clone()
        idx, weights = original(all_reps, complete_examples, total)
        captured["idx"] = idx
        return idx, weights

    trainer._select_on_main = select_on_main
    torch.manual_seed(7)
    microbatches = trainer._select_microbatches([dict(b) for b in batch_samples])
    captured["rng"] = torch.get_rng_state()
    captured["m"], captured["v"] = (
        None if t is None else t.clone() for t in (trainer.prev_m_t, trainer.prev_v_t)
    )
    captured["microbatches"] = microbatches
    trainer._select_on_main = original
    return captured


FLAG_SETS = {
    "lazy_mode_switch": dict(lazy_mode_switch=True),
    "skip_unused_features": dict(skip_unused_features=True),
    "zo_label_positions_only": dict(zo_label_positions_only=True),
    "zo_packing_sdpa": dict(zo_packing=True, zo_attn_implementation="model"),
    "zo_packing_sdpa_rows": dict(
        zo_packing=True, zo_attn_implementation="model", zo_pack_max_tokens=64
    ),
    "zo_packing_varlen": dict(zo_packing=True, zo_attn_implementation="colm_varlen"),
    "all": dict(
        lazy_mode_switch=True,
        skip_unused_features=True,
        zo_label_positions_only=True,
        zo_packing=True,
        zo_attn_implementation="colm_varlen",
    ),
}


@pytest.mark.parametrize("mezo_optim", ["adam", "sgd"])
@pytest.mark.parametrize("name", list(FLAG_SETS))
def test_selection_matches_original(tmp_path, tokenizer, mixture_file, lora_phi, name, mezo_optim):
    ref = _trainer(tmp_path, tokenizer, mixture_file, lora_phi, mezo_optim=mezo_optim, **ALL_OFF)
    new = _trainer(
        tmp_path,
        tokenizer,
        mixture_file,
        lora_phi,
        mezo_optim=mezo_optim,
        **{**ALL_OFF, **FLAG_SETS[name]},
    )
    skipped_any = False
    for batch_samples in _large_batches(ref, num_steps=4):
        a, b = _run_selection(ref, batch_samples), _run_selection(new, batch_samples)
        assert b["idx"] == a["idx"]
        computed = b["reps"].abs().sum(dim=1) > 0
        skipped_any |= bool((~computed).any())
        torch.testing.assert_close(b["reps"][computed], a["reps"][computed], rtol=1e-6, atol=1e-9)
        if mezo_optim == "adam":
            torch.testing.assert_close(b["m"], a["m"], rtol=1e-6, atol=1e-9)
            torch.testing.assert_close(b["v"], a["v"], rtol=1e-6, atol=1e-12)
        # Same RNG stream for the training dropout afterwards.
        assert torch.equal(b["rng"], a["rng"])
        for x, y in zip(a["microbatches"], b["microbatches"], strict=True):
            assert torch.equal(x["input_ids"], y["input_ids"])
    if name in ("skip_unused_features", "all"):
        assert skipped_any


def test_no_forward_when_kept_examples_fill_the_budget(tmp_path, tokenizer, mixture_file, lora_phi):
    # 3 of 4 sources kept: most steps need no MeZO forward at all; RNG, Adam state and the
    # selection must still match the original (which forwards everything).
    kw = dict(keep_sources="0_1_2")
    ref = _trainer(tmp_path, tokenizer, mixture_file, lora_phi, **kw, **ALL_OFF)
    new = _trainer(tmp_path, tokenizer, mixture_file, lora_phi, **kw, **FLAG_SETS["all"])
    forwards = []
    original = new._zo_projected_grads
    new._zo_projected_grads = lambda group: forwards.append(1) or original(group)
    for batch_samples in _large_batches(ref, num_steps=4):
        before = len(forwards)
        a, b = _run_selection(ref, batch_samples), _run_selection(new, batch_samples)
        assert b["idx"] == a["idx"]
        assert torch.equal(b["rng"], a["rng"])
        if len(forwards) == before:
            assert (b["reps"] == 0).all()
    assert len(forwards) < 4  # at least one step without any forward


def test_features_needed_budget_rules():
    # keep {0}; 4 candidates of source 1, 1 of source 2, 3 of source 3; budget 8 - 2 = 6 of 8.
    sources = [0, 0, 1, 1, 1, 1, 2, 3, 3, 3]
    kw = dict(keep_sources=[0], strategy="proportional", per_class_start="floor")
    adam = features_needed(sources, total=8, need_selected=True, per_source_rng=False, **kw)
    assert not adam[:2].any() and adam[2:].all()
    # sgd: sources selected in full (budget == count) need no feature.
    sgd = features_needed(sources, total=8, need_selected=False, per_source_rng=False, **kw)
    assert sgd.tolist() == [False] * 2 + [True] * 4 + [False] + [True] * 3
    # Budget 2 for 3 candidate sources: one source gets no budget and needs no feature.
    small = features_needed(sources, total=4, need_selected=True, per_source_rng=False, **kw)
    _, _, quotas = _per_class_budget(2, 8, np.array(sources[2:]), "floor", "proportional")
    per_source = dict(zip([1, 2, 3], quotas, strict=True))
    assert 0 in quotas
    assert small.tolist() == [False] * 2 + [bool(per_source[s] > 0) for s in sources[2:]]
    # Kept examples fill the budget: nothing is needed.
    assert not features_needed(
        sources, total=2, need_selected=True, per_source_rng=False, **kw
    ).any()
    # Random tie breaking / per-source sampling: every candidate is needed.
    for rng_kw in (dict(strategy="balanced"), dict(per_source_rng=True)):
        call = {**kw, "need_selected": True, "per_source_rng": False, **rng_kw}
        assert features_needed(sources, total=5, **call).tolist() == [False] * 2 + [True] * 8


def _padded_batch(tokenizer):
    texts = ["Hello world, this is CoLM.", "Short one.", "A third, slightly longer example!"]
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    labels[:, :3] = -100
    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": labels}


@pytest.mark.parametrize("impl", ["sdpa", attention.NAME])
@pytest.mark.parametrize("max_tokens", [0, 30])
def test_packed_decomposer_matches_padded(tokenizer, lora_phi, impl, max_tokens):
    model = lora_phi.eval()
    base = model.get_base_model()
    attention.register()
    batch = _padded_batch(tokenizer)
    decomposer = DecomposedPhiCausalLM(base)
    sequences = [packing.unpad(batch, r) for r in range(3)]
    rows = packing.rows_by_token_budget([len(s[0]) for s in sequences], max_tokens)
    row = packing.pack(sequences, tokenizer.pad_token_id, rows)
    with torch.no_grad():
        ref = decomposer.forward_till_penultimate(batch["input_ids"], batch["attention_mask"])
        ref_loss = decomposer.final_layer_token_losses(ref, packing.shift_left(batch["labels"]))
        base.config._attn_implementation = impl
        kwargs = row.attention_kwargs("cpu") if impl == attention.NAME else {}
        mid = decomposer.forward_till_penultimate(
            row.input_ids, None, row.position_ids, attention_kwargs=kwargs
        )
        loss = decomposer.final_layer_token_losses(mid, packing.shift_left(row.labels))
        base.config._attn_implementation = "sdpa"
    for i, (ids, _) in enumerate(sequences):
        r, offset = next((r, row_) for r, row_ in enumerate(rows) if i in row_)
        start = sum(len(sequences[j][0]) for j in offset[: offset.index(i)])
        n = len(ids)
        torch.testing.assert_close(
            mid["hidden_states"][r, start : start + n],
            ref["hidden_states"][i, :n],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            loss[r, start : start + n].sum(), ref_loss[i].sum(), rtol=1e-5, atol=1e-6
        )


def test_packed_training_forward_matches_padded_and_needs_no_cache(tokenizer, lora_phi):
    model = lora_phi.eval()
    batch = _padded_batch(tokenizer)
    sequences = [packing.unpad(batch, r) for r in range(3)]
    row = packing.pack(sequences, tokenizer.pad_token_id)
    with torch.no_grad():
        ref = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        packed = model(input_ids=row.input_ids, position_ids=row.position_ids, use_cache=False)
        leaky = model(input_ids=row.input_ids, position_ids=row.position_ids)  # use_cache=True
    offset = 0
    for i, (ids, _) in enumerate(sequences):
        n = len(ids)
        torch.testing.assert_close(
            packed.logits[0, offset : offset + n], ref[i, :n], rtol=1e-5, atol=1e-5
        )
        offset += n
    # F5: with the default DynamicCache packing is not detected and sequences attend across
    # example boundaries.
    assert not torch.allclose(leaky.logits, packed.logits, atol=1e-3)


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("mode", ["sub_batch", "merged", "merged_rows"])
def test_packed_training_matches_original(tmp_path, tokenizer, mixture_file, lora_phi, mode):
    flags = {
        "sub_batch": {"train_packing": "sub_batch"},
        "merged": {"train_packing": "merged", "train_pack_max_tokens": 0},
        "merged_rows": {"train_packing": "merged", "train_pack_max_tokens": 1100},
    }[mode]
    ref = _trainer(tmp_path, tokenizer, mixture_file, lora_phi, **ALL_OFF)
    new = _trainer(tmp_path, tokenizer, mixture_file, lora_phi, **{**ALL_OFF, **flags})
    for trainer in (ref, new):
        trainer.create_optimizer_and_scheduler(num_training_steps=1)
    (batch_samples,) = _large_batches(ref, num_steps=1)
    results = []
    for trainer in (ref, new):
        microbatches = _run_selection(trainer, batch_samples)["microbatches"]
        assert len(microbatches) == GAS
        trainer.model.zero_grad()
        total = sum(float(trainer.training_step(trainer.model, dict(mb))) for mb in microbatches)
        results.append((total, _grads(trainer.model), microbatches))
    (loss_a, grads_a, _), (loss_b, grads_b, mbs) = results
    if mode == "merged":
        assert sum(1 for mb in mbs if mb) == 1
    elif mode == "merged_rows":
        assert 1 < sum(1 for mb in mbs if mb) < GAS
    assert loss_b == pytest.approx(loss_a, rel=1e-6)  # HF computes the CE in fp32
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        torch.testing.assert_close(grads_b[name], grads_a[name], rtol=1e-5, atol=1e-9)


def test_cached_flops_and_lazy_mode(tmp_path, tokenizer, mixture_file, lora_phi):
    ref = _trainer(tmp_path, tokenizer, mixture_file, lora_phi, **ALL_OFF)
    new = _trainer(tmp_path, tokenizer, mixture_file, lora_phi)
    (batch_samples,) = _large_batches(ref, num_steps=1)
    for inputs in batch_samples + [{}]:
        assert new.floating_point_ops(inputs) == ref.floating_point_ops(inputs)
    new.create_optimizer_and_scheduler(num_training_steps=1)
    new.model.train()
    microbatches = new._select_microbatches([dict(b) for b in batch_samples])
    assert not new.model.training and not any(m.training for m in new.model.modules())
    for mb in microbatches:
        new.training_step(new.model, dict(mb))
    assert all(m.training for m in new.model.modules())
    assert np.isfinite(new.state.global_step)


def test_features_needed_uses_the_global_budget(
    tmp_path, tokenizer, mixture_file, lora_phi, monkeypatch
):
    # Rank 1 of 2: budgets come from both ranks' source ids, the mask is this rank's slice.
    import colm.train.trainers as trainers

    trainer = _trainer(tmp_path, tokenizer, mixture_file, lora_phi)
    per_rank = [[0, 0, 1, 1, 2, 3, 3, 3], [1, 1, 1, 1, 0, 2, 3, 3]]
    monkeypatch.setattr(trainers, "_all_gather_object", lambda obj: per_rank)
    monkeypatch.setattr(type(trainer.args), "process_index", property(lambda self: 1))
    monkeypatch.setattr(type(trainer.args), "world_size", property(lambda self: 2))
    got = trainer._features_needed(per_rank[1], num_per_rank=4)
    expected = features_needed(
        per_rank[0] + per_rank[1],
        total=8,
        keep_sources=[0],
        strategy="proportional",
        per_class_start="floor",
        need_selected=True,
        per_source_rng=False,
    )[8:]
    assert got.tolist() == expected.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel of colm_varlen")
def test_colm_varlen_cuda_matches_per_sequence_sdpa():
    gen = torch.Generator().manual_seed(0)
    lengths = [37, 1, 128, 300, 5]
    total, heads, dim = sum(lengths), 4, 80
    q, k, v = (torch.randn(1, heads, total, dim, generator=gen).cuda() for _ in range(3))
    cu = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32).cuda()

    class Module:
        is_causal = True

    with torch.inference_mode():
        out, _ = attention.varlen_attention_forward(
            Module(), q, k, v, None, scaling=dim**-0.5, cu_seq_lens_q=cu, max_length_q=max(lengths)
        )
        ref = attention._varlen_causal(
            *(t.transpose(1, 2).cpu() for t in (q, k, v)), cu.cpu(), max(lengths), dim**-0.5
        )
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)
