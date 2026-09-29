"""Execution-only step optimisations: same features, selections and gradients as the plain path.

Every test compares the optimised code with the straightforward computation it replaces, in
float64 on CPU (fp32 makes the MeZO feature decided at rounding level, `docs/errors.md`).
"""

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from equivalence.helpers import build, make_args
from torch.func import functional_call

from colm.selection.batching import UNLIMITED
from colm.selection.features import example_means
from colm.selection.packing import (
    IGNORE_INDEX,
    balanced_shares,
    greedy_groups,
    label_counts,
    label_positions,
    pack,
)
from colm.selection.select import CoresetSelector

# ---------------------------------------------------------------------------------------------
# Label geometry computed on the CPU at pack time (no device synchronisation later)
# ---------------------------------------------------------------------------------------------


def test_label_geometry_of_a_pack_equals_the_device_computation(tokenizer, mixture_file):
    from colm.data.get_training_dataset import get_training_dataset

    dataset = get_training_dataset([mixture_file], tokenizer=tokenizer, max_seq_length=512)
    examples = [dataset[i] for i in range(9)]
    from colm.data.get_training_dataset import make_collator

    args = make_args("unused", efficient_mezo=True)
    batch = make_collator(args, tokenizer)(examples)["examples"]
    row = pack(batch)
    # The former computation, on the packed labels themselves.
    labels = row["labels"][0]
    targets = torch.cat([labels[1:], labels.new_full((1,), IGNORE_INDEX)])
    positions = (targets != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    segment = torch.bucketize(positions, row["cu_seq_lens_q"][1:].long(), right=True)
    got = label_positions(row)
    assert torch.equal(got[0], positions)
    assert torch.equal(got[1], targets[positions])
    assert torch.equal(got[2], segment)
    assert torch.equal(label_counts(row), torch.bincount(segment, minlength=len(batch)))
    assert label_counts(row).tolist() == [e.num_labels for e in batch]


# ---------------------------------------------------------------------------------------------
# balanced_shares
# ---------------------------------------------------------------------------------------------


def test_balanced_shares_cover_the_wanted_items_and_balance_the_tokens():
    rng = np.random.default_rng(0)
    for parts in (1, 2, 3, 4):
        lengths = rng.integers(5, 400, size=23).tolist()
        wanted = rng.random(23) < 0.7
        shares = balanced_shares(lengths, wanted, parts)
        flat = sorted(i for share in shares for i in share)
        assert flat == np.flatnonzero(wanted).tolist()  # each wanted item exactly once
        assert all(share == sorted(share) for share in shares)  # pool order kept
        loads = [sum(lengths[i] for i in share) for share in shares]
        assert max(loads) - min(loads) <= max(lengths)
        assert shares == balanced_shares(lengths, wanted, parts)  # deterministic: all ranks agree
    assert balanced_shares([3, 4, 5], np.array([False, False, False]), 2) == [[], []]


# ---------------------------------------------------------------------------------------------
# O1: which features the selection can use
# ---------------------------------------------------------------------------------------------

SELECTOR_CASES = [
    dict(mezo_optim="adam", num_per_class_start="floor", mezo_topk="largest"),
    dict(mezo_optim="sgd", num_per_class_start="floor", mezo_topk="largest"),
    dict(mezo_optim="adam", num_per_class_start="ceil", mezo_topk="smallest"),
    dict(mezo_optim="sgd", num_per_class_start="ceil", mezo_topk="largest_smallest"),
    dict(mezo_optim="adam", num_per_class_start="floor", mezo_topk="random"),
]


@pytest.mark.parametrize("case", range(len(SELECTOR_CASES)))
def test_unneeded_features_do_not_change_the_selection(tmp_path, case):
    """Zeroing every feature `needed` says is unused leaves indices, weights and Adam state alone."""
    rng = np.random.default_rng(case)
    skipped = 0
    for trial in range(60):
        n, num_sources = int(rng.integers(8, 40)), int(rng.integers(2, 9))
        keep = "_".join(str(s) for s in rng.choice(num_sources, size=int(rng.integers(0, 3))))
        args = make_args(
            tmp_path,
            efficient_mezo=True,
            keep_sources=keep if keep else "",
            zo_dim=8,
            **SELECTOR_CASES[case],
        )
        total = int(rng.integers(1, n))
        draws = [
            (
                rng.choice(num_sources, size=n, p=rng.dirichlet(np.ones(num_sources) * 0.6)),
                torch.from_numpy(rng.normal(size=(n, 20))),
            )
            for _ in range(3)
        ]
        selectors = [CoresetSelector(args, 2), CoresetSelector(args, 2)]
        try:  # some random pools are not selectable at all (e.g. only kept sources)
            for step, (sources, feats) in enumerate(draws):
                torch.manual_seed(step)
                CoresetSelector(args, 2)(feats, sources.tolist(), total, 0)
        except (IndexError, ValueError):
            continue
        for step, (sources, feats) in enumerate(draws):  # the Adam moments carry over
            wanted = selectors[0].needed(sources.tolist(), total)
            assert wanted.shape == (n,)
            skipped += int((~wanted).sum())
            zeroed = feats * torch.from_numpy(wanted)[:, None]
            torch.manual_seed(step)
            full = selectors[0](feats, sources.tolist(), total, step)
            torch.manual_seed(step)
            part = selectors[1](zeroed, sources.tolist(), total, step)
            assert part.indices == full.indices, (case, trial, step)
            assert part.weights == full.weights
            for a, b in (
                (selectors[0].prev_m, selectors[1].prev_m),
                (selectors[0].prev_v, selectors[1].prev_v),
            ):
                assert (a is None) == (b is None)
                if a is not None:
                    assert torch.equal(a, b)
    assert skipped > 0  # the test is not vacuous


def test_needed_uses_every_candidate_where_the_rows_interact(tmp_path):
    sources = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    common = dict(efficient_mezo=True, keep_sources="0", zo_dim=8)
    for extra in (
        dict(source_wise_selection="balanced"),
        dict(source_wise_selection="none"),
        dict(mezo_topk="sampling"),
    ):
        selector = CoresetSelector(make_args(tmp_path, **common, **extra), 2)
        assert selector.needed(sources, 4).tolist() == [False] + [True] * 9, extra
    selector = CoresetSelector(make_args(tmp_path, **common), 2)
    assert not selector.needed(sources, 1).any()  # the kept example fills the budget
    # A zero quota (budget 1 over three sources: floor gives 0, 0, 0 and one is filled up).
    wanted = selector.needed(sources, 2)
    assert wanted.sum() < 9 and not wanted[0]


# ---------------------------------------------------------------------------------------------
# O1 + O3 in the trainer: same selection, fewer forwards; g_i equal an independent estimate
# ---------------------------------------------------------------------------------------------


def _skewed_mixture(path):
    """64 rows over 6 sources of very different sizes (so that some get a zero quota)."""
    cycle = [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
    rows = []
    for i in range(64):
        rows.append(
            {
                "instruction": f"Add {i} and {i % 7}.",
                "input": "" if i % 2 else f"numbers {i}",
                "output": f"The answer is {i + i % 7}." + " ok" * (i % 5),
                "source": f"source_{cycle[i % len(cycle)]}",
                "original_index": i,
                "completion_length": 5 + i % 5,
            }
        )
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")


def _efficient_trainer(tmp_path, tokenizer, data, **kwargs):
    args = make_args(
        tmp_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        small_batch_ratio=0.5,
        efficient_mezo=True,
        **kwargs,
    )
    return build(args, tokenizer, data)


def _plain_features(trainer, examples):
    """The features of the whole pool the former way: every example, features per pack."""
    packs = trainer.batching.feature_batches(examples)
    g = torch.cat([trainer.extractor.extract(trainer._prepare_inputs(p)) for p in packs])
    return trainer.extractor.expand(g)


@pytest.mark.parametrize("keep", ["", "0", "0_1"])
@pytest.mark.parametrize("optim", ["adam", "sgd"])
def test_skipping_unused_features_selects_the_same_examples(
    tmp_path, tokenizer, keep, optim, monkeypatch
):
    data = str(tmp_path / "skewed.jsonl")
    _skewed_mixture(data)
    trainer, model = _efficient_trainer(
        tmp_path, tokenizer, data, keep_sources=keep, mezo_optim=optim
    )
    model.eval()
    loader = iter(trainer.get_train_dataloader())
    forwarded = []
    extract = trainer.extractor.extract

    def counting(pack_):
        forwarded.append(len(pack_["cu_seq_lens_q"]) - 1)
        return extract(pack_)

    monkeypatch.setattr(trainer.extractor, "extract", counting)
    skipped_any = False
    for step in range(4):
        inputs = next(loader)
        reference = CoresetSelector(trainer.args, 2)
        reference.prev_m, reference.prev_v = trainer.selector.prev_m, trainer.selector.prev_v
        pool = inputs["examples"]
        feats = _plain_features(trainer, pool)
        sources = [e.source for e in pool]
        total = trainer._per_rank(trainer.args.pool_micro_batches)
        expected = reference(feats, sources, total, step)

        trainer.state.global_step = step
        forwarded.clear()
        sub_batches, _ = trainer._select(inputs)
        skipped_any |= sum(forwarded) < len(pool)
        got = [i for b, _ in sub_batches for i in b["colm_meta"]["indices"].tolist()]
        want = [pool[i].index for i in expected.indices]
        assert got == want, step
        for got_state, want_state in (
            (trainer.selector.prev_m, reference.prev_m),
            (trainer.selector.prev_v, reference.prev_v),
        ):
            assert (got_state is None) == (want_state is None)
            if got_state is not None:
                assert torch.equal(got_state, want_state)
    if keep:
        assert skipped_any  # the kept sources at least were not forwarded


def test_the_estimate_equals_an_independent_two_forward_estimate(tmp_path, tokenizer, mixture_file):
    """g_i = (L(theta + eps z) - L(theta - eps z)) / 2 eps of the whole model, example by example."""
    trainer, model = _efficient_trainer(tmp_path, tokenizer, mixture_file, keep_sources="")
    model.eval()
    inputs = next(iter(trainer.get_train_dataloader()))
    examples = inputs["examples"][:6]
    extractor = trainer.extractor
    g = torch.cat(
        [
            extractor.extract(trainer._prepare_inputs(p))
            for p in trainer.batching.feature_batches(examples)
        ]
    )
    eps = trainer.args.mezo_eps
    zs = extractor.perturbation.z()
    names = extractor.perturbation.names
    params = extractor.perturbation.params

    def loss(e, sign):
        overrides = {n: p + sign * eps * z for n, p, z in zip(names, params, zs, strict=True)}
        out = functional_call(
            model,
            overrides,
            (),
            {"input_ids": torch.tensor(e.input_ids)[None], "use_cache": False},
        )
        logits = out.logits[0, :-1]
        target = torch.tensor(e.labels)[1:]
        return F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)

    with torch.no_grad():
        want = torch.stack([(loss(e, 1) - loss(e, -1)) / (2 * eps) for e in examples])
    torch.testing.assert_close(g, want, rtol=1e-8, atol=1e-12)
    # The features are the same g_i z the gather used to move.
    feats = extractor.expand(g)
    z = torch.cat([z.flatten() for z in zs])
    torch.testing.assert_close(feats, g[:, None] * z[None], rtol=0, atol=0)


def test_choose_from_the_shares_of_two_ranks_equals_one_pool(tmp_path, tokenizer, mixture_file):
    """Rank 0 assembles the pool's g_i from (positions, values) of each rank: same selection."""
    trainer, model = _efficient_trainer(tmp_path, tokenizer, mixture_file, keep_sources="0")
    model.eval()
    pool = next(iter(trainer.get_train_dataloader()))["examples"]
    sources = [e.source for e in pool]
    total = trainer._per_rank(trainer.args.pool_micro_batches)
    wanted = trainer.selector.needed(sources, total)
    shares = balanced_shares([len(e) for e in pool], wanted, 2)
    gathered = []
    for positions in shares:
        packs = trainer.batching.feature_batches([pool[i] for i in positions])
        values = torch.cat([trainer.extractor.extract(trainer._prepare_inputs(p)) for p in packs])
        gathered.append((positions, values))
    trainer.state.global_step = 0
    indices, weights = trainer._choose(pool, sources, gathered, total)
    reference = CoresetSelector(trainer.args, 2)
    expected = reference(_plain_features(trainer, pool), sources, total, 0)
    assert indices == expected.indices and weights == expected.weights


# ---------------------------------------------------------------------------------------------
# O8: all selected examples in one forward = the micro-batch loop
# ---------------------------------------------------------------------------------------------


def _gradients(trainer, model, examples, weights, groups):
    total = trainer.batching.total_labels(examples)
    model.zero_grad(set_to_none=True)
    model.train()
    loss = 0.0
    for group in groups:
        batch = pack([examples[i] for i in group])
        w = torch.tensor([weights[i] for i in group])
        part = trainer.batching.loss(trainer, model, batch, w, total)
        part.backward()
        loss += float(part)
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return loss, grads


@pytest.mark.parametrize("weighted", [False, True])
def test_one_forward_has_the_gradient_of_the_micro_batch_loop(
    tmp_path, tokenizer, mixture_file, weighted
):
    trainer, model = _efficient_trainer(tmp_path, tokenizer, mixture_file, keep_sources="")
    examples = next(iter(trainer.get_train_dataloader()))["examples"]
    weights = np.linspace(0.5, 2.0, len(examples)).tolist() if weighted else [1.0] * len(examples)
    lengths = [len(e) for e in examples]
    micro = greedy_groups(lengths, 2 * int(np.mean(lengths)))  # the former, micro-batch sized packs
    assert len(micro) > 2
    loss_loop, grads_loop = _gradients(trainer, model, examples, weights, micro)
    everything = [list(range(len(examples)))]
    loss_one, grads_one = _gradients(trainer, model, examples, weights, everything)
    budget = greedy_groups(lengths, sum(lengths) // 3)  # rows under a token budget
    assert 1 < len(budget) < len(micro)
    loss_rows, grads_rows = _gradients(trainer, model, examples, weights, budget)
    assert loss_one == pytest.approx(loss_loop, rel=1e-12)
    assert loss_rows == pytest.approx(loss_loop, rel=1e-12)
    torch.testing.assert_close(grads_one, grads_loop, rtol=1e-9, atol=1e-13)
    torch.testing.assert_close(grads_rows, grads_loop, rtol=1e-9, atol=1e-13)
    assert grads_loop.norm() > 0


def _step_gradients(trainer, model, examples, weights):
    """Loss and gradients of the real training loop (`_train_packs`) on the given selection."""
    model.zero_grad(set_to_none=True)
    model.train()
    sub_batches = trainer.batching.train_batches(examples, weights)
    loss = trainer._train_packs(model, sub_batches, trainer.batching.total_labels(examples))
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return len(sub_batches), float(loss), grads


def test_unlimited_and_bounded_budgets_train_the_same_step(tmp_path, tokenizer, mixture_file):
    """`train_max_tokens=0` (one pack) and a small budget (several packs): same loss and gradient."""
    trainer, model = _efficient_trainer(
        tmp_path, tokenizer, mixture_file, keep_sources="", train_max_tokens=0
    )
    assert trainer.batching.train_tokens == UNLIMITED
    examples = next(iter(trainer.get_train_dataloader()))["examples"]
    weights = np.linspace(0.5, 2.0, len(examples)).tolist()
    packs_all, loss_all, grads_all = _step_gradients(trainer, model, examples, weights)
    trainer.batching.train_tokens = sum(len(e) for e in examples) // 4
    packs_few, loss_few, grads_few = _step_gradients(trainer, model, examples, weights)
    assert packs_all == 1 and packs_few > 2
    assert loss_few == pytest.approx(loss_all, rel=1e-12)
    torch.testing.assert_close(grads_few, grads_all, rtol=1e-9, atol=1e-13)
    assert grads_all.norm() > 0


def test_the_budget_is_a_bounded_default_and_out_of_memory_says_what_to_set(
    tmp_path, tokenizer, mixture_file, monkeypatch
):
    trainer, model = _efficient_trainer(tmp_path, tokenizer, mixture_file, keep_sources="")
    assert trainer.args.train_max_tokens > 0
    assert trainer.batching.train_tokens == trainer.args.train_max_tokens
    examples = next(iter(trainer.get_train_dataloader()))["examples"]

    def out_of_memory(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(trainer.batching, "loss", out_of_memory)
    sub_batches = trainer.batching.train_batches(examples, [1.0] * len(examples))
    with pytest.raises(torch.OutOfMemoryError, match="lower it"):
        trainer._train_packs(model, sub_batches, 1)
    trainer.batching.train_tokens = UNLIMITED
    with pytest.raises(torch.OutOfMemoryError, match="set train_max_tokens to a positive value"):
        trainer._train_packs(model, sub_batches, 1)


def test_train_batches_pack_the_selection_under_the_token_budget(tmp_path, tokenizer, mixture_file):
    trainer, _ = _efficient_trainer(tmp_path, tokenizer, mixture_file, keep_sources="")
    examples = next(iter(trainer.get_train_dataloader()))["examples"]
    weights = [1.0] * len(examples)
    assert trainer.batching.train_tokens == trainer.args.train_max_tokens
    trainer.batching.train_tokens = 2**62  # no limit: one forward
    assert len(trainer.batching.train_batches(examples, weights)) == 1
    trainer.batching.train_tokens = sum(len(e) for e in examples) // 2
    batches = trainer.batching.train_batches(examples, weights)
    assert len(batches) > 1
    assert all(
        int(b["cu_seq_lens_q"][-1]) <= trainer.batching.train_tokens or len(w) == 1
        for b, w in batches
    )
    assert sorted(i for b, _ in batches for i in b["colm_meta"]["indices"].tolist()) == sorted(
        e.index for e in examples
    )


def test_example_means_use_the_counts_of_the_pack(tokenizer):
    logits = torch.randn(5, 7, dtype=torch.float64)
    targets = torch.tensor([1, 2, 3, 4, 5])
    segment = torch.tensor([0, 0, 1, 2, 2])
    counts = torch.tensor([2, 1, 2])
    token = F.cross_entropy(logits, targets, reduction="none")
    expected = torch.stack([token[:2].mean(), token[2:3].mean(), token[3:].mean()])
    torch.testing.assert_close(example_means(logits, targets, segment, counts), expected)


def test_mode_switch_equals_model_train(tokenizer):
    from equivalence.fixtures import model_fp64

    from colm.train.trainers import ModeSwitch

    model = model_fp64(tokenizer, lora_dropout=0.1, resid_pdrop=0.1)
    switch = ModeSwitch(model)
    assert switch.flat  # PEFT + transformers modules do not override train()
    for training in (True, False, True):
        switch(training)
        assert all(m.training == training for m in model.modules())

    # A module that overrides train() sends the whole model back to the recursive method.
    class Custom(torch.nn.Linear):
        def train(self, mode=True):
            self.calls = getattr(self, "calls", 0) + 1
            return super().train(mode)

    net = torch.nn.Sequential(torch.nn.Linear(2, 2), Custom(2, 2))
    fallback = ModeSwitch(net)
    assert not fallback.flat
    fallback(False)
    assert not net.training and not net[1].training and net[1].calls == 1


def test_a_loader_worker_yields_the_same_pools(tmp_path, tokenizer, mixture_file):
    """The default loader has a worker (it tokenises the next pool during the step): same batches."""
    pools = []
    for workers in (0, 1):
        args = make_args(
            tmp_path,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            efficient_mezo=True,
            dataloader_num_workers=workers,
        )
        trainer, _ = build(args, tokenizer, mixture_file)
        loader = iter(trainer.get_train_dataloader())
        pools.append([[e.index for e in next(loader)["examples"]] for _ in range(3)])
    assert pools[0] == pools[1]
