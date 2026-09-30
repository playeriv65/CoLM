"""Padding-free batches: layout, and that packing changes nothing about what the model computes."""

import numpy as np
import pytest
import torch
from equivalence.fixtures import model_fp64
from equivalence.helpers import build, make_args

from colm.selection.features import build_extractor, example_means
from colm.selection.packing import (
    Example,
    greedy_groups,
    label_counts,
    label_positions,
    model_inputs,
    pack,
)
from colm.selection.zo import LastLayerSplit


def _examples(tokenizer, texts, prompt=3, start=0):
    out = []
    for i, text in enumerate(texts):
        ids = tokenizer(text)["input_ids"]
        labels = np.array([-100] * prompt + ids[prompt:])
        out.append(
            Example(
                np.array(ids),
                labels,
                source=i % 2,
                index=start + i,
                completion_length=len(ids) - prompt,
            )
        )
    return out


TEXTS = ["Hello world, this is CoLM.", "Short one.", "A third, slightly longer example!", "x y z w"]


@pytest.fixture()
def model(tokenizer):
    model = model_fp64(tokenizer)
    model.eval()
    return model


def test_pack_layout(tokenizer):
    examples = _examples(tokenizer, TEXTS)
    batch = pack(examples)
    lengths = [len(e) for e in examples]
    assert batch["input_ids"].shape == (1, sum(lengths))
    assert batch["cu_seq_lens_q"].tolist() == np.cumsum([0] + lengths).tolist()
    assert batch["max_length_q"] == max(lengths)
    starts = batch["cu_seq_lens_q"][:-1].long()
    assert (batch["position_ids"][0, starts] == 0).all() and (
        batch["labels"][0, starts] == -100
    ).all()
    assert batch["position_ids"][0, 1].item() == 1
    positions, targets, segment = label_positions(batch)
    # every position predicts a token of the same example, and that token is a label
    for p, t, s in zip(positions.tolist(), targets.tolist(), segment.tolist(), strict=True):
        assert batch["cu_seq_lens_q"][s] <= p < batch["cu_seq_lens_q"][s + 1] - 1
        assert t == examples[s].input_ids[p - int(batch["cu_seq_lens_q"][s]) + 1]
    assert torch.bincount(segment).tolist() == [e.num_labels for e in examples]


def _torch_geometry(examples):
    """The label geometry of a pack with torch operations (how `pack` computed it before it
    was moved to numpy to avoid the intra-op thread pool)."""
    lengths = torch.tensor([len(e) for e in examples])
    cu = torch.zeros(len(examples) + 1, dtype=torch.int32)
    cu[1:] = lengths.cumsum(0)
    labels = torch.from_numpy(np.concatenate([e.labels for e in examples]))
    labels[cu[:-1].long()] = -100
    starts = cu[:-1].long()
    position_ids = torch.arange(int(cu[-1])) - torch.repeat_interleave(starts, lengths)
    targets = torch.cat([labels[1:], labels.new_full((1,), -100)])
    positions = (targets != -100).nonzero(as_tuple=True)[0]
    segment = torch.bucketize(positions, cu[1:].long(), right=True)
    return {
        "position_ids": position_ids[None],
        "cu_seq_lens_q": cu,
        "labels": labels[None],
        "label_positions": positions,
        "label_targets": targets[positions],
        "label_segment": segment,
        "label_counts": torch.bincount(segment, minlength=len(examples)),
    }


def test_pack_equals_the_torch_computation_including_examples_without_labels():
    rng = np.random.default_rng(0)
    for trial in range(50):
        examples = []
        for _ in range(int(rng.integers(1, 8))):
            n = int(rng.integers(2, 40))
            ids = rng.integers(0, 100, n)
            cut = int(rng.choice([0, 1, n // 2, n]))  # n: nothing to predict in this example
            labels = ids.copy()
            labels[:cut] = -100
            examples.append(Example(ids, labels, int(rng.integers(0, 4)), trial))
        batch, expected = pack(examples), _torch_geometry(examples)
        meta = batch["colm_meta"]
        got = {**batch, **meta}
        for key, value in expected.items():
            assert got[key].dtype == value.dtype and torch.equal(got[key], value), key


def test_greedy_groups():
    assert greedy_groups([3, 3, 3, 10, 1], 6) == [[0, 1], [2], [3], [4]]
    assert greedy_groups([5], 1) == [[0]]


def test_packed_logits_equal_the_separate_ones(tokenizer, model):
    """Sequences of a pack do not see each other: logits per sequence = logits of the sequence alone."""
    examples = _examples(tokenizer, TEXTS)
    batch = pack(examples)
    with torch.no_grad():
        packed = model(**model_inputs(batch)).logits[0]
        for k, e in enumerate(examples):
            start, end = int(batch["cu_seq_lens_q"][k]), int(batch["cu_seq_lens_q"][k + 1])
            alone = model(input_ids=torch.tensor(e.input_ids)[None], use_cache=False).logits[0]
            torch.testing.assert_close(packed[start:end], alone, rtol=0, atol=1e-10)


def test_last_layer_split_on_a_pack(tokenizer, model):
    batch = pack(_examples(tokenizer, TEXTS))
    split = LastLayerSplit(model.get_base_model())
    with torch.no_grad():
        state = split.prefix(**model_inputs(batch))  # also runs the built-in verification
        replay = split.hidden(state)
        full = split.decoder(**model_inputs(batch)).last_hidden_state
    torch.testing.assert_close(replay, full, rtol=0, atol=1e-10)


def test_model_inputs_carry_the_flash_layout_and_no_cache(tokenizer):
    """`position_ids` and the cumulative lengths of the flash kernels; a cache would end the
    packed-batch detection of the stock mask / flash paths, so it is switched off."""
    batch = pack(_examples(tokenizer, TEXTS))
    inputs = model_inputs(batch)
    assert inputs["use_cache"] is False and "attention_mask" not in inputs
    assert (
        inputs["cu_seq_lens_q"] is batch["cu_seq_lens_q"] and inputs["position_ids"].shape[0] == 1
    )


def test_the_loss_of_an_example_does_not_depend_on_its_pack(tmp_path, tokenizer, mixture_file):
    """MeZO feature (E2): identical alone, with short companions or with long ones."""
    args = make_args(
        tmp_path, per_device_train_batch_size=4, gradient_accumulation_steps=2, efficient_mezo=True
    )
    trainer, model = build(args, tokenizer, mixture_file)
    model.eval()
    extractor = trainer.extractor
    a, b, c, d = _examples(tokenizer, TEXTS)
    long = _examples(
        tokenizer, ["A much longer companion example that is far longer than the others, yes."]
    )[0]
    alone = extractor.extract(trainer._prepare_inputs(pack([a])))
    with_short = extractor.extract(trainer._prepare_inputs(pack([a, b])))
    with_long = extractor.extract(trainer._prepare_inputs(pack([long, a, c])))
    torch.testing.assert_close(with_short[0], alone[0], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(with_long[1], alone[0], rtol=1e-9, atol=1e-12)


def test_example_means_are_token_means(tokenizer, model):
    examples = _examples(tokenizer, TEXTS)
    batch = pack(examples)
    positions, targets, segment = label_positions(batch)
    with torch.no_grad():
        logits = model(**model_inputs(batch), logits_to_keep=positions).logits[0]
        means = example_means(logits, targets, segment, label_counts(batch))
        for k, e in enumerate(examples):
            loss = model(
                input_ids=torch.tensor(e.input_ids)[None], labels=torch.tensor(e.labels)[None]
            ).loss
            torch.testing.assert_close(
                means[k].float(), loss, rtol=1e-6, atol=1e-7
            )  # HF casts the logits to fp32


def test_every_extractor_runs_on_packs(tmp_path, tokenizer, mixture_file):
    for unit in ("rep", "mezo", "masked_grad", "completion_length", "length_loss_weighted"):
        args = make_args(
            tmp_path,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            data_selection_unit=unit,
            keep_sources="",
        )
        trainer, model = build(args, tokenizer, mixture_file)
        extractor = build_extractor(args, model, trainer.zo_params, 1)
        model.train(extractor.mode == "train")
        values = extractor.extract(trainer._prepare_inputs(pack(_examples(tokenizer, TEXTS[:1]))))
        assert values.shape[0] == 1 and torch.isfinite(values.double()).all(), unit
