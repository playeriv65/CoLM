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
from colm.train import attention


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
    attention.register()
    model = model_fp64(tokenizer, attn=attention.NAME)
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


def test_a_pack_of_a_cache_is_not_the_trap(tokenizer, model):
    """The varlen attention reads cu_seq_lens, so even a cache cannot make sequences attend to each other."""
    batch = pack(_examples(tokenizer, TEXTS))
    with torch.no_grad():
        logits = model(**{**model_inputs(batch), "use_cache": True}).logits
        reference = model(**model_inputs(batch)).logits
    torch.testing.assert_close(logits, reference, rtol=0, atol=1e-10)


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
