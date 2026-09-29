"""SuperGLUE-style training data: option collators, classification loss, trainers on them."""

import math

import pytest
import torch
import torch.nn.functional as F
from equivalence.helpers import make_args, model_fp64

from colm.data.get_training_dataset import ExampleCollator
from colm.data.superglue import (
    ClassificationCollator,
    ListDataset,
    OptionCollator,
    classification_loss,
    option_examples,
)
from colm.train import attention
from colm.train.trainers import CustomTrainer, SubsetTrainer


def _ids(tokenizer, text):
    return tokenizer(text)["input_ids"]


def test_option_examples_mask_the_prompt(tokenizer):
    short, long = _ids(tokenizer, "abc de"), _ids(tokenizer, "abc def ghi")
    features = [
        {"input_ids": short, "option_len": 2, "sources": 1, "indices": 0},
        {"input_ids": long, "option_len": 3, "sources": 2, "indices": 1},
    ]
    examples = option_examples(features)
    assert [e.num_labels for e in examples] == [2, 3]  # the first token is never predicted
    assert (
        examples[0].labels[-2:].tolist() == short[-2:] and (examples[0].labels[:-2] == -100).all()
    )
    assert [e.source for e in examples] == [1, 2] and examples[1].completion_length == 3
    # legacy (E16): counted back from the padded width, the shorter example loses tokens
    padded = OptionCollator(tokenizer.pad_token_id)(features)["labels"]
    assert (padded[0] != -100).sum() < 2 and (padded[1] != -100).sum() == 3


def _candidates(tokenizer, texts, option_len, label):
    return [
        {
            "input_ids": _ids(tokenizer, t),
            "labels": label,
            "option_len": option_len,
            "num_options": len(texts),
        }
        for t in texts
    ]


def test_classification_loss_matches_a_per_candidate_computation(tokenizer, phi):
    features = [
        _candidates(tokenizer, ["q: 1 yes", "q: 1 no long"], 2, 1),
        _candidates(tokenizer, ["z: 2 a", "z: 2 bb", "z: 2 c"], 1, 0),
    ]
    batch = ClassificationCollator(tokenizer.pad_token_id)(features)
    phi.eval()
    with torch.no_grad():
        logits = phi(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = classification_loss(logits, batch)
        legacy = classification_loss(logits, batch, legacy=True)

        scores, row = [], 0
        for example in features:
            per = []
            for cand in example:
                ids = torch.tensor(cand["input_ids"])
                lg = phi(input_ids=ids[None]).logits[0, :-1]
                lp = F.log_softmax(lg, -1)[torch.arange(len(ids) - 1), ids[1:]]
                per.append(lp[-cand["option_len"] :].mean())
                row += 1
            scores.append((torch.stack(per), example[0]["labels"]))
    expected = torch.stack([F.cross_entropy(s[None], torch.tensor([y])) for s, y in scores]).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(loss, legacy)  # the padded candidates change the legacy value


def test_classification_baseline_trains(tmp_path, tokenizer):
    data = [_candidates(tokenizer, [f"q{i}: yes", f"q{i}: no"], 1, i % 2) for i in range(8)]
    args = make_args(
        tmp_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        data_selection_method="none",
        max_steps=2,
    )
    trainer = CustomTrainer(
        model=model_fp64(tokenizer),
        args=args,
        train_dataset=ListDataset(data),
        processing_class=tokenizer,
        data_collator=ClassificationCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    assert all(math.isfinite(h["loss"]) for h in trainer.state.log_history if "loss" in h)


@pytest.mark.parametrize("unit", ["mezo", "completion_length"])
def test_generation_task_with_selection(tmp_path, tokenizer, unit):
    data = [
        {
            "input_ids": _ids(tokenizer, f"question {i} answer {'x' * (i % 5)}"),
            "option_len": 2 + i % 3,
            "sources": i % 2,
            "indices": i,
        }
        for i in range(32)
    ]
    args = make_args(
        tmp_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        data_selection_unit=unit,
        keep_sources="",
        max_steps=2,
    )
    trainer = SubsetTrainer(
        model=model_fp64(tokenizer, attn=attention.register()),
        args=args,
        train_dataset=ListDataset(data),
        processing_class=tokenizer,
        data_collator=ExampleCollator(option_examples),
    )
    trainer.train()
    assert trainer.state.global_step == 2
