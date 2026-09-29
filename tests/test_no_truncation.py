"""Nothing is truncated: an example above the model context is dropped and counted (CPU)."""

import json
import logging
import re
from pathlib import Path

import pytest

from colm.data import utils
from colm.data.get_training_dataset import IGNORE_INDEX, get_training_dataset
from colm.data.superglue import convert_samples
from colm.data.tasks import Sample
from colm.eval.eval_loss import load_gsm8k_test
from colm.train.config import parse_args
from colm.train.data_arguments import DataArguments

REPO = Path(__file__).resolve().parents[1]
CONTEXT = 400  # the prompt template alone has ~130 tokens
SHORT, LONG = "a" * 10, "b" * 1000  # characters are tokens in the test tokenizer


def write_jsonl(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def dropped_log(caplog, count, total):
    text = " ".join(r.getMessage() for r in caplog.records)
    assert f"Dropped {count} of {total} examples longer than {CONTEXT}" in text
    return text


def test_instruction_output_examples_above_the_context_are_dropped(tmp_path, tokenizer, caplog):
    rows = [
        {"instruction": SHORT, "output": SHORT, "source": "s0"},
        {"instruction": SHORT, "output": LONG, "source": "s1"},
        {"instruction": LONG, "output": SHORT, "source": "s1"},
        {"instruction": SHORT, "output": SHORT, "source": "s1"},
    ]
    with caplog.at_level(logging.INFO, logger="colm.data.get_training_dataset"):
        data = get_training_dataset([write_jsonl(tmp_path, rows)], tokenizer, CONTEXT)
    text = dropped_log(caplog, 2, 4)
    assert "'s1': 2" in text  # counted per source
    assert len(data) == 2 and all(t.endswith(SHORT + "</s>") for t in data.targets)


def test_prompt_completion_format_drops_and_never_cuts(tmp_path, tokenizer, caplog):
    rows = [
        {"prompt": SHORT, "completion": SHORT, "dataset": "d0"},
        {"prompt": SHORT, "completion": LONG, "dataset": "d1"},
        {"prompt": LONG, "completion": SHORT, "dataset": "d1"},
    ]
    with caplog.at_level(logging.INFO, logger="colm.data.get_training_dataset"):
        data = get_training_dataset([write_jsonl(tmp_path, rows)], tokenizer, CONTEXT)
    text = dropped_log(caplog, 2, 3)
    assert "'d1': 2" in text and len(data) == 1
    ids, labels = data[0]["input_ids"].tolist(), data[0]["labels"].tolist()
    assert ids == tokenizer(SHORT)["input_ids"] + tokenizer(SHORT + "</s>")["input_ids"]
    prompt = len(tokenizer(SHORT)["input_ids"])
    assert labels[:prompt] == [IGNORE_INDEX] * prompt and labels[prompt:] == ids[prompt:]
    assert ids[-1] == tokenizer.eos_token_id  # the EOS is kept


def test_messages_format_drops_and_never_cuts(tmp_path, tokenizer, caplog):
    def chat(question, answer, dataset):
        return {
            "dataset": dataset,
            "id": dataset,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    rows = [chat(SHORT, SHORT, "d0"), chat(SHORT, LONG, "d1"), chat(LONG, SHORT, "d1")]
    with caplog.at_level(logging.INFO, logger="colm.data.get_training_dataset"):
        data = get_training_dataset([write_jsonl(tmp_path, rows)], tokenizer, CONTEXT)
    text = dropped_log(caplog, 2, 3)
    assert "'d1': 2" in text and len(data) == 1
    ids, labels = data[0]["input_ids"], data[0]["labels"]
    whole = tokenizer(f"<|user|>\n{SHORT}\n<|assistant|>\n{SHORT}</s>\n")["input_ids"]
    assert ids.tolist() == whole  # the whole text
    kept = ids[labels != IGNORE_INDEX].tolist()
    assert (
        kept == tokenizer(f"{SHORT}</s>\n", add_special_tokens=False)["input_ids"]
    )  # assistant only


def test_context_boundary_is_inclusive(tmp_path, tokenizer):
    rows = [{"instruction": "", "output": "c" * n, "source": "s"} for n in (1, 2)]
    path = write_jsonl(tmp_path, rows)
    whole = get_training_dataset([path], tokenizer, CONTEXT)
    exact = len(tokenizer(whole.sources[0])["input_ids"]) + 2  # prompt + 1 char + EOS
    assert len(get_training_dataset([path], tokenizer, exact)) == 1  # the 1-char example fits
    with pytest.raises(ValueError, match="no example fits"):
        get_training_dataset([path], tokenizer, exact - 1)


def test_eval_loss_gsm8k_drops_examples_above_the_context(tmp_path, tokenizer):
    rows = [
        {"question": "What is 1+1?", "answer": "1+1=<<1+1=2>>2\n#### 2"},
        {"question": "x" * 1000, "answer": "y\n#### 1"},
    ]
    path = write_jsonl(tmp_path, rows)
    assert len(load_gsm8k_test(path, tokenizer, 2000)) == 2
    assert len(load_gsm8k_test(path, tokenizer, CONTEXT)) == 1


class StubTemplate:
    def encode(self, sample):
        return sample.data["text"]

    def verbalize(self, sample, candidate):
        return sample.data["text"] + " " + candidate


class StubTask:
    generation, classification, train_sep = True, False, "\n\n"

    def get_template(self):
        return StubTemplate()


def superglue_samples():
    return [
        Sample(id=i, data={"text": text}, correct_candidate="yes", candidates=["yes"])
        for i, text in enumerate([SHORT, LONG, SHORT])
    ]


def test_superglue_prompts_never_truncate_they_raise(tokenizer):
    task, sample = StubTask(), superglue_samples()[1]
    with pytest.raises(utils.PromptTooLong, match=f"does not fit the context window of {CONTEXT}"):
        utils.encode_prompt(
            task, task.get_template(), [], sample, tokenizer, CONTEXT, generation=True
        )
    encoded, _ = utils.encode_prompt(
        task, task.get_template(), [], sample, tokenizer, 2000, generation=True
    )
    assert len(encoded[0]) >= len(LONG)  # the whole prompt
    short = superglue_samples()[0]  # room for the generated tokens is part of the budget
    fits = len(tokenizer.encode(SHORT))
    utils.encode_prompt(task, task.get_template(), [], short, tokenizer, fits, generation=True)
    with pytest.raises(utils.PromptTooLong):
        utils.encode_prompt(
            task, task.get_template(), [], short, tokenizer, fits, generation=True, max_new_tokens=5
        )


def test_superglue_training_drops_samples_above_the_context(tokenizer, caplog):
    with caplog.at_level(logging.INFO, logger="colm.data.superglue"):
        data = convert_samples(superglue_samples(), StubTask(), tokenizer, CONTEXT, True)
    assert len(data) == 2 and [d["indices"] for d in data] == [0, 2]
    assert f"Dropped 1 of 3 samples longer than {CONTEXT} tokens" in caplog.text


def test_no_option_sets_a_sequence_cut(tmp_path):
    fields = {f for f in DataArguments.__dataclass_fields__}
    assert not [f for f in fields if re.search(r"max_seq|max_length|truncat|cutoff", f)]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_seq_length": 512}))
    with pytest.raises(ValueError, match="unknown config keys"):
        parse_args([str(config)])
    for script in ("math_eval/run_open.py", "superglue_eval/eval_superglue.py"):
        assert not re.search(
            r"--(max_length|model_max_length|max_seq_length)", (REPO / script).read_text()
        )


CUT_PATTERNS = [
    r"truncation\s*=\s*True",
    r"truncation\s*=\s*['\"]",
    r"max_seq_length",
    r"sequence_limit",
    r"--model_max_length",
    r"model_max_length\"\s*:\s*\d",
    r"\[\s*-\s*\(?max_length",
    r"padding\s*=\s*['\"]max_length",
]


def test_no_truncation_call_remains():
    """Grep-style guard over the code and the configs (`max_length_q/k` are flash kwargs)."""
    files = [
        p
        for folder in ("colm", "math_eval", "superglue_eval", "scripts", "configs")
        for p in (REPO / folder).rglob("*")
        if p.suffix in (".py", ".json", ".sh") and "dataset" not in p.parts
    ]
    assert files
    found = [
        f"{p.relative_to(REPO)}: {pattern}"
        for p in files
        for pattern in CUT_PATTERNS
        if re.search(pattern, p.read_text())
    ]
    assert not found, found
