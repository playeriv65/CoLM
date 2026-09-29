"""Held-out split, teacher-forced eval loss, trainer callback and standalone CLI (CPU)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from conftest import make_phi

from colm.data.get_training_dataset import SupervisedCollator, get_training_dataset
from colm.data.holdout import select_examples, split_holdout
from colm.eval.arguments import HeldoutEvalArguments
from colm.eval.eval_loss import (
    build_eval_sets,
    clean_gsm8k_solution,
    evaluate_loss,
    evaluate_sets,
    load_gsm8k_test,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def dataset(mixture_file, tokenizer):
    return get_training_dataset([mixture_file], tokenizer, 512)


@pytest.fixture()
def gsm8k_file(tmp_path):
    rows = [
        {
            "question": f"What is {i}+{i}?",
            "answer": f"{i}+{i}=<<{i}+{i}={2 * i}>>{2 * i}\n#### {2 * i}",
        }
        for i in range(6)
    ]
    path = tmp_path / "gsm8k.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_clean_gsm8k_solution():
    raw = "Janet sells 16 - 3 = <<16-3=13>>13 eggs.\nShe makes 13 * 2 = $<<13*2=26>>26.\n#### 26"
    assert clean_gsm8k_solution(raw) == (
        "Janet sells 16 - 3 = 13 eggs.\nShe makes 13 * 2 = $26.\nThe answer is 26"
    )


def test_gsm8k_dataset_renders_like_training(gsm8k_file, tokenizer):
    data = load_gsm8k_test(gsm8k_file, tokenizer, 512)
    assert len(data) == 6 and data.all_data_sources == ["gsm8k_test"]
    assert data.sources[0].endswith("### Response:") and "What is 0+0?" in data.sources[0]
    assert data.targets[1] == "1+1=2\nThe answer is 2" + tokenizer.eos_token
    assert len(load_gsm8k_test(gsm8k_file, tokenizer, 512, limit=2)) == 2


def test_split_holdout_is_deterministic_and_disjoint(dataset):
    train, held = split_holdout(dataset, 10, seed=3)
    train2, held2 = split_holdout(dataset, 10, seed=3)
    assert len(train) == 54 and len(held) == 10
    assert held.indices == held2.indices and train.indices == train2.indices
    assert not set(held.indices) & set(train.indices)
    assert sorted(held.indices + train.indices) == list(range(64))
    assert held.indices == sorted(held.indices)  # order preserved
    assert held.all_data_sources == dataset.all_data_sources and held.num_sources == 4
    assert split_holdout(dataset, 10, seed=4)[1].indices != held.indices
    # Per-example lists stay aligned.
    for k, index in enumerate(held.indices):
        assert (
            held.sources[k] == dataset.sources[index] and held.targets[k] == dataset.targets[index]
        )
    with pytest.raises(ValueError):
        split_holdout(dataset, 0, seed=0)


def test_evaluate_loss_matches_per_example_hf_loss(phi, tokenizer, dataset):
    """Pooled loss = sum of per-example HF losses weighted by completion tokens; batch-free."""
    subset = select_examples(dataset, range(7))
    phi.eval()
    pooled = evaluate_loss(phi, subset, tokenizer, 3, torch.device("cpu"))
    single = evaluate_loss(phi, subset, tokenizer, 1, torch.device("cpu"))
    assert pooled["loss"] == pytest.approx(single["loss"], rel=1e-5)
    assert pooled["n_examples"] == 7

    collator = SupervisedCollator(tokenizer)
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(7):
            batch = collator([subset[i]])
            n = int((batch["labels"][:, 1:] != -100).sum())
            loss = phi(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss
            total += float(loss) * n
            count += n
    assert pooled["n_tokens"] == count
    assert pooled["loss"] == pytest.approx(total / count, rel=1e-4)
    assert set(pooled["per_source"]) == set(subset.all_data_sources)
    assert pooled["perplexity"] == pytest.approx(torch.exp(torch.tensor(pooled["loss"])).item())


def test_evaluate_sets_restores_training_mode(phi, tokenizer, dataset):
    phi.train()
    evaluate_sets(phi, {"a": select_examples(dataset, range(3))}, tokenizer, 2, torch.device("cpu"))
    assert phi.training


def test_build_eval_sets_validates(gsm8k_file, tokenizer, dataset):
    args = HeldoutEvalArguments(eval_loss_gsm8k_file=gsm8k_file)
    with pytest.raises(ValueError, match="holdout_size is 0"):
        build_eval_sets(args, tokenizer, None, 512)
    _, held = split_holdout(dataset, 5, seed=0)
    sets = build_eval_sets(args, tokenizer, held, 512, limit=2)
    assert {k: len(v) for k, v in sets.items()} == {"heldout": 2, "gsm8k": 2}


def test_train_callback_and_standalone_cli_agree(tmp_path, tokenizer, mixture_file, gsm8k_file):
    model_dir = tmp_path / "tiny-phi"
    make_phi(tokenizer).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = {
        "model_name_or_path": str(model_dir),
        "train_files": [mixture_file],
        "output_dir": str(tmp_path / "out"),
        "max_steps": 2,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "efficient_mezo": True,
        "last_layer_index": 1,
        "zo_dim": 16,
        "keep_sources": "0",
        "lora_r": 4,
        "lora_alpha": 16,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
        "use_cpu": True,
        "precision": "fp32",
        "selection_prefix_dtype": "float32",
        "save_strategy": "no",
        "holdout_size": 8,
        "holdout_seed": 1,
        "eval_loss_steps": [0, 2],
        "eval_loss_gsm8k_file": gsm8k_file,
        "eval_loss_batch_size": 3,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
    run = subprocess.run(
        [sys.executable, "-m", "colm.train.train", str(config_path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]

    out = tmp_path / "out"
    assert json.loads((out / "train_results.json").read_text())["train_samples"] == 56
    held = json.loads((out / "holdout_indices.json").read_text())["original_index"]
    assert len(held) == 8
    records = [json.loads(line) for line in (out / "eval_loss.jsonl").read_text().splitlines()]
    assert [(r["step"], r["set"]) for r in records] == [
        (0, "heldout"),
        (0, "gsm8k"),
        (2, "heldout"),
        (2, "gsm8k"),
    ]
    assert all(r["loss"] > 0 and r["n_tokens"] > 0 for r in records)
    state = json.loads((out / "trainer_state.json").read_text())
    logged = {h["step"]: h for h in state["log_history"] if "eval_heldout_loss" in h}
    assert set(logged) == {0, 2}
    final = {r["set"]: r["loss"] for r in records if r["step"] == 2}

    cli_out = tmp_path / "cli.json"
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "colm.eval.eval_loss",
            "--train_config",
            str(config_path),
            "--adapter",
            str(out),
            "--base",
            "--output",
            str(cli_out),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert cli.returncode == 0, cli.stdout[-3000:] + cli.stderr[-3000:]
    payload = json.loads(cli_out.read_text())
    by_label = {r["label"]: r["results"] for r in payload["records"]}
    assert set(by_label) == {"base", "out"}
    for name, loss in final.items():
        assert by_label["out"][name]["loss"] == pytest.approx(loss, rel=1e-4)
    # Step 0 of the callback is the untrained model: LoRA B starts at zero, so it equals the base.
    step0 = {r["set"]: r["loss"] for r in records if r["step"] == 0}
    for name, loss in step0.items():
        assert by_label["base"][name]["loss"] == pytest.approx(loss, rel=1e-4)


def test_holdout_keeps_the_questions_of_the_heldout_set_out_of_training(tokenizer, tmp_path):
    from colm.data.holdout import question_key

    rows = []
    for q in range(40):  # every question twice: a CoT and a PoT solution with the hint
        for pot in (False, True):
            rows.append(
                {
                    "instruction": f"What is {q} plus {q}?"
                    + (" Let's write a program." if pot else ""),
                    "output": f"{q + q}",
                    "source": "pot" if pot else "cot",
                    "original_index": len(rows),
                }
            )
    path = tmp_path / "twins.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    data = get_training_dataset([str(path)], tokenizer=tokenizer, context_length=512)
    train, held = split_holdout(data, 10, seed=0)
    assert len(held) == 10 and len(train) == 70
    assert not {question_key(p) for p in held.sources} & {question_key(p) for p in train.sources}
    assert question_key(
        "### Instruction:\nWhat is 3 plus 3? Let's write a program.\n"
    ) == question_key("### Instruction:\nwhat is 3 plus 3?")
