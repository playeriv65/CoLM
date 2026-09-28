"""math_eval/run_open.py: argument checks, prompt building and the evaluation loop (stub engine)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def run_open(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "math_eval"))
    import run_open as module

    return module


def make_adapter(path: Path, rank=8, base="microsoft/phi-2"):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": rank,
                "lora_alpha": 32,
                "target_modules": ["q_proj"],
                "base_model_name_or_path": base,
            }
        )
    )
    return str(path)


class StubGenerator:
    """Answers every prompt with a fixed program; records what it was asked."""

    def __init__(self, text="print(3)"):
        self.text, self.calls = text, []

    def generate(self, model_path, questions, examples, form):
        self.calls.append((model_path, len(questions), form))
        return [self.text] * len(questions)


def eval_args(run_open, **overrides):
    args = run_open.build_parser().parse_args(
        ["--model", "x", "--dataset", "simuleq", "--stem_flan_type", "pot_prompt", "--cot_backup"]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.batch_size = -1
    return args


def test_parser_accepts_the_sweep_command_line(run_open, tmp_path):
    ckpts = [make_adapter(tmp_path / f"checkpoint-{i}", rank=16) for i in (512, 1024)]
    args = run_open.build_parser().parse_args(
        [
            "--model",
            *ckpts,
            "--dataset",
            "gsm8k",
            "math",
            "numglue",
            "svamp",
            "deepmind",
            "simuleq",
            "--shots",
            "0",
            "--stem_flan_type",
            "pot_prompt",
            "--batch_size",
            "8",
            "--model_max_length",
            "2048",
            "--cot_backup",
            "--use_vllm",
            "--dtype",
            "float16",
            "--enable_lora",
            "--limit",
            "20",
        ]
    )
    run_open.validate_models(args)
    assert (
        args.model == ckpts and args.limit == 20 and run_open.adapter_ranks(args.model) == [16, 16]
    )


def test_validate_rejects_silent_base_evaluation(run_open, tmp_path):
    adapter = make_adapter(tmp_path / "ckpt")
    parser = run_open.build_parser()
    no_lora = parser.parse_args(["--model", adapter, "--dataset", "gsm8k"])
    with pytest.raises(SystemExit, match="without --enable_lora"):
        run_open.validate_models(no_lora)
    mixed = parser.parse_args(
        ["--model", adapter, "microsoft/phi-2", "--dataset", "gsm8k", "--enable_lora"]
    )
    with pytest.raises(SystemExit, match="mixes"):
        run_open.validate_models(mixed)
    other = make_adapter(tmp_path / "other", base="another/base")
    bases = parser.parse_args(["--model", adapter, other, "--dataset", "gsm8k", "--enable_lora"])
    with pytest.raises(SystemExit, match="different base"):
        run_open.validate_models(bases)


def test_dry_run_builds_every_prompt_without_loading_a_model():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "math_eval/run_open.py"),
            "--model",
            "microsoft/phi-2",
            "--dataset",
            "gsm8k",
            "simuleq",
            "--stem_flan_type",
            "pot_prompt",
            "--dry_run",
            "--limit",
            "5",
        ],
        cwd="/",
        capture_output=True,
        text=True,
        timeout=300,
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": ""},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "[dry run] gsm8k: 5 prompts" in result.stdout
    assert "### Instruction:" in result.stdout and "Let's write a program." in result.stdout


def test_build_prompts_matches_training_template(run_open):
    prompts = run_open.build_prompts([], ["What is 2+2?"], "alpaca")
    assert prompts == [
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\nWhat is 2+2?\n\n### Response:"
    ]


def test_evaluate_dataset_writes_metrics_and_skips_finished(run_open, tmp_path):
    model = tmp_path / "ckpt"
    model.mkdir()
    args = eval_args(run_open, limit=5)
    generator = StubGenerator("print(3)")
    metrics = run_open.evaluate_dataset(args, generator, str(model), "simuleq")
    assert (
        metrics["total"] == 5 and metrics["dataset"] == "simuleq" and 0 <= metrics["accuracy"] <= 1
    )
    outputs = list((model / "outputs").iterdir())
    names = sorted(p.name for p in outputs)
    assert any(n.endswith(".jsonl") for n in names) and any(
        n.endswith(".metrics.json") for n in names
    )
    assert not any(n.endswith(".partial") for n in names)
    lines = [
        json.loads(x)
        for x in next(p for p in outputs if p.suffix == ".jsonl").read_text().splitlines()
    ]
    assert len(lines) == 5 and all(x["task"] == "simuleq" for x in lines)
    calls = len(generator.calls)
    assert run_open.evaluate_dataset(args, generator, str(model), "simuleq") is None
    assert len(generator.calls) == calls  # finished output is not recomputed


def test_output_dir_overrides_the_default_location(run_open, tmp_path):
    args = eval_args(run_open, output_dir=str(tmp_path / "base" / "outputs"))
    path = Path(run_open.output_path(args, "microsoft/phi-2", "simuleq"))
    assert path.parent == tmp_path / "base" / "outputs" and path.parent.is_dir()
    args.model = ["a", "b"]
    with pytest.raises(SystemExit, match="output_dir"):
        run_open.validate_models(args)


def test_partial_output_is_recomputed(run_open, tmp_path):
    model = tmp_path / "ckpt"
    args = eval_args(run_open, limit=3)
    (model / "outputs").mkdir(parents=True)
    path = Path(run_open.output_path(args, str(model), "simuleq"))
    Path(str(path) + ".partial").write_text("half a line")
    assert run_open.evaluate_dataset(args, StubGenerator(), str(model), "simuleq")["total"] == 3
    assert path.exists() and not Path(str(path) + ".partial").exists()


def test_cot_backup_reruns_questions_without_answer(run_open, tmp_path):
    model = tmp_path / "ckpt"
    model.mkdir()
    args = eval_args(run_open, limit=4)
    generator = StubGenerator("no digits here")
    metrics = run_open.evaluate_dataset(args, generator, str(model), "simuleq")
    assert metrics["cot_backup_reruns"] == 4 and metrics["total"] == 4
    assert len(generator.calls) == 2  # PoT pass, then the CoT rerun
