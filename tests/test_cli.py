"""Entry points and configuration: defaults of the paper recipe, validation, resolved config."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from equivalence.helpers import make_args

from colm import cli
from colm.train.config import parse_args

REPO = Path(__file__).resolve().parents[1]


def _parse(tmp_path, *flags, config=None):
    argv = []
    if config is not None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        argv.append(str(path))
    return parse_args([*argv, "--model_name_or_path", "microsoft/phi-2", *flags])


def test_a_plain_run_is_the_paper_recipe(tmp_path):
    model, data, training, _ = _parse(tmp_path)
    assert data.train_files == ["data/MathInstruct.jsonl"]
    assert (training.max_steps, training.learning_rate, training.warmup_steps) == (1024, 2e-5, 0.03)
    assert training.efficient_mezo and training.data_selection_unit == "mezo"
    # per-device 4 x accumulation 8 = the pool of 32 examples, one HF batch
    assert (training.micro_batch_size, training.per_device_train_batch_size) == (4, 32)
    assert training.gradient_accumulation_steps == 1 and training.small_batch_ratio == 0.5
    assert training.keep_source_ids == [0, 1, 3, 5, 7, 8, 9, 10, 11, 13]
    assert not training.legacy and training.report_to == [] and training.save_only_model
    # the recipe of phi: fp16 AMP over fp32 weights, LoRA on q k v fc1 fc2
    assert training.fp16 and model.torch_dtype == "none"
    assert model.lora_target_modules == ["q_proj", "k_proj", "v_proj", "fc1", "fc2"]
    assert (model.lora_r, model.lora_alpha) == (128, 512)


def test_config_files_hold_only_the_differences(tmp_path):
    _, _, training, _ = _parse(tmp_path, "--max_steps", "7", config={"lora_r": 8, "max_steps": 3})
    assert training.max_steps == 7  # flags override the file


def test_unknown_keys_and_wrong_choices_fail_at_load(tmp_path):
    with pytest.raises(ValueError, match="unknown config keys"):
        _parse(tmp_path, config={"data_selection_unt": "mezo"})
    with pytest.raises(ValueError, match="data_selection_unit"):
        _parse(tmp_path, config={"data_selection_unit": "typo"})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(efficient_mezo=True, data_selection_unit="rep"), "efficient_mezo"),
        (dict(efficient_mezo=True, small_batch_ratio=0.1), "small_batch_ratio"),
        (dict(efficient_mezo=False), "per_device_train_batch_size=1"),
        (dict(small_batch_ratio=1.5), "small_batch_ratio"),
    ],
)
def test_inconsistent_selection_settings_fail_early(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        make_args(
            tmp_path, per_device_train_batch_size=4, gradient_accumulation_steps=2, **overrides
        )


def test_help_lists_every_option_and_the_gpus(tmp_path):
    out = io.StringIO()
    old, sys.stdout = sys.stdout, out
    try:
        with pytest.raises(SystemExit):
            cli.train(["--help"])
    finally:
        sys.stdout = old
    text = out.getvalue()
    assert "--gpus" in text and "--max_steps" in text and "--zo_dim" in text and "--legacy" in text


def test_train_needs_gpus_and_builds_one_torchrun(tmp_path, monkeypatch):
    monkeypatch.delenv("COLM_GPUS", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    with pytest.raises(SystemExit, match="--gpus"):
        cli.train(["config.json"])
    monkeypatch.setenv("COLM_LOG_DIR", str(tmp_path))
    started = {}

    class Fake:
        stdout = io.StringIO("line\n")
        returncode = 0

        def __init__(self, command, env, **kw):
            started.update(command=command, env=env)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cli.subprocess, "Popen", Fake)
    assert cli.train(["config.json", "--gpus", "2,3", "--max_steps", "5"]) == 0
    assert started["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert started["command"][-5:] == ["-m", "colm.train.train", "config.json", "--max_steps", "5"]
    assert "--nproc_per_node" in started["command"] and "2" in started["command"]
    (log,) = tmp_path.glob("config-gpu2_3-np2-*.log")
    assert log.read_text() == "line\n"


def test_eval_accuracy_gets_the_paper_protocol(tmp_path):
    adapter = tmp_path / "ckpt"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 8,
                "base_model_name_or_path": "microsoft/phi-2",
            }
        )
    )
    argv = cli._accuracy_defaults(["--model", str(adapter)])
    assert "--enable_lora" in argv and "--use_vllm" in argv and "--cot_backup" in argv
    assert argv[argv.index("--dtype") + 1] == "float16"  # the recipe of phi
    assert argv[argv.index("--stem_flan_type") + 1] == "pot_prompt"
    assert argv[argv.index("--dataset") + 1 : argv.index("--dataset") + 7][0] == "gsm8k"


def test_train_writes_the_resolved_config(tmp_path, tokenizer, mixture_file):
    """python -m colm.train.train: the full resolved config is logged and saved next to the outputs."""
    from conftest import make_phi

    model_dir = tmp_path / "tiny-phi"
    make_phi(tokenizer).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = {
        "model_name_or_path": str(model_dir), "train_files": [mixture_file], "output_dir": str(tmp_path / "out"),
        "max_steps": 1, "gradient_accumulation_steps": 2, "keep_sources": "0", "last_layer_index": 1,
        "zo_dim": 16, "lora_r": 4, "lora_alpha": 16, "use_cpu": True, "precision": "fp32", "save_strategy": "no",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
    }  # fmt: skip
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run(
        [sys.executable, "-m", "colm.train.train", str(path)],
        check=True,
        env=env,
        cwd=REPO,
        capture_output=True,
    )
    resolved = json.loads((tmp_path / "out" / "resolved_config.json").read_text())
    assert resolved["training"]["small_batch_ratio"] == 0.5  # a default
    assert resolved["training"]["max_steps"] == 1 and resolved["model"]["lora_r"] == 4
    derived = resolved["derived"]
    assert (
        derived["max_seq_length"] == 512 and derived["selected_per_rank"] == 4
    )  # phi context of the fixture
    assert derived["zo_parameters"] and derived["attn_implementation"] == "colm_varlen"
