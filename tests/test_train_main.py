"""End-to-end `python -m colm.train.train config.json` on a tiny local Phi (CPU)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import add_lora, make_phi  # noqa: F401

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("efficient", [True, False])
def test_train_main_runs(tmp_path, tokenizer, mixture_file, efficient):
    model_dir = tmp_path / "tiny-phi"
    make_phi(tokenizer).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = {
        "model_name_or_path": str(model_dir),
        "train_files": [mixture_file],
        "token_cache_dir": "",  # the default would add a file to the shared cache per test run
        "output_dir": str(tmp_path / "out"),
        "max_steps": 2,
        "per_device_train_batch_size": 4 if efficient else 1,
        "gradient_accumulation_steps": 2 if efficient else 4,
        "efficient_mezo": efficient,
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
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "colm.train.train", str(config_path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    out = tmp_path / "out"
    assert (out / "adapter_model.safetensors").exists()
    metrics = json.loads((out / "train_results.json").read_text())
    assert metrics["train_samples"] == 64
    state = json.loads((out / "trainer_state.json").read_text())
    losses = [h["loss"] for h in state["log_history"] if "loss" in h]
    assert len(losses) == 2 and all(v == v for v in losses)
    assert "Using SubsetTrainerEfficient" in result.stdout + result.stderr or not efficient
