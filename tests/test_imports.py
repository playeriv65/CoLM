import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import colm

REPO = Path(__file__).resolve().parents[1]


def test_import_every_colm_module():
    names = [m.name for m in pkgutil.walk_packages(colm.__path__, prefix="colm.")]
    assert "colm.train.trainers" in names
    for name in names:
        importlib.import_module(name)


def test_trak_and_old_trainer_copies_are_gone():
    assert not (REPO / "colm/train/subset_trainer_distributed.py").exists()
    assert not (REPO / "colm/train/huggingface_trainer.py").exists()
    source = (REPO / "colm/train/utils.py").read_text()
    assert "trak" not in source


@pytest.mark.parametrize("module", ["utils", "prompt_utils", "data_loader", "run_open"])
def test_import_math_eval(module, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "math_eval"))
    importlib.import_module(module)


def test_import_superglue_eval(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "superglue_eval"))
    importlib.import_module("eval_superglue")


def test_stock_vllm_api():
    pytest.importorskip("vllm")
    from vllm import LLM, SamplingParams  # noqa: F401
    from vllm.lora.request import LoRARequest

    assert LoRARequest("adapter", 1, "/tmp/adapter").lora_int_id == 1


def test_wandb_not_imported_by_default():
    code = (
        "import sys, colm.train.train, colm.train.trainers, transformers;"
        "from colm.train.training_arguments import TrainingArguments;"
        "args = TrainingArguments(output_dir='unused', use_cpu=True);"
        "assert args.report_to == [], args.report_to;"
        "assert 'wandb' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)
