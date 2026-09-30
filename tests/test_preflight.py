"""Checks before a run or a resume: incomplete checkpoints, model-only resumes, disk space."""

import pytest
import torch

from colm.train import preflight
from colm.train.preflight import (
    check_disk_space,
    check_resumable,
    checkpoint_bytes,
    newest_complete_checkpoint,
    planned_saves,
)
from colm.train.training_arguments import TrainingArguments


def _checkpoint(root, step, complete=True, optimizer=False):
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(b"0")
    if complete:  # the trainer state is written last
        (path / "trainer_state.json").write_text("{}")
    if optimizer:
        (path / "optimizer.pt").write_bytes(b"0")
        (path / "scheduler.pt").write_bytes(b"0")
    return path


def test_the_newest_complete_checkpoint_skips_an_interrupted_save(tmp_path):
    _checkpoint(tmp_path, 256)
    _checkpoint(tmp_path, 512)
    _checkpoint(tmp_path, 1024, complete=False)  # the save was killed after the weights
    (tmp_path / "checkpoint-notes").mkdir()
    assert newest_complete_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-512")
    assert newest_complete_checkpoint(str(tmp_path / "missing")) is None


def test_resuming_needs_a_complete_checkpoint(tmp_path):
    bad = _checkpoint(tmp_path, 8, complete=False)
    with pytest.raises(FileNotFoundError, match="trainer_state.json"):
        check_resumable(str(bad))
    with pytest.raises(FileNotFoundError):
        check_resumable(str(tmp_path / "checkpoint-9"))


def test_a_model_only_checkpoint_resumes_with_a_warning(tmp_path, caplog):
    only = _checkpoint(tmp_path / "a", 8)
    full = _checkpoint(tmp_path / "b", 8, optimizer=True)
    with caplog.at_level("WARNING", logger=preflight.logger.name):
        check_resumable(str(full))
        assert not caplog.records
        check_resumable(str(only))
    assert "optimizer.pt / scheduler.pt" in caplog.text and "save_only_model" in caplog.text


def test_checkpoint_size_counts_trainable_weights_and_adam_moments():
    layer = torch.nn.Linear(10, 10)  # 110 fp32 values
    layer.bias.requires_grad_(False)  # 100 trainable
    assert checkpoint_bytes(layer.parameters(), save_only_model=True) == 400
    assert checkpoint_bytes(layer.parameters(), save_only_model=False) == 1200


def test_too_small_a_disk_fails_before_the_run(tmp_path, monkeypatch):
    class Usage:
        free = 5 * 10**9

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: Usage)
    check_disk_space(
        str(tmp_path / "not" / "yet" / "there"), 10**9, saves=5
    )  # walks up to a parent
    with pytest.raises(OSError, match="5.0 GB free but the checkpoints need about 6.0 GB"):
        check_disk_space(str(tmp_path / "out"), 10**9, saves=6)


@pytest.mark.parametrize(
    ("options", "saves"),
    [
        (dict(save_strategy="no"), 1),  # the final model only
        (dict(save_strategy="steps", save_steps=512, max_steps=1024), 3),
        (dict(save_strategy="steps", save_steps=0.25, max_steps=1024), 5),  # a ratio of the run
        (dict(save_strategy="steps", save_steps=100, max_steps=1000, save_total_limit=2), 4),
    ],
)
def test_planned_saves(tmp_path, options, saves):
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, data_selection_method="none", **options
    )
    assert planned_saves(args) == saves
