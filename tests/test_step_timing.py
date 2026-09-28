"""Step timing: no-op when off, hierarchical closure when on (CPU)."""

import json

import pytest
from equivalence.helpers import build, make_args

from colm.train import step_timing
from colm.train.step_timing import (
    OTHER,
    WALL,
    StepTimer,
    StepTimingCallback,
    build_tree,
    summarize,
)


def test_off_timer_is_a_noop(monkeypatch):
    def fail():
        raise AssertionError("synchronize called with profiling off")

    monkeypatch.setattr(step_timing, "_sync", fail)
    timer = StepTimer("off")
    with timer.section("a"), timer.fine("b"):
        timer.start("c")
        timer.stop("c")
        timer.count("n", 3)
    assert timer.pop_step() == ({}, {})


def test_levels_and_paths():
    coarse = StepTimer("coarse")
    with coarse.section("a"):
        with coarse.fine("hidden"):
            with coarse.section("b"):
                pass
    assert set(coarse.sections) == {"a", "a/b"}
    fine = StepTimer("fine")
    with fine.section("a"), fine.fine("f"):
        pass
    assert set(fine.sections) == {"a", "a/f"}
    with pytest.raises(RuntimeError):
        fine.start("x")
        fine.stop("y")


def test_tree_residuals():
    steps = [
        {"step": 1, WALL: 1.0, "sections": {"a": 0.6, "a/x": 0.4, "b": 0.3}, "counts": {}},
        {"step": 2, WALL: 2.0, "sections": {"a": 1.0, "a/x": 0.5, "b": 0.5}, "counts": {}},
    ]
    tree = build_tree(steps)
    assert tree[OTHER].tolist() == pytest.approx([0.1, 0.5])
    assert tree["a/" + OTHER].tolist() == pytest.approx([0.2, 0.5])


def _closes(steps):
    """Every parent >= the sum of its children in every step (timers nest)."""
    for rec in steps:
        tree = build_tree([rec])
        residuals = [v[0] for k, v in tree.items() if k.endswith(OTHER)]
        assert min(residuals) > -1e-6, rec


@pytest.mark.parametrize("level", ["coarse", "fine"])
def test_efficient_trainer_timing(tmp_path, tokenizer, mixture_file, level):
    args = make_args(
        tmp_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        efficient_mezo=True,
        keep_sources="0",
        max_steps=3,
        profile_timing=level,
        profile_timing_dir=str(tmp_path / "timing"),
        profile_census_steps=1,
    )
    trainer, _ = build(args, tokenizer, mixture_file)
    trainer.train()
    (out,) = (tmp_path / "timing").glob("step_timing-*.jsonl")
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    steps = [x for x in lines if x["type"] == "step"]
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert [s["census"] for s in steps] == [True, False, False]
    assert any(k.startswith("census/") for k in steps[0]["counts"])
    for s in steps[1:]:
        sec = s["sections"]
        for name in [
            "selection",
            "selection/features",
            "selection/gather",
            "selection/select",
            "train",
            "train/forward",
            "train/backward",
            "optimizer",
            "optimizer/step",
        ]:
            assert name in sec, name
    _closes(steps)
    summary = summarize(str(out), warmup=1, window=2)
    assert summary["num_steps_used"] == 2
    assert summary["closure"]["other_mean_ms"] >= 0


def test_off_writes_nothing(tmp_path, tokenizer, mixture_file):
    args = make_args(
        tmp_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        efficient_mezo=True,
        profile_timing_dir=str(tmp_path / "timing"),
    )
    trainer, _ = build(args, tokenizer, mixture_file)
    assert not trainer._timer.enabled
    assert not any(isinstance(cb, StepTimingCallback) for cb in trainer.callback_handler.callbacks)
    trainer.train()
    assert not (tmp_path / "timing").exists()
