"""Sweep expansion into a queue, and the summary of a (fake) finished sweep."""

import json
import shutil
from pathlib import Path

import pytest

from colm.jobs import summarize, worker
from colm.jobs.file_queue import JobQueue
from colm.jobs.rank_sweep import arm_dir, create_queue, load_spec

REPO = Path(__file__).resolve().parents[1]
SWEEP = "configs/rank_sweep/sweep.json"


@pytest.fixture()
def repo(tmp_path):
    """A scratch 'repo' holding only the sweep configs (outputs must not touch the real out/)."""
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    return tmp_path


def test_real_sweep_spec_is_the_agreed_design():
    spec, base = load_spec(SWEEP)
    assert [(a["lora_r"], a["lora_alpha"]) for a in spec["arms"]] == [
        (128, 512),
        (16, 64),
        (16, 512),
        (8, 32),
        (8, 512),
    ]
    assert base["max_steps"] == 1024 and base["per_device_train_batch_size"] == 4
    assert base["gradient_accumulation_steps"] == 8 and base["efficient_mezo"] is True
    assert base["report_to"] == "none" and base["seed"] == 0
    assert base["save_steps"] == 512 and base["holdout_size"] == 1000
    assert spec["eval"]["checkpoints"] == [512, 1024]


def test_queue_order_and_contents(repo):
    root = create_queue(SWEEP, "queues/sweep", ["/mnt/net"], repo)
    names = [p.name for p in JobQueue(root).jobs("pending")]
    assert names == [
        "000-evalloss-base.json",
        "001-train-r128-a512.json",
        "002-evalloss-r128-a512.json",
        "003-evalacc-r128-a512.json",
        "004-train-r16-a64.json",
        "005-evalloss-r16-a64.json",
        "006-evalacc-r16-a64.json",
        "007-train-r16-a512.json",
        "008-evalloss-r16-a512.json",
        "009-evalacc-r16-a512.json",
        "010-train-r8-a32.json",
        "011-evalloss-r8-a32.json",
        "012-evalacc-r8-a32.json",
        "013-train-r8-a512.json",
        "014-evalloss-r8-a512.json",
        "015-evalacc-r8-a512.json",
        "016-evalacc-base.json",
        "017-summary.json",
    ]
    jobs = {p.name: json.loads(p.read_text()) for p in JobQueue(root).jobs("pending")}
    train = jobs["004-train-r16-a64.json"]
    assert train["log_stem"] == "rank-sweep-train-r16-a64-1024steps-seed0"
    assert train["env"]["HF_HUB_OFFLINE"] == "1"
    config_file = train["argv"][-1]
    config = json.loads((repo / config_file).read_text())
    assert (config["lora_r"], config["lora_alpha"], config["output_dir"]) == (
        16,
        64,
        "out/rank-sweep/phi-2-r16-a64-1024steps-seed0",
    )
    assert config["seed"] == 0 and config["max_steps"] == 1024 and config["holdout_size"] == 1000
    acc = jobs["003-evalacc-r128-a512.json"]
    assert "--use_vllm" in acc["argv"] and "--enable_lora" in acc["argv"]
    assert acc["argv"].count("out/rank-sweep/phi-2-r128-a512-1024steps-seed0/checkpoint-512") == 1
    assert jobs["017-summary.json"]["gpu"] is False
    assert acc["argv"][acc["argv"].index("--gpu_memory_utilization") + 1] == "0.9"
    base_acc = jobs["016-evalacc-base.json"]
    assert "--enable_lora" not in base_acc["argv"] and "--use_vllm" in base_acc["argv"]
    assert base_acc["argv"][base_acc["argv"].index("--model") + 1] == "microsoft/phi-2"
    assert "--output_dir" in base_acc["argv"] and "requires" not in base_acc
    for flag in ("--shots", "--stem_flan_type", "--dtype", "--model_max_length", "--dataset"):
        i, j = acc["argv"].index(flag), base_acc["argv"].index(flag)
        assert acc["argv"][i : i + 2] == base_acc["argv"][j : j + 2]
    meta = JobQueue(root).read_meta()
    assert meta["forbidden_cache_prefixes"] == ["/mnt/net"] and "HF_HOME" in meta["require_env"]
    with pytest.raises(SystemExit, match="already has jobs"):
        create_queue(SWEEP, "queues/sweep", (), repo)


def test_existing_checkpoints_are_never_overwritten(repo):
    spec, base = load_spec(SWEEP, repo)
    checkpoint = repo / arm_dir(spec, spec["arms"][1], base) / "checkpoint-512"
    checkpoint.mkdir(parents=True)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        create_queue(SWEEP, "queues/x", (), repo)


def test_generated_queue_dry_runs_through_the_worker(repo, capsys):
    root = create_queue(SWEEP, "queues/sweep", (), repo)
    assert worker.main(["--queue", str(root), "--gpu", "0", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("CUDA_VISIBLE_DEVICES=0") == 17 and "18 pending jobs" in out
    assert (
        "colm.train.train out/rank-sweep/phi-2-r128-a512-1024steps-seed0/train_config.json" in out
    )


def _fake_arm(repo, spec, base, arm, scale):
    directory = repo / arm_dir(spec, arm, base)
    (directory / "checkpoint-1024").mkdir(parents=True, exist_ok=True)
    import torch
    from safetensors.torch import save_file

    save_file(
        {"a": torch.zeros(arm["lora_r"], 10), "b": torch.zeros(5, arm["lora_r"])},
        str(directory / "checkpoint-1024" / "adapter_model.safetensors"),
    )
    history = [
        {"step": s, "loss": scale, "step_time_s": 3.0, "peak_mem_gb": 20.0 + s / 1000}
        for s in range(1, 1025)
    ]
    history[255]["step_time_s"] = 60.0  # step 256 contains an evaluation
    (directory / "trainer_state.json").write_text(json.dumps({"log_history": history}))
    lines = [
        {"step": s, "set": name, "loss": scale + s / 1e4, "train_peak_before_eval_gb": 25.0}
        for s in (256, 512)
        for name in ("heldout", "gsm8k")
    ]
    (directory / "eval_loss.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    outputs = directory / "checkpoint-1024" / "outputs"
    outputs.mkdir()
    for d in spec["eval"]["datasets"]:
        (outputs / f"{d}_x.metrics.json").write_text(json.dumps({"dataset": d, "accuracy": 0.5}))


def test_summary_of_fake_results(repo):
    spec, base = load_spec(SWEEP, repo)
    for i, arm in enumerate(spec["arms"][:2]):
        _fake_arm(repo, spec, base, arm, scale=1.0 + i)
    base_outputs = repo / spec["eval"]["base_output_dir"]
    base_outputs.mkdir(parents=True)
    for d in spec["eval"]["datasets"]:
        (base_outputs / f"{d}_x.metrics.json").write_text(
            json.dumps({"dataset": d, "accuracy": 0.25})
        )
    result = summarize.summarize(SWEEP, repo)
    assert result["base_accuracy"]["gsm8k"] == 0.25 and result["notes"]
    first, second, third = result["arms"][:3]
    assert first["trainable_params"] == 128 * 10 + 5 * 128
    assert second["trainable_params"] == 16 * 10 + 5 * 16
    assert first["step_time_mean_s"] == pytest.approx(3.0)  # step 256/257 excluded
    assert first["train_peak_mem_gb"] == 25.0
    assert first["final_train_loss"] == pytest.approx(1.0)
    assert first["eval_loss_curve"]["heldout"][512] == pytest.approx(1.0512)
    assert first["accuracy_mean"][1024] == 0.5 and first["accuracy_mean"][512] is None
    assert third["trainable_params"] is None and third["step_time_mean_s"] is None
    markdown = summarize.to_markdown(result, spec["eval"]["checkpoints"])
    assert "r128-a512" in markdown and "## Eval loss: heldout" in markdown
    assert "| base (no LoRA) | - | 0.2500" in markdown and "contaminated" in markdown
    summarize.main(["--sweep", SWEEP], repo)
    assert (repo / spec["output_root"] / "summary.md").exists()
