"""Several CPU ranks (gloo): the same config on 1, 2 and 4 processes selects and trains alike."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from dist_eval_worker import run as evaluate
from dist_worker import run

WORKER = Path(__file__).with_name("dist_worker.py")
EVAL_WORKER = Path(__file__).with_name("dist_eval_worker.py")
STEPS = 2


def _launch(tmp_path, data, case, world, worker=WORKER):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    port = 29700 + (os.getpid() + world * 7 + (case == "regular")) % 200
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(world),
        "--master_port",
        str(port),
        str(worker),
        *([case] if worker == WORKER else []),
        data,
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True, env=env, cwd=WORKER.parent.parent, timeout=900)
    return [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(world)]


def _per_step(trained, key="indices"):
    steps = {}
    for t in trained:
        steps.setdefault(t["step"], []).extend(t[key])
    return steps


@pytest.mark.parametrize("case", ["efficient", "regular"])
@pytest.mark.parametrize("world", [2, 4])
def test_ranks_agree_with_one_process(tmp_path, mixture_file, case, world):
    ranks = _launch(tmp_path, mixture_file, case, world)
    single_dir = tmp_path / "single"
    single_dir.mkdir()
    single = run(case, mixture_file, str(single_dir), gas_scale=world, steps=STEPS)
    per_rank = [_per_step(r["trained"]) for r in ranks]
    reference = _per_step(single["trained"])

    assert all(r["steps"] == STEPS for r in ranks)
    for step in range(STEPS):
        shares = [p[step] for p in per_rank]
        assert all(shares)  # every rank trains something (a rank without a backward would hang DDP)
        union = [i for s in shares for i in s]
        assert len(union) == len(set(union))  # disjoint shares
        # The pool of `world` ranks is the pool of one rank world times as large, in the same
        # order, so the selection is the same set of examples.
        assert sorted(union) == sorted(reference[step]), f"step {step}"
        # Equal work: the token counts of the ranks differ by less than the longest example
        # (the guarantee of longest-first assignment); round-robin shares do not have it.
        lengths = [_per_step(r["trained"], "lengths")[step] for r in ranks]
        tokens = [sum(x) for x in lengths]
        assert max(tokens) - min(tokens) <= max(max(x) for x in lengths), tokens
    # DDP: identical weights on every rank, equal to the single-process weights.
    for r in ranks[1:]:
        for name, values in r["lora"].items():
            torch.testing.assert_close(
                torch.tensor(values), torch.tensor(ranks[0]["lora"][name]), rtol=0, atol=1e-12
            )
    for name, values in single["lora"].items():
        torch.testing.assert_close(
            torch.tensor(ranks[0]["lora"][name]),
            torch.tensor(values),
            rtol=0,
            atol=1e-9,
            msg=lambda m, n=name: f"{n}: {m}",
        )
    assert ranks[0]["loss"] == pytest.approx(single["loss"], rel=1e-6)
    # The evaluation after step 1 is a collective (every rank takes its share of the batches);
    # rank 0 records it, and it is the single-process number of the same weights.
    (recorded,) = ranks[0]["eval"]
    assert all(not r["eval"] for r in ranks[1:]) and recorded["step"] == 1
    assert recorded["loss"] == pytest.approx(single["eval"][0]["loss"], rel=1e-9)
    assert recorded["n_tokens"] == single["eval"][0]["n_tokens"]
    # One gradient all-reduce per optimizer step, not one per sub-batch.
    assert [r["all_reduces"] for r in ranks] == [STEPS] * world


def test_check_replicas_compares_the_ranks(tmp_path, mixture_file):
    ranks = _launch(tmp_path, mixture_file, "efficient", 2)
    assert (
        all(r["replicas"] == ranks[0]["replicas"] for r in ranks) and len(ranks[0]["replicas"]) == 2
    )


@pytest.mark.parametrize("world", [2, 3])
def test_evaluation_loss_is_sharded_over_the_ranks_without_changing_it(
    tmp_path, mixture_file, world
):
    ranks = _launch(tmp_path, mixture_file, "", world, EVAL_WORKER)
    single = evaluate(mixture_file)
    for result in ranks:  # every rank holds the pooled result, equal to the single-process one
        assert result == json.loads(json.dumps(single))
