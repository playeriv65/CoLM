"""Several CPU ranks (gloo): the same config on 1, 2 and 4 processes selects and trains alike."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from dist_worker import run

WORKER = Path(__file__).with_name("dist_worker.py")
STEPS = 2


def _launch(tmp_path, data, case, world):
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
        str(WORKER),
        case,
        data,
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True, env=env, cwd=WORKER.parent.parent, timeout=900)
    return [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(world)]


def _per_step(trained):
    steps = {}
    for t in trained:
        steps.setdefault(t["step"], []).extend(t["indices"])
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
        assert len({len(s) for s in shares}) == 1  # equal work on every rank
        union = [i for s in shares for i in s]
        assert len(union) == len(set(union))  # disjoint shares
        # The pool of `world` ranks is the pool of one rank world times as large, in the same
        # order, so the selection is the same set of examples.
        assert sorted(union) == sorted(reference[step]), f"step {step}"
        # Every rank gets the same mixture: the kept-source examples (id 0) are spread evenly.
        kept = [sum(i % 4 == 0 for i in s) for s in shares]
        assert max(kept) - min(kept) <= 1, kept
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
    # One gradient all-reduce per optimizer step, not one per sub-batch.
    assert [r["all_reduces"] for r in ranks] == [STEPS] * world
