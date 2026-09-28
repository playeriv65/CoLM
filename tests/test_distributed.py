"""Two CPU ranks: gather on rank 0, select, broadcast, train disjoint shares in sync."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).with_name("dist_worker.py")


@pytest.mark.parametrize("mode", ["efficient", "regular"])
def test_two_rank_selection(tmp_path, mixture_file, mode):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "--master_port",
        str(29500 + os.getpid() % 1000 + (mode == "regular")),
        str(WORKER),
        mixture_file,
        str(tmp_path),
        mode,
    ]
    subprocess.run(cmd, check=True, env=env, cwd=WORKER.parent.parent, timeout=600)
    ranks = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(2)]
    assert all(r["steps"] == 2 for r in ranks)
    per_rank = 2 * 2 if mode == "efficient" else 2  # selected examples per rank per step
    for r in ranks:
        assert len(r["trained"]) == 2 * per_rank
    # The two ranks train disjoint shares of each selected set ...
    for step in range(2):
        a = set(ranks[0]["trained"][step * per_rank : (step + 1) * per_rank])
        b = set(ranks[1]["trained"][step * per_rank : (step + 1) * per_rank])
        assert not a & b
    # ... and DDP keeps their weights identical.
    assert ranks[0]["lora"] == pytest.approx(ranks[1]["lora"], rel=1e-5, abs=1e-6)
