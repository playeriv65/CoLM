"""File queue and the single worker (CPU, tiny python jobs)."""

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from colm.jobs import kernel_preflight, worker
from colm.jobs.file_queue import JobQueue, QueueBusyError

REPO = Path(__file__).resolve().parents[1]


def job(name, code, **extra):
    return {
        "name": name,
        "argv": ["{python}", "-u", "-c", code],
        "log_stem": f"test-{name}-r8-a32-steps2-seed0",
        **extra,
    }


def run_worker(queue, tmp_path):
    return worker.drain(queue, "3", tmp_path / "logs", REPO)


def test_queue_orders_claims_and_finishes(tmp_path):
    queue = JobQueue(tmp_path / "q")
    queue.add(1, job("b", "pass"))
    queue.add(0, job("a", "pass"))
    with pytest.raises(FileExistsError):
        queue.add(0, job("a", "pass"))
    first, second = queue.jobs("pending")
    assert first.name == "000-a.json" and second.name == "001-b.json"
    running = queue.claim(first)
    assert running.parent.name == "running" and queue.claim(first) is None
    done = queue.finish(running, True, {"exit_code": 0})
    assert done.parent.name == "done"
    assert json.loads(done.read_text())["result"]["exit_code"] == 0
    line = json.loads((queue.root / "results.jsonl").read_text().splitlines()[0])
    assert line["job"] == "000-a.json" and line["state"] == "done"


def test_lock_is_exclusive_and_released(tmp_path):
    queue = JobQueue(tmp_path / "q")
    with queue.lock():
        with pytest.raises(QueueBusyError), JobQueue(tmp_path / "q").lock():
            pass
    with queue.lock():
        pass


def test_worker_runs_in_order_with_logs_env_and_failures(tmp_path):
    queue = JobQueue(tmp_path / "q")
    marker = tmp_path / "order.txt"
    code = (
        "import os,sys;"
        f"open({str(marker)!r},'a').write(sys.argv[1]+' '+os.environ['CUDA_VISIBLE_DEVICES']+'\\n')"
    )
    queue.add(0, {**job("first", code), "argv": ["{python}", "-c", code, "one"]})
    queue.add(1, job("boom", "import sys; print('about to fail'); sys.exit(3)"))
    queue.add(2, {**job("second", code), "argv": ["{python}", "-c", code, "two"]})
    queue.add(3, {**job("needs", "pass"), "requires": [str(tmp_path / "nothing-*.bin")]})
    queue.add(4, {**job("promises", "pass"), "expects": [str(tmp_path / "never")]})
    queue.add(5, {**job("cpu", code), "gpu": False, "argv": ["{python}", "-c", code, "cpu"]})

    failures = run_worker(queue, tmp_path)

    assert failures == 1
    assert marker.read_text().splitlines() == ["one 3"]
    assert [p.name for p in queue.jobs("done")] == ["000-first.json"]
    assert [p.name for p in queue.jobs("pending")] == [
        "002-second.json",
        "003-needs.json",
        "004-promises.json",
        "005-cpu.json",
    ]
    failed = {p.name: json.loads(p.read_text())["result"] for p in queue.jobs("failed")}
    assert failed["001-boom.json"]["exit_code"] == 3
    log = Path(failed["001-boom.json"]["log"])
    assert log.name.startswith("test-boom-r8-a32-steps2-seed0-") and log.suffix == ".log"
    assert "about to fail" in log.read_text() and "gpu=3" in log.read_text()
    assert failed["001-boom.json"]["wall_s"] >= 0
    assert run_worker(queue, tmp_path) == 1
    assert marker.read_text().splitlines() == ["one 3"]


@pytest.mark.parametrize(
    ("field", "error"),
    [("requires", "missing requirement"), ("expects", "missing expected output")],
)
def test_worker_stops_on_missing_dependency_or_output(tmp_path, field, error):
    queue = JobQueue(tmp_path / "q")
    queue.add(0, {**job("first", "pass"), field: [str(tmp_path / "missing")]})
    queue.add(1, job("dependent", "pass"))
    assert run_worker(queue, tmp_path) == 1
    assert error in json.loads(queue.jobs("failed")[0].read_text())["result"]["error"]
    assert [p.name for p in queue.jobs("pending")] == ["001-dependent.json"]


def test_relative_paths_resolve_against_job_cwd(tmp_path):
    (tmp_path / "work").mkdir()
    queue = JobQueue(tmp_path / "q")
    queue.add(
        0,
        job(
            "rel",
            "open('made.txt','w').write('x')",
            cwd=str(tmp_path / "work"),
            expects=["made.txt"],
        ),
    )
    assert run_worker(queue, tmp_path) == 0
    assert (tmp_path / "work" / "made.txt").exists()


def test_stale_running_jobs_are_requeued(tmp_path):
    queue = JobQueue(tmp_path / "q")
    path = queue.add(0, job("crashed", "pass"))
    queue.claim(path)  # worker "died" here
    assert run_worker(queue, tmp_path) == 0
    done = json.loads(queue.jobs("done")[0].read_text())
    assert done["attempts"] == 1 and "stale" in done["notes"][0]


def test_worker_requires_gpu_and_dry_run_runs_nothing(tmp_path, capsys):
    queue = JobQueue(tmp_path / "q")
    queue.add(0, job("x", "raise SystemExit(9)", env={"A": "{gpu}"}))
    with pytest.raises(SystemExit):
        worker.main(["--queue", str(queue.root)])  # no default GPU
    with pytest.raises(SystemExit):
        worker.main(["--queue", str(queue.root), "--gpu", "gpu0"])
    assert worker.main(["--queue", str(queue.root), "--gpu", "5", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "CUDA_VISIBLE_DEVICES=5" in out and '"A": "5"' in out
    assert len(queue.jobs("pending")) == 1 and not queue.jobs("done") and not queue.jobs("failed")


def test_second_worker_is_refused(tmp_path):
    queue = JobQueue(tmp_path / "q")
    queue.add(0, job("x", "pass"))
    with queue.lock():
        assert worker.main(["--queue", str(queue.root), "--gpu", "0"]) == 3
    assert len(queue.jobs("pending")) == 1


def test_sigterm_requeues_running_job_and_stops(tmp_path):
    queue = JobQueue(tmp_path / "q")
    queue.add(0, job("slow", "import time; print('started', flush=True); time.sleep(120)"))
    queue.add(1, job("later", "pass"))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "colm.jobs.worker",
            "--queue",
            str(queue.root),
            "--gpu",
            "0",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:  # wait for the job's own output, not for a timer
        if "started" in line and not line.startswith("#"):
            break
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=90)
    assert [p.name for p in queue.jobs("pending")] == ["000-slow.json", "001-later.json"]
    assert not queue.jobs("running") and not queue.jobs("done")
    assert json.loads(queue.jobs("pending")[0].read_text())["attempts"] == 1


def test_preflight_checks_environment_and_local_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("COLM_TEST_VAR", raising=False)
    with pytest.raises(SystemExit, match="COLM_TEST_VAR is not set"):
        worker.preflight({"require_env": ["COLM_TEST_VAR"]})
    monkeypatch.setenv("COLM_TEST_CACHE", "/network/disk/hf")
    with pytest.raises(SystemExit, match="forbidden prefix"):
        worker.preflight(
            {"local_cache_env": ["COLM_TEST_CACHE"], "forbidden_cache_prefixes": ["/network"]}
        )
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(SystemExit, match="missing"):
        worker.preflight({"require_hf_files": {"some/model": ["config.json"]}})
    worker.preflight(
        {"local_cache_env": ["COLM_TEST_CACHE"], "forbidden_cache_prefixes": ["/other"]}
    )


def test_local_kernel_preflight_is_offline_and_propagates_to_jobs(monkeypatch, tmp_path):
    repo = "kernels-community/flash-attn2"
    revision = "abc123"
    snapshot = tmp_path / "hub" / "kernels--kernels-community--flash-attn2" / "snapshots" / revision
    requirement = {"repo": repo, "revision": revision, "version": 3, "symbols": ["flash_attn_func"]}
    meta = {"require_kernels": [requirement], "preflight_env": {"HF_HUB_OFFLINE": "0"}}
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    with pytest.raises(SystemExit, match="kernel .* missing at"):
        worker.local_kernel_env(meta)
    snapshot.mkdir(parents=True)
    extra_env = worker.local_kernel_env(meta)
    assert extra_env == {"LOCAL_KERNELS": f"{repo}={snapshot}"}

    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    worker.preflight(meta, "2", extra_env)
    argv, kwargs = calls[0]
    assert json.loads(argv[-1]) == requirement
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert kwargs["env"]["LOCAL_KERNELS"] == f"{repo}={snapshot}"
    assert (
        worker.resolve_job(job("x", "pass"), "2", extra_env=extra_env)["env"]["LOCAL_KERNELS"]
        == f"{repo}={snapshot}"
    )

    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, "", "binary is broken"),
    )
    with pytest.raises(SystemExit, match="binary is broken"):
        worker.preflight(meta, "2", extra_env)


def test_missing_kernel_stops_worker_before_claiming_any_job(monkeypatch, tmp_path):
    queue = JobQueue(tmp_path / "q")
    queue.add(0, job("first", "pass"))
    queue.write_meta(
        {
            "require_kernels": [
                {
                    "repo": "kernels-community/flash-attn2",
                    "revision": "missing-commit",
                    "version": 3,
                }
            ]
        }
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    with pytest.raises(SystemExit, match="missing at"):
        worker.main(["--queue", str(queue.root), "--gpu", "2"])
    assert [p.name for p in queue.jobs("pending")] == ["000-first.json"]
    assert not queue.jobs("running") and not queue.jobs("failed")


def test_kernel_loader_requires_expected_callables(monkeypatch):
    import kernels

    monkeypatch.setattr(kernels, "get_kernel", lambda repo, version: object())
    with pytest.raises(RuntimeError, match="missing callable symbols"):
        kernel_preflight.check(
            {"repo": "kernels-community/flash-attn2", "version": 3, "symbols": ["flash_attn_func"]}
        )
