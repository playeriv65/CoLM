"""Drain a `JobQueue` serially on one GPU (the only queue worker; no orchestration).

    python -u -m colm.jobs.worker --queue queues/rank-sweep --gpu 0

Job fields (JSON): name, argv, env, cwd, log_stem, gpu (bool, default true), requires (glob patterns that
must match before the job starts), expects (globs that must match after it exits 0).
Placeholders in argv/env/cwd/paths: {python} (this interpreter), {repo}, {gpu}.
Each job's output is teed to stdout and to `<log_dir>/<log_stem>-<timestamp>.log`.
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from colm.jobs.file_queue import META_FILENAME, JobQueue, QueueBusyError

REPO = Path(__file__).resolve().parents[2]
TERMINATE_GRACE_SECONDS = 60


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _say(message: str) -> None:
    print(f"[worker {_now()}] {message}", flush=True)


def substitute(value, context: dict):
    if isinstance(value, str):
        for key, replacement in context.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [substitute(v, context) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, context) for k, v in value.items()}
    return value


def resolve_job(job: dict, gpu: str, repo: Path = REPO) -> dict:
    """Job with placeholders filled in and the child environment assembled."""
    context = {"python": sys.executable, "repo": str(repo), "gpu": gpu}
    resolved = substitute(job, context)
    env = dict(os.environ)
    env.update(resolved.get("env", {}))
    env["PYTHONUNBUFFERED"] = "1"
    # The worker's --gpu is authoritative; a CPU-only job sees no GPU at all.
    env["CUDA_VISIBLE_DEVICES"] = gpu if resolved.get("gpu", True) else ""
    resolved["env"] = env
    resolved["cwd"] = resolved.get("cwd", str(repo))
    return resolved


def preflight(meta: dict) -> None:
    """Fail before any job starts if the environment breaks the queue's requirements."""
    for name in meta.get("require_env", []):
        if not os.environ.get(name):
            raise SystemExit(f"preflight: environment variable {name} is not set")
    for name in meta.get("local_cache_env", []):
        value = os.environ.get(name, "")
        for prefix in meta.get("forbidden_cache_prefixes", []):
            if value.startswith(prefix):
                raise SystemExit(
                    f"preflight: {name}={value} points under forbidden prefix {prefix}"
                )
    if meta.get("require_hf_files"):
        from huggingface_hub import try_to_load_from_cache

        for model, filenames in meta["require_hf_files"].items():
            missing = [
                f for f in filenames if not isinstance(try_to_load_from_cache(model, f), str)
            ]
            if missing:
                raise SystemExit(f"preflight: {model} is missing {missing} in the local HF cache")
            _say(f"preflight: {model} complete in the local HF cache ({len(filenames)} files)")


def _matches(cwd: str, pattern: str) -> bool:
    """Whether the (cwd-relative) glob `pattern` matches anything."""
    return bool(glob.glob(os.path.join(cwd, pattern)))


def _copy_stream(proc, log_file) -> None:
    """Tee the child's combined output byte-for-byte (progress bars use \\r) to stdout and log."""
    stdout = sys.stdout.buffer
    while chunk := os.read(proc.stdout.fileno(), 65536):
        stdout.write(chunk)
        stdout.flush()
        log_file.write(chunk)
        log_file.flush()


class _Interrupted(Exception):
    pass


_state = {"in_job": False, "interrupted": False}


def _on_signal(signum, frame):
    """Terminate the running job (it is requeued) and stop the worker."""
    _state["interrupted"] = True
    if _state["in_job"]:
        raise _Interrupted


def run_job(job: dict, gpu: str, log_dir: Path, repo: Path = REPO) -> dict:
    """Run one job to completion; returns the result record (exit_code, wall_s, log, error)."""
    resolved = resolve_job(job, gpu, repo)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = Path(log_dir) / f"{resolved['log_stem']}-{timestamp}.log"
    result = {"log": str(log_path), "started": _now(), "gpu": gpu, "exit_code": None, "error": None}
    missing = [p for p in resolved.get("requires", []) if not _matches(resolved["cwd"], p)]
    if missing:
        result["error"] = f"missing requirement: {missing}"
        return result

    start = time.perf_counter()
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log_file:
        header = (
            f"# job {resolved['name']} started {result['started']} gpu={gpu}\n"
            f"# cwd {resolved['cwd']}\n# argv {' '.join(resolved['argv'])}\n"
            f"# env {json.dumps({k: v for k, v in job.get('env', {}).items()})}\n"
            f"# HF_HOME={os.environ.get('HF_HOME')} HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}\n"
            f"# loadavg {os.getloadavg()}\n"
        )
        sys.stdout.write(header)
        sys.stdout.flush()
        log_file.write(header.encode())
        log_file.flush()
        proc = subprocess.Popen(
            resolved["argv"],
            cwd=resolved["cwd"],
            env=resolved["env"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _state["in_job"] = True
        try:
            _copy_stream(proc, log_file)
            result["exit_code"] = proc.wait()
        except _Interrupted:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            result["exit_code"] = proc.returncode
            result["error"] = "interrupted"
        finally:
            _state["in_job"] = False
    result["wall_s"] = round(time.perf_counter() - start, 1)
    result["finished"] = _now()
    if result["exit_code"] == 0 and result["error"] is None:
        absent = [p for p in resolved.get("expects", []) if not _matches(resolved["cwd"], p)]
        if absent:
            result["error"] = f"missing expected output: {absent}"
    return result


def describe(job: dict, gpu: str, log_dir: Path, repo: Path = REPO) -> str:
    resolved = resolve_job(job, gpu, repo)
    env = substitute(job.get("env", {}), {"python": sys.executable, "repo": str(repo), "gpu": gpu})
    return (
        f"{job['name']}\n  cwd   {resolved['cwd']}\n  argv  {' '.join(resolved['argv'])}\n"
        f"  env   {json.dumps(env)} CUDA_VISIBLE_DEVICES="
        f"{resolved['env'].get('CUDA_VISIBLE_DEVICES')}\n"
        f"  log   {Path(log_dir) / (resolved['log_stem'] + '-<timestamp>.log')}\n"
        f"  requires {resolved.get('requires', [])} expects {resolved.get('expects', [])}"
    )


def drain(queue: JobQueue, gpu: str, log_dir: Path, repo: Path = REPO) -> int:
    """Run pending jobs in order until none is left; returns the number of failed jobs."""
    _state["interrupted"] = False
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    failures = 0
    for stale in queue.requeue_stale():
        _say(f"requeued stale job {stale.name}")
    while not _state["interrupted"]:
        pending = queue.jobs("pending")
        if not pending:
            break
        running = queue.claim(pending[0])
        if running is None:
            continue
        job = json.loads(running.read_text())
        _say(f"start {running.name} ({len(pending) - 1} more pending)")
        result = run_job(job, gpu, log_dir, repo)
        if result["error"] == "interrupted":
            queue.requeue(running, "interrupted by signal")
            _say(f"interrupted {running.name}; requeued")
            break
        ok = result["exit_code"] == 0 and result["error"] is None
        queue.finish(running, ok, result)
        failures += not ok
        _say(
            f"{'done' if ok else 'FAILED'} {running.name} exit={result['exit_code']} "
            f"wall={result.get('wall_s')}s error={result['error']} log={result['log']}"
        )
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--queue", required=True, help="Queue directory.")
    parser.add_argument("--gpu", required=True, help="Physical GPU id(s) for CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--log-dir", default=str(REPO / "logs"))
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved jobs; run none.")
    args = parser.parse_args(argv)
    if not args.gpu.replace(",", "").isdigit():
        parser.error(f"--gpu must be a comma separated list of ids, got {args.gpu!r}")

    queue = JobQueue(args.queue)
    log_dir = Path(args.log_dir)
    if args.dry_run:
        for path in queue.jobs("pending"):
            print(describe(json.loads(path.read_text()), args.gpu, log_dir), flush=True)
        print(
            f"{len(queue.jobs('pending'))} pending jobs; nothing executed ({META_FILENAME}: "
            f"{queue.read_meta()})"
        )
        return 0
    try:
        with queue.lock():
            preflight(queue.read_meta())
            _say(f"queue {queue.root} gpu={args.gpu} log_dir={log_dir}")
            failures = drain(queue, args.gpu, log_dir)
    except QueueBusyError as exc:
        print(exc, file=sys.stderr)
        return 3
    _say(f"queue drained, {failures} failed job(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
