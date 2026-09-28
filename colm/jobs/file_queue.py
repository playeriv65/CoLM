"""A directory-backed job queue: one JSON file per job, state = the sub-directory it is in.

    <queue>/pending/NNN-name.json   waiting (drained in file-name order)
    <queue>/running/...             claimed by the worker (atomic rename out of pending/)
    <queue>/done/... , failed/...   finished; the job file gains a "result" record
    <queue>/worker.lock             flock held by the single worker (mutual exclusion)
    <queue>/results.jsonl           one line per finished job

There is no daemon and nothing polls: the worker drains `pending/` and exits.
"""

import contextlib
import fcntl
import json
import os
from pathlib import Path

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"
STATES = (PENDING, RUNNING, DONE, FAILED)
LOCK_FILENAME = "worker.lock"
RESULTS_FILENAME = "results.jsonl"
META_FILENAME = "queue.json"


class QueueBusyError(RuntimeError):
    """Another worker holds the queue lock."""


class JobQueue:
    def __init__(self, root):
        self.root = Path(root)
        for state in STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)

    def path(self, state: str, name: str) -> Path:
        return self.root / state / name

    def jobs(self, state: str) -> list[Path]:
        return sorted((self.root / state).glob("*.json"))

    def add(self, position: int, job: dict) -> Path:
        """Write `job` as pending job number `position` (file name = order)."""
        target = self.path(PENDING, f"{position:03d}-{job['name']}.json")
        if target.exists():
            raise FileExistsError(target)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=1) + "\n")
        os.rename(tmp, target)
        return target

    def claim(self, pending_path: Path) -> Path | None:
        """Move a pending job to running/; None if somebody else claimed it first."""
        target = self.path(RUNNING, pending_path.name)
        try:
            os.rename(pending_path, target)
        except FileNotFoundError:
            return None
        return target

    def finish(self, running_path: Path, ok: bool, result: dict) -> Path:
        """Record `result` in the job file and move it to done/ or failed/."""
        job = json.loads(running_path.read_text())
        job["result"] = result
        tmp = running_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=1) + "\n")
        os.replace(tmp, running_path)
        target = self.path(DONE if ok else FAILED, running_path.name)
        os.rename(running_path, target)
        with open(self.root / RESULTS_FILENAME, "a") as f:
            f.write(
                json.dumps({"job": running_path.name, "state": target.parent.name, **result}) + "\n"
            )
        return target

    def requeue(self, running_path: Path, note: str) -> Path:
        """Put a running job back to pending (interrupted or stale), counting the attempt."""
        job = json.loads(running_path.read_text())
        job["attempts"] = job.get("attempts", 0) + 1
        job.setdefault("notes", []).append(note)
        running_path.write_text(json.dumps(job, indent=1) + "\n")
        target = self.path(PENDING, running_path.name)
        os.rename(running_path, target)
        return target

    def requeue_stale(self) -> list[Path]:
        """Jobs left in running/ by a dead worker. Only call while holding the lock."""
        return [self.requeue(p, "stale: worker died while running") for p in self.jobs(RUNNING)]

    @contextlib.contextmanager
    def lock(self):
        """Exclusive non-blocking lock; released by the OS even if the worker is killed."""
        handle = open(self.root / LOCK_FILENAME, "w")
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise QueueBusyError(f"another worker holds {self.root / LOCK_FILENAME}") from exc
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            yield
        finally:
            handle.close()

    def read_meta(self) -> dict:
        path = self.root / META_FILENAME
        return json.loads(path.read_text()) if path.exists() else {}

    def write_meta(self, meta: dict) -> None:
        (self.root / META_FILENAME).write_text(json.dumps(meta, indent=1) + "\n")
