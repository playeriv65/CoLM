"""Wall-clock record of the phases of one process (`startup.json` next to `memory.json`).

A one-shot timer around each phase of a run (imports, configuration, tokenizer, model load, data
tokenisation, trainer set-up, first step, steady steps, evaluation, checkpoints, final save). It
adds no synchronisation and no per-step work. Phases are flat; a name with a `/` (`data/tokenise`)
is a detail of the phase before the slash and is not added to the total.
"""

import json
import os
import time
from contextlib import contextmanager

LAUNCH_ENV = "COLM_LAUNCH_TIME"  # epoch seconds, set by `colm-train` before torchrun starts


def process_start_time() -> float:
    """Epoch seconds at which this process was created (Linux /proc); now if unavailable."""
    try:
        with open("/proc/self/stat") as f:
            ticks = int(f.read().rsplit(")", 1)[1].split()[19])  # field 22, after the command
        with open("/proc/stat") as f:
            boot = next(int(line.split()[1]) for line in f if line.startswith("btime"))
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, StopIteration):
        return time.time()


class PhaseClock:
    def __init__(self):
        self.process_start = process_start_time()
        launch = os.environ.get(LAUNCH_ENV)
        self.launch = float(launch) if launch else None
        self.last = self.process_start  # the first mark covers the interpreter and the imports
        self.seconds: dict[str, float] = {}

    def mark(self, name: str) -> None:
        """The phase that ends now, started at the previous mark."""
        now = time.time()
        self.add(name, now - self.last)
        self.last = now

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds

    @contextmanager
    def detail(self, name: str):
        """Time a block that lies inside a phase (name it `phase/what`)."""
        start = time.time()
        try:
            yield
        finally:
            self.add(name, time.time() - start)

    def total(self) -> float:
        return sum(v for k, v in self.seconds.items() if "/" not in k)

    def save(self, path: str, **extra) -> None:
        end = time.time()
        record = {
            "process_start": self.process_start,
            "launch": self.launch,
            "end": end,
            "process_seconds": end - self.process_start,
            "launcher_to_process_seconds": None
            if self.launch is None
            else self.process_start - self.launch,
            "phases_total_seconds": self.total(),
            "seconds": {k: round(v, 3) for k, v in self.seconds.items()},
            **extra,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=1)


CLOCK = PhaseClock()
