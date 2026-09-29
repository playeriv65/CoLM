"""Peak GPU memory of every rank, per phase of the step.

`torch.cuda.max_memory_allocated()` is a per-process, cumulative counter: reading it on rank 0
only (or after an evaluation that reset it) says nothing about the other ranks. The meter resets
the counters at the start of each phase (`selection`, `train`), records the peaks of the phase
at its end, and gathers the peaks of all ranks whenever the trainer logs. Allocated memory is
what tensors hold; reserved memory adds the caching allocator's free blocks. The CUDA context
and NCCL buffers are outside both.
"""

import logging

import torch

from colm.selection.pool import all_gather_object

logger = logging.getLogger(__name__)

GB = 1024**3


class MemoryMeter:
    def __init__(self):
        self.enabled = torch.cuda.is_available()
        self.window: dict[
            str, dict[str, float]
        ] = {}  # phase -> {allocated, reserved} since the last log
        self.run: dict[str, dict[str, float]] = {}  # the same over the whole run

    def start(self) -> None:
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()

    def stop(self, phase: str) -> None:
        if not self.enabled:
            return
        peaks = {
            "allocated": torch.cuda.max_memory_allocated() / GB,
            "reserved": torch.cuda.max_memory_reserved() / GB,
        }
        for record in (self.window, self.run):
            old = record.setdefault(phase, dict.fromkeys(peaks, 0.0))
            record[phase] = {k: max(old[k], v) for k, v in peaks.items()}

    def gather(self, run: bool = False) -> list[dict]:
        """Peaks of every rank (one dict per rank, phase -> {allocated, reserved}); a collective."""
        record = self.run if run else self.window
        ranks = all_gather_object(record)
        if not run:
            self.window = {}
        return ranks

    @staticmethod
    def summary(ranks: list[dict]) -> dict[str, float]:
        """Log entries: the maximum over the ranks and phases, and every rank's own."""
        if not ranks or not any(ranks):
            return {}
        per_rank = [max((p["allocated"] for p in r.values()), default=0.0) for r in ranks]
        logs = {
            "peak_mem_gb": round(max(per_rank), 3),
            "peak_reserved_gb": round(max(p["reserved"] for r in ranks for p in r.values()), 3),
        }
        for phase in sorted({k for r in ranks for k in r}):
            logs[f"peak_mem_{phase}_gb"] = round(
                max(r[phase]["allocated"] for r in ranks if phase in r), 3
            )
        if len(ranks) > 1:
            logs.update({f"peak_mem_gb_rank{i}": round(v, 3) for i, v in enumerate(per_rank)})
        return logs
