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


def train_token_budget(probe, tokens: tuple[int, int], fraction: float) -> int:
    """Tokens of one training forward + backward that fit in the memory left to this process.

    `probe(n)` runs a forward and a backward on about `n` tokens. Activation memory is linear in
    the tokens of a pack, so two small probes give the slope (bytes per token) and the intercept
    (bytes held whatever the size: weights, optimizer state, the fp16 copies of the frozen
    weights); the budget is what remains of `fraction` of the memory available to this process
    (free memory plus what its allocator has reserved), divided by the slope. The line is
    extrapolated far beyond the probes, so a probe at the derived size checks it and shrinks the
    budget (or halves it after an out-of-memory error) until it fits. Probes use every position
    as a label, the worst case for the LM head.
    """
    small, large = tokens
    free, _ = torch.cuda.mem_get_info()
    available = free + torch.cuda.memory_reserved()
    limit = fraction * available

    def peak_of(n: int) -> float:
        torch.cuda.reset_peak_memory_stats()
        probe(n)
        return torch.cuda.max_memory_allocated()

    peaks = [peak_of(n) for n in tokens]
    slope = (peaks[1] - peaks[0]) / (large - small)
    if slope <= 0:
        raise RuntimeError(f"memory does not grow with the tokens of a pack: peaks {peaks}")
    intercept = peaks[0] - slope * small
    budget = max(small, int((limit - intercept) / slope))
    logger.info(
        f"Training memory: {slope / 2**20:.2f} MiB per token, {intercept / GB:.2f} GB fixed, "
        f"{available / GB:.1f} GB available, fraction {fraction}; first budget {budget} tokens"
    )
    for _ in range(4):
        if budget <= large:  # inside the probed range
            break
        try:
            peak = peak_of(budget)
        except torch.OutOfMemoryError:
            peak = None
        if peak is None:  # (outside the except block, which still holds the failed graph)
            torch.cuda.empty_cache()
            budget = max(large, budget // 2)
            continue
        logger.info(f"Check at {budget} tokens: peak {peak / GB:.1f} GB of {limit / GB:.1f} GB")
        if peak <= limit:
            break
        budget = max(large, int(budget * (limit - intercept) / (peak - intercept)))
    return budget
