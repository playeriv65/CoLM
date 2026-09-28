"""Opt-in per-phase wall-clock breakdown of CoLM optimizer steps.

`StepTimer` records hierarchical named sections. With `enabled=False` every call is a
no-op (no CUDA synchronize, no clock read). With `enabled=True` each section boundary
calls `torch.cuda.synchronize()` before reading the wall clock, so GPU work is charged to
the section that launched it. Nested sections are addressed by '/'-joined paths
("selection/features/perturb"); repeated entries of the same path within one step add up.

`StepTimingCallback` closes a step at `on_step_end` (the time between two consecutive
optimizer steps is the step wall clock), brackets `optimizer.step()` and the
scheduler/zero_grad tail with sections, and writes one JSON line per step plus meta
lines (config, GPU, `os.getloadavg()` at start and end).

`python -m colm.train.step_timing <file.jsonl>` summarises a run: per node mean/std/p50/p90
over the post-warmup steps, an explicit `other` residual for every parent, the closure
error of the top level against the step clock, token counts and a sliding-window
stability check.
"""

import argparse
import json
import os
import socket
import sys
import time
import warnings
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import TrainerCallback

STEP = "step"  # root node of the summary tree: the measured step wall clock
WALL = "step_wall_s"  # record field holding it
LAYER_GROUP = "layers"  # children "00", "01", ... are collapsed into one row of the table
OTHER = "other"
_NULL = nullcontext()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


LEVELS = ("off", "coarse", "fine")


class StepTimer:
    """Hierarchical section timer; every call is a no-op at level 'off'.

    `section` is timed at levels 'coarse' and 'fine', `fine` only at level 'fine'.
    """

    def __init__(self, level: str = "off"):
        if level not in LEVELS:
            raise ValueError(f"profile level {level!r} not in {LEVELS}")
        self.level = level
        self.enabled = level != "off"
        self.fine_enabled = level == "fine"
        self._stack: list[tuple[str, float]] = []
        self.sections: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    def path(self, depth: int | None = None) -> str:
        names = [n for n, _ in self._stack]
        return "/".join(names[:depth] if depth else names)

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        _sync()
        self._stack.append((name, time.perf_counter()))

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        _sync()
        now = time.perf_counter()
        path = self.path()
        top, begin = self._stack.pop()
        if top != name:
            raise RuntimeError(f"StepTimer: stop({name!r}) while {top!r} is open")
        self.sections[path] = self.sections.get(path, 0.0) + now - begin

    def is_open(self, name: str) -> bool:
        return bool(self._stack) and self._stack[-1][0] == name

    @contextmanager
    def _section(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def section(self, name: str):
        """Coarse section `name` under the currently open sections."""
        return self._section(name) if self.enabled else _NULL

    def fine(self, name: str):
        """Section timed only at level 'fine'."""
        return self._section(name) if self.fine_enabled else _NULL

    def count(self, name: str, value) -> None:
        if self.enabled:
            self.counts[name] = self.counts.get(name, 0) + float(value)

    def pop_step(self) -> tuple[dict, dict]:
        """Hand over this step's sections and counts and start empty ones."""
        sections, counts = self.sections, self.counts
        self.sections, self.counts = {}, {}
        return sections, counts


_TRANSFER_OPS = None
_SYNC_WARNING = "called a synchronizing CUDA operation"


def _transfer_ops():
    global _TRANSFER_OPS
    if _TRANSFER_OPS is None:
        aten = torch.ops.aten
        _TRANSFER_OPS = {
            aten._to_copy.default,
            aten.copy_.default,
            aten.to.device,
            aten.to.dtype_layout,
            aten.to.other,
            aten.to.prim_Device,
        }
    return _TRANSFER_OPS


def _device_of(x):
    return x.device if isinstance(x, torch.Tensor) else None


class TransferCensus(TorchDispatchMode):
    """Counts aten ops, host<->device copies (bytes by direction) and synchronizing CUDA calls.

    Copies are seen as aten ops through a dispatch mode; synchronizing calls through
    `torch.cuda.set_sync_debug_mode('warn')` (explicit `torch.cuda.synchronize()` is not
    reported by it). Both are attributed to the open timer sections (first `depth`
    levels, default: the full path). It adds Python overhead to every op: only for
    diagnostic (warmup) steps.
    """

    def __init__(self, timer: StepTimer, depth: int | None = None):
        super().__init__()
        self.timer = timer
        self.depth = depth
        self._warnings = None

    def _add(self, what: str, value: float) -> None:
        key = f"census/{self.timer.path(self.depth) or '(none)'}/{what}"
        self.timer.counts[key] = self.timer.counts.get(key, 0) + value

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._add("aten_ops", 1)
        if func in (torch.ops.aten.item.default, torch.ops.aten._local_scalar_dense.default):
            if _device_of(args[0]) is not None and args[0].device.type == "cuda":
                self._add("d2h_calls", 1)
                self._add("d2h_bytes", args[0].element_size())
        elif func in _transfer_ops():
            src, dst = (
                (args[1], args[0]) if func is torch.ops.aten.copy_.default else (args[0], out)
            )
            s, d = _device_of(src), _device_of(dst)
            if s is not None and d is not None and s.type != d.type:
                direction = "d2h" if s.type == "cuda" else "h2d"
                self._add(f"{direction}_calls", 1)
                self._add(f"{direction}_bytes", dst.nbytes)
        return out

    def _on_warning(self, message, category, filename, lineno, file=None, line=None):
        if _SYNC_WARNING in str(message):
            self._add("syncs", 1)
        else:
            self._showwarning(message, category, filename, lineno, file, line)

    def __enter__(self):
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        # Every sync warning must reach the counter; other warnings keep their filters.
        warnings.filterwarnings("always", message=_SYNC_WARNING)
        self._showwarning = warnings.showwarning
        warnings.showwarning = self._on_warning
        if torch.cuda.is_available():
            torch.cuda.set_sync_debug_mode("warn")
        return super().__enter__()

    def __exit__(self, *exc):
        out = super().__exit__(*exc)
        if torch.cuda.is_available():
            torch.cuda.set_sync_debug_mode(0)
        self._warnings.__exit__(*exc)
        self._warnings = None
        return out


class StepTimingCallback(TrainerCallback):
    """Closes one timing record per optimizer step and writes it as a JSON line."""

    def __init__(
        self, timer: StepTimer, out_file: str, census_steps: int = 0, meta: dict | None = None
    ):
        self.timer = timer
        self.out_file = out_file
        self.census_steps = census_steps
        self.meta = meta or {}
        self._last = None
        self._fh = None
        self._census = None

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(os.path.dirname(os.path.abspath(self.out_file)), exist_ok=True)
        self._fh = open(self.out_file, "w")
        meta = {
            "type": "meta",
            "event": "train_begin",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": socket.gethostname(),
            "loadavg": os.getloadavg(),
            "cpu_count": os.cpu_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "world_size": args.world_size,
            "profile_level": self.timer.level,
            "census_steps": self.census_steps,
            **self.meta,
        }
        self._write(meta)
        if self.census_steps > 0:
            self._census = TransferCensus(self.timer)
            self._census.__enter__()
        _sync()
        self._last = time.perf_counter()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not self.timer.is_open("optimizer"):
            self.timer.start("optimizer")  # max_grad_norm = 0: no clipping section opened it
        self.timer.start("step")

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.timer.stop("step")
        self.timer.start("sched_zero_grad")

    def on_step_end(self, args, state, control, **kwargs):
        self.timer.stop("sched_zero_grad")
        self.timer.stop("optimizer")
        _sync()
        now = time.perf_counter()
        sections, counts = self.timer.pop_step()
        census = self._census is not None
        if census and state.global_step >= self.census_steps:
            self._census.__exit__(None, None, None)
            self._census = None
        self._write(
            {
                "type": "step",
                "step": state.global_step,
                "census": census,
                WALL: now - self._last,
                "sections": sections,
                "counts": counts,
            }
        )
        # Writing the record belongs to the next step; charge it explicitly.
        self._last = now
        self.timer.sections["profiler_io"] = time.perf_counter() - now

    def on_train_end(self, args, state, control, **kwargs):
        if self._census is not None:
            self._census.__exit__(None, None, None)
            self._census = None
        if self._fh is None:
            return
        self._write(
            {
                "type": "meta",
                "event": "train_end",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "loadavg": os.getloadavg(),
                "global_step": state.global_step,
            }
        )
        self._fh.close()
        self._fh = None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def load_records(path: str) -> tuple[list[dict], list[dict]]:
    steps, metas = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            (steps if rec["type"] == "step" else metas).append(rec)
    return steps, metas


def _stats(values) -> dict:
    v = np.asarray(values, dtype=np.float64) * 1e3
    return {
        "mean_ms": float(v.mean()),
        "std_ms": float(v.std()),
        "p50_ms": float(np.percentile(v, 50)),
        "p90_ms": float(np.percentile(v, 90)),
    }


def _count_stats(steps: list[dict], keep) -> dict:
    names = sorted({c for s in steps for c in s["counts"] if keep(c)})
    out = {}
    for c in names:
        v = np.array([s["counts"].get(c, 0.0) for s in steps])
        out[c] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    return out


def build_tree(steps: list[dict]) -> dict[str, np.ndarray]:
    """Per-step seconds of every node, with an `other` residual under every parent.

    The root `step` is the measured step clock; top-level sections are its children.
    """
    paths = sorted({p for s in steps for p in s["sections"]})
    series = {STEP: np.array([s[WALL] for s in steps])}
    for p in paths:
        series[p] = np.array([s["sections"].get(p, 0.0) for s in steps])
    parents = {STEP: [p for p in paths if "/" not in p]}
    for p in paths:
        if "/" in p:
            parents.setdefault(p.rsplit("/", 1)[0], []).append(p)
    for parent, children in parents.items():
        if parent not in series:
            raise ValueError(f"children {children} have no timed parent {parent!r}")
        residual = series[parent] - sum(series[c] for c in children)
        series[OTHER if parent == STEP else f"{parent}/{OTHER}"] = residual
    return series


def stability(step_s: np.ndarray, window: int, tolerance: float) -> dict:
    mean = float(step_s.mean())
    if len(step_s) < window:
        return {"window": window, "ok": None, "reason": "fewer steps than the window"}
    sliding = np.convolve(step_s, np.ones(window) / window, mode="valid")
    dev = sliding / mean - 1
    return {
        "window": window,
        "tolerance": tolerance,
        "max_rel_dev": float(np.abs(dev).max()),
        "min_sliding_ms": float(sliding.min() * 1e3),
        "max_sliding_ms": float(sliding.max() * 1e3),
        "ok": bool(np.abs(dev).max() <= tolerance),
    }


def summarize(path: str, warmup: int = 10, window: int = 50, tolerance: float = 0.05) -> dict:
    steps, metas = load_records(path)
    kept = [s for s in steps if s["step"] > warmup]
    if not kept:
        raise ValueError(f"no steps after warmup={warmup} in {path}")
    if any(s.get("census") for s in kept):
        raise ValueError("census steps inside the timed window: census_steps must be <= warmup")
    # Step 1 carries one-off start-up work; census over the remaining census steps.
    census_steps = [s for s in steps if s.get("census") and s["step"] > 1]
    series = build_tree(kept)
    step_mean = series[STEP].mean()
    nodes = {}
    for name in sorted(series, key=lambda p: (p != STEP, p)):
        entry = _stats(series[name])
        entry["pct_of_step"] = float(100 * series[name].mean() / step_mean)
        parent = None if name == STEP else (name.rsplit("/", 1)[0] if "/" in name else STEP)
        if parent is not None:
            entry["pct_of_parent"] = float(100 * series[name].mean() / series[parent].mean())
        nodes[name] = entry
    layer_groups = {}
    for name in nodes:
        if name.rsplit("/", 1)[-1] == LAYER_GROUP:
            layers = [n for n in nodes if n.startswith(name + "/") and n[len(name) + 1 :].isdigit()]
            if layers:
                means = np.array([nodes[n]["mean_ms"] for n in layers])
                layer_groups[name] = {
                    "num_layers": len(layers),
                    "mean_ms_per_layer": float(means.mean()),
                    "min_ms": float(means.min()),
                    "min_layer": layers[int(means.argmin())].rsplit("/", 1)[-1],
                    "max_ms": float(means.max()),
                    "max_layer": layers[int(means.argmax())].rsplit("/", 1)[-1],
                    "sum_ms": float(means.sum()),
                }
    count_stats = _count_stats(kept, lambda c: not c.startswith("census/"))
    census_stats = _count_stats(census_steps, lambda c: c.startswith("census/"))
    summary = {
        "file": os.path.abspath(path),
        "num_steps_total": len(steps),
        "warmup_dropped": warmup,
        "num_steps_used": len(kept),
        "closure": {
            "step_mean_ms": float(step_mean * 1e3),
            "sum_top_level_mean_ms": float(
                sum(series[p].mean() for p in series if "/" not in p and p not in (STEP, OTHER))
                * 1e3
            ),
            "other_mean_ms": float(series[OTHER].mean() * 1e3),
            "other_pct_of_step": float(100 * series[OTHER].mean() / step_mean),
        },
        "stability": stability(series[STEP], window, tolerance),
        "nodes": nodes,
        "layer_groups": layer_groups,
        "counts_per_step": count_stats,
        "census_per_step": {"steps": [s["step"] for s in census_steps], **census_stats},
        "meta": metas,
    }
    return summary


def format_table(summary: dict) -> str:
    lines = [
        "| phase | mean ms | % step | % parent | std | p50 | p90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, e in summary["nodes"].items():
        depth = 0 if name == STEP else name.count("/") + 1
        label = "&nbsp;&nbsp;" * depth + (name.rsplit("/", 1)[-1] if name != STEP else STEP)
        lines.append(
            f"| {label} | {e['mean_ms']:.1f} | {e['pct_of_step']:.1f} | "
            f"{e.get('pct_of_parent', 100.0):.1f} | {e['std_ms']:.1f} | {e['p50_ms']:.1f} | "
            f"{e['p90_ms']:.1f} |"
        )
    return "\n".join(_tree_order(lines, summary))


def _tree_order(lines, summary):
    # Rows are emitted depth-first so every child sits under its parent.
    names = list(summary["nodes"])
    rows = dict(zip(names, lines[2:], strict=True))

    def children(parent):
        prefix = "" if parent == STEP else parent + "/"
        kids = [
            n
            for n in names
            if n != STEP and n.startswith(prefix) and "/" not in n[len(prefix) :] and n != parent
        ]
        # Largest first, residual last.
        return sorted(
            kids,
            key=lambda n: (n.endswith(OTHER), -summary["nodes"][n]["mean_ms"]),
        )

    out = lines[:2]

    groups = summary.get("layer_groups", {})

    def visit(n):
        out.append(rows[n])
        if n in groups:
            g = groups[n]
            depth = n.count("/") + 2
            out.append(
                f"| {'&nbsp;&nbsp;' * depth}per layer ({g['num_layers']}): mean "
                f"{g['mean_ms_per_layer']:.2f}, min {g['min_ms']:.2f} (#{g['min_layer']}), "
                f"max {g['max_ms']:.2f} (#{g['max_layer']}) | | | | | | |"
            )
            kids = [k for k in children(n) if not k.rsplit("/", 1)[-1].isdigit()]
        else:
            kids = children(n)
        for k in kids:
            visit(k)

    visit(STEP)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("jsonl")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--out", default=None, help="Summary JSON (default: <jsonl>.summary.json)")
    args = parser.parse_args(argv)
    summary = summarize(args.jsonl, args.warmup, args.window, args.tolerance)
    out = args.out or os.path.splitext(args.jsonl)[0] + ".summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(format_table(summary))
    keys = ("closure", "stability", "counts_per_step", "census_per_step")
    print(json.dumps({k: summary[k] for k in keys}, indent=2))
    print(f"summary written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
