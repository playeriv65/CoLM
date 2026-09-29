"""Markdown table of the paired runs written by `scripts/diagnostics/train_precision_arm.py`.

Diagnostic (kept because it documents a measurement, not part of the library). Result: docs/selection-precision.md.

    python scripts/diagnostics/summarize_precision_runs.py $ROOT/F-seed0-300steps $ROOT/F-seed1-300steps ...

Per run: held-out and GSM8K loss at every evaluation step, mean train loss of the last 100 steps,
mean step time (steps after the first 10), peak memory, load average.
"""

import json
import sys
from pathlib import Path

import numpy as np

LAST = 100
SKIP_TIMING = 10


def summarize(run: Path) -> dict:
    evals = {}
    for line in (run / "eval_loss.jsonl").read_text().splitlines():
        row = json.loads(line)
        evals.setdefault(row["step"], {})[row["set"]] = row["loss"]
    state = json.loads((run / "trainer_state.json").read_text())
    logs = [row for row in state["log_history"] if "loss" in row]
    times = [
        row["step_time_s"] for row in logs if "step_time_s" in row and row["step"] > SKIP_TIMING
    ]
    memory = json.loads((run / "memory.json").read_text()) if (run / "memory.json").exists() else {}
    info = json.loads((run / "precision_arm.json").read_text())
    return {
        "run": run.name,
        "evals": evals,
        "train_loss_last": float(np.mean([row["loss"] for row in logs[-LAST:]])),
        "step_time_s": float(np.mean(times)) if times else float("nan"),
        "peak_gb": memory.get("peak_mem_gb"),
        "peak_reserved_gb": memory.get("peak_reserved_gb"),
        "loadavg": [round(info["loadavg_start"][0], 1), round(info["loadavg_end"][0], 1)],
    }


def main():
    rows = [summarize(Path(p)) for p in sys.argv[1:]]
    steps = sorted({s for r in rows for s in r["evals"]})
    heads = [f"{kind} @{s}" for s in steps for kind in ("held-out", "gsm8k")]
    print("| run | " + " | ".join(heads) + " | train loss (last 100) | step s | peak GB | load |")
    print("|---|" + "---|" * (len(heads) + 4))
    for r in rows:
        cells = [
            f"{r['evals'].get(s, {}).get(kind, float('nan')):.4f}"
            for s in steps
            for kind in ("heldout", "gsm8k")
        ]
        print(
            f"| {r['run']} | " + " | ".join(cells) + f" | {r['train_loss_last']:.4f} | "
            f"{r['step_time_s']:.3f} | {r['peak_gb']} | {r['loadavg'][0]}-{r['loadavg'][1]} |"
        )
    print("\n" + json.dumps(rows))


if __name__ == "__main__":
    main()
