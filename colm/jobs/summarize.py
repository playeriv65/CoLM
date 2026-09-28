"""Collect the results of a rank sweep into one table (JSON + markdown).

    python -m colm.jobs.summarize --sweep configs/rank_sweep/sweep.json

Per arm: trainable parameters, step time (steps disturbed by an evaluation or a checkpoint save
are excluded), training peak memory, final training loss, evaluation loss at every evaluated
step (in-training callback) and on the saved checkpoints (standalone), and accuracy per dataset
for every evaluated checkpoint. Missing pieces are reported as null, never guessed.
"""

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

from safetensors import safe_open

from colm.eval.eval_loss import EVAL_LOSS_FILENAME
from colm.jobs.rank_sweep import (
    BASE_LOSS_FILE,
    CKPT_LOSS_FILE,
    REPO,
    arm_dir,
    arm_label,
    load_spec,
)

FINAL_LOSS_WINDOW = 64


def trainable_parameters(adapter_file: Path) -> int | None:
    if not adapter_file.exists():
        return None
    total = 0
    with safe_open(adapter_file, "pt") as f:
        for key in f.keys():  # noqa: SIM118 - safetensors handle, not a dict
            n = 1
            for dim in f.get_slice(key).get_shape():
                n *= dim
            total += n
    return total


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def training_stats(directory: Path, eval_steps: list[int], save_steps: list[int]) -> dict:
    state_file = directory / "trainer_state.json"
    if not state_file.exists():
        return dict.fromkeys(
            [
                "steps_logged",
                "step_time_mean_s",
                "step_time_median_s",
                "train_peak_mem_gb",
                "final_train_loss",
            ]
        )
    history = json.loads(state_file.read_text())["log_history"]
    rows = [h for h in history if "loss" in h and "step_time_s" in h]
    disturbed = {s + d for s in [*eval_steps, *save_steps] for d in (0, 1)}
    clean = [h["step_time_s"] for h in rows if h["step"] not in disturbed]
    losses = [h["loss"] for h in history if "loss" in h]
    memory = [h["peak_mem_gb"] for h in rows if "peak_mem_gb" in h]
    # Evaluations reset the peak-memory counter; the callback records the training peak it saw.
    memory += [
        r["train_peak_before_eval_gb"]
        for r in _read_jsonl(directory / EVAL_LOSS_FILENAME)
        if "train_peak_before_eval_gb" in r
    ]
    return {
        "steps_logged": len(rows),
        "step_time_mean_s": statistics.fmean(clean) if clean else None,
        "step_time_median_s": statistics.median(clean) if clean else None,
        "train_peak_mem_gb": max(memory) if memory else None,
        "final_train_loss": statistics.fmean(losses[-FINAL_LOSS_WINDOW:]) if losses else None,
    }


def curve(directory: Path) -> dict:
    """{set: {step: loss}} from the in-training evaluation callback."""
    result: dict[str, dict[int, float]] = {}
    for record in _read_jsonl(directory / EVAL_LOSS_FILENAME):
        result.setdefault(record["set"], {})[record["step"]] = record["loss"]
    return result


def checkpoint_losses(directory: Path) -> dict:
    path = directory / CKPT_LOSS_FILE
    if not path.exists():
        return {}
    return {
        r["label"]: {name: v["loss"] for name, v in r["results"].items()}
        for r in json.loads(path.read_text())["records"]
    }


def accuracies(outputs_dir: Path) -> dict:
    """{dataset: accuracy} from the ``*.metrics.json`` files of one outputs directory."""
    found = {}
    for path in glob.glob(str(outputs_dir / "*.metrics.json")):
        metrics = json.loads(Path(path).read_text())
        found[metrics["dataset"]] = metrics["accuracy"]
    return found


def summarize(sweep_path, repo: Path = REPO) -> dict:
    spec, base = load_spec(sweep_path, repo)
    eval_steps = base.get("eval_loss_steps", [])
    save_steps = [
        s for s in range(base["save_steps"], base["max_steps"] + 1, int(base["save_steps"]))
    ]
    datasets = spec["eval"]["datasets"]
    base_file = repo / spec["output_root"] / BASE_LOSS_FILE
    summary = {
        "base_eval_loss": (
            {
                r["label"]: {n: v["loss"] for n, v in r["results"].items()}
                for r in json.loads(base_file.read_text())["records"]
            }
            if base_file.exists()
            else None
        ),
        "datasets": datasets,
        "base_accuracy": accuracies(repo / spec["eval"]["base_output_dir"]),
        "notes": spec.get("summary_notes", []),
        "arms": [],
    }
    for arm in spec["arms"]:
        directory = repo / arm_dir(spec, arm, base)
        accs = {
            c: accuracies(directory / f"checkpoint-{c}" / "outputs")
            for c in spec["eval"]["checkpoints"]
        }
        summary["arms"].append(
            {
                "arm": arm_label(arm),
                **arm,
                "trainable_params": trainable_parameters(
                    directory
                    / f"checkpoint-{spec['eval']['checkpoints'][-1]}"
                    / "adapter_model.safetensors"
                ),
                **training_stats(directory, eval_steps, save_steps),
                "eval_loss_curve": curve(directory),
                "eval_loss_checkpoints": checkpoint_losses(directory),
                "accuracy": accs,
                "accuracy_mean": {
                    c: (
                        statistics.fmean(a[d] for d in datasets)
                        if all(d in a for d in datasets)
                        else None
                    )
                    for c, a in accs.items()
                },
            }
        )
    return summary


def _fmt(value, spec="{:.4f}"):
    return "-" if value is None else spec.format(value)


def to_markdown(summary: dict, checkpoints: list[int]) -> str:
    lines = [f"> {note}" for note in summary.get("notes", [])]
    lines += [
        "",
        "## Training cost",
        "",
        "| arm | r | alpha | trainable params | step time mean / median (s) | train peak mem (GB) | final train loss (last 64) |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in summary["arms"]:
        params = "-" if a["trainable_params"] is None else f"{a['trainable_params']:,}"
        lines.append(
            f"| {a['arm']} | {a['lora_r']} | {a['lora_alpha']:g} | {params} | "
            f"{_fmt(a.get('step_time_mean_s'), '{:.3f}')} / {_fmt(a.get('step_time_median_s'), '{:.3f}')} | "
            f"{_fmt(a.get('train_peak_mem_gb'), '{:.1f}')} | {_fmt(a.get('final_train_loss'))} |"
        )
    base = (summary["base_eval_loss"] or {}).get("base", {})
    for name in sorted({n for a in summary["arms"] for n in a["eval_loss_curve"]} | set(base)):
        steps = sorted({s for a in summary["arms"] for s in a["eval_loss_curve"].get(name, {})})
        lines += [
            "",
            f"## Eval loss: {name} (mean token NLL; base model {_fmt(base.get(name))})",
            "",
            "| arm | " + " | ".join(f"step {s}" for s in steps) + " |",
            "|---|" + "---|" * len(steps),
        ]
        for a in summary["arms"]:
            row = a["eval_loss_curve"].get(name, {})
            lines.append(f"| {a['arm']} | " + " | ".join(_fmt(row.get(s)) for s in steps) + " |")
    lines += [
        "",
        "## Accuracy",
        "",
        "| arm | checkpoint | " + " | ".join(summary["datasets"]) + " | mean |",
        "|---|---|" + "---|" * (len(summary["datasets"]) + 1),
    ]
    base_acc = summary.get("base_accuracy") or {}
    if base_acc:
        mean = statistics.fmean(base_acc[d] for d in summary["datasets"] if d in base_acc)
        complete = all(d in base_acc for d in summary["datasets"])
        lines.append(
            "| base (no LoRA) | - | "
            + " | ".join(_fmt(base_acc.get(d)) for d in summary["datasets"])
            + f" | {_fmt(mean if complete else None)} |"
        )
    for a in summary["arms"]:
        for c in checkpoints:
            acc = a["accuracy"].get(c, {})
            lines.append(
                f"| {a['arm']} | {c} | "
                + " | ".join(_fmt(acc.get(d)) for d in summary["datasets"])
                + f" | {_fmt(a['accuracy_mean'].get(c))} |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None, repo: Path = REPO):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", required=True)
    args = parser.parse_args(argv)
    spec, _ = load_spec(args.sweep, repo)
    summary = summarize(args.sweep, repo)
    out = repo / spec["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    markdown = to_markdown(summary, spec["eval"]["checkpoints"])
    (out / "summary.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    sys.exit(main())
