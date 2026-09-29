"""Expand a sweep spec (configs/rank_sweep/*.json) into a job queue.

    python -m colm.jobs.rank_sweep --sweep configs/rank_sweep/sweep.json --queue queues/rank-sweep

Queue order: a cheap base-model eval-loss job (fails fast if the eval-loss path is broken),
then per arm its train job, eval-loss job (standalone, on the saved checkpoints) and accuracy
job (vLLM), the base model's accuracy job, and finally a CPU-only summary job. The per-arm
training config is written to ``<output_root>/<arm>/train_config.json`` (self-contained
provenance of the run).
"""

import argparse
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from colm.jobs.file_queue import JobQueue
from colm.train.training_arguments import TrainingArguments

REPO = Path(__file__).resolve().parents[2]
TRAIN_CONFIG = "train_config.json"
BASE_LOSS_FILE = "base_eval_loss.json"
CKPT_LOSS_FILE = "eval_loss_checkpoints.json"


def arm_label(arm: dict) -> str:
    return f"r{arm['lora_r']}-a{arm['lora_alpha']:g}"


def arm_run_name(spec: dict, arm: dict, base_config: dict) -> str:
    model = base_config["model_name_or_path"].rstrip("/").split("/")[-1]
    return f"{model}-{arm_label(arm)}-{base_config['max_steps']}steps-seed{base_config['seed']}"


def load_spec(path, repo: Path = REPO) -> tuple[dict, dict]:
    """(spec, base training config with the sweep's overrides applied); `path` is under `repo`."""
    spec = json.loads((repo / path).read_text())
    base = json.loads((repo / spec["base_config"]).read_text())
    base.update(spec.get("train_overrides", {}))
    # The run names and the summary need these; a config only lists what differs from the defaults.
    defaults = {f.name: f.default for f in fields(TrainingArguments)}
    for key in ("max_steps", "seed", "save_steps"):
        base.setdefault(key, defaults[key])
    return spec, base


def arm_dir(spec: dict, arm: dict, base_config: dict) -> str:
    return f"{spec['output_root']}/{arm_run_name(spec, arm, base_config)}"


def train_config(spec: dict, arm: dict, base_config: dict) -> dict:
    return {
        **base_config,
        "lora_r": arm["lora_r"],
        "lora_alpha": arm["lora_alpha"],
        "output_dir": arm_dir(spec, arm, base_config),
    }


def _git_commit(repo: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def accuracy_argv(ev: dict, models: list[str], lora: bool, extra: list[str]) -> list[str]:
    """`math_eval/run_open.py` command line shared by the arms' and the base model's accuracy jobs."""
    argv = [
        "{python}",
        "-u",
        "math_eval/run_open.py",
        "--model",
        *models,
        "--dataset",
        *ev["datasets"],
        "--shots",
        str(ev["shots"]),
        "--stem_flan_type",
        ev["stem_flan_type"],
        "--batch_size",
        str(ev["batch_size"]),
        "--max_new_tokens",
        str(ev["max_new_tokens"]),
        "--cot_backup",
        "--use_vllm",
        "--dtype",
        ev["dtype"],
    ]
    if lora:
        argv.append("--enable_lora")
    if ev.get("gpu_memory_utilization"):
        argv += ["--gpu_memory_utilization", str(ev["gpu_memory_utilization"])]
    return argv + extra


def base_accuracy_job(spec: dict, base_config: dict) -> dict:
    """Accuracy of the un-tuned base model, same eval settings as the arms (the reference row)."""
    ev = spec["eval"]
    steps, seed = base_config["max_steps"], base_config["seed"]
    out_dir = ev["base_output_dir"]
    limit_args = ["--limit", str(ev["limit"])] if ev.get("limit") else []
    return {
        "name": "evalacc-base",
        "kind": "eval_acc",
        "argv": accuracy_argv(
            ev,
            [base_config["model_name_or_path"]],
            lora=False,
            extra=["--output_dir", out_dir, *limit_args],
        ),
        "env": dict(spec["env"]),
        "log_stem": f"{spec['name']}-evalacc-base-{steps}steps-seed{seed}",
        "expects": [f"{out_dir}/{d}_*.metrics.json" for d in ev["datasets"]],
    }


def build_jobs(spec: dict, base_config: dict, sweep_arg: str) -> list[dict]:
    steps, seed = base_config["max_steps"], base_config["seed"]
    env = dict(spec["env"])
    ev = spec["eval"]
    limit_args = ["--limit", str(ev["limit"])] if ev.get("limit") else []
    first_dir = arm_dir(spec, spec["arms"][0], base_config)
    root = spec["output_root"]

    jobs = [
        {
            "name": "evalloss-base",
            "kind": "eval_loss",
            "argv": [
                "{python}",
                "-u",
                "-m",
                "colm.eval.eval_loss",
                "--train_config",
                f"{first_dir}/{TRAIN_CONFIG}",
                "--base",
                "--output",
                f"{root}/{BASE_LOSS_FILE}",
                *limit_args,
            ],
            "env": env,
            "log_stem": f"{spec['name']}-evalloss-base-{steps}steps-seed{seed}",
            "requires": [f"{first_dir}/{TRAIN_CONFIG}"],
            "expects": [f"{root}/{BASE_LOSS_FILE}"],
        }
    ]
    for arm in spec["arms"]:
        directory = arm_dir(spec, arm, base_config)
        label = arm_label(arm)
        key = f"{label}-{steps}steps-seed{seed}"
        checkpoints = [f"{directory}/checkpoint-{c}" for c in ev["checkpoints"]]
        adapter_files = [f"{c}/adapter_model.safetensors" for c in checkpoints]
        jobs.append(
            {
                "name": f"train-{label}",
                "kind": "train",
                "arm": arm,
                "argv": ["{python}", "-u", "-m", "colm.train.train", f"{directory}/{TRAIN_CONFIG}"],
                "env": env,
                "log_stem": f"{spec['name']}-train-{key}",
                "expects": adapter_files + [f"{directory}/eval_loss.jsonl"],
            }
        )
        jobs.append(
            {
                "name": f"evalloss-{label}",
                "kind": "eval_loss",
                "arm": arm,
                "argv": [
                    "{python}",
                    "-u",
                    "-m",
                    "colm.eval.eval_loss",
                    "--train_config",
                    f"{directory}/{TRAIN_CONFIG}",
                    "--adapter",
                    *checkpoints,
                    "--output",
                    f"{directory}/{CKPT_LOSS_FILE}",
                    *limit_args,
                ],
                "env": env,
                "log_stem": f"{spec['name']}-evalloss-{key}",
                "requires": adapter_files,
                "expects": [f"{directory}/{CKPT_LOSS_FILE}"],
            }
        )
        jobs.append(
            {
                "name": f"evalacc-{label}",
                "kind": "eval_acc",
                "arm": arm,
                "argv": accuracy_argv(ev, checkpoints, lora=True, extra=limit_args),
                "env": env,
                "log_stem": f"{spec['name']}-evalacc-{key}",
                "requires": adapter_files,
                "expects": [
                    f"{c}/outputs/{d}_*.metrics.json" for c in checkpoints for d in ev["datasets"]
                ],
            }
        )
    jobs.append(base_accuracy_job(spec, base_config))
    jobs.append(
        {
            "name": "summary",
            "kind": "summary",
            "gpu": False,
            "argv": ["{python}", "-u", "-m", "colm.jobs.summarize", "--sweep", sweep_arg],
            "env": {},
            "log_stem": f"{spec['name']}-summary-{steps}steps-seed{seed}",
            "expects": [f"{root}/summary.md"],
        }
    )
    return jobs


def create_queue(sweep_path, queue_dir, forbidden_cache_prefixes=(), repo: Path = REPO) -> Path:
    """`sweep_path` and `queue_dir` are relative to `repo` (the worker's working directory)."""
    sweep_arg = str(sweep_path)
    spec, base = load_spec(sweep_path, repo)
    queue = JobQueue(repo / queue_dir)
    if any(queue.jobs(state) for state in ("pending", "running", "done", "failed")):
        raise SystemExit(f"{queue.root} already has jobs; use a fresh queue directory")

    for arm in spec["arms"]:
        directory = repo / arm_dir(spec, arm, base)
        if list(directory.glob("checkpoint-*")):
            raise SystemExit(f"{directory} already has checkpoints; refusing to overwrite")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / TRAIN_CONFIG).write_text(
            json.dumps(train_config(spec, arm, base), indent=4) + "\n"
        )

    for position, job in enumerate(build_jobs(spec, base, sweep_arg)):
        queue.add(position, job)
    queue.write_meta(
        {
            "sweep": sweep_arg,
            "commit": _git_commit(repo),
            "require_env": spec.get("require_env", []),
            "local_cache_env": spec.get("local_cache_env", []),
            "forbidden_cache_prefixes": list(forbidden_cache_prefixes),
            "require_hf_files": spec.get("require_hf_files", {}),
        }
    )
    return queue.root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", required=True, help="Sweep spec json.")
    parser.add_argument("--queue", required=True, help="New queue directory.")
    parser.add_argument(
        "--forbid-cache-prefix",
        action="append",
        default=[],
        help="The worker refuses to start if HF_HOME/HF_HUB_CACHE lie under this path (repeatable).",
    )
    args = parser.parse_args(argv)
    root = create_queue(args.sweep, args.queue, args.forbid_cache_prefix)
    queue = JobQueue(root)
    for path in queue.jobs("pending"):
        print(path.name)
    print(f"queue {root}: {len(queue.jobs('pending'))} jobs")


if __name__ == "__main__":
    sys.exit(main())
