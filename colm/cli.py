"""Entry points: `colm-train`, `colm-eval`, `colm-sweep` (see the README quickstart)."""

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from colm.phases import LAUNCH_ENV

REPO = Path(__file__).resolve().parents[1]

TRAIN_USAGE = """usage: colm-train [config.json] [--gpus 0,1] [--<option> <value> ...]

Trains on the GPUs given by --gpus (or $COLM_GPUS; there is no default, GPUs are shared and
reserved): one process per GPU, launched with torchrun. The config file gives the options that
differ from the defaults of the paper recipe, flags override it. Output goes to the terminal and
to logs/<config>-gpu<ids>-np<n>-<time>.log ($COLM_LOG_DIR). Under torchrun / inside a launched
process the options are used as they are.

The options are those of the run (defaults included):
"""


@contextlib.contextmanager
def forwarding_termination(process):
    """Pass SIGTERM / SIGHUP received by the launcher on to `process` (torchrun) while it runs.

    Without it `kill <launcher>` ends only the launcher: torchrun and its workers keep the GPUs
    (and write into a closed pipe) until they fail on it. torchrun stops its workers on SIGTERM.
    """
    signals = (signal.SIGTERM, signal.SIGHUP)
    previous = [
        signal.signal(s, lambda number, frame: process.send_signal(signal.SIGTERM)) for s in signals
    ]
    try:
        yield
    finally:
        for number, handler in zip(signals, previous, strict=True):
            signal.signal(number, handler)


def train(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "RANK" in os.environ:  # already launched: one of the processes of a run
        from colm.train.train import main

        main(argv)
        return 0
    if "-h" in argv or "--help" in argv:
        print(TRAIN_USAGE)
        from colm.train.config import parse_args

        parse_args(["--help"])
    gpus = os.environ.get("COLM_GPUS", "")
    if "--gpus" in argv:
        i = argv.index("--gpus")
        gpus = argv[i + 1]
        del argv[i : i + 2]
    if not gpus.replace(",", "").isdigit():
        sys.exit("colm-train: give the reserved GPU ids with --gpus 0,1 (or COLM_GPUS)")
    processes = len(gpus.split(","))
    name = Path(argv[0]).stem if argv and argv[0].endswith(".json") else "run"
    log_dir = Path(os.environ.get("COLM_LOG_DIR", REPO / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (
        log_dir
        / f"{name}-gpu{gpus.replace(',', '_')}-np{processes}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    )
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", str(processes), "-m", "colm.train.train", *argv,
    ]  # fmt: skip
    print(f"gpus={gpus} processes={processes} log={log}", flush=True)
    launched = time.time()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpus,
        "PYTHONUNBUFFERED": "1",
        LAUNCH_ENV: str(launched),  # lets startup.json count the launcher and torchrun start-up
    }
    with (
        subprocess.Popen(
            command, env=env, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process,
        open(log, "w") as file,
        forwarding_termination(process),
    ):
        for line in process.stdout:
            sys.stdout.write(line)
            file.write(line)
            file.flush()
    done = f"wall clock {time.time() - launched:.0f} s, exit code {process.returncode}"
    print(done, flush=True)
    with open(log, "a") as file:
        file.write(done + "\n")
    return process.returncode


EVAL_USAGE = """usage: colm-eval {loss,accuracy,superglue} [options]

  loss      teacher-forced loss on the held-out set and the GSM8K solutions (colm.eval.eval_loss)
  accuracy  answer accuracy on the math datasets with vLLM (math_eval/run_open.py); defaults are
            the paper protocol: 0-shot PoT with CoT backup on gsm8k math numglue svamp deepmind
            simuleq, the dtype of the model's recipe, LoRA enabled for adapters
  superglue SuperGLUE tasks (superglue_eval/eval_superglue.py)

`colm-eval <command> --help` lists the options of a command.
"""


def _accuracy_defaults(argv: list[str]) -> list[str]:
    """`run_open.py` flags of the paper protocol that the user did not give."""
    if "-h" in argv or "--help" in argv:
        return argv
    from colm.train.config import eval_dtype

    def given(flag):
        return flag in argv

    out = list(argv)
    if not given("--stem_flan_type"):
        out += ["--stem_flan_type", "pot_prompt"]
    for flag in ("--cot_backup", "--use_vllm"):
        if not given(flag):
            out.append(flag)
    if not given("--batch_size"):
        out += ["--batch_size", "8"]
    if not given("--dataset"):
        out += ["--dataset", "gsm8k", "math", "numglue", "svamp", "deepmind", "simuleq"]
    models = out[out.index("--model") + 1 :] if given("--model") else []
    first = next((m for m in models if not m.startswith("--")), None)
    if first:
        if not given("--dtype"):
            out += ["--dtype", eval_dtype(first)]
        if not given("--enable_lora") and (Path(first) / "adapter_config.json").exists():
            out.append("--enable_lora")
    return out


def evaluate(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(EVAL_USAGE)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "loss":
        from colm.eval.eval_loss import main

        main(rest)
    elif command in ("accuracy", "superglue"):
        folder, script = (
            ("math_eval", "run_open.py")
            if command == "accuracy"
            else ("superglue_eval", "eval_superglue.py")
        )
        import runpy

        sys.path.insert(0, str(REPO / folder))
        sys.argv = [script, *(_accuracy_defaults(rest) if command == "accuracy" else rest)]
        runpy.run_path(str(REPO / folder / script), run_name="__main__")
    else:
        sys.exit(f"colm-eval: unknown command {command!r}\n{EVAL_USAGE}")
    return 0


SWEEP_USAGE = """usage: colm-sweep {create,work,summary} [options]

  create   expand a sweep spec into a job queue        (--sweep spec.json --queue dir)
  work     run the jobs of a queue on the given GPU(s) (--queue dir --gpu 2)
  summary  collect the results of a sweep              (--sweep spec.json)

`colm-sweep <command> --help` lists the options of a command.
"""


def sweep(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(SWEEP_USAGE)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "create":
        from colm.jobs.rank_sweep import main
    elif command == "work":
        from colm.jobs.worker import main
    elif command == "summary":
        from colm.jobs.summarize import main
    else:
        sys.exit(f"colm-sweep: unknown command {command!r}\n{SWEEP_USAGE}")
    return main(rest) or 0
