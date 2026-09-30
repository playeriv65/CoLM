"""Checks made before a run, or before a resume, that would otherwise fail (or mislead) much later.

A checkpoint is written into its final directory (`Trainer._save_checkpoint`), `trainer_state.json`
last, so a directory without that file is a save that did not finish; the newest directory is not
necessarily a usable one. The save also fails only at its first step (256 or 512, minutes into a
run) when the disk is too small.
"""

import logging
import os
import re
import shutil

from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

_CHECKPOINT = re.compile(rf"^{PREFIX_CHECKPOINT_DIR}-(\d+)$")
_WEIGHTS_SUFFIX = ".safetensors"


def is_complete(checkpoint: str) -> bool:
    """Whether a checkpoint directory was written to the end (state file last, weights inside)."""
    return os.path.isfile(os.path.join(checkpoint, TRAINER_STATE_NAME)) and any(
        name.endswith(_WEIGHTS_SUFFIX) for name in os.listdir(checkpoint)
    )


def newest_complete_checkpoint(output_dir: str) -> str | None:
    """The `checkpoint-N` of the highest N that was written completely; None if there is none."""
    if not os.path.isdir(output_dir):
        return None
    steps = sorted(
        (int(m[1]), name) for name in os.listdir(output_dir) if (m := _CHECKPOINT.match(name))
    )
    for _, name in reversed(steps):
        path = os.path.join(output_dir, name)
        if is_complete(path):
            return path
        logger.warning(f"{path} is incomplete (an interrupted save?): skipped")
    return None


def check_resumable(checkpoint: str) -> None:
    """Fail on an incomplete checkpoint; warn that a model-only one restores no optimizer state."""
    if not os.path.isdir(checkpoint) or not is_complete(checkpoint):
        raise FileNotFoundError(
            f"{checkpoint} is not a complete checkpoint: it needs {TRAINER_STATE_NAME} (written "
            "last) and a weights file. Resume from an earlier checkpoint."
        )
    missing = [
        name for name in (OPTIMIZER_NAME, SCHEDULER_NAME)
        if not os.path.isfile(os.path.join(checkpoint, name))
    ]  # fmt: skip
    if missing:
        logger.warning(
            f"{checkpoint} has no {' / '.join(missing)} (`save_only_model`): only the weights, the "
            "step counter and the selection state are restored; the optimizer state, the learning-"
            "rate schedule position and the random state start again, so the run is not the "
            "uninterrupted one. Save with `--save_only_model false` to resume exactly."
        )


def checkpoint_bytes(parameters, save_only_model: bool) -> int:
    """Size of one checkpoint: the trainable weights (the adapter), plus Adam's two moments."""
    weights = sum(p.numel() * p.element_size() for p in parameters if p.requires_grad)
    return weights if save_only_model else 3 * weights


def planned_saves(args) -> int:
    """Checkpoints on disk at the peak: the periodic ones (`save_total_limit` keeps the newest,
    the one just written included before the oldest is removed) and the final model."""
    if args.save_strategy != "steps" or not args.save_steps:
        return 1
    interval = args.save_steps if args.save_steps >= 1 else args.save_steps * args.max_steps
    periodic = int(args.max_steps // interval)
    if args.save_total_limit:
        periodic = min(periodic, args.save_total_limit + 1)
    return periodic + 1


def check_disk_space(output_dir: str, per_checkpoint: int, saves: int) -> None:
    """Raise if the disk of `output_dir` cannot hold `saves` checkpoints (the periodic ones plus
    the final model), instead of failing at the first save."""
    path = os.path.abspath(output_dir)
    while not os.path.exists(path):
        path = os.path.dirname(path)
    free, needed = shutil.disk_usage(path).free, per_checkpoint * saves
    if free < needed:
        raise OSError(
            f"{output_dir}: {free / 1e9:.1f} GB free but the checkpoints need about "
            f"{needed / 1e9:.1f} GB ({saves} x {per_checkpoint / 1e9:.2f} GB); free space, "
            "point `output_dir` at another disk, or save fewer checkpoints (`save_steps`)."
        )
