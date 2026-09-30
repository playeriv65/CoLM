"""Selection state (the coreset selector's Adam moments) in and out of Trainer checkpoints.

The moments live on rank 0 only (the only rank that selects), so rank 0 writes one small file next
to every checkpoint (`on_save`) and reads it back before `Trainer.train` resumes. The optimizer,
scheduler and RNG are the stock Trainer's business. A checkpoint without the file resumes with a
warning: the moments restart from zero while the Adam bias correction still counts the restored
global step (`CoresetSelector._adam`), so the first selections after the resume see rescaled features.
"""

import logging
import os

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from colm.selection.select import CoresetSelector

logger = logging.getLogger(__name__)

SELECTION_STATE_FILE = "selection_state.pt"


def save_selection_state(selector: CoresetSelector, directory: str, zo_seed: int) -> str | None:
    """Write the selector state into `directory` (atomically); None if there is nothing to save."""
    if not selector.has_state:
        return None
    state = {k: None if v is None else v.detach().cpu() for k, v in selector.state_dict().items()}
    path = os.path.join(directory, SELECTION_STATE_FILE)
    tmp = f"{path}.tmp"
    torch.save({**state, "zo_seed": zo_seed}, tmp)
    os.replace(tmp, path)
    return path


def load_selection_state(
    selector: CoresetSelector, directory: str, zo_seed: int, device: torch.device | str = "cpu"
) -> bool:
    """Restore the selector state from the checkpoint `directory`; False (and a warning) if absent."""
    if not selector.has_state:
        return False
    path = os.path.join(directory, SELECTION_STATE_FILE)
    if not os.path.isfile(path):
        logger.warning(
            f"{directory} has no {SELECTION_STATE_FILE}: the selection Adam moments restart from "
            "zero at global-step bias correction, so the selections right after the resume are not "
            "those of an uninterrupted run (checkpoint written before the state was saved?)"
        )
        return False
    state = torch.load(path, map_location=device, weights_only=True)
    if state["zo_seed"] != zo_seed:
        raise ValueError(
            f"{path} was written with zo_seed={state['zo_seed']} but this run has zo_seed="
            f"{zo_seed}: the moments belong to another random direction (use the seed of the run)"
        )
    selector.load_state_dict(state)
    logger.info(f"Restored the selection Adam moments from {path}")
    return True


class SelectionStateCallback(TrainerCallback):
    """Save the selector state next to every checkpoint (process 0 only)."""

    def __init__(self, selector: CoresetSelector, zo_seed: int):
        self.selector = selector
        self.zo_seed = zo_seed

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        directory = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        if os.path.isdir(directory):
            save_selection_state(self.selector, directory, self.zo_seed)
