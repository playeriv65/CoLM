"""Splits the time inside `Trainer.train` into first step, steady steps, evaluation and saves."""

import time

from transformers import TrainerCallback

from colm.phases import PhaseClock

EVAL_PHASE = "train/eval_loss"  # added by `EvalLossCallback`


class PhaseCallback(TrainerCallback):
    """One-shot timestamps at the begin, first step, saves and end of the training loop.

    `train` (the whole loop) = `first_step` + `steady_steps` + `checkpoint_save` + `eval_loss`;
    the evaluation callback adds its own seconds to the clock, and the other phases exclude them.
    """

    def __init__(self, clock: PhaseClock):
        self.clock = clock
        self.begin = self.first_step_seconds = self.step_end = None
        self.eval_at_step_end = 0.0
        self.saves = 0.0

    def _eval_seconds(self) -> float:
        return self.clock.seconds.get(EVAL_PHASE, 0.0)

    def on_train_begin(self, args, state, control, **kwargs):
        self.begin = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.step_end = time.time()
        self.eval_at_step_end = self._eval_seconds()
        if self.first_step_seconds is None:  # an evaluation before step 1 is not the first step
            self.first_step_seconds = self.step_end - self.begin - self.eval_at_step_end

    def on_save(self, args, state, control, **kwargs):
        # step end -> checkpoint written (logging included, the evaluation in between excluded)
        self.saves += time.time() - self.step_end - (self._eval_seconds() - self.eval_at_step_end)

    def on_train_end(self, args, state, control, **kwargs):
        loop = time.time() - self.begin
        first = self.first_step_seconds or 0.0
        evaluation = self._eval_seconds()
        self.clock.add("first_step", first)
        self.clock.add("checkpoint_save", self.saves)
        self.clock.add("eval_loss", evaluation)
        self.clock.add("steady_steps", loop - first - self.saves - evaluation)
        self.clock.seconds.pop(EVAL_PHASE, None)
