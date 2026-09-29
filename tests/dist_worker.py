"""Worker of test_distributed.py: CPU ranks (gloo) running a coreset trainer on the float64 fixtures."""

import json
import os
import sys
from pathlib import Path

import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from equivalence.fixtures import tokenizer  # noqa: E402
from equivalence.helpers import build, lora_state, make_args  # noqa: E402

CASES = {  # per-device batch, gradient accumulation (per rank), extra arguments
    "efficient": (2, 2, dict(efficient_mezo=True, keep_sources="0")),
    "regular": (1, 4, dict(data_selection_unit="mezo", keep_sources="0")),
}


def run(case: str, data: str, out: str, gas_scale: int = 1, steps: int = 2) -> dict:
    """Train; `gas_scale` = W makes one process hold the pools of W ranks (the reference run)."""
    bs, gas, extra = CASES[case]
    args = make_args(
        out,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=gas * gas_scale,
        max_steps=steps,
        **({"ddp_backend": "gloo"} if "RANK" in os.environ else {}),
        **extra,
    )
    trainer, model = build(args, tokenizer(), data)
    trained, reduces = [], []
    make = trainer.batching.train_batches

    def sub_batches(examples, weights):
        out = make(examples, weights)
        for batch, _ in out:
            trained.append(
                {
                    "step": trainer.state.global_step,
                    "indices": batch["colm_meta"]["indices"].tolist(),
                }
            )
        return out

    trainer.batching.train_batches = sub_batches
    step = trainer.training_step

    def training_step(model_, inputs, num_items_in_batch=None):
        if not reduces and hasattr(model_, "register_comm_hook"):
            reduces.append(0)  # count the gradient all-reduces of the DDP wrapper

            def hook(state, bucket):
                reduces[0] += 1
                return (
                    dist.all_reduce(bucket.buffer(), async_op=True)
                    .get_future()
                    .then(lambda f: f.value()[0] / dist.get_world_size())
                )

            model_.register_comm_hook(None, hook)
        return step(model_, inputs, num_items_in_batch)

    trainer.training_step = training_step
    trainer.train()
    replicas = trainer.check_replicas()
    return {
        "replicas": replicas,
        "trained": trained,
        "lora": {k: v.tolist() for k, v in lora_state(model).items()},
        "steps": trainer.state.global_step,
        "all_reduces": reduces[0] if reduces else 0,
        "loss": [h["loss"] for h in trainer.state.log_history if "loss" in h],
    }


if __name__ == "__main__":
    case, data, out = sys.argv[1:4]
    Path(out, f"rank{os.environ['RANK']}.json").write_text(json.dumps(run(case, data, out)))
    dist.barrier()
