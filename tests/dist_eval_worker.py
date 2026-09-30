"""Worker of test_distributed.py: CPU ranks (gloo) evaluating the loss of a tiny float64 Phi."""

import json
import os
import sys
from pathlib import Path

import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from equivalence.fixtures import model_fp64, tokenizer  # noqa: E402

from colm.data.get_training_dataset import get_training_dataset  # noqa: E402
from colm.eval.eval_loss import evaluate_loss  # noqa: E402


def run(data: str, batch_size: int = 3) -> dict:
    tok = tokenizer()
    dataset = get_training_dataset([data], tok, 512)
    model = model_fp64(tok).eval()
    return evaluate_loss(model, dataset, tok, batch_size, "cpu")


if __name__ == "__main__":
    data, out = sys.argv[1:3]
    dist.init_process_group("gloo")
    result = run(data)
    Path(out, f"rank{os.environ['RANK']}.json").write_text(json.dumps(result))
    dist.barrier()
