"""Peak GPU memory of the worst-case selection and training forwards of a run.

    CUDA_VISIBLE_DEVICES=<gpu> uv run python scripts/memory_worst_case.py configs/math_phi2_efficient.json

Builds the model of the config (as train.py does), then runs one selection forward (the MeZO
estimate) and one forward + backward of the training step on packs of maximum-length examples
(`--length` tokens, half of them label tokens): a pack of `micro_batch_size` examples for the
selection; for training `--train_examples` examples go through `batching.train_batches`, i.e.
packs of at most `train_max_tokens` (an example is never split), a forward + backward per pack.
Prints the peak allocated and reserved memory of each phase.
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from colm.selection.packing import Example, pack
from colm.train.config import context_length, parse_args
from colm.train.model_arguments import add_padding_to_tokenizer
from colm.train.train import build_model
from colm.train.trainers import SubsetTrainerEfficient


def example(length: int, vocab: int, rng) -> Example:
    ids = rng.integers(0, vocab, length)
    labels = ids.copy()
    labels[: length // 2] = -100
    return Example(ids, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--length", type=int, default=None, help="tokens per example (default: the context window)"
    )
    parser.add_argument(
        "--train_examples", type=int, default=2, help="selected examples of the training step"
    )
    parser.add_argument("--output_dir", default="/tmp/colm-memory")
    cli, rest = parser.parse_known_args()
    model_args, _, training_args, _ = parse_args(
        [cli.config, "--output_dir", cli.output_dir, "--report_to", "none", *rest]
    )
    length = cli.length or context_length(model_args)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    add_padding_to_tokenizer(tokenizer)
    model = build_model(model_args, training_args, tokenizer)
    dataset = TensorDataset(torch.zeros(1))
    dataset.mean_tokens = length
    trainer = SubsetTrainerEfficient(
        model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer
    )
    rng = np.random.default_rng(0)
    vocab = model.config.vocab_size
    micro = training_args.micro_batch_size

    select_pack = trainer._prepare_inputs(pack([example(length, vocab, rng) for _ in range(micro)]))
    trainer.memory.start()
    trainer.model.eval()
    trainer.extractor.extract(select_pack)
    trainer.memory.stop("selection")

    selected = [example(length, vocab, rng) for _ in range(cli.train_examples)]
    packs = trainer.batching.train_batches(selected, [1.0] * len(selected))
    total = trainer.batching.total_labels(selected)
    trainer.model.train()
    trainer.memory.start()
    for train_pack, weights in packs:
        train_pack = trainer._prepare_inputs(train_pack)
        with torch.autocast("cuda", dtype=torch.float16 if training_args.fp16 else torch.bfloat16):
            loss = trainer.batching.loss(trainer, trainer.model, train_pack, weights, total)
        loss.backward()
    trainer.memory.stop("train")
    print(
        json.dumps(
            {
                "tokens_per_example": length,
                "train_packs": len(packs),
                **trainer.memory.gather(run=True)[0],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
