"""Peak GPU memory of the worst-case selection and training forwards of a run.

    CUDA_VISIBLE_DEVICES=<gpu> uv run python scripts/memory_worst_case.py configs/math_phi2_efficient.json

Builds the model of the config (as train.py does), then runs one selection forward (the MeZO
estimate) and one forward + backward of the training step on packs of maximum-length examples
(`--length` tokens, half of them label tokens): a pack of `micro_batch_size` examples for the
selection and of `micro_batch_size * small_batch_ratio` for training. Prints the peak allocated
and reserved memory of each phase.
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from colm.selection.packing import Example, pack
from colm.train.config import parse_args, sequence_limit
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
    parser.add_argument("--output_dir", default="/tmp/colm-memory")
    cli, rest = parser.parse_known_args()
    model_args, data_args, training_args, _ = parse_args(
        [cli.config, "--output_dir", cli.output_dir, "--report_to", "none", *rest]
    )
    length = cli.length or sequence_limit(model_args, data_args, training_args.legacy)
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
    train_micro = max(1, int(micro * training_args.small_batch_ratio))

    select_pack = trainer._prepare_inputs(pack([example(length, vocab, rng) for _ in range(micro)]))
    trainer.memory.start()
    trainer.model.eval()
    trainer.extractor.extract(select_pack)
    trainer.memory.stop("selection")

    train_pack = trainer._prepare_inputs(
        pack([example(length, vocab, rng) for _ in range(train_micro)])
    )
    trainer.model.train()
    trainer.memory.start()
    with torch.autocast("cuda", dtype=torch.float16 if training_args.fp16 else torch.bfloat16):
        loss = trainer.batching.loss(
            trainer, trainer.model, train_pack, torch.ones(train_micro), 1, 1
        )
    loss.backward()
    trainer.memory.stop("train")
    print(
        json.dumps({"tokens_per_example": length, **trainer.memory.gather(run=True)[0]}, indent=1)
    )


if __name__ == "__main__":
    main()
