"""Build trainers on the float64 fixtures (shared by the golden and property tests)."""

import pathlib

import torch

from colm.data.get_training_dataset import get_training_dataset, make_collator
from colm.train import attention
from colm.train.trainers import CustomTrainer, SubsetTrainer, SubsetTrainerEfficient
from colm.train.training_arguments import TrainingArguments
from equivalence.fixtures import NUM_LAYERS, mixture, model_fp64, tokenizer

__all__ = [
    "NUM_LAYERS",
    "build",
    "make_args",
    "mixture",
    "model_fp64",
    "tokenizer",
    "trainer_class",
]


def make_args(out, **kw) -> TrainingArguments:
    base = dict(
        output_dir=str(out),
        use_cpu=True,
        max_steps=3,
        learning_rate=1e-3,
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        last_layer_index=NUM_LAYERS - 1,
        zo_dim=16,
        dataloader_num_workers=0,
    )
    base.update(kw)
    return TrainingArguments(**base)


def trainer_class(args):
    if not args.coreset:
        return CustomTrainer
    return SubsetTrainerEfficient if args.efficient_mezo else SubsetTrainer


def build(args, tok, data, lora_dropout=0.0, model=None):
    attn = "sdpa" if args.legacy else attention.register()
    model = model or model_fp64(tok, lora_dropout=lora_dropout, attn=attn)
    dataset = get_training_dataset([data], tokenizer=tok, max_seq_length=512)
    trainer = trainer_class(args)(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tok,
        data_collator=make_collator(args, tok),
    )
    return trainer, model


def lora_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def data_file(tmp_path: pathlib.Path) -> str:
    path = str(tmp_path / "mixture.jsonl")
    mixture(path)
    return path
