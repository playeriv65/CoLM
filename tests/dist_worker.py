"""Worker for test_distributed.py: 2 CPU ranks (gloo) running SubsetTrainerEfficient."""

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import conftest  # noqa: E402

from colm.data.get_training_dataset import (  # noqa: E402
    DataCollatorForSupervisedDatasetWithSource,
    get_training_dataset,
)
from colm.train.trainers import SubsetTrainer, SubsetTrainerEfficient  # noqa: E402
from colm.train.training_arguments import TrainingArguments  # noqa: E402


def main():
    mixture_file, out_dir, efficient = sys.argv[1], sys.argv[2], sys.argv[3] == "efficient"
    tokenizer = conftest.tokenizer.__wrapped__()
    bs = 4 if efficient else 1
    args = TrainingArguments(
        output_dir=out_dir,
        use_cpu=True,
        ddp_backend="gloo",
        max_steps=2,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=2 if efficient else 4,
        small_batch_ratio=0.5,
        efficient_mezo=efficient,
        last_layer_index=conftest.NUM_LAYERS - 1,
        zo_dim=16,
        learning_rate=1e-3,
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        keep_sources="0",
    )
    args.keep_sources = [0]
    args.last_layers = [n + ".lora_B" for n in args.last_layers]
    model = conftest.add_lora(conftest.make_phi(tokenizer))
    dataset = get_training_dataset([mixture_file], tokenizer=tokenizer, max_seq_length=512)
    trainer_cls = SubsetTrainerEfficient if efficient else SubsetTrainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSupervisedDatasetWithSource(tokenizer=tokenizer),
    )
    trained = []
    original = trainer.training_step

    def training_step(model, inputs, num_items_in_batch=None):
        if inputs:
            trained.extend(int(i) for i in inputs["indices"])
        return original(model, inputs, num_items_in_batch)

    trainer.training_step = training_step
    trainer.train()
    lora = {n: p.detach().sum().item() for n, p in model.named_parameters() if "lora_" in n}
    rank = int(os.environ["RANK"])
    Path(out_dir, f"rank{rank}.json").write_text(
        json.dumps({"trained": trained, "lora": lora, "steps": trainer.state.global_step})
    )
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
