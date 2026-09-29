#!/usr/bin/env python
"""Train a (LoRA) language model with the CoLM trainers.

torchrun --nproc_per_node N -m colm.train.train config.json [--flag value ...]
"""

import json
import logging
import os
import sys
import time

import datasets
import torch
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)

from colm.data.get_training_dataset import SupervisedDataset, get_training_dataset, make_collator
from colm.data.holdout import save_holdout_indices, split_holdout
from colm.data.superglue import build_superglue
from colm.eval.eval_loss import add_eval_loss_callback
from colm.phases import CLOCK
from colm.train.config import context_length, parse_args, resolved_config, save_resolved_config
from colm.train.data_arguments import get_data_statistics
from colm.train.model_arguments import add_padding_to_tokenizer
from colm.train.phase_callback import PhaseCallback
from colm.train.trainers import CustomTrainer, SubsetTrainer, SubsetTrainerEfficient

logger = logging.getLogger(__name__)
PHASES_FILENAME = "startup.json"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "auto": "auto",
    # transformers 4.x loaded weights in fp32 when no dtype was given; v5 defaults to
    # the checkpoint dtype, so "none" is pinned to fp32 to keep the original recipe.
    "none": torch.float32,
}


def default_output_dir(model_args, data_args, training_args) -> str:
    """Descriptive run directory, e.g. out/phi-2-MathInstruct-lora-gas8-bs4-mezo-eff-...-seed0."""
    model_name = model_args.model_name_or_path.rstrip("/").split("/")[-1]
    task = os.path.splitext(os.path.basename(data_args.train_files[0]))[0]
    micro = training_args.micro_batch_size or training_args.per_device_train_batch_size
    parts = [
        model_name,
        task,
        "lora" if model_args.lora else "full",
        f"gas{training_args.pool_micro_batches if training_args.coreset else training_args.gradient_accumulation_steps}",
        f"bs{micro}",
    ]
    if training_args.coreset:
        parts += [
            training_args.data_selection_method,
            training_args.data_selection_unit + ("-eff" if training_args.efficient_mezo else ""),
            f"r{training_args.small_batch_ratio}",
            "+".join(training_args.last_layers),
            f"{training_args.zo_dim}_{training_args.mezo_topk}_{training_args.mezo_selection}",
        ]
    parts += [f"{training_args.max_steps}steps", f"seed{training_args.seed}"]
    return os.path.join(data_args.output_root, "-".join(parts))


def configure_wandb(training_args):
    """W&B is opt-in: only touched when report_to explicitly contains it."""
    report_to = training_args.report_to or []
    if isinstance(report_to, str):
        report_to = [report_to]
    if "wandb" not in report_to:
        return
    for env, value in [
        ("WANDB_ENTITY", training_args.wandb_entity),
        ("WANDB_PROJECT", training_args.wandb_project),
        ("WANDB_NOTES", training_args.wandb_notes),
    ]:
        if value:
            os.environ[env] = value
    os.environ.setdefault("WANDB_NAME", f"{training_args.run_name}_{os.uname()[1]}")


def build_model(model_args, training_args, tokenizer):
    kwargs = dict(
        dtype=DTYPES[model_args.torch_dtype],
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
    )
    if not model_args.enable_dropout:
        logger.info("Set dropout to 0")
        config = AutoConfig.from_pretrained(
            model_args.config_name or model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
        for name in ("resid_pdrop", "embd_pdrop", "attention_dropout"):
            if hasattr(config, name):
                setattr(config, name, 0.0)
        model_args.lora_dropout = 0
        kwargs["config"] = config
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **kwargs)

    # Resize embeddings if needed (e.g. for LlamaTokenizer)
    modules_to_save = []
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        # https://github.com/huggingface/peft/issues/334
        modules_to_save = ["lm_head", "embed_tokens"]

    if model_args.lora and not isinstance(model, PeftModel):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        # The fp16 gradient scaler cannot unscale fp16 parameters: trainable ones are fp32 (the
        # upstream code upcast the embedding and the head unconditionally instead).
        for p in model.parameters():
            if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
        model.print_trainable_parameters()
        model.enable_input_require_grads()
    if training_args.should_log:
        logger.info(
            f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
    return model


def build_data(data_args, training_args, eval_args, tokenizer, context):
    """(train dataset, collator, analysis dataset, held-out examples)."""
    if "superglue" in data_args.train_files[0]:
        return (*build_superglue(data_args, training_args, tokenizer, context), None)
    dataset = get_training_dataset(
        data_args.train_files,
        tokenizer=tokenizer,
        context_length=context,
        sample_percentage=data_args.percentage,
        subset_index_files=data_args.subset_index_files,
        seed=data_args.sample_data_seed,
        hf_datasets_cache_dir=data_args.hf_datasets_cache_dir,
        subset_selection=data_args.subset_selection,
        token_cache_dir=data_args.token_cache_dir,
    )
    heldout = None
    if eval_args.holdout_size:
        if not isinstance(dataset, SupervisedDataset):
            raise ValueError("holdout_size needs an instruction/output (SupervisedDataset) file")
        with CLOCK.detail("data/holdout"):
            dataset, heldout = split_holdout(
                dataset, eval_args.holdout_size, eval_args.holdout_seed
            )
        logger.info(f"Held out {len(heldout)} examples; training on {len(dataset)}")
        if training_args.should_save:
            save_holdout_indices(heldout, training_args.output_dir)
    with CLOCK.detail("data/statistics"):
        get_data_statistics(dataset, is_custom_dataset=isinstance(dataset, SupervisedDataset))
    if isinstance(dataset, SupervisedDataset):
        collator = make_collator(training_args, tokenizer)
        for source in training_args.keep_source_ids:
            if not 0 <= source < dataset.num_sources:
                raise ValueError(
                    f"keep_sources has source {source}, the data has {dataset.num_sources}"
                )
            logger.info(f"Kept in full: {dataset.all_data_sources[source]}")
    else:  # pre-tokenised (LESS) data: no per-example bookkeeping, so no selection
        if training_args.coreset:
            raise ValueError("coreset selection needs an instruction / output data file")
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
        if "dataset" in dataset.column_names:
            dataset = dataset.remove_columns(["dataset", "id", "messages"])
    return dataset, collator, None, heldout


def main(argv=None):
    CLOCK.mark("imports")
    model_args, data_args, training_args, eval_args = parse_args(argv)
    if training_args.output_dir_is_auto:
        training_args.output_dir = default_output_dir(model_args, data_args, training_args)
    if training_args.run_name is None or training_args.run_name == training_args.output_dir:
        training_args.run_name = os.path.basename(training_args.output_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    logging.getLogger("colm").setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.warning(
        f"rank {training_args.process_index}/{training_args.world_size}, device {training_args.device}, "
        f"fp16 {training_args.fp16}, bf16 {training_args.bf16}"
    )
    logger.info(f"{training_args}\n{model_args}\n{data_args}")

    set_seed(training_args.seed)
    CLOCK.mark("config")
    context = context_length(model_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        model_max_length=context,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    add_padding_to_tokenizer(tokenizer)
    CLOCK.mark("tokenizer")
    model = build_model(model_args, training_args, tokenizer)
    CLOCK.mark("model_load")
    train_dataset, collator, analysis_dataset, heldout = build_data(
        data_args, training_args, eval_args, tokenizer, context
    )
    CLOCK.mark("data")

    if not training_args.coreset:
        trainer_class = CustomTrainer
    elif training_args.efficient_mezo:
        trainer_class = SubsetTrainerEfficient
    else:
        trainer_class = SubsetTrainer
    logger.info(f"Using {trainer_class.__name__}")
    configure_wandb(training_args)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=analysis_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[PhaseCallback(CLOCK)],
    )
    CLOCK.mark("trainer_init")
    add_eval_loss_callback(trainer, eval_args, heldout, training_args.output_dir, context)
    CLOCK.mark("eval_setup")
    config = resolved_config(
        model_args, data_args, training_args, eval_args,
        {"context_length": context, "train_examples": len(train_dataset), **trainer.describe()},
    )  # fmt: skip
    logger.info(f"Resolved config:\n{json.dumps(config, indent=1, default=str)}")
    if training_args.should_save:
        save_resolved_config(config, training_args.output_dir)
    CLOCK.mark("config_save")

    result = trainer.train(resume_from_checkpoint=model_args.checkpoint_path)
    CLOCK.last = time.time()  # the loop was split into its phases by PhaseCallback
    trainer.check_replicas()
    trainer.save_model()
    metrics = result.metrics
    metrics["train_samples"] = len(train_dataset)
    memory = trainer.save_memory_report()  # peaks of every rank, not only rank 0
    # (transformers formats every metric named `*_mem_*` as an integer number of bytes)
    metrics.update({k.replace("_mem_", "_memory_"): v for k, v in memory.items()})
    logger.info(f"Peak GPU memory (allocated, max over ranks): {memory.get('peak_mem_gb')} GB")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    CLOCK.mark("final_save")
    if trainer.is_world_process_zero():
        CLOCK.save(
            os.path.join(training_args.output_dir, PHASES_FILENAME), steps=trainer.state.global_step
        )


if __name__ == "__main__":
    main()
