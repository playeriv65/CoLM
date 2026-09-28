#!/usr/bin/env python
import json
import logging
import os
import sys

import datasets
import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PhiConfig,
    set_seed,
)

from colm.data.get_training_dataset import (
    DataCollatorForSupervisedDataset,
    DataCollatorForSupervisedDatasetWithSource,
    HFDataset,
    SupervisedDataset,
    convert_superglue_to_hf,
    convert_superglue_to_hf_source,
    get_training_dataset,
)
from colm.data.tasks import Sample, get_task
from colm.data.utils import (
    DataCollatorWithPaddingAndNesting,
    NondiffCollator,
    forward_wrap_with_option_len,
)
from colm.train.data_arguments import DataArguments, get_data_statistics
from colm.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from colm.train.trainers import CustomTrainer, SubsetTrainer, SubsetTrainerEfficient
from colm.train.training_arguments import TrainingArguments

logger = logging.getLogger(__name__)
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
    parts = [
        model_name,
        task,
        "lora" if model_args.lora else "full",
        f"gas{training_args.gradient_accumulation_steps}",
        f"bs{training_args.per_device_train_batch_size}",
    ]
    if training_args.data_selection_method != "none":
        layers = "+".join(layer.split(".")[-1] for layer in training_args.last_layers)
        parts += [
            training_args.data_selection_method,
            training_args.data_selection_unit + ("-eff" if training_args.efficient_mezo else ""),
            f"r{training_args.small_batch_ratio}",
            layers,
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


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not model_args.lora_target_modules:
        if "phi-2" in model_args.model_name_or_path:
            model_args.lora_target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2"]
        else:  # Llama, zephyr
            model_args.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if model_args.precision == "auto" and not training_args.fp16 and not training_args.bf16:
        if "phi-2" in model_args.model_name_or_path or "Llama" in model_args.model_name_or_path:
            training_args.fp16 = True
            model_args.torch_dtype = "none"
        else:  # zephyr
            training_args.bf16 = True
            model_args.torch_dtype = "bfloat16"
    elif model_args.precision == "fp32":
        training_args.fp16 = training_args.bf16 = False
        model_args.torch_dtype = "float32"
    # Mixed precision is resolved in TrainingArguments.__post_init__; keep it in sync.
    training_args.mixed_precision = (
        "fp16" if training_args.fp16 else "bf16" if training_args.bf16 else "no"
    )

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
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {training_args.world_size > 1}, "
        f"fp16: {training_args.fp16}, bf16: {training_args.bf16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    model_kwargs = dict(
        dtype=DTYPES[model_args.torch_dtype],
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
    )
    if not model_args.enable_dropout:
        logger.info("Set dropout to 0")
        model_config = AutoConfig.from_pretrained(
            model_args.config_name or model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
        assert isinstance(model_config, PhiConfig), "Only support no dropout for Phi-2!"
        model_config.resid_pdrop = 0
        model_args.lora_dropout = 0
        model_kwargs["config"] = model_config
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if training_args.fsdp and (training_args.fsdp_config or {}).get(
        "activation_checkpointing", False
    ):
        logger.info("Enable gradient checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    add_padding_to_tokenizer(tokenizer)

    # Resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    modules_to_save = []
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # https://github.com/huggingface/peft/issues/334
        modules_to_save = ["lm_head", "embed_tokens"]

    if not isinstance(model, PeftModel) and model_args.lora:
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
        # ValueError: Attempting to unscale FP16 gradients
        # https://github.com/huggingface/peft/issues/341
        base = model.get_base_model()
        base.model.embed_tokens.weight.data = base.model.embed_tokens.weight.data.float()
        base.lm_head.weight.data = base.lm_head.weight.data.float()
        logger.info("Applied LoRA to model.")
        model.print_trainable_parameters()
        model.enable_input_require_grads()
        # The perturbed / selected tensors are the LoRA B matrices of the last layer.
        training_args.last_layers = [name + ".lora_B" for name in training_args.last_layers]

    logger.info(
        f"trainable model_params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(model)

    analysis_dataset = None
    if "superglue" in data_args.train_files[0]:
        task_name = data_args.train_files[0].split("-")[-1]
        task = get_task(task_name)
        if data_args.train_files[0].split("-")[0] == "load":
            with open(os.path.join(data_args.data_dir, f"{data_args.train_files[0]}.jsonl")) as f:
                train_samples = [Sample(**json.loads(line)) for line in f]
        else:
            train_samples = task.sample_subset(num=1000)
        convert = (
            convert_superglue_to_hf_source
            if training_args.source_wise_selection != "none"
            else convert_superglue_to_hf
        )
        convert_kwargs = dict(
            task=task,
            tokenizer=tokenizer,
            max_length=model_args.model_max_length,
            max_new_tokens=training_args.max_new_tokens,
            non_diff=training_args.non_diff,
            train_as_classification=task.classification,
            only_train_option=training_args.only_train_option,
        )
        train_dataset = HFDataset(convert(train_samples, **convert_kwargs))
        logger.info(
            f"Train dataset of task {task_name} has {len(train_samples)} examples with attributes "
            f"generation = {task.generation} and classification = {task.classification}"
        )
        logger.info(f"TRAIN DATASET EXAMPLE: {train_samples[0]}")
        if training_args.analysis_mode:
            analysis_dataset = HFDataset(
                convert_superglue_to_hf(task.samples["valid"], **convert_kwargs)
            )

        # Change forward pass of model for SuperGLUE
        if training_args.only_train_option and not training_args.non_diff:
            training_args.modify_forward = True
            model.original_forward = model.forward
            model.forward = forward_wrap_with_option_len.__get__(model, type(model))

        if task.classification:
            data_collator = DataCollatorWithPaddingAndNesting(tokenizer, pad_to_multiple_of=8)
        elif training_args.non_diff:
            data_collator = NondiffCollator(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    else:
        train_dataset = get_training_dataset(
            data_args.train_files,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            sample_percentage=data_args.percentage,
            subset_index_files=data_args.subset_index_files,
            seed=data_args.sample_data_seed,
            hf_datasets_cache_dir=data_args.hf_datasets_cache_dir,
        )
        logger.info(f"TRAIN DATASET: {train_dataset[0].keys()}")
        logger.info(f"TRAIN DATASET EXAMPLE: {train_dataset[0]}")

        if isinstance(train_dataset, SupervisedDataset):
            if (
                training_args.source_wise_selection != "none"
                or not training_args.remove_unused_columns
            ):
                data_collator = DataCollatorForSupervisedDatasetWithSource(tokenizer=tokenizer)
            else:
                data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        else:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model, padding="longest"
            )

        get_data_statistics(
            train_dataset, is_custom_dataset=isinstance(train_dataset, SupervisedDataset)
        )

        if (
            not isinstance(train_dataset, SupervisedDataset)
            and "dataset" in train_dataset.column_names
        ):
            train_dataset = train_dataset.remove_columns(["dataset", "id", "messages"])

    logger.info(f"Using data collator {type(data_collator)}")

    if len(training_args.keep_sources) and isinstance(
        data_collator, DataCollatorForSupervisedDatasetWithSource
    ):
        training_args.keep_sources = [int(idx) for idx in training_args.keep_sources.split("_")]
        logger.info("Keep all examples of the following sources in the mini-batch.")
        for source_idx in training_args.keep_sources:
            logger.info(train_dataset.all_data_sources[source_idx])
    else:
        training_args.keep_sources = []
    logger.info(f"Keep source indices in {training_args.keep_sources}")

    if training_args.data_selection_method == "none":
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
        data_collator=data_collator,
    )

    train_result = trainer.train(resume_from_checkpoint=model_args.checkpoint_path)
    if torch.cuda.is_available() and trainer.is_world_process_zero():
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    if torch.cuda.is_available():
        metrics["peak_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Only the adapter is needed; drop a full FSDP state dict if one was written.
    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(training_args.output_dir, "pytorch_model_fsdp.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)


if __name__ == "__main__":
    main()
