"""Teacher-forced evaluation loss of a (LoRA) model on reference solutions.

Two sets are supported:

* ``heldout``: examples of the training mixture that were excluded from training
  (`colm.data.holdout.split_holdout`), rendered exactly like training examples;
* ``gsm8k``: the reference solutions of the GSM8K test split, in the MathInstruct CoT style
  (calculator annotations dropped, ``#### n`` rewritten to ``The answer is n``).

The loss is the mean negative log-likelihood per completion token (prompt tokens are masked,
the EOS token counts), pooled over the whole set. It is used in two ways: as a trainer callback
(learning curve at chosen steps) and as a standalone CLI on saved adapters
(``python -m colm.eval.eval_loss``).
"""

import argparse
import contextlib
import json
import logging
import math
import os
import re
import time

import torch
import torch.nn.functional as F
from transformers import HfArgumentParser, TrainerCallback

from colm.data.get_training_dataset import (
    IGNORE_INDEX,
    SupervisedCollator,
    SupervisedDataset,
    get_training_dataset,
)
from colm.data.holdout import select_examples, split_holdout
from colm.eval.arguments import GSM8K_SET, HELDOUT_SET, HeldoutEvalArguments

logger = logging.getLogger(__name__)

EVAL_LOSS_FILENAME = "eval_loss.jsonl"
GSM8K_SOURCE = "gsm8k_test"
_CALCULATOR_ANNOTATION = re.compile(r"<<[^>]*>>")
_FINAL_ANSWER = re.compile(r"\n?#### *(.*)\s*$")


# ---------------------------------------------------------------------------
# Sets
# ---------------------------------------------------------------------------
def clean_gsm8k_solution(answer: str) -> str:
    """GSM8K reference solution in the MathInstruct CoT style."""
    answer = _CALCULATOR_ANNOTATION.sub("", answer)
    return _FINAL_ANSWER.sub(lambda m: f"\nThe answer is {m.group(1).strip()}", answer).strip()


def load_gsm8k_test(path: str, tokenizer, limit: int | None = None) -> SupervisedDataset:
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rows.append(
                {
                    "instruction": row["question"].strip(),
                    "output": clean_gsm8k_solution(row["answer"]),
                    "source": GSM8K_SOURCE,
                }
            )
    if limit:
        rows = rows[:limit]
    return SupervisedDataset(list_data_dict=rows, tokenizer=tokenizer, template_variation=False)


def build_eval_sets(
    eval_args: HeldoutEvalArguments,
    tokenizer,
    heldout: SupervisedDataset | None,
    limit: int | None = None,
) -> dict[str, SupervisedDataset]:
    """The sets named in `eval_args.eval_loss_sets`; `limit` truncates each (smoke runs)."""
    sets = {}
    for name in eval_args.eval_loss_sets:
        if name == HELDOUT_SET:
            if heldout is None:
                raise ValueError("eval_loss_sets contains 'heldout' but holdout_size is 0")
            sets[name] = (
                select_examples(heldout, range(min(limit, len(heldout)))) if limit else heldout
            )
        elif name == GSM8K_SET:
            sets[name] = load_gsm8k_test(eval_args.eval_loss_gsm8k_file, tokenizer, limit)
        else:
            raise ValueError(f"Unknown eval loss set {name!r}")
    return sets


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(
    model,
    dataset: SupervisedDataset,
    tokenizer,
    batch_size: int,
    device,
    autocast=contextlib.nullcontext,
) -> dict:
    """Pooled mean token NLL of `dataset` (plus per-source breakdown and counts)."""
    collator = SupervisedCollator(tokenizer)
    # Length-sorted batches keep padding small; the result does not depend on the batching.
    order = sorted(
        range(len(dataset)), key=lambda i: len(dataset.sources[i]) + len(dataset.targets[i])
    )
    nll_sum = torch.zeros(len(dataset), dtype=torch.float64)
    tok_count = torch.zeros(len(dataset), dtype=torch.int64)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        batch = collator([dataset[i] for i in chunk])
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        targets = labels[:, 1:]
        token_nll = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(targets)
        idx = torch.tensor(chunk)
        nll_sum[idx] = token_nll.sum(dim=1).double().cpu()
        tok_count[idx] = (targets != IGNORE_INDEX).sum(dim=1).cpu()

    total_tokens = int(tok_count.sum())
    counted = tok_count > 0
    per_source = {}
    for source_id, name in enumerate(dataset.all_data_sources):
        mask = torch.tensor([s == source_id for s in dataset.data_sources])
        tokens = int(tok_count[mask].sum())
        if tokens:
            per_source[name] = {
                "loss": float(nll_sum[mask].sum() / tokens),
                "n_examples": int(mask.sum()),
                "n_tokens": tokens,
            }
    loss = float(nll_sum.sum() / total_tokens)
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "example_mean_loss": float((nll_sum[counted] / tok_count[counted]).mean()),
        "n_examples": len(dataset),
        "n_tokens": total_tokens,
        "per_source": per_source,
    }


def evaluate_sets(model, sets, tokenizer, batch_size, device, autocast=contextlib.nullcontext):
    """`evaluate_loss` over several named sets, with the model in eval mode; restores the mode."""
    was_training = model.training
    model.eval()
    try:
        results = {}
        for name, dataset in sets.items():
            start = time.perf_counter()
            results[name] = evaluate_loss(model, dataset, tokenizer, batch_size, device, autocast)
            results[name]["seconds"] = round(time.perf_counter() - start, 2)
        return results
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------
class EvalLossCallback(TrainerCallback):
    """Evaluation loss on fixed sets after chosen optimizer steps (0 = before training).

    Results go to the trainer log (``eval_<set>_loss``, so they land in ``trainer_state.json``)
    and to ``<output_dir>/eval_loss.jsonl``. The wall-clock of each evaluation is recorded
    (``seconds``); the step that follows an evaluation is longer by that amount.
    """

    def __init__(self, trainer, sets, steps, batch_size, out_file):
        self.trainer = trainer
        self.sets = sets
        self.steps = set(steps)
        self.batch_size = batch_size
        self.out_file = out_file

    def on_train_begin(self, args, state, control, **kwargs):
        if 0 in self.steps:
            self._evaluate(0)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            self._evaluate(state.global_step)

    def _evaluate(self, step: int) -> None:
        trainer = self.trainer
        if not trainer.is_world_process_zero():
            return
        # The evaluation must not hide the training peak: read it, evaluate, and reset so that the
        # logged `peak_mem_gb` keeps describing training steps only.
        memory = {}
        if torch.cuda.is_available():
            memory["train_peak_before_eval_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
        results = evaluate_sets(
            trainer.model,
            self.sets,
            trainer.processing_class,
            self.batch_size,
            trainer.args.device,
            trainer.compute_loss_context_manager,
        )
        if torch.cuda.is_available():
            memory["eval_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
        logs = {}
        for name, result in results.items():
            logs[f"eval_{name}_loss"] = round(result["loss"], 6)
            logger.info(
                f"[eval loss] step {step} {name}: loss {result['loss']:.4f} "
                f"({result['n_examples']} examples, {result['n_tokens']} tokens, "
                f"{result['seconds']}s)"
            )
        with open(self.out_file, "a") as f:
            for name, result in results.items():
                f.write(json.dumps({"step": step, "set": name, **result, **memory}) + "\n")
        trainer.log(logs)


def add_eval_loss_callback(trainer, eval_args: HeldoutEvalArguments, heldout, output_dir):
    """Register `EvalLossCallback` on `trainer` if `eval_loss_steps` is set."""
    if not eval_args.eval_loss_steps:
        return None
    sets = build_eval_sets(eval_args, trainer.processing_class, heldout)
    callback = EvalLossCallback(
        trainer,
        sets,
        eval_args.eval_loss_steps,
        eval_args.eval_loss_batch_size,
        os.path.join(output_dir, EVAL_LOSS_FILENAME),
    )
    trainer.add_callback(callback)
    logger.info(
        f"Eval loss on {list(sets)} ({[len(s) for s in sets.values()]} examples) after steps "
        f"{sorted(eval_args.eval_loss_steps)} -> {callback.out_file}"
    )
    return callback


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train_config", required=True, help="Training config json of the run.")
    parser.add_argument("--adapter", nargs="*", default=[], help="LoRA checkpoint directories.")
    parser.add_argument("--base", action="store_true", help="Also evaluate the base model.")
    parser.add_argument("--output", required=True, help="Result json.")
    parser.add_argument("--limit", type=int, default=None, help="Examples per set (smoke).")
    parser.add_argument(
        "--autocast_dtype",
        default="float16",
        choices=["float16", "bfloat16", "none"],
        help="Autocast of the fp32 weights, as in training (phi-2: float16).",
    )
    args = parser.parse_args(argv)
    if not args.adapter and not args.base:
        parser.error("give at least one --adapter or --base")
    return args


def main(argv=None):
    import transformers
    from peft import PeftModel

    from colm.train.data_arguments import DataArguments
    from colm.train.model_arguments import ModelArguments, add_padding_to_tokenizer

    args = _parse_args(argv)
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, HeldoutEvalArguments))
    with open(args.train_config) as f:
        config = json.load(f)
    model_args, data_args, eval_args = hf_parser.parse_dict(config, allow_extra_keys=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    add_padding_to_tokenizer(tokenizer)
    heldout = None
    if eval_args.holdout_size:
        full = get_training_dataset(
            data_args.train_files,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            sample_percentage=data_args.percentage,
            subset_index_files=data_args.subset_index_files,
            seed=data_args.sample_data_seed,
            hf_datasets_cache_dir=data_args.hf_datasets_cache_dir,
        )
        _, heldout = split_holdout(full, eval_args.holdout_size, eval_args.holdout_seed)
    sets = build_eval_sets(eval_args, tokenizer, heldout, limit=args.limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.float32,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
    )
    assert len(tokenizer) <= model.get_input_embeddings().weight.shape[0], "embedding resize"
    model.to(device)
    autocast_dtype = None if args.autocast_dtype == "none" else getattr(torch, args.autocast_dtype)

    def autocast():
        if autocast_dtype is None or device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=autocast_dtype)

    def run(label):
        results = evaluate_sets(
            model, sets, tokenizer, eval_args.eval_loss_batch_size, device, autocast
        )
        for name, result in results.items():
            print(
                f"[eval loss] {label} {name}: loss {result['loss']:.4f} "
                f"({result['n_examples']} examples, {result['n_tokens']} tokens, "
                f"{result['seconds']}s)",
                flush=True,
            )
        return {"label": label, "results": results}

    records = []
    if args.base:
        records.append(run("base"))
    peft_model = None
    for path in args.adapter:
        name = os.path.basename(os.path.normpath(path))
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            peft_model.load_adapter(path, adapter_name=name)
        peft_model.set_adapter(name)
        record = run(name)
        record["adapter"] = os.path.abspath(path)
        records.append(record)

    payload = {"train_config": os.path.abspath(args.train_config), "records": records}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
