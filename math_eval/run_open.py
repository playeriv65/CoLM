"""Math evaluation of base models and LoRA checkpoints (PoT with CoT backup), vLLM or HF.

One process evaluates every (model, dataset) pair given, and with vLLM builds the engine once:

    python -u run_open.py --model out/run/checkpoint-512 out/run/checkpoint-1024 \
        --dataset gsm8k math numglue svamp deepmind simuleq --use_vllm --enable_lora ...

All models must be LoRA checkpoints of the same base model (or a single full model). Results go
to ``<model>/outputs/<name>.jsonl`` (one line per example) plus ``.metrics.json`` (accuracy,
counts) and the legacy ``.csv``; a finished output is never recomputed, a partial one is.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
import utils
from data_loader import BatchDatasetLoader
from peft import LoraConfig, PeftModel
from prompt_utils import get_examples, get_prompt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
DATASETS = ["gsm8k", "svamp", "math", "numglue", "deepmind", "simuleq"]
STOP_TOKENS = [  # the model starting a new turn / prompt
    "Question:",
    "USER:",
    "ASSISTANT:",
    "Instruction:",
    "Response:",
    "### Instruction",
]
DEFAULT_MAX_NEW_TOKENS = 1024
REPO_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        nargs="+",
        default=[],
        help="Model paths / hub ids (LoRA checkpoints share a base).",
    )
    parser.add_argument("--output", default="", type=str, help="Single model and dataset only.")
    parser.add_argument("--stem_flan_type", default="", choices=["", "pot_prompt"], type=str)
    parser.add_argument("--dtype", default="bfloat16", type=str)
    parser.add_argument("--dataset", nargs="+", required=True, choices=DATASETS)
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--form", default="alpaca", type=str)
    parser.add_argument("--shots", default=0, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--print", action="store_true", default=False)
    parser.add_argument(
        "--max_new_tokens",
        default=DEFAULT_MAX_NEW_TOKENS,
        type=int,
        help="Tokens generated per answer (vLLM `max_tokens` / HF `max_new_tokens`). Prompts are "
        "never cut.",
    )
    parser.add_argument("--cot_backup", action="store_true", default=False)
    parser.add_argument("--enable_lora", action="store_true", default=False)
    parser.add_argument(
        "--cache_dir", default=None, type=str, help="HF cache override (default: $HF_HOME)"
    )
    parser.add_argument("--gpu_memory_utilization", default=0.9, type=float)
    parser.add_argument(
        "--exec_workers",
        default=16,
        type=int,
        help="Threads that run the generated programs concurrently (each in its own process).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Directory for the outputs of a single hub-id model (default: <repo>/out/<name>).",
    )
    parser.add_argument(
        "--max_lora_rank",
        default=None,
        type=int,
        help="vLLM LoRA rank capacity (default: largest rank among the adapters).",
    )
    parser.add_argument("--limit", default=None, type=int, help="First N examples per dataset.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Check the arguments and build the prompts of every dataset; load no model.",
    )
    return parser


def is_adapter(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def adapter_ranks(models: list[str]) -> list[int]:
    return [LoraConfig.from_pretrained(m).r for m in models if is_adapter(m)]


def validate_models(args):
    """Fail fast on argument combinations that would silently evaluate the wrong thing."""
    if not args.model:
        raise SystemExit("--model is required")
    if args.output_dir and len(args.model) > 1:
        raise SystemExit("--output_dir needs exactly one --model")
    adapters = [is_adapter(m) for m in args.model]
    if any(adapters) and not all(adapters):
        raise SystemExit("--model mixes LoRA checkpoints and full models")
    if any(adapters):
        if not args.enable_lora:
            raise SystemExit(
                "LoRA checkpoints given without --enable_lora (would evaluate the base)"
            )
        bases = {LoraConfig.from_pretrained(m).base_model_name_or_path for m in args.model}
        if len(bases) != 1:
            raise SystemExit(f"LoRA checkpoints have different base models: {sorted(bases)}")
    elif len(args.model) > 1:
        raise SystemExit("several --model values are only supported for LoRA checkpoints")
    if args.output and (len(args.model) > 1 or len(args.dataset) > 1):
        raise SystemExit("--output needs exactly one --model and one --dataset")


def build_prompts(examples, questions, form):
    prompt_no_input, prefix = get_prompt(examples, form)
    return [prompt_no_input + prefix.format(query=q) for q in questions]


# ---------------------------------------------------------------------------
# Generators: text completions for prompts, under one model path
# ---------------------------------------------------------------------------
class VllmGenerator:
    """One stock-vLLM engine; LoRA checkpoints are served as adapters of the shared base."""

    def __init__(self, args):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.sampling_params = SamplingParams(
            temperature=0, top_p=1, max_tokens=args.max_new_tokens, stop=STOP_TOKENS
        )
        adapters = [m for m in args.model if is_adapter(m)]
        self.lora_requests = {
            path: LoRARequest(f"adapter-{i}", i + 1, path) for i, path in enumerate(adapters)
        }
        base_model = (
            LoraConfig.from_pretrained(adapters[0]).base_model_name_or_path
            if adapters
            else args.model[0]
        )
        self.llm = LLM(
            model=base_model,
            # The tokenizer (with the added pad token) is read from the checkpoint directory.
            tokenizer=args.model[0],
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=args.enable_lora,
            max_lora_rank=args.max_lora_rank or max(adapter_ranks(args.model), default=16),
            download_dir=args.cache_dir,
        )
        print("Using VLLM, we do not need to set batch size!", flush=True)

    def generate(self, model_path, questions, examples, form):
        prompts = build_prompts(examples, questions, form)
        outputs = self.llm.generate(
            prompts, self.sampling_params, lora_request=self.lora_requests.get(model_path)
        )
        return [output.outputs[0].text for output in outputs]


class HfGenerator:
    """Plain transformers generation (one model loaded at a time)."""

    def __init__(self, args):
        self.args = args
        self.loaded = None

    def _load(self, path):
        if self.loaded and self.loaded[0] == path:
            return self.loaded[1:]
        args = self.args
        tokenizer = AutoTokenizer.from_pretrained(
            path, padding_side="left", cache_dir=args.cache_dir
        )
        if is_adapter(path):
            config = LoraConfig.from_pretrained(path)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                dtype=DTYPES[args.dtype],
                device_map="auto",
                cache_dir=args.cache_dir,
            )
            model = PeftModel.from_pretrained(base_model, path, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path, device_map="auto", dtype=DTYPES[args.dtype], cache_dir=args.cache_dir
            )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        self.loaded = (path, model, tokenizer)
        return model, tokenizer

    def generate(self, model_path, questions, examples, form):
        model, tokenizer = self._load(model_path)
        return utils.get_answer(
            examples=examples,
            questions=questions,
            model=model,
            tokenizer=tokenizer,
            form=form,
            max_new_tokens=self.args.max_new_tokens,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def extract_answer(dataset: str, output: str) -> tuple[str, str]:
    """(kept output, answer) of one generation: run its program if it has one (5 s timeout)."""
    if "print(" in output:
        output = output.split("### Instruction")[0]
        tmp = "The answer is" + " " + utils.execute_with_timeout(output)
        return output, utils.answer_clean(dataset, ("####", "The answer is"), tmp)
    return output, utils.answer_clean(dataset, ("####", "The answer is"), output)


def run_question_answer(
    args, generator, model_path, questions, groundtruths, collect_rerun: bool = False
):
    used_examples = get_examples(args.dataset_name, args.shots, args.stem_flan_type)
    outputs = generator.generate(model_path, questions, used_examples, args.form)

    # We need to collect the values and possibly the rerun questions;
    returned_value = []
    rerun_questions = []
    rerun_groundtruths = []
    with ThreadPoolExecutor(max_workers=max(1, args.exec_workers)) as pool:
        extracted = list(pool.map(lambda o: extract_answer(args.dataset_name, o), outputs))
    for (output, answer), question, groundtruth in zip(
        extracted, questions, groundtruths, strict=True
    ):
        if answer == "" and collect_rerun:
            rerun_questions.append(utils.remove_flan_tag(question, args.stem_flan_type))
            rerun_groundtruths.append(groundtruth)
            continue

        returned_value.append((question, output, answer, groundtruth))

    if collect_rerun:
        assert len(returned_value) + len(rerun_questions) == len(questions) == len(groundtruths)
        return returned_value, rerun_questions, rerun_groundtruths
    return returned_value


def is_correct(dataset: str, answer, groundtruth) -> bool:
    if dataset == "math":
        assert len(groundtruth) == 2, groundtruth
        groundtruth_str, groundtruth_num = groundtruth
        return utils.compare_both_string_and_number_format(answer, groundtruth_str, groundtruth_num)
    return answer == groundtruth


def output_path(args, model_path: str, dataset: str) -> str:
    if args.output:
        return args.output
    suffix = "PoT" if "pot" in args.stem_flan_type.lower() else "CoT"
    filename = f"{dataset}_{args.shots}shots_{args.form}_new{args.max_new_tokens}"
    if args.cot_backup:
        filename += "_CoTBackup"
    filename += f"_bs{args.batch_size}_{suffix}_import"
    if args.output_dir:
        out_dir = args.output_dir
    elif os.path.exists(model_path):
        out_dir = os.path.join(model_path, "outputs")
    else:
        out_dir = os.path.join(REPO_OUT, model_path.split("/")[-1], "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{filename}.jsonl")


def append_accuracy_csv(args, path: str, dataset: str, accuracy: float):
    filename = path.replace(".jsonl", ".csv").replace(f"{dataset}_{args.shots}shots_", "")
    row = pd.DataFrame({"dataset": [dataset], "accuracy": [accuracy], "shots": [args.shots]})
    if os.path.exists(filename):
        row = pd.concat([pd.read_csv(filename), row], ignore_index=True)
    row.to_csv(filename, index=False)


def evaluate_dataset(args, generator, model_path: str, dataset: str) -> dict | None:
    """Evaluate one dataset; returns the metrics, or None when a finished output exists."""
    args.dataset_name = dataset
    path = output_path(args, model_path, dataset)
    if os.path.exists(path):
        print(f"Output file {path} exists, skipping", flush=True)
        return None
    partial = path + ".partial"
    correct = wrong = reruns = 0
    start = time.perf_counter()
    with open(partial, "w") as file_handle:
        for questions, groundtruths in tqdm(
            BatchDatasetLoader(dataset, args.batch_size, limit=args.limit)
        ):
            # First pass to use PoT
            processed_questions = utils.process_question_with_flan_tag(
                questions, args.stem_flan_type
            )
            if args.stem_flan_type == "pot_prompt" and args.cot_backup:
                # hybrid decoding: try PoT first and fall back to CoT when no answer came out
                returned_values, rerun_questions, rerun_groundtruths = run_question_answer(
                    args, generator, model_path, processed_questions, groundtruths, True
                )
                reruns += len(rerun_questions)
                if rerun_questions:
                    processed_questions = utils.process_question_with_flan_tag(rerun_questions, "")
                    returned_values += run_question_answer(
                        args, generator, model_path, processed_questions, rerun_groundtruths
                    )
            else:
                returned_values = run_question_answer(
                    args, generator, model_path, processed_questions, groundtruths
                )

            for question, output, answer, groundtruth in returned_values:
                if is_correct(dataset, answer, groundtruth):
                    correct += 1
                else:
                    wrong += 1
                if args.print:
                    print(answer, "#", groundtruth, "#", correct / (correct + wrong))
                example = {
                    "question": question,
                    "correct": groundtruth,
                    "solution": output,
                    "pred": answer,
                    "task": dataset,
                }
                file_handle.write(json.dumps(example) + "\n")
            print("finished one epoch", flush=True)
    os.replace(partial, path)

    accuracy = correct / (correct + wrong)
    metrics = {
        "model": model_path,
        "dataset": dataset,
        "accuracy": accuracy,
        "correct": correct,
        "total": correct + wrong,
        "cot_backup_reruns": reruns,
        "limit": args.limit,
        "seconds": round(time.perf_counter() - start, 1),
    }
    with open(path.replace(".jsonl", ".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    append_accuracy_csv(args, path, dataset, accuracy)
    print(f"final accuracy: {accuracy}", flush=True)
    print(
        f"[eval acc] {model_path} {dataset}: {accuracy:.4f} ({correct}/{correct + wrong}, "
        f"{metrics['seconds']}s)",
        flush=True,
    )
    return metrics


def dry_run(args):
    """Argument check plus the prompts of every dataset; no model is loaded."""
    print(f"models: {args.model}")
    if any(is_adapter(m) for m in args.model):
        ranks = adapter_ranks(args.model)
        print(
            f"LoRA ranks {ranks}; base {LoraConfig.from_pretrained(args.model[0]).base_model_name_or_path}"
        )
    used_examples = get_examples(args.dataset[0], args.shots, args.stem_flan_type)
    for dataset in args.dataset:
        questions, _ = next(iter(BatchDatasetLoader(dataset, -1, limit=args.limit)))
        questions = utils.process_question_with_flan_tag(questions, args.stem_flan_type)
        prompts = build_prompts(used_examples, questions, args.form)
        mean_chars = sum(map(len, prompts)) / len(prompts)
        print(f"[dry run] {dataset}: {len(prompts)} prompts, mean {mean_chars:.0f} chars")
        print(prompts[0][-300:].replace("\n", "\\n"))


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_models(args)
    if args.dry_run:
        dry_run(args)
        return
    generator = VllmGenerator(args) if args.use_vllm else HfGenerator(args)
    if args.use_vllm:
        args.batch_size = -1
    all_metrics = []
    for model_path in args.model:
        print(
            f"Using finetuned model at {model_path}" if os.path.exists(model_path) else model_path
        )
        for dataset in args.dataset:
            metrics = evaluate_dataset(args, generator, model_path, dataset)
            if metrics:
                all_metrics.append(metrics)
    print(f"evaluated {len(all_metrics)} (model, dataset) pairs", flush=True)
    return all_metrics


if __name__ == "__main__":
    main()
