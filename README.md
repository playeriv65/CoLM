# CoLM
![Python 3.12](https://img.shields.io/badge/python-3.12-green)
![Pytorch 2.13](https://img.shields.io/badge/pytorch-2.13-green)
![Transformers 5](https://img.shields.io/badge/transformers-5.x-green)
![License MIT](https://img.shields.io/badge/license-MIT-blue)

This repository is the official implementation of our ICLR 2025 paper [Mini-batch Coresets for Memory-efficient Language Model Training on Data Mixtures](https://arxiv.org/pdf/2407.19580).

## 🔗 Quick Links
- [CoLM](#colm)
  - [🔗 Quick Links](#-quick-links)
  - [Install Requirements](#install-requirements)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Tests](#tests)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


## Install Requirements
The environment is managed by [uv](https://docs.astral.sh/uv/) (Python 3.12, PyTorch 2.13 + CUDA 13.0,
transformers 5.x, PEFT, accelerate; versions pinned in `uv.lock`).
```bash
uv sync                  # training: creates .venv with colm installed in editable mode
uv sync --extra eval     # + stock vLLM for math evaluation (LoRA through LoRARequest)
uv sync --extra wandb    # + Weights & Biases (opt-in, see below)
uv sync --all-extras     # everything
```
`submodlib` (facility location) is built from its git repository; `flash-attn`, `traker`, the vLLM
fork and `bitsandbytes` are no longer needed (attention uses PyTorch SDPA).

W&B is off by default (`report_to="none"`, nothing imports `wandb`). To log a run, install the extra
and pass `--report_to wandb` (optionally `--wandb_project/--wandb_entity/--wandb_notes`, or the
`WANDB_*` environment variables).

## Data Preparation
Download MathInstruct with the additional annotations
[here](https://drive.google.com/file/d/1kpYMJ0xrn0eLyv-uwhUZCTjFWT6Zlb-Q/view?usp=sharing)
(e.g. `uvx gdown 1kpYMJ0xrn0eLyv-uwhUZCTjFWT6Zlb-Q`) into the shared data directory and link it:
```bash
bash scripts/link-external.sh    # data -> $COLM_ARTIFACT_ROOT/datasets/colm, out -> .../artifacts/CoLM/out
```
The linked paths are declared in `external-paths.json` (`COLM_ARTIFACT_ROOT` defaults to
`/mnt/data2/zelin4593`). Configs read `data/MathInstruct.jsonl`.

## Training
```bash
scripts/run_math_efficient.sh <gpu_ids>     # e.g. scripts/run_math_efficient.sh 2,3
scripts/run.sh <config.json> <gpu_ids> [--extra_flag value ...]
```
GPU ids can also come from `COLM_GPUS`; there is no default. `nproc_per_node` is derived from the list.
Logs go to `logs/<config>-gpu<ids>-np<n>-<timestamp>.log`; every loss log also records
`step_time_s`, `select_time_s` and `peak_mem_gb`. Runs without `output_dir` are written to
`out/<model>-<data>-lora-gas..-bs..-<method>-<unit>-...-<steps>steps-seed<seed>`.
All hyperparameters live in the dataclasses of `colm/train/*_arguments.py`
(paper defaults) and the JSON files in `configs/`.

### Step timing
`--profile_timing coarse|fine` (default `off`: no synchronize, no overhead) writes a per-phase
wall-clock breakdown of every optimizer step to
`<profile_timing_dir or output_dir>/step_timing-<run>-<level>-rank<r>-<timestamp>.jsonl`.
Each section boundary calls `torch.cuda.synchronize()`, so GPU work is charged to the phase that
launched it; `fine` adds per-layer / per-op sections (more syncs, slightly inflated totals).
`--profile_census_steps N` counts aten ops, host<->device copies (bytes) and synchronizing calls per
phase during the first N steps (slow; keep N within the summary warmup). Summarise with
```bash
python -m colm.train.step_timing logs/step_timing-....jsonl --warmup 10   # table + .summary.json
```
Every parent node is reported with an explicit `other` residual; the root is the measured time
between consecutive optimizer steps. Configs: `configs/timing_phi2_efficient.json` (fine, 130 steps,
census on the 10 warmup steps) and `configs/timing_phi2_efficient_coarse.json` (coarse, 60 steps).
Measured breakdown of the default config: `docs/optimization-backlog.md` ("Measured baseline").

Note: We implement CoLM with an efficient last-layer zeroth-order gradient estimation that requires approximately only one forward pass of the model. While the selection time is negligible (<0.1s), CoLM still introduces additional overhead, such as synchronizing gradients before selection, broadcasting selected indices back, padding after selection (which can make some samples longer), transferring tensors between CPU and GPU, context switching, and so on. In the paper, we report the ideal training time of our method which is the forward pass time for a batch size of 128 + the forward and backward pass time for a batch size of 64.

Note: the logged training `loss` of the CoLM trainers is the mean loss divided by `small_batch_ratio`
(kept from the original implementation so curves remain comparable).

## Evaluation
Accuracy (vLLM, PoT with CoT backup) of base models and LoRA checkpoints; one process and one vLLM
engine serve every checkpoint and dataset given:
```bash
CUDA_VISIBLE_DEVICES=<gpu> python -u math_eval/run_open.py --model out/run/checkpoint-512 out/run/checkpoint-1024 \
    --dataset gsm8k math numglue svamp deepmind simuleq --shots 0 --stem_flan_type pot_prompt \
    --model_max_length 2048 --cot_backup --use_vllm --dtype float16 --enable_lora
# --dry_run builds the prompts of every dataset and loads no model; --limit N evaluates N examples
```
Results: `<checkpoint>/outputs/<name>.jsonl` (+ `.metrics.json` with accuracy and counts); finished
outputs are skipped on a rerun, partial ones (`.partial`) are recomputed. The legacy
`math_eval/eval_finetuned.sh` / `eval_pretrained.sh` still work.

**Loss** (`colm/eval/eval_loss.py`): mean token NLL of teacher-forced reference solutions, pooled
over the whole set, on (a) `heldout`: `holdout_size` MathInstruct examples removed from training
(`holdout_seed`, drawn uniformly; indices saved as `holdout_indices.json`) and (b) `gsm8k`: the
GSM8K test solutions in the MathInstruct CoT style. Off by default (`holdout_size=0`,
`eval_loss_steps=[]` = the paper recipe); `holdout_size > 0` is a deviation from the paper (fewer
training examples) and must be the same across compared runs. With `eval_loss_steps` the trainer
evaluates after those steps (`<output_dir>/eval_loss.jsonl`, `eval_<set>_loss` in the trainer log;
the step after an evaluation is longer, the training peak memory is recorded separately). A
saved adapter is evaluated with
`python -m colm.eval.eval_loss --train_config <json> --adapter <ckpt>... [--base] --output <json>`.

### LoRA rank sweep (queue)
`configs/rank_sweep/sweep.json` expands into a file queue (`colm/jobs`): one JSON job per file,
one worker drains it serially on one GPU (atomic claim by rename, `done/` / `failed/` with exit
code, wall clock and log path; a lock file, no polling, no process-name matching).
```bash
python -m colm.jobs.rank_sweep --sweep configs/rank_sweep/sweep.json --queue queues/rank-sweep
python -m colm.jobs.worker --queue queues/rank-sweep --gpu 0 --dry-run    # print resolved jobs
python -u -m colm.jobs.worker --queue queues/rank-sweep --gpu 0          # run (GPU id is required)
python -m colm.jobs.summarize --sweep configs/rank_sweep/sweep.json       # out/rank-sweep/summary.md
```
Design, arms and timing: `TODO.md` ("LoRA rank sweep").

## Tests
```bash
CUDA_VISIBLE_DEVICES="" uv run pytest -q     # CPU: tiny random Phi, selection, trainers, 2-rank gloo, eval loss, job queue
uv run ruff format . && uv run ruff check .
```

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Dang Nguyen (nguyentuanhaidang@gmail.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{nguyen2025mini,
  title = {Mini-batch Coresets for Memory-efficient Language Model Training on Data Mixtures},
  author = {Nguyen, Dang and Yang, Wenhan and Anand, Rathul and Yang, Yu and Mirzasoleiman, Baharan},
  journal = {International Conference on Learning Representations (ICLR)},
  year = {2025}
}
```

## Acknowledgements
The structure of this repository is largely based on the official implementation of [LESS](https://github.com/princeton-nlp/LESS) and [MeZO](https://github.com/princeton-nlp/MeZO). We are grateful for their open sources.
