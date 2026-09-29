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
  - [Quickstart](#quickstart)
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
`submodlib` (facility location) is built from its git repository; `traker`, the vLLM fork and
`bitsandbytes` are no longer needed, and `flash-attn` is not compiled: attention is stock
transformers (`flash_attention_2` for the fp16 training forward, loaded as the hub kernel
`kernels-community/flash-attn2` through the `kernels` package unless `flash-attn` is installed;
`sdpa` for the fp32 selection forward). On CPU, on GPUs older than Ampere or without mixed
precision pass `--attn_implementation sdpa`.

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

## Quickstart
Three commands (installed by `uv sync`), each with `--help`:

| command | what |
|---|---|
| `colm-train` | train (CoLM or the full-batch baseline) on 1, 2 or N GPUs |
| `colm-eval loss \| accuracy \| superglue` | evaluation loss, answer accuracy (vLLM), SuperGLUE |
| `colm-sweep create \| work \| summary` | job queue of the LoRA rank sweep |

```bash
colm-train --gpus 2 --model_name_or_path microsoft/phi-2            # 1 GPU: the paper recipe
colm-train --gpus 2,3 --model_name_or_path microsoft/phi-2          # 2 GPUs: same config, same command
colm-train configs/math_phi2.json --gpus 0,1,2,3 --max_steps 100    # a config file + flags
```
GPU ids are always explicit (`--gpus`, or `COLM_GPUS`): the GPUs are shared and reserved. One process
runs per GPU under `torchrun`; the logs go to `logs/<config>-gpu<ids>-np<n>-<time>.log`. With more
GPUs the pool of every step grows with the number of ranks (the per-rank pool is
`per_device_train_batch_size x gradient_accumulation_steps` examples, 32 by default), and the
selection is made over all ranks together.

**Configuration.** Every option has the value of the paper recipe as its default, so a run needs the
model and little else; `colm-train --help` lists every option with its default and meaning. A config
file (JSON, keys = option names) holds only what differs from the defaults, flags override it, and
unknown keys or wrong values fail at load. `configs/math_phi2_efficient.json` is the plain
recipe, `configs/math_phi2.json` the one-example-per-micro-batch variant. Values derived from the
model (the last layer, LoRA targets, precision, context length) are computed; the resolved
configuration (defaults included) is printed at the start and saved as
`<output_dir>/resolved_config.json`. Runs without `output_dir` go to
`out/<model>-<data>-lora-gas..-bs..-<method>-<unit>-...-<steps>steps-seed<seed>`.

```
before  configs/math_phi2_efficient.json    {"model_name_or_path": "microsoft/phi-2", "train_files": ["data/MathInstruct.jsonl"],
                                             "max_steps": 1024, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 8,
                                             "efficient_mezo": true}      scripts/run_math_efficient.sh 2,3
after   configs/math_phi2_efficient.json    {"model_name_or_path": "microsoft/phi-2"}      colm-train configs/math_phi2_efficient.json --gpus 2,3
```

**What a step does** (`colm/train/trainers.py`, `colm/selection/`): the pool of all ranks (one HF
batch per rank, packed without padding) is gathered on every rank and `CoresetSelector.needed`
decides from the source ids which features can influence the selection: examples of `keep_sources`
and of sources with a zero quota are not forwarded at all (about 25% of the pool with the paper
recipe). The rest is shared over the ranks by token count and goes through the extractor of
`data_selection_unit` (default: the batched last-layer MeZO estimate); for it a feature is `g_i z`
with one fixed direction z, so each rank sends one scalar per example and rank 0 builds the
features. Facility location picks `small_batch_ratio` of the pool source by source, the picks are
broadcast and every rank trains on its share, in packed forwards chosen by `train_max_tokens`:
N > 0 (memory mode, default 1536) packs the examples greedily into forwards of at most N tokens and
accumulates the gradients (phi-2: 1382 ms per step, 32.3 GB peak); `0` (speed mode) puts the whole
step into one forward (1383 ms, i.e. no faster on phi-2, but 57 GB peak / 94 GB reserved; table in
`docs/optimization-backlog.md`). The loss of a step is
the mean over all label tokens of the examples trained in the step (all ranks) whatever the
grouping. Peak GPU memory is measured on every rank and per phase (`memory.json`, `peak_mem_*` in
the log).

**Known errors of the upstream code** are fixed (`docs/errors.md`, with the evidence). The
alignment with the upstream behaviour was proven with a temporary `legacy` switch (float64 CPU
goldens and a phi-2 GPU run); it has been removed again: the tag `pre-refactor` is the upstream
code, `legacy-bridge` the last commit that still has the switch and its tests.

### Step timing
`--profile_timing coarse|fine` (default `off`: no synchronize, no overhead) writes a per-phase
wall-clock breakdown of every optimizer step to
`<profile_timing_dir or output_dir>/step_timing-<run>-<level>-rank<r>-<timestamp>.jsonl`.
Each section boundary calls `torch.cuda.synchronize()`, so GPU work is charged to the phase that
launched it. `--profile_census_steps N` counts aten ops, host<->device copies (bytes) and
synchronizing calls per phase during the first N steps (slow; keep N within the summary warmup).
Summarise with
```bash
python -m colm.train.step_timing logs/step_timing-....jsonl --warmup 10   # table + .summary.json
```
Every parent node is reported with an explicit `other` residual; the root is the measured time
between consecutive optimizer steps. Configs: `configs/timing_phi2_efficient.json` (fine, 130 steps)
and `configs/timing_phi2_efficient_coarse.json` (coarse, 60 steps).

Note: CoLM introduces overhead besides the selection forward: gathering features and examples,
broadcasting the selected indices, host/device transfers. In the paper we report the ideal
training time: the forward pass of the pool + the forward and backward pass of the trained
fraction.

## Evaluation
Accuracy (vLLM, PoT with CoT backup) of base models and LoRA checkpoints; one process and one vLLM
engine serve every checkpoint and dataset given. The defaults are the paper protocol (0-shot,
gsm8k math numglue svamp deepmind simuleq, the dtype of the model's recipe, LoRA for adapters):
```bash
colm-eval accuracy --model out/run/checkpoint-512 out/run/checkpoint-1024    # on CUDA_VISIBLE_DEVICES
colm-eval accuracy --model microsoft/phi-2 --dataset gsm8k --limit 20 --dry_run   # prompts only
```
Results: `<checkpoint>/outputs/<name>.jsonl` (+ `.metrics.json` with accuracy and counts); finished
outputs are skipped on a rerun, partial ones (`.partial`) are recomputed.

**Loss** (`colm-eval loss`, `colm/eval/eval_loss.py`): mean token NLL of teacher-forced reference
solutions, pooled over the whole set, on (a) `heldout`: `holdout_size` MathInstruct examples
removed from training (`holdout_seed`; whole groups of the same question, so no held-out question
is trained on in another solution format; indices saved as `holdout_indices.json`) and (b) `gsm8k`:
the GSM8K test solutions in the MathInstruct CoT style. Off by default (`holdout_size=0`,
`eval_loss_steps=[]` = the paper recipe); `holdout_size > 0` is a deviation from the paper (fewer
training examples) and must be the same across compared runs. With `eval_loss_steps` the trainer
evaluates after those steps on rank 0 (`<output_dir>/eval_loss.jsonl`, `eval_<set>_loss` in the
trainer log; the step after an evaluation is longer). A saved adapter is evaluated with
`colm-eval loss --train_config <json> --adapter <ckpt>... [--base] --output <json>`.

### LoRA rank sweep (queue)
`configs/rank_sweep/sweep.json` expands into a file queue (`colm/jobs`): one JSON job per file,
one worker drains it serially on one GPU (atomic claim by rename, `done/` / `failed/` with exit
code, wall clock and log path; a lock file, no polling, no process-name matching).
```bash
colm-sweep create --sweep configs/rank_sweep/sweep.json --queue queues/rank-sweep-v2
colm-sweep work --queue queues/rank-sweep-v2 --gpu 0 --dry-run    # print resolved jobs
colm-sweep work --queue queues/rank-sweep-v2 --gpu 0              # run (the GPU id is required)
colm-sweep summary --sweep configs/rank_sweep/sweep.json       # out/rank-sweep-v2/summary.md
```
Design, arms and timing: `TODO.md` ("LoRA rank sweep").

## Tests
```bash
CUDA_VISIBLE_DEVICES="" uv run pytest -q     # CPU: tiny random Phi, selection, trainers, 2- and 4-rank gloo, eval loss, job queue
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
