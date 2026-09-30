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
  - [Documentation](#documentation)
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
`sdpa` for the selection forward). On CPU, on GPUs older than Ampere or without mixed
precision pass `--attn_implementation sdpa`.

With the Phi-2 recipe, selection runs the unperturbed decoder prefix under
FP16 autocast except its last two blocks, which run in FP32, and keeps the
perturbed last layer, head and loss in FP32. The Phi-2 model profile sets
`selection_prefix_dtype=float16` and `selection_prefix_fp32_tail=2` and requires
FP32 model weights; pass `--selection_prefix_dtype float32` for the full FP32
prefix or `--selection_prefix_fp32_tail 0` for the plain FP16 prefix. The tail
cuts the FP16 error of the per-example MeZO scalars from a median 18 % to about
1 % (selection overlap with the exact selection within the FP32 noise floor) for
about +3.6 % step time. It also sets `pack_tokens=1536` for selection;
`--pack_tokens 0` restores the data-derived budget (about 1003 tokens on
MathInstruct). Learning quality has not been compared; see
[`docs/fp16-prefix.md`](docs/fp16-prefix.md) and
[`docs/selection-precision.md`](docs/selection-precision.md).
Other model profiles keep an FP32 prefix (tail 0) until tested; an
explicit command-line or JSON value overrides the profile. The resolved run
configuration records the selected mode, and an unsupported FP16-prefix/weight
combination or a tail without the FP16 prefix fails at startup.

The TRAINING forward has an FP32 tail of its own: `train_fp32_tail=k` runs the q/k
projections (with their LoRA) and the attention of the last k blocks in FP32 instead
of under FP16 autocast (`colm/train/precision.py`). Autocast rounds q and k inside the
projections, which corrupts the LoRA gradient of the last three Phi-2 blocks: its error
against the exact FP32 gradient is 1.0 (cosine 0.65) with the plain FP16 forward and
0.04 (cosine 0.999) with the tail. The Phi-2 profile sets `train_fp32_tail=3` (+12 % of a
pack's forward + backward, about +6 % step time, +0.7 GiB); other profiles 0;
`--train_fp32_tail 0` restores the plain FP16 training forward. It needs FP16/BF16 autocast
over FP32 model weights and fails at startup otherwise; see
[`docs/training-precision.md`](docs/training-precision.md).

W&B is off by default (`report_to="none"`, nothing imports `wandb`). To log a run, install the extra
and pass `--report_to wandb` (optionally `--wandb_project/--wandb_entity/--wandb_notes`, or the
`WANDB_*` environment variables).

## Data Preparation
Download MathInstruct with the additional annotations
[here](https://drive.google.com/file/d/1kpYMJ0xrn0eLyv-uwhUZCTjFWT6Zlb-Q/view?usp=sharing)
(e.g. `uvx gdown 1kpYMJ0xrn0eLyv-uwhUZCTjFWT6Zlb-Q`) into the shared data directory and link it:
```bash
bash scripts/link-external.sh    # data -> $COLM_ARTIFACT_ROOT/datasets/colm, out and cache -> .../artifacts/CoLM/
```
The linked paths are declared in `external-paths.json` (`COLM_ARTIFACT_ROOT` defaults to
`/mnt/data2/zelin4593`). Configs read `data/MathInstruct.jsonl`. `cache/tokens` holds the token-count
cache (below); it is rebuilt when missing.

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

**What a step does** (`colm/train/trainers.py`, `colm/selection/`): the pool of all ranks (one HF
batch per rank, packed without padding) is gathered on every rank and `CoresetSelector.needed`
decides from the source ids which features can influence the selection: examples of `keep_sources`
and of sources with a zero quota are not forwarded at all (about 25% of the pool with the paper
recipe). The rest is shared over the ranks by token count and goes through the extractor of
`data_selection_unit` (default: the batched last-layer MeZO estimate); for it a feature is `g_i z`
with one fixed direction z, so each rank sends one scalar per example and rank 0 builds the
features. Facility location picks `small_batch_ratio` of the pool source by source, the picks are
broadcast and every rank trains on a token-balanced share (the step lasts as long as the slowest rank), in packed forwards chosen by `train_max_tokens`:
N > 0 (memory mode, default 1536) packs the examples greedily into forwards of at most N tokens and
accumulates the gradients (phi-2, default recipe: 0.82 s per step and 23.7 GB peak training memory since
the frozen weights are stored in fp16, `docs/system-audit.md`; 0.89 s and 32.0 GB before, `docs/startup-overhead.md`); `0` (speed mode) puts the whole step into one forward (no faster on
phi-2, but 57 GB peak / 94 GB reserved; measured with the FP32 prefix, table in
`docs/optimization-backlog.md`). The loss of a step is
the mean over all label tokens of the examples trained in the step (all ranks) whatever the
grouping. Peak GPU memory is measured on every rank and per phase (`memory.json`, `peak_mem_*` in
the log).

**No truncation.** No option cuts a sequence (`max_seq_length`, `model_max_length` and the
SuperGLUE left-truncation are gone; the tokenisers are never called with `truncation=True`). The
only limit is the context window of the model (`max_position_embeddings`, read from its config).
An example above it is dropped and counted per source in the log (`Dropped N of M examples longer
than L tokens`), in every data path: MathInstruct-style `instruction`/`output`, the LESS
`prompt`/`completion` and `messages` formats, SuperGLUE training samples, the held-out and GSM8K
eval-loss sets. Evaluation never cuts a prompt either (SuperGLUE raises
`PromptTooLong` when a prompt does not fit; the accuracy evaluation has no cut at all). Memory is controlled only by how many whole examples
go into one packed forward (`train_max_tokens`).

**Known errors of the upstream code** are fixed (`docs/errors.md`, with the evidence). The
alignment with the upstream behaviour was proven with a temporary `legacy` switch (float64 CPU
goldens and a phi-2 GPU run); it has been removed again: the tag `pre-refactor` is the upstream
code, `legacy-bridge` the last commit that still has the switch and its tests.

### Start-up cost and the token cache
Deciding which of the 262k MathInstruct examples fit the context window needs their token counts;
tokenising them took ~30 s (minutes on a loaded host) in every training run and in every standalone
`colm-eval loss`. The counts are now cached in `--token_cache_dir` (default `cache/tokens`, empty
disables): the file name is a hash of the texts (data file, prompt template, EOS, sampling), the
context limit and the tokenizer, so any change of them reads another file, and the result is
identical to the uncached path (`tests/test_token_cache.py`). Every run writes `startup.json` (next to
`memory.json`) with the wall clock of each phase (imports, config, model load, data, first step,
steady steps, evaluation, checkpoints, final save) and prints the wall clock of the whole launch;
the measured table is in [`docs/startup-overhead.md`](docs/startup-overhead.md).

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
between consecutive optimizer steps. Configs: `configs/diagnostics/timing_phi2_efficient.json` (fine, 130 steps)
and `configs/diagnostics/timing_phi2_efficient_coarse.json` (coarse, 60 steps).

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
`--max_new_tokens` (default 1024) is the number of tokens generated per answer (vLLM `max_tokens`,
HF `max_new_tokens`); prompts are never cut. Results: `<checkpoint>/outputs/<name>.jsonl` (+ `.metrics.json` with accuracy and counts); finished
outputs are skipped on a rerun, partial ones (`.partial`) are recomputed.

**Loss** (`colm-eval loss`, `colm/eval/eval_loss.py`): mean token NLL of teacher-forced reference
solutions, pooled over the whole set, on (a) `heldout`: `holdout_size` MathInstruct examples
removed from training (`holdout_seed`; whole groups of the same question, so no held-out question
is trained on in another solution format; indices saved as `holdout_indices.json`) and (b) `gsm8k`:
the GSM8K test solutions in the MathInstruct CoT style. Off by default (`holdout_size=0`,
`eval_loss_steps=[]` = the paper recipe); `holdout_size > 0` is a deviation from the paper (fewer
training examples) and must be the same across compared runs. With `eval_loss_steps` the trainer
evaluates after those steps (with several ranks each takes every N-th batch and the sums are added
up: the same numbers, less time; rank 0 writes `<output_dir>/eval_loss.jsonl` and `eval_<set>_loss`
into the trainer log; the logged step time of an evaluation step includes the evaluation). A saved adapter is evaluated with
`colm-eval loss --train_config <json> --adapter <ckpt>... [--base] --output <json>`.

### LoRA rank sweep (queue)
`configs/rank_sweep/sweep.json` expands into a file queue (`colm/jobs`): one JSON job per file,
one worker drains it serially on one GPU (atomic claim by rename, `done/` / `failed/` with exit
code, wall clock and log path; a lock file, no polling, no process-name matching). The worker
stops at the first failed job and leaves dependent jobs pending; a queue with failures cannot be
resumed. Before claiming any job, it checks the model cache and loads the configured FlashAttention
hub kernel in an offline subprocess. The sweep pins the kernel repo, version and commit, while the
worker resolves its snapshot under the active `HF_HUB_CACHE` or `HF_HOME/hub` and passes
`LOCAL_KERNELS` to every job. The selected local build must already be cached and loadable on the
assigned GPU. `--dry-run` only prints jobs; it does not perform this preflight.
```bash
colm-sweep create --sweep configs/rank_sweep/sweep.json --queue queues/rank-sweep-v5
colm-sweep work --queue queues/rank-sweep-v5 --gpu 0 --dry-run    # print resolved jobs
colm-sweep work --queue queues/rank-sweep-v5 --gpu 0              # run (the GPU id is required)
colm-sweep summary --sweep configs/rank_sweep/sweep.json       # out/rank-sweep-v5/summary.md
```
Design, arms and timing: `TODO.md` ("LoRA rank sweep").

## Documentation
`docs/README.md` says which document answers what (upstream errors, optimisation backlog, FP16 selection
prefix, selection precision, start-up cost). `TODO.md` is the short current task list.

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
