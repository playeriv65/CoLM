# CoLM — agent notes

ICLR 2025 "Mini-batch Coresets for Memory-efficient LM Training on Data Mixtures" (LESS + MeZO
based), ported to transformers 5.x. This fork is a **public** GitHub repo: never commit secrets,
tokens, W&B keys or machine-private data.

## Environment

- `uv sync` builds `.venv` (Python 3.12, torch 2.13.0+cu130 from the PyTorch cu130 index,
  transformers 5.x, peft, accelerate, submodlib from git). Extras: `--extra eval` (vLLM 0.30,
  pinned to the same torch), `--extra wandb`. Lock file `uv.lock` is committed; upgrade with
  `uv lock --upgrade` and re-run the tests.
- The uv cache comes from `UV_CACHE_DIR` (machine env), not from pyproject.
- HF caches must point at the local NVMe (`HF_HOME`), never at a network disk. Pass no
  `cache_dir` in configs unless overriding.
- After a fresh clone / new worktree run `bash scripts/link-external.sh` (data and out symlinks,
  declared in `external-paths.json`; root from `COLM_ARTIFACT_ROOT`).

## Code map

- `colm/train/trainers.py` — thin subclasses of the HF 5.x `Trainer`:
  `CustomTrainer` (full-batch baseline), `SubsetTrainer` (bs=1 per micro-batch, all selection units),
  `SubsetTrainerEfficient` (batched last-layer MeZO; the paper's method). CoLM hooks only
  `get_batch_samples` (feature → gather on rank 0 → facility location → broadcast → micro-batches)
  and `training_step` (loss scaling). Do not copy HF loop internals back in.
- `colm/train/custom_phi.py` — Phi forward split before the last decoder layer (5.x modeling API).
- `colm/train/facility_location.py` — source-wise facility location (submodlib).
- `colm/train/step_timing.py` — opt-in per-phase step timer (`--profile_timing coarse|fine`),
  transfer/sync census, and the summariser (`python -m colm.train.step_timing <jsonl>`). At level
  `off` every timer call is a no-op; keep new timing sections behind `timer.section` / `timer.fine`.
- `colm/train/*_arguments.py` — all hyperparameters; defaults are the paper recipe.
- `colm/eval/` — teacher-forced eval loss (`eval_loss.py`: sets, `evaluate_loss`, trainer callback,
  standalone CLI) and its arguments; `colm/data/holdout.py` — deterministic held-out split.
- `colm/jobs/` — file queue + the single worker (`file_queue.py`, `worker.py`), sweep expansion
  (`rank_sweep.py`, specs in `configs/rank_sweep/`) and `summarize.py`. Queues live in the
  gitignored `queues/`; the worker takes `--gpu` explicitly.
- `math_eval/` (vLLM / HF generation; `run_open.py` takes several models/datasets per process),
  `superglue_eval/` — evaluation scripts.

## Rules

- GPUs are shared and reserved per period: scripts never default to a GPU list; pass ids or
  `COLM_GPUS`. Tests run on CPU: `CUDA_VISIBLE_DEVICES="" uv run pytest -q`.
- W&B is opt-in (`--report_to wandb`); with it off nothing may import `wandb` (tested).
- Before committing: `uv run ruff format . && uv run ruff check . && CUDA_VISIBLE_DEVICES="" uv run pytest -q`.
- Behaviour preserved from the original implementation on purpose (do not "fix" silently):
  logged CoLM loss is divided by `small_batch_ratio`; `torch.manual_seed(zo_random_seed)` is
  re-applied on every MeZO estimate (same z every step); per-sample MeZO loss averages over the
  padded length; base weights are fp32 with fp16 AMP for phi-2 (`torch_dtype=none`).

## Optimisation work

- Backlog, findings and open decisions: `docs/optimization-backlog.md`. Read it before touching
  the selection path; update item status there when an item lands or is measured.
