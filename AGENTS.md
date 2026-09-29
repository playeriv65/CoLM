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

## Entry points

`colm-train` (1 / 2 / N GPUs with `--gpus`, torchrun underneath), `colm-eval loss|accuracy|superglue`,
`colm-sweep create|work|summary`; all print `--help`. README "Quickstart" has the commands. There
are no other launch scripts: do not add shell wrappers, extend the entry points.

## Code map

- `colm/train/trainers.py` — thin subclasses of the HF 5.x `Trainer`: `CustomTrainer` (full-batch
  baseline), `SubsetTrainer` (one example per forward, every selection unit) and
  `SubsetTrainerEfficient` (batched last-layer MeZO, the paper's method). The selection pool of a
  step (`per_device_train_batch_size x gradient_accumulation_steps` examples) is ONE HF batch
  (`TrainingArguments` sets HF's gradient accumulation to 1); the coreset trainers override only
  `training_step` (documented extension point): plan (gather the pool, which features are needed)
  -> features of the needed examples (a token-balanced share per rank) -> gather (scalars g_i for
  MeZO) -> rank-0 selection -> broadcast -> forward/backward of the selected examples in packs of
  `batching.train_tokens` (= `train_max_tokens`, a plain token budget; `accelerator.no_sync` for all
  but the last). Optimizer step, clipping, scheduler, logging, checkpointing are stock. Do not
  copy HF loop internals back in or touch private HF attributes.
- `colm/selection/` — `features.py` (one extractor per `data_selection_unit`, on packed batches;
  `extract` -> per-example values, `expand` -> features; the MeZO extractor returns g_i),
  `zo.py` (`Perturbation`: fixed-seed z, out-of-place +-eps through `functional_call`;
  `LastLayerSplit`: the model's own forward stopped by a pre-hook on the last layer, then the
  last layer replayed; works for any decoder), `select.py` (`CoresetSelector`: `needed` = which
  features can matter, keep sources, transform, Adam, coordinate mask, facility location; keeps
  the Adam moments), `facility_location.py` (`class_budgets` = the per-source quotas),
  `packing.py` (padding-free batches with the label geometry computed on the CPU, `balanced_shares`),
  `batching.py` (packed batching, the step loss), `pool.py` (collectives that are no-ops in one
  process).
- Attention is stock transformers (no custom kernel): packed rows carry `position_ids` and the
  flash cumulative lengths (`packing.model_inputs`); the training forward runs
  `attn_implementation` (`flash_attention_2`, recipe default in `configs/model_profiles.json`; the
  hub kernel through `kernels` when `flash-attn` is not installed) and the no-grad selection
  forward `selection_attn_implementation` (`sdpa`, dense block mask; fp16 prefix and fp32
  suffix for the Phi-2 recipe); the trainer switches with
  `model.set_attn_implementation` (`_Trainer.set_attention`). `colm/train/memory.py` — peak memory
  of every rank per phase (`MemoryMeter`).
  `colm/train/step_timing.py` — opt-in per-phase step timer and its summariser
  (`--profile_timing coarse|fine`; keep new timing sections behind `timer.section`).
- `colm/train/config.py` (JSON + flags, model recipe from `configs/model_profiles.json`, resolved
  config), `training_arguments.py` / `model_arguments.py` / `data_arguments.py` — all options; the
  defaults are the paper recipe with the corrections of `docs/errors.md`.
- `colm/data/` — datasets and collators (`get_training_dataset.py`, `superglue.py`, `holdout.py`,
  `tasks.py`, `templates.py`); `colm/cli.py` — the entry points.
- `colm/eval/` — teacher-forced eval loss (`eval_loss.py`: sets, `evaluate_loss`, trainer callback,
  CLI); `colm/jobs/` — file queue + the single worker, sweep expansion (`rank_sweep.py`, specs in
  `configs/rank_sweep/`) and `summarize.py`. Queues live in the gitignored `queues/`.
- `math_eval/` (vLLM / HF generation; `run_open.py` takes several models/datasets per process),
  `superglue_eval/` — evaluation code behind `colm-eval`.
- `tests/equivalence/` — the tiny float64 Phi, tokenizer and data used by the tests.

## Rules

- GPUs are shared and reserved per period: nothing defaults to a GPU list; pass `--gpus` / `--gpu`
  or `COLM_GPUS`. Tests run on CPU: `CUDA_VISIBLE_DEVICES="" uv run pytest -q` (includes 2- and
  4-rank gloo runs).
- W&B is opt-in (`--report_to wandb`); with it off nothing may import `wandb` (tested).
- Before committing: `uv run ruff format . && uv run ruff check . && CUDA_VISIBLE_DEVICES="" uv run pytest -q`.
- The upstream behaviour is not kept as an option: the tag `pre-refactor` is the upstream code and
  `legacy-bridge` the last commit with the temporary `legacy` switch and its float64 goldens
  (`tests/test_equivalence.py`); `docs/errors.md` lists every error and its fix. The default path
  must contain no known error.
- Exactness is float64 identity on CPU: in fp32 the MeZO feature is decided at rounding level
  (a fp32 run of the upstream code agrees with itself in 3 of 20 steps), so end-to-end fp32 runs
  are compared statistically (selection overlap against that noise floor), not bitwise.
- The Phi-2 model profile sets `selection_prefix_dtype=float16`: the unperturbed
  prefix uses fp16 autocast and requires fp32 model weights, while the perturbed
  last layer and loss remain fp32. Other profiles retain their explicit precision
  until tested. CLI/JSON overrides take precedence; the resolved configuration
  records the choice. The Phi-2 path changes selected sets beyond
  the fp32 packing noise (`docs/fp16-prefix.md`); learning quality has not yet
  been compared. Packed inputs need `use_cache=False` (a cache ends the
  packed-batch detection of transformers).

## Optimisation work

- The execution-only optimisations are done and are the code path (no switches): backlog, measured
  tables and the candidates left (O11-O13) are in `docs/optimization-backlog.md`. Keep them exact:
  `tests/test_opt.py` compares every one with the plain computation in float64 on CPU,
  `scripts/check_opt.py` does the phi-2 teacher-forced comparison on a GPU (selection overlap
  against the fp32 noise floor, gradients against an exact fp32 reference).
- fp32 gradient references use `sdpa_kernel(SDPBackend.MATH)`: the fp32 memory-efficient SDPA
  backward is ~0.3 off on phi-2 on this GPU (`docs/errors.md`).
- Timing follows the protocol in the backlog (`--profile_timing fine`, 130 steps, 10 warm-up,
  closure, GPU alone, loadavg recorded); a step's time is 0.154 ms per forwarded token + 0.130 ms
  per trained token, so compare runs on the same pools.
