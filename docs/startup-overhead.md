# Where the time of a job goes (start-up, evaluation, saves)

2026-09-29, phi-2, one RTX PRO 6000 (physical GPU 2, alone during the timed run), code `c9a124a`
(phase timers and token cache) on top of `09b4602`. Every run writes `<output_dir>/startup.json`
(`colm/phases.py`, `colm/train/phase_callback.py`: one timestamp per phase, no synchronisation,
nothing per step); `colm-train` also prints and logs the wall clock of the whole launch.

## Phases of the default sweep job (1024 steps)

`configs/rank_sweep/base.json` on the defaults of the paper recipe: `holdout_size` 1000, evaluation
loss (held-out + GSM8K) after steps 256/512/768/1024, model-only checkpoints at 512 and 1024,
FP16 selection prefix with an FP32 tail of 2, `pack_tokens` = `train_max_tokens` = 1536, W&B off, one
GPU. Output `artifacts/CoLM/wrapup/accept-a-1024steps-20260929-181244/`, log
`logs/base-gpu2-np1-20260929-181244.log`. Host load average (1 min) 28.7 at the start, 12.3 at the
end (a shared host: start-up phases are CPU work and move with it).

| phase | seconds | share |
|---|---|---|
| launcher + torchrun start | 1.2 | 0.1 % |
| python imports (torch, transformers, peft, ...) | 5.3 | 0.5 % |
| configuration (argument parsing, model config) | 0.8 | 0.1 % |
| tokenizer, model load (memory-mapped) | 2.3 | 0.2 % |
| **data** (raw rows 0.03, prompts + token counts 6.4, hold-out split 3.5, statistics 0.3) | 10.3 | 1.0 % |
| trainer init (model to the GPU, accelerator; the weights are memory-mapped until then) | 9.2 | 0.9 % |
| GSM8K evaluation set | 0.2 | 0.0 % |
| **start-up until the first step** | **28.1** | **2.6 %** |
| first step (kernel loading, autotune) | 2.6 | 0.2 % |
| 1021 steady steps (0.887 s each, median logged step 0.874 s) | 906.1 | 84.1 % |
| 4 evaluation-loss passes (15.9 s held-out + 12.8 s GSM8K each) | 114.7 | 10.6 % |
| 2 checkpoints (671 MB adapter each, network disk) | 13.5 | 1.3 % |
| final `save_model` + state + metrics (network disk) | 6.8 | 0.6 % |
| interpreter exit | ~3 | 0.3 % |
| **wall clock of the launch** | **1077** | 100 % |

Train loss 0.774 at step 1, held-out loss 0.5224 and GSM8K loss 0.7240 at step 1024
(0.5459 / 0.7509 at step 256; base model 0.8812 / 1.3428). Peak memory (allocated / reserved,
`memory.json`): selection 13.2 GB, training 32.0 GB / 82.1 GB reserved.

Reading: 84 % of the job is training steps. The rest is dominated by the evaluation-loss passes
(10.6 %): 4 x 463k tokens (249k held-out + 214k GSM8K, prompts included) in 115 s is about 16k
tokens/s, roughly 85 TFLOP/s for a 2.7B model in fp16 autocast over fp32 weights with unmerged LoRA
and fp32 logits over every position. The batches are length-sorted, so padding is small and packing
would not help; a larger batch (one fp32 -> fp16 weight cast per forward) or logits at the label
positions only might give 1.3x on a 10 % phase, i.e. about 2-3 % of a job, and changes the rounding of
the reported loss, so it is left. Checkpoint writes are network-disk I/O (about 100 MB/s).

## The token cache (what was changed)

Before, every training process and every standalone `colm-eval loss` tokenised all 262,039
MathInstruct examples once (only to decide which fit the context window; the collator tokenises the
batches again), 25-32 s on an idle host, minutes under load. `TokenCountCache`
(`colm/data/get_training_dataset.py`) now stores the boolean "fits" and the length of every example:

* key = SHA-256 of the prompts and completions (so the data file, prompt template, EOS token and
  sampling are covered), the context limit and the tokenizer (its backend definition, init arguments,
  `transformers` / `tokenizers` versions); any change reads another file, a corrupt file is
  recomputed, slow tokenizers are not cached;
* location `--token_cache_dir` (default `cache/tokens`, a link to
  `$COLM_ARTIFACT_ROOT/artifacts/CoLM/cache`, declared in `external-paths.json`; empty disables);
  1.3 MB per key; atomic write, so the ranks of a multi-GPU run may write the same file;
* the outcome is identical: the hash of all per-example lists (`sources`, `targets`, source ids,
  indices, completion lengths) and `mean_tokens` of the real data is the same with the cache and
  with the unmodified code at `5a6f778`
  (`d3197543...a032d`, 262,008 examples); `tests/test_token_cache.py` checks this on the test data,
  including contexts that drop examples, the key, corrupt files and a counting tokenizer proving that a hit
  does not tokenise. Step 1 of a cached run has the loss of an uncached run (0.7595 on the
  4-step probes); later steps differ from run to run anyway (the fp16 prefix decides the selection at
  rounding level: two identical cached runs diverge from step 2, 0.9972 against 0.9999).

| | tokenisation of the training data | wall clock of the job |
|---|---|---|
| before (60-step probe, GPU alone) | 31.8 s | start-up 60.7 s of 198 s |
| cache miss (first run, writes the file) | 29.3 s | |
| cache hit | 6.4 s (of which 5.7 s is the row loop that builds the prompts, unchanged, and 0.4 s the key) | |
| default sweep job, estimated before | | ~1102 s, start-up 53.5 s (4.9 %) |
| default sweep job, measured after | | 1077 s, start-up 28.1 s (2.6 %) |

The saving is 25 s per process that reads the training data: 2.3 % of a training job, and 23 of the
~100 s of the standalone loss job (its load time went from ~45 s to ~22 s: imports 5 s, model 8 s,
data 6 s, GSM8K set).

## Other candidates, measured and left alone

| candidate | finding |
|---|---|
| `dataloader_num_workers` | already 1 (the next pool is tokenised while the GPU works); tokenising a pool is 11 ms |
| model / tokenizer / config loaded twice | the config is read three times from the local snapshot (0.8 s in total); the weights once (2.3 s memory-mapped, the 9 s of trainer init is the copy to the GPU) |
| `save_only_model` | already the default (671 MB adapter, no optimizer state); the 6.7 s per checkpoint is the network disk |
| final `save_model` duplicates checkpoint-1024 | 6.8 s (0.6 %), below the 3 % bar; the final adapter in `output_dir` is what the docs and `summarize` read |
| hold-out split (3.5 s), the row loop of the prompts (5.7 s), imports (5.3 s) | 0.3-0.5 % each |
| evaluation-loss passes | ~85 TFLOP/s, little padding (see above); a larger batch or logits at label positions only might save 2-3 % of a job and change the rounding of the reported loss |
| standalone `evalloss` job per arm (~1.3 min with the cache, 1.6 min before) | duplicates the in-training numbers at steps 512 / 1024 (they agree to 6e-5, see the acceptance section below); kept as the independent check, dropping it is a user decision (`TODO.md`) |
| `evalloss` and `evalacc` in one process | not done: the HF model and the vLLM engine (`gpu_memory_utilization` 0.9) cannot share a GPU, and a shared process would need the worker to pass state between jobs |
| vLLM engine start | 73 s of an accuracy job that runs 11-18 min: 17 s process + import, 3 s weights, 16 s `torch.compile` (its cache lives in `VLLM_CACHE_ROOT`, local), 25 s of CUDA-graph capture in two passes; `enforce_eager` or fewer capture sizes would save ~25 s at the price of slower decoding or a vLLM-specific knob |

## Acceptance of the chain (2026-09-29, GPU 2)

* `colm-train` default recipe, 1 GPU, 1024 steps: finished (table above).
* `colm-eval loss` on checkpoint-512 / 1024 and the base model, standalone, 100.9 s
  (base + 2 checkpoints, 3 passes), agrees with the in-training evaluation:

  | | held-out standalone / in training | GSM8K standalone / in training |
  |---|---|---|
  | checkpoint-512 | 0.53186 / 0.53193 | 0.73551 / 0.73549 |
  | checkpoint-1024 | 0.52230 / 0.52236 | 0.72397 / 0.72402 |
  | base | 0.88123 | 1.34282 |

  (differences <= 6.3e-5: the passes use different batch compositions, so the fp16 rounding differs).
* `colm-eval accuracy` on both checkpoints with `--limit 20` (vLLM 0.30 + LoRA, all six datasets):
  finished in 2 min 48 s, engine start 88 s (compile cache warm), 12 result files. Accuracy on 20
  examples each, checkpoint-512 / 1024: gsm8k 0.60 / 0.60, math 0.20 / 0.20, numglue 0.60 / 0.60,
  svamp 0.65 / 0.75, deepmind 0.65 / 0.70, simuleq 0.20 / 0.35 (a chain check, not a result). vLLM
  prints `ERROR ... Unrecognized model in .../checkpoint-512` while it probes the adapter directory as
  a model config; the run is not affected.
* `colm-eval superglue` (HF, LoRA checkpoint-1024, one GPU): CB 0.393 (56 examples, 17 s) and RTE
  0.601 (277 examples, 30 s). Both failed before this wrap-up: the task datasets were loaded with the
  bare hub ids `super_glue` / `squad` / `drop`, which `datasets` >= 4 rejects (all 11 tasks load
  now, offline from the local cache), and `sample_subset(num=-1)` dropped the last example (CB was
  scored on 55, RTE on 276; `docs/errors.md` E17).
* `colm-sweep create --sweep configs/rank_sweep/smoke.json --queue ...` then `colm-sweep work --queue
  ... --gpu 2 --dry-run` resolve (6 jobs; the queue and the run directory it made were deleted again);
  `colm-sweep create --dry-run` (new) prints the 18 jobs of the real sweep without writing anything.
  `--help` of `colm-train`, `colm-eval` (loss, accuracy, superglue) and `colm-sweep` (create, work,
  summary) exits 0.
* Multi-GPU: only GPU 2 was assigned, so a 2-GPU run was not repeated. The multi-rank path is covered
  by the CPU 2- and 4-rank gloo tests (`tests/test_distributed.py`) and by the earlier real 2-GPU check
  on GPUs 0 and 2 (`docs/errors.md` E7 / MG numbers); `startup.json` is written by rank 0 only.
* CPU tests: 174 passed (`CUDA_VISIBLE_DEVICES="" pytest -q`, 130 s), `ruff format` and `ruff check` clean.
