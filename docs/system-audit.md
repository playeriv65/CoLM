# System audit (2026-09-29): speed, memory, multi-GPU, robustness, structure

Second pass over the implementation as a system (not the algorithm). Base: `0b06680`. Method: static
reading of `colm/`, the timing data already on disk (`docs/optimization-backlog.md`,
`docs/startup-overhead.md`, the fine-timing jsonl of `artifacts/CoLM/fp16-select-pack1536-20260929/`),
CPU experiments on the real data lengths, then a few GPU runs (physical GPU 2 only, shared with other
jobs: only memory and correctness are read from shared runs, timings were taken alone, see "GPU runs").
Nothing here changes what is selected or which weights are trained; the changes are listed per area with
the evidence that they do not.

## Summary

| area | verdict | changed |
|---|---|---|
| (a) multi-GPU | two real defects found and fixed, one large imbalance fixed; the serial rank-0 part is small now | eval-loss callback hang (`should_log`), eval sharded over the ranks, token-balanced training shares, DDP unused-parameter search off |
| (b) per-step fixed costs | host part was 2.4 % of the step, all of it tail latency; the rest is GPU work, of which 6 % was a repeated weight cast | `pack` in numpy (-19 ms), frozen fp32 Linear weights stored in fp16 (-55 ms, -8.6 GB peak) |
| (c) checkpoints / eval duplicates | 1.9 % of a job, below the bar; nothing changed | - |
| (d) robustness | resume from an incomplete or model-only checkpoint, disk-full at the first save, launcher SIGTERM, missing loss rows | preflight module, launcher signal forwarding, validation, OOM hint |
| (e) structure / tests | no dead options; `train.main` split; tests for the collective paths were missing | test for eval callback and sharding on 2/3 ranks, token balance, pack geometry, resume |

## (a) Multi-GPU

### What a step does with W ranks (code: `colm/train/trainers.py::_select`, `colm/selection/pool.py`)

| stage | who | collective | size |
|---|---|---|---|
| plan | all | `all_gather_object` of the pool's `Example`s (numpy int64 ids and labels) | ~0.13 MB per rank |
| features | all, token-balanced shares (`balanced_shares`) | - | forward of ~1/W of the needed tokens |
| gather | all -> 0 | `gather_object((positions, g_i))` | 25 scalars per rank |
| select | rank 0 only | - | `expand` -> Adam -> mask -> facility location |
| scatter | 0 -> all | `broadcast_object_list((indices, weights))` | ~1 KB |
| train | all, packs of <= 1536 tokens | DDP all-reduce of the LoRA gradients on the last pack (`no_sync` before) | 671 MB fp32 |
| log | all | `_nested_gather(loss)` (stock), `MemoryMeter.gather` (`all_gather_object`) | scalars |

Ten small object collectives per step (pickle, stream synchronisation, ~0.1-0.3 ms each, ~3 ms) plus the
gradient all-reduce. The one-scalar gather of the earlier optimisation (O3) already removed what made
the first real 2-GPU run slow (42 MB of features per rank pickled: 3.71 s per step against 2.39 s on one
GPU, throughput 1.29x, measured before the optimisations; not repeated, only GPU 2 was assigned).

### Serial rank-0 selection: measured, nothing to change

`CoresetSelector.__call__` on synthetic g_i with the real selector and submodlib (GPU 2, shared, 50
steps, features `[N, 327680]` built from g_i and z as in the trainer):

| ranks | pool N | selected | mean ms | p50 ms |
|---|---:|---:|---:|---:|
| 1 | 32 | 16 | 10.7 (warm-up outliers) | 4.2 |
| 2 | 64 | 32 | 4.9 | 3.7 |
| 4 | 128 | 64 | 6.3 | 5.4 |
| 8 | 256 | 128 | 8.7 | 7.9 |

The fine timing of a real one-GPU step has `select` = 6.4 ms (0.7 %). At the paper's 4 GPUs the ranks
wait ~6 ms per step (0.6 %); at 8 GPUs ~9 ms. Rank 0 does not need to be relieved, and overlapping it is
not worth the complexity.

### Defect 1: the evaluation callback broke the collectives (fixed)

`EvalLossCallback._evaluate` called `Trainer.log` from `on_step_end`. `CallbackHandler.on_log` sets
`control.should_log = False`, and `_maybe_log_save_evaluate` (which runs after `on_step_end`) then skips
its logging block **on rank 0 only**. Two consequences:

* single GPU: the train-loss row of every evaluation step is missing. The 1024-step acceptance run has
  1020 loss rows; the missing ones are exactly 256, 512, 768, 1024 (`trainer_state.json`);
* several ranks: rank 0 skips the loss gather and the `MemoryMeter.gather` that ranks 1..W-1 enter
  (they were also idle for the length of the evaluation). The collectives no longer match: gloo aborts
  with "Received data size doesn't match expected size"; NCCL would hang or read wrong sizes. Every
  multi-GPU run with `eval_loss_steps` (the rank sweep configuration) was affected; the sweep has only run
  on one GPU so far, so nothing published is wrong.

Found because a 2-rank test with the callback failed (`tests/test_distributed.py`, evaluation after
step 1; the same test with the pre-change callback fails identically). Fix: the callback restores
`control.should_log` after logging. Regression tests: the single-process callback test asserts a loss row
at every step including the evaluation step (it fails on the old code with `[1] != [1, 2]`), the 2- and
4-rank runs pass through the evaluation.

### Sharded evaluation (implemented)

With `eval_loss_steps` only rank 0 evaluated: 4 x 28.7 s = 115 s per job (10.6 % of the 1077 s of a
one-GPU job) while W-1 GPUs idle, and the same 115 s on 4 GPUs is 10 % of a job. Now every rank takes
every W-th batch of the length-sorted batches and the per-example NLL sums and token counts are added up
(`evaluate_loss`, `pool.all_reduce_sum`). Each example goes through the same batch as in one process,
so the pooled numbers are identical, not merely close (`test_evaluation_loss_is_sharded_over_the_ranks_
without_changing_it`: 2 and 3 ranks, float64, `==`). Expected: ~29 s per job on 4 GPUs (ideal 1/W;
the last batch of a rank and the all-reduce add a fraction of a second), i.e. ~7-8 % of the job wall clock
on 4 GPUs. Not measured on GPUs (one GPU assigned).

### Training shares: round-robin was 8 % slower at 4 GPUs (implemented)

The selected list is `[kept..., picks per source...]` and the ranks took every W-th entry. The step loss
is a sum over the selected examples divided by the step's label count, so which rank trains which
example changes nothing (`test_ranks_agree_with_one_process`: weights equal to 1e-9), but the step lasts
as long as the rank with the most tokens (the gradient all-reduce waits for it) and round-robin does not
balance tokens. Simulation on the real per-example token counts (262,008 MathInstruct examples, recipe
keep sources, proportional quotas, random picks inside a source, 3000 steps per row, step model from the
fine timing: 65 us per forwarded token, 121 us per trained token, 47 ms fixed):

| ranks | trained tokens, slowest / mean: round-robin | longest-first | step ms, round-robin | longest-first | ideal (mean work) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 908 | 908 | 908 |
| 2 | 1.084 | 1.005 | 949 | 912 | 908 |
| 4 | 1.161 | 1.008 | 990 | 918 | 912 |
| 8 | 1.233 | 1.011 | 1023 | 919 | 911 |

The feature stage was already balanced (`balanced_shares`: slowest / mean 1.004-1.007). Now the selected
examples use the same function (`balanced_shares` with every example wanted: longest first onto the
least loaded rank, each rank at least one example, the same split on all ranks, one rank identical to
before). At 4 GPUs 990 -> 918 ms (7.9 %) before communication. The selection's picks are not independent
of length in reality, so this is an estimate; the token counts per rank can be read from `train_tokens`
of a real multi-GPU timing run. Tests: token counts of the ranks differ by less than the longest example
(`test_ranks_agree_with_one_process`), and a unit test with a skewed selection on which round-robin is
off by 548 tokens (`test_selected_examples_are_shared_over_the_ranks_by_tokens`). E7 in `errors.md`
wanted "the same mixture on every rank"; the mixture does not enter the loss, so equal tokens replace it.

### DDP unused-parameter search (implemented)

The Trainer sets `find_unused_parameters=True` for a PEFT model (it is not a `PreTrainedModel`); PyTorch
warns that this walks the autograd graph after every backward. Every LoRA weight is used in every
forward. `ddp_find_unused_parameters` now defaults to `False` (a parameter that is really unused fails
loudly at the backward). The 2- and 4-rank gloo runs pass without the warning. Gain unmeasured (needs
>= 2 GPUs; a graph walk of ~22 k backward nodes, an estimated few ms per step).

### Estimate for the paper setting (4 GPUs, pool 4 x 4 x 8 = 128, one step)

| term | ms | basis |
|---|---:|---|
| one-GPU step (same tokens per rank) | 821 (885-908 before the fp16-stored weights and the numpy `pack`) | fine timing, "GPU runs" |
| imbalance of the slowest rank (longest-first) | +10 | simulation (round-robin: +82) |
| rank-0 selection at N = 128 | +6 | measured above |
| ten object collectives | +3 | estimate |
| gradient all-reduce, 671 MB fp32, exposed part | 0-50 | ring: 2 (W-1)/W x 671 MB = 1 GB per rank at 10-25 GB/s PCIe (GPUs 0-3 are PIX/PXB, no NVLink), 40-100 ms, of which up to the last pack's backward (~75 ms) overlaps; unmeasured |
| **step** | **~0.84-0.89 s** | |

Throughput scaling 4 x 0.821 / (0.84-0.89) = 3.7-3.9x on the training steps (with round-robin shares
and the fp32 weights, on the same model: 3.4-3.6x); the earlier real measurement was 1.29x on 2 GPUs.
With the sharded evaluation the job (1024 steps, 4 evaluations, 3 saves, start-up) is ~0.98e3 s against
~1.07e3 s with an evaluation on rank 0 only (0.82-0.89 s x 1024 + 29 s (115 s) + 20 s + ~35 s). Candidates that need >= 2 GPUs to measure (not done):
put the largest pack last so that its backward hides more of the all-reduce (a model in which the
all-reduce starts with the last backward, 64 us per token, on the simulated 4-rank shares: gain 7 / 27 /
48 ms per step for an all-reduce of 40 / 70 / 100 ms, i.e. 0.8 / 2.8 / 5 % of the step),
fp16 gradient compression (`ddp_comm_hook`, halves the traffic but changes the gradient rounding: a
decision), one object collective instead of two for the memory meter (~1 ms).

## (b) Per-step fixed costs (one GPU, fine timing `fp16-select-pack1536`, mean 885 ms)

| phase | ms | % | verdict |
|---|---:|---:|---|
| prefix (fp16 + fp32 tail 2, ~5 packs) | 286 | 32.3 | GPU compute; includes 5 fp32 -> fp16 casts of the weights (see below) |
| train forward / backward | 209 / 240 | 23.7 / 27.1 | GPU compute (~135 TFLOP/s); the forward includes 3.2 casts |
| ±eps suffix: layer / head / loss | 55 / 38 / 2.5 | 6.3 / 4.3 / 0.3 | fp32 (head 2.2 TFLOP in SIMT fp32: TF32 is a precision decision) |
| `pack` (host) | 14.5 (p50 6.5, p90 30.6, max 112) | 1.6 | **tail latency of torch CPU ops, fixed** |
| `select` other (host: training packs) | 6.2 (p50 1.2, p90 17) | 0.7 | same cause, fixed |
| rank-0 selection | 6.4 | 0.7 | nothing to change |
| optimizer / scheduler | 8.3 | 0.9 | AdamW over 168 M fp32 parameters is at the bandwidth floor (4.7 GB, ~3 ms) |
| mode switch, prepare, log, other | 12 | 1.4 | nothing to change |

* Host syncs (`.item()` / `.cpu()` / `.tolist()`): the census per step is 7 syncing calls outside the
  copies of the packs (60 + 36 pageable host-to-device copies of ~120-235 KB in total); the values are
  needed on the host (`to_cpu` is 0.1 ms). The python loops run over at most 32 examples. No timed
  section besides the ones in the table exceeds ~0.3 ms, so `dict` copies and repeated tensor
  constructions do not show. `pin_memory` / `non_blocking`: the copies are < 0.3 MB per step; nothing
  to gain. Dataloader: one worker, 11 ms per
  pool, hidden behind the step (`dataloader_num_workers` already 1).
* **Tail latency of `pack` (implemented).** The 30-110 ms spikes of `pack` and of the host part after
  `scatter` (12 % of the steps in the fine run, loadavg 22-35) are not the algorithm: `pack` (and
  `train_batches`) built ~15 tiny tensors with torch CPU ops (`cumsum`, `repeat_interleave`, `bucketize`,
  `bincount`, ...), which use the intra-op OpenMP pool; on a loaded host its barriers stall. A CPU
  micro-benchmark of one step's packs (5 packs of a 32-example pool, same load): torch ops mean 1.7 ms,
  p99 7 ms, p99.9 55-83 ms, max 107-118 ms; numpy: mean 0.36 ms, p99 0.5 ms, p99.9 0.6 ms, max 0.8-1.3 ms.
  `pack` is now numpy only with identical outputs (2000 random packs equal in values and dtypes against the
  old function; `tests/test_packing.py` keeps a torch reference, including examples without labels).
  Measured in the timed run below: -19 ms per step (2.1 %), no host spike above 2 ms left.
* **Weight casts (implemented, `colm/train/frozen_weights.py`).** Autocast casts a Linear's fp32 weight to
  fp16 at every forward, and it does not cache frozen weights (only leaves that require grad). Per
  step: ~5 selection packs (29 fp16 layers) + 3.2 training packs (32 layers) = ~8 forwards x 13.7-15.1 GB
  of traffic (~9-10 ms at 1.5-1.7 TB/s) = ~75 ms (8.5 % of the step), and the autograd graph keeps the
  casts alive for the backward (5.6 GB). The frozen Linear weights of the layers that only run under
  autocast are now stored in fp16 once (the fp32 tail, the perturbed last layer, head, norms,
  embeddings and all LoRA tensors stay fp32; not applied with an fp32 selection prefix or a
  non-efficient extractor). The numbers are the same because autocast's cast is this cast: prefix
  state, training loss and LoRA gradients are bit-identical (CPU bf16 tests, and the tiny model on GPU 2
  with fp16, `test_stored_fp16_weights_equal_autocast_on_the_gpu_bit_for_bit`; phi-2: prefix states
  bit-identical, see "GPU runs"). Measured (130-step fine runs, "GPU runs"): **-54.6 ms per step
  (-6.2 %)**, training peak memory 32.3 -> 23.7 GB, selection 13.1 -> 8.7 GB, reserved 43.0 -> 33.8 GB.

## (c) Checkpoints and duplicated work

* Checkpoint saving: 671 MB fp32 adapter, 6.7 s per save on the network disk (~100 MB/s); a default job has
  2 saves + the final `save_model` = 20 s of 1077 s (1.9 %, below the 3 % bar). Writing to the local NVMe
  and moving in a background thread would hide most of it, but HF saves in place, `on_save` writes
  `selection_state.pt` into the directory afterwards and rotation looks at it: it needs a queue of
  finished saves and a join at the end, ~60 lines and a race to test for ~15 s per job. Not done. The
  final `save_model` repeats `checkpoint-1024` when `max_steps` is a multiple of `save_steps`
  (6.8 s, 0.6 %); a hard link would remove it; not done (the adapter in `output_dir` is what the queue reads).
* The standalone `evalloss` job (~1.3 min per arm, 3 % of an arm) duplicates the in-training numbers at
  steps 512 / 1024 (agreement 6e-5). It is the independent check; dropping it is the user's decision
  (`TODO.md`). With the frozen fp16 weights the standalone loss job and the in-training evaluation both
  lose the per-batch weight cast (the pass is ~60 us per token: ~10 % of it was the cast).
* Evaluation loss passes (10.6 % of a job on one GPU): 16.8 k tokens/s, batches of 8; larger batches would
  amortise the fixed part but change the rounding of the reported loss; left.

## (d) Robustness

Checked and found sound: a dying rank (torchrun `--standalone` has `max_restarts=0`: it terminates the
others and exits 1; the worker kills the whole process group on SIGTERM); NCCL timeouts (`ddp_timeout`
1800 s, longer than any rank-0-only stretch: checkpoint 7 s, holdout save); partial checkpoints on a
crash during a save fail loudly at load (truncated safetensors), they are not silently used; the
`selection_state.pt` is written atomically; the launcher tees everything to `logs/<config>-...log` with
flush, per-step loss and timing are in it, so a crash loses nothing but the final json files.

Found and fixed (`colm/train/preflight.py`, `colm/cli.py`, `training_arguments.py`, `trainers.py`,
`train.py`):

| gap | change | test |
|---|---|---|
| `resume_from_checkpoint=True` took the newest `checkpoint-N` directory, complete or not (HF 5.17 writes into the final directory, `trainer_state.json` last) | the newest complete checkpoint; an explicit incomplete path is an error | `test_resume_true_takes_the_newest_complete_checkpoint` |
| resuming a model-only checkpoint (the default `save_only_model`) silently restarts the optimizer, the LR schedule position and the RNG | a warning naming the missing files and the fix | `test_a_model_only_checkpoint_resumes_with_a_warning` |
| no free-space check: a full disk fails at the first save (step 256/512, minutes in) | checked before the data is read: trainable bytes x planned saves (x3 without `save_only_model`, `save_total_limit` respected) | `test_too_small_a_disk_fails_before_the_run`, `test_planned_saves` |
| `kill <colm-train pid>` left torchrun and its workers on the GPU | SIGTERM / SIGHUP forwarded to torchrun while it runs | `test_the_launcher_passes_termination_on_to_torchrun` |
| negative or zero `pack_tokens`, `train_max_tokens`, `zo_dim`, `mezo_eps` were accepted | validated at load | `test_inconsistent_selection_settings_fail_early` |
| an out-of-memory error in the selection forward had no hint (the training one had) | names `pack_tokens` | `test_the_budget_is_a_bounded_default_and_out_of_memory_says_what_to_set` |
| `eval_loss_steps` beyond `max_steps` never run, silently | warning | - |
| NCCL "destroy_process_group() was not called" warning at every exit | destroyed at the end of `main` | - |
| pytest runs wrote ~5 files per run into the shared token cache (`cache/tokens`): 50 files of 828 bytes had accumulated | the tests' configs disable the cache | (the cache directory is unchanged by the suite) |

Left (decisions): `save_only_model=True` (default) means a crash at step 900 of a 1024-step arm restarts
that arm; `False` costs 3 x the bytes per checkpoint (~2 GB, ~20 s per save on the network disk) and gives
an exact resume; with the frozen weights nothing else changes. A 20-arm queue at ~18 min per arm makes
the choice a question of how often a run dies, which the history does not say.

## (e) Structure and tests

* Dead options: every field of the four argument classes outside HF's own is read somewhere (script over
  the code base); the upstream data options `subset_selection` / `percentage` / `subset_index_files` /
  `sample_data_seed` are only exercised at `percentage=1` (`load_raw_dataset` is the longest function of
  the core, 153 lines, 4 of its 5 branches unreachable in the recipe): removing them changes the
  interface, a decision.
* Aliases: `per_device_train_batch_size` is rewritten in `__post_init__` to the pool size
  (`micro_batch_size` keeps the configured value; `resolved_config.json` shows 32 for the default). It is
  documented in the help, but it is the one naming trap left; renaming the key would break every config.
* Long functions: `train.main` was 112 lines; the logging set-up, the model checks and the run's
  epilogue are now `configure_logging`, `prepare_model` and `finish` (`main` ~50 lines). The rest of
  the core is < 90 lines (`CoresetSelector.__call__` 70, `_select` 55). No duplicated logic between the
  trainers (`SubsetTrainer` / `SubsetTrainerEfficient` differ in three small methods).
* Test gaps closed: the eval callback under 2 ranks (rank 0 records, all ranks take part), the sharded
  loss on 2 and 3 ranks (`==`), the token balance of the shares, `pack` against its torch reference,
  resume of an incomplete directory, model-only warning, launcher signal, disk, validation, the stored
  fp16 weights (bit-identical). Still uncovered: a 2-rank resume (only rank 0 loads the selection state; the
  other ranks are not checked), a 2-rank run with `keep_sources` covering more than the budget, NCCL (the
  CPU tests use gloo; the object collectives behave the same, the ordering rules are what the tests check).

## GPU runs

Physical GPU 2 (`00000000:4E:00.0`), phi-2, `configs/diagnostics/timing_phi2_efficient.json` (paper recipe,
`--profile_timing fine`, 130 steps, census on steps 1-10, seed 0, so the same pools in the same order), one
run per arm, the two arms one after the other (A `--frozen_base_low_precision false`, B `true`), loadavg
20.1 before A, 19.6 between, 18.6 after B. The card was alone when the runs started (a check before
launch); from step 118 of B every section, the host `pack` included, inflates (`pack` 1 -> 125 ms, prefix
290 -> 640 ms): another job started on GPU 2. Those 13 steps are dropped, and both arms are compared on
steps 11-117 (107 steps; 50-step sliding means within 2.4 % (A), 3.4 % (B), 4.4 % (the old run)). Raw
files and the analysis script: `$COLM_ARTIFACT_ROOT/artifacts/CoLM/system-audit-20260929/`
(`step_timing-timed-frozen-*.jsonl`, `timed-ab-summary-steps11-117.txt`, `analyze_ab.py`).

| ms per step, steps 11-117, mean | old code, fine run of 2026-09-29 13:19 (`fp16-select-pack1536`) | A: this branch, fp32 weights | B: this branch, fp16 frozen weights |
|---|---:|---:|---:|
| **step** (wall clock) | 887.6 | 875.9 | **821.3** (median 850 / 856 / 807) |
| prefix (5.9 k forwarded tokens) | 287.5 | 319.4 | 275.7 |
| ±eps replay | 98.5 | 98.2 | 98.6 |
| `pack` (host) mean / p90 / max | 14.6 / 34 / 112 | 1.1 / 1.4 / 2 | 1.2 / 1.5 / 2 |
| host part after `scatter` mean / max | 6.6 / 73 | 0.7 / 1 | 0.7 / 1 |
| train forward / backward | 209.3 / 239.5 | 201.9 / 226.2 | 183.8 / 231.0 |
| trained tokens per step | 3753 | 3766 | 3739 |
| peak memory, training / selection (allocated) | - | 32.3 / 13.1 GB | 23.7 / 8.7 GB |
| peak reserved | - | 43.0 GB | 33.8 GB |

* **Host tails (numpy `pack`)**: `pack` 14.6 -> 1.1 ms (max 112 -> 2 ms), the host part after `scatter`
  6.6 -> 0.7 ms (max 73 -> 1 ms): -19 ms per step (2.1 %), and no step of the run has a host spike. The
  old column is another run at another time; its GPU sections differ from A by more than the code can
  explain (prefix 287 against 319 ms with identical GPU code), so the GPU part of that comparison is
  machine noise: only the host rows are attributed to the change.
* **Frozen fp16 weights, A -> B**: 875.9 -> 821.3 ms, **-54.6 ms (-6.2 %)**, median -5.8 %. The prefix
  loses 43.7 ms (5 packs x 8.7 ms: the predicted ~9 ms per forward of casting, 15 GB of traffic), the
  training forward 18 ms (3.2 forwards x 5.6 ms), the backward gains 4.8 ms (noise class). The
  estimate of ~75 ms was high for the training side; the measured saving is 55 ms. Start-up: model load +1.0 s
  (the conversion on the CPU), trainer init -2.9 s (4.6 GB less to copy), net -1.9 s. Mean train loss over the
  130 steps 0.698 (A) / 0.685 (B): one seed each, the selection of a step is decided at rounding level, so
  this is not a learning comparison (the numbers of a step are identical, see below).
* **Equality check on phi-2** (`phi2_frozen_equal.py`, one real pool of 32 examples, 8477 tokens, shared
  GPU, twice): the prefix states of all packs are bit-identical with fp32 weights and with the stored fp16
  weights; g_i differs by a median relative 2.0e-4 (5.3e-4 in the first run) against a repeat noise of the same
  weights of 1.6e-4 (3.9e-4): the difference is the run-to-run nondeterminism of the fp32 replay, not the
  storage; the training loss is bit-identical (0.87964942); the LoRA gradient of the selected half differs
  between two runs of the same weights by up to 1.2e2 in max abs (the flash backward has atomics; the
  gradient's own scale is ~1e3), and the stored-weight gradient differs from those by no more (9.1e1).
  The same check on the tiny model on GPU 2 is bit-identical for prefix, loss and gradients.
* Not measured: anything with more than one GPU (only GPU 2 was assigned); the estimates in (a) are model and
  simulation, and the sharded evaluation, the token balance and `find_unused` are verified for correctness
  on 2 / 3 / 4 CPU ranks only.

## Remaining candidates (prioritised)

| # | candidate | gain | risk | needs |
|---|---|---|---|---|
| 1 | measure a real 2/4-GPU run (`train_tokens` per rank, exposed all-reduce, eval sharding, `find_unused`), then order the packs of a step so the largest is last (only the accumulation order changes) | 1-5 % of the step at 4 GPUs (model above) | low | 2-4 GPUs |
| 2 | asynchronous / local-disk checkpoint saves and a hard link for the final adapter | ~15-20 s per job (1.5-1.9 %) | medium (race with `on_save`, rotation) | - |
| 3 | LoRA merged into the fp16 prefix weights for the selection forward (O12 for fp16) | ~30-40 ms of the prefix (3-4 %) | changes rounding of g_i (precision decision) | user |
| 4 | TF32 for the fp32 head of the ±eps replay (2.2 TFLOP in SIMT fp32, 38 ms) | ~30 ms (3.4 %) | precision decision (D2) | user |
| 5 | `save_only_model=False` for long arms | resume exactness | ~20 s per save | user |
| 6 | drop the unused upstream data-sampling options (`subset_selection`, `percentage`, ...) | readability | interface change | user |
| 7 | eval batch size 8 -> 32 for the loss passes | ~1 % of a job | changes reported loss at 1e-5 | user |
| 8 | fused AdamW | ~1-2 ms | none | - |
