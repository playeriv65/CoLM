# FP16 decoder prefix with an FP32 tail and an FP32 MeZO suffix

2026-09-29, implementation based on `3244c97`; default changed to an fp32 tail of two blocks
(user decision, see "Default: fp32 tail of 2 blocks" below). The computation is: run the first 31
Phi-2 decoder layers (the prefix) under CUDA FP16 autocast, except the last
`selection_prefix_fp32_tail` prefix blocks (default 2: blocks 29 and 30), which run in FP32; promote
the captured floating inputs of the last layer to FP32, then replay the perturbed last layer,
output head, and per-example cross entropy in FP32. The base weights and LoRA parameters remain
stored in FP32. Selection attention remains stock SDPA; training attention remains stock
FlashAttention-2. The Phi-2 model profile sets `selection_prefix_dtype=float16` and
`selection_prefix_fp32_tail=2` and requires FP32 model weights; `selection_prefix_dtype=float32`
retains the previous full FP32 prefix, and `selection_prefix_fp32_tail=0` the plain FP16 prefix
that the sections below measured. Command-line and JSON settings override the profile, and the
resolved configuration records the chosen mode.

Everything from "Numerical check" to "Selection pack budget" below was measured with the plain FP16
prefix (`selection_prefix_fp32_tail=0`) and is kept as the record of that arm (arm P of
`docs/selection-precision.md`); its selection-overlap and timing numbers do not describe the new
default.

## Default: fp32 tail of 2 blocks

`selection_prefix_fp32_tail=k` (integer, `0 <= k <= 31`, only with `selection_prefix_dtype=float16`;
the Phi-2 profile default is 2, every other profile 0, and an explicit `float32` prefix gets 0
unless a tail is asked for, which is an error) puts a forward pre-hook on prefix block `31 - k`
(`LastLayerSplit._fp32_tail` in `colm/selection/zo.py`). The hook casts the block's floating inputs
(hidden states, position embeddings, masks) to FP32 and enters `autocast(enabled=False)`; the
prefix forward stops at the perturbed last layer, and the hook and the autocast switch are removed
when `LastLayerSplit.prefix` returns. `k = 31` reproduces the `float32` prefix and `k = 0` the plain
FP16 prefix (CPU tests with bf16 autocast, `tests/test_fp32_tail.py`).

Evidence (`docs/selection-precision.md`, 512 examples x 5 directions against the exact float64
derivative): median relative error of g_i 18.5 % (k = 0), 11.3 % (1), **1.15 % (2)**, 1.09 % (4),
0.31 % (8), 0.16 % (full FP32); sign flips 152, 110, **19**, 15, 2, 2 of 2560. The error comes from the
attention branch of blocks 29 and 30, whose attention logits reach ~1e5 and are rounded in FP16.
The selection overlap of k = 2 with the exact selection is inside the FP32 noise floor. Cost: about
+3.6 % step time (~957 vs ~924 ms, estimated from per-layer prefix timings). A cheaper variant that
keeps only q, k and softmax of blocks 29 and 30 in FP32 is not implemented.
The library-path check of the shipped default is in the section "Library check of the default" at
the end of this file.

## Numerical check

The frozen adapter is the 100-step rank-128, alpha-512 Phi-2 adapter and the
fixed MathInstruct pools from `docs/layer-signal.md` (adapter SHA-256
`81daf515ca1c6e4caba0377c942e112279ac98135b4afb7c315afae655a03971`,
pool SHA-256 `befd7dd8150cbb5e549de543f246e8d8ee90d44c263a5f12bcdacafd050c0e6d`).
Eight examples, packed four at a time, were evaluated at three fixed directions
(seeds 734221–734223) with epsilon 1e-3 and TF32 disabled. Both arms used the
same FP32 last layer, head, and loss. The relative L2 error of their 24
per-example loss differences was **0.1234**; signs agreed in **22/24**.

For selection, six saved pools of 32 examples were used with the recipe's fixed
ZO direction (`zo_random_seed=534895718`), selecting 16 examples per pool. Each
arm maintained its own selector Adam state across the six pools. The FP32
reference and the FP32 packing-order floor used the same weights and examples;
the floor reversed the packed example order and restored the scalar order before
selection. Mean selected-set overlap with FP32 was **15.17/16** for the FP32
floor and **13.0/16** for the FP16 prefix. Per-pool overlaps were 15, 16, 14,
15, 15, 16 (floor) and 12, 14, 10, 14, 14, 14 (FP16 prefix). The difference is
larger than the measured FP32 packing noise. A direct call to the new
`MezoEfficient.extract` path reproduced the diagnostic projected gradients to a
maximum absolute difference of 6e-5; the corresponding FP32 repeat differed
by up to 2e-5.

These checks establish a selection change at this checkpoint; they do not
establish a learning-quality loss. No learning-curve comparison has been run.

### Fixed-seed rerun

On 2026-09-29, the same checkpoint, pools, packed order, epsilon, and seeds
were run again on GPU 2. Against the first run, the 24 loss differences changed
by relative L2 **0.0079%** (FP32 prefix) and **0.0082%** (FP16 prefix); the
largest absolute changes were 1.2e-7 and 2.4e-7. The FP16-vs-FP32 difference
remained **12.34%**. For the three fixed direction seeds 734221–734223, the
per-direction relative L2 differences were 11.31%, 13.47%, and 11.67%.
The full comparison has correlation 0.993. Its two sign changes occur at FP32
loss differences of only 1.0e-5 and 1.8e-5, while the median absolute FP32
loss difference is 3.2e-4.

With the same selection seed and six pools, the two FP32 runs selected a mean
of **14.67/16** common examples per pool. The two FP16-prefix runs selected
**14.83/16** in common. The FP16-prefix vs FP32 overlap was **13.0/16** in the
first run and **13.33/16** in the rerun. Thus a fixed seed makes the ZO
direction and pool repeatable but does not make floating-point execution or
near-tied facility-location choices bitwise repeatable. Changing the seed is
not a correction for the systematic precision difference. This experiment
does not isolate whether the remaining tiny same-precision variation arises
in the model kernels, the selector's tie handling, or both.

The rerun's raw JSON and logs are under
`$COLM_ARTIFACT_ROOT/artifacts/CoLM/seed-recheck-20260929-131110/`.

## One full-step timing run

Physical GPU 2, one rank; `configs/timing_phi2_efficient.json` with
`selection_prefix_dtype=float16`, 130 steps, 10 warm-up steps and 120 measured
steps, fine timing and transfer census in the warm-up only. Other resolved
parameters include pool 32, train 16, micro-batch 4, selection pack budget
1003 tokens, training pack budget 1536 tokens, FP16 AMP training, learning rate
2e-5, and default LoRA rank 128/alpha 512/dropout 0.05. W&B was disabled. The
machine load average was 141–145 during the run. The 50-step sliding mean
deviated by at most 4.15% from the overall measured mean. Internal timing
closed to 1.09% of the wall-clock step.

| Phase | FP16 prefix, ms/step | Earlier stock FP32 selection, ms/step |
|---|---:|---:|
| Full step | 1011 | 1425 |
| Selection | 509 | 967 |
| Prefix | 305 | 845 |
| Train forward/backward | 220 / 256 | 206 / 231 |
| Forwarded / trained tokens | 5863 / 3777 | 6097 / 3780 |

The whole-step ratio is **1.41x** against the earlier stock-attention run, and
the prefix is **2.77x** faster. This is a cross-run comparison: the earlier run
used physical GPU 0 with load average 22–35 and slightly different selected
training examples. The present run's peak was 32.30 GB allocated and 41.73 GB
reserved, so the prefix change did not increase the training memory peak.

The full resolved configuration, step JSONL, summary JSON, trainer state,
memory JSON, and logs are in
`$COLM_ARTIFACT_ROOT/artifacts/CoLM/fp16-prefix-20260929/timing/`. The pairwise
losses and selection sets are in the sibling `precision.json`, `selection.json`,
and their logs. Raw outputs are excluded from Git.

The numerical check can be reproduced from the saved adapter and pool:

```bash
ROOT="$COLM_ARTIFACT_ROOT/artifacts/CoLM/layer-signal-20260928"
python -u scripts/check_prefix_precision.py \
  --config configs/prefix_precision_phi2.json \
  --pool-file "$ROOT/inputs/pools.pkl" \
  --adapter "$ROOT/inputs/adapter_model.safetensors" \
  --out "$COLM_ARTIFACT_ROOT/artifacts/CoLM/fp16-prefix-check.json" \
  --examples 8 --pack-size 4 --directions 3 --seed 734221 --eps 0.001 \
  --selection-pools 6 --check-extractor
```

## Decision

The user first chose the plain FP16 prefix split (k = 0) for the Phi-2 recipe. Selection differed
beyond the FP32 packing noise, which is disclosed above, and the result demonstrated stable
execution and a speed gain, not equal learning quality. After the precision study
(`docs/selection-precision.md`) the user made the FP16 prefix with an FP32 tail of 2 blocks the
default: it removes the measured selection difference at about +3.6 % step time. A paired
learning-curve comparison of k = 2 is still not run.

## Selection pack budget after the FP16 prefix

One 130-step run on GPU 2 set `pack_tokens=1536`, while keeping
`train_max_tokens=1536`. The previous run used the data-derived selection
budget of 1003 tokens. Both used seed 0, the same 120 measured pool token
counts and the same mean 5,863 forwarded selection tokens per step. The new
run was on `efa5d63` (the earlier run on `f77f6c3`); the intervening commit
removed over-context examples from the dataset. Measured steps 11–130:

| Phase | Selection budget 1003 | Selection budget 1536 |
|---|---:|---:|
| Full step, ms | 1011.36 | 885.28 |
| Selection, ms | 508.56 | 413.12 |
| Selection packing, ms | 62.53 | 14.47 |
| Selection prefix, ms | 304.97 | 285.84 |
| Selection suffix, ms | 101.37 | 96.84 |
| Training, ms | 481.87 | 454.72 |
| Peak allocated / reserved, GB | 32.297 / 41.730 | 32.289 / 41.725 |

The 50-step window deviation from each run's overall mean was at most 4.15%
and 4.63% respectively. The timing sections closed to about 1% of step wall
time. The machine load differed substantially between runs; even training,
whose pack budget was unchanged, became 27 ms faster. Therefore 126 ms is
the observed cross-run difference, not a causal estimate of the pack setting
alone. The larger budget ran without an OOM or an increase in the training
memory peak. The Phi-2 profile now uses 1536; `--pack_tokens 0` restores the
data-derived budget. Raw config, step JSONL, memory, and logs are under
`$COLM_ARTIFACT_ROOT/artifacts/CoLM/fp16-select-pack1536-20260929/`.

## Library check of the default (fp32 tail of 2 blocks)

2026-09-29, physical GPU 2, shared: the machine load average was 17-33 and other users' jobs
appeared on the card during the run, so timings below are not exclusive.

Accuracy through the library path (`MezoEfficient` with `selection_prefix_fp32_tail=2`, not the
script-local arm): `scripts/measure_selection_precision.py --pools 8 --directions 3
--library-tail 2` (256 examples x 3 directions = 768 values of g_i, the adapter and pools of
`docs/layer-signal.md`, TF32 off, eps 1e-3, 1536-token packs) against the exact float64
derivative R:

| arm | median rel. err. | p90 rel. err. | sign flips / 768 | Pearson |
|---|---:|---:|---:|---:|
| P (plain fp16 prefix, k = 0) | 20.3 % | 117 % | 44 | 0.9849 |
| library, k = 2 | **1.18 %** | 9.7 % | **4** | 0.9999 |
| F (fp32 prefix) | 0.15 % | 1.1 % | 0 | 1.0000 |

This reproduces the study's k = 2 numbers (1.15 % median, 19 of 2560 = 0.7 % sign flips). The
library arm also agrees with F to 1.16 % (median), 4 sign flips.

One 30-step `colm-train` run with the new Phi-2 default (`configs/timing_phi2_efficient_coarse`
settings with `max_steps=30`, coarse timing, seed 0, W&B off; the resolved config records
`selection_prefix_dtype=float16`, `selection_prefix_fp32_tail=2`, `pack_tokens=1536`) trained
without error: loss 0.79-0.84 at steps 29-30. Steps 11-30 (20 steps, not exclusive, only an
indication): 1031 ms per step, of which selection 447 ms (features 442 ms) and training 464 ms;
6,199 forwarded selection tokens per step on average. Peak allocated memory 32.30 GB in the
training phase (13.0 GB in the selection phase), reserved 41.24 GB: unchanged from the plain FP16
prefix (32.30 / 41.73 GB above). The stage-level cost of the tail (+3.6 % of the step) is the
per-layer estimate of `docs/selection-precision.md`; this run cannot resolve it because the
machine load varied. Raw outputs are under
`$COLM_ARTIFACT_ROOT/artifacts/CoLM/fp32-tail-20260929/`.
