# Training attention precision (Phi-2)

2026-09-29. **Decision: `train_fp32_tail` (Phi-2 profile: 3), implemented in the library** (see
"Library option" at the end); the measurements below were taken with the script-local prototype at
`4fffb2c`. Question: the LoRA gradients of the training
forward (fp16 autocast, `flash_attention_2` hub kernel on packed rows) are far from the exact fp32
gradient (`docs/errors.md`, "fp16 attention"). The selection forward owes its error to the attention
of blocks 29 and 30 (`docs/selection-precision.md`, "Why the last two blocks"). Is it the same
place in training, and how cheaply can it be removed?

Script: `scripts/diagnostics/measure_training_attention.py` (variants, per-layer dispatch that lives
only in the script, timing); the learning check is `scripts/diagnostics/train_precision_arm.py
--train-attention`. Raw data: `$COLM_ARTIFACT_ROOT/artifacts/CoLM/train-precision-20260929/`.

## Protocol

phi-2 fp32 weights + the saved LoRA r=128 / alpha=512 adapter (`layer-signal-20260928/inputs`),
dropout off, 6 real packs of 1134-1458 tokens (two steps of 3 packs), the trainer's loss (token
cross-entropy over the labels of the step, times the fp16 loss scale 65536), gradient of all 167.8 M
LoRA parameters. Reference: fp32, no autocast, exact `sdpa_kernel(MATH)` attention (it differs by
3.0e-3 from stock sdpa MATH on packs of one example). Error = relative L2 of the gradient
(cosine in brackets); "pack" is the mean over the 6 packs, "step" the mean over the 2 steps (the sum
of the 3 packs of a step). Noise of V0 (same input twice / examples of every pack reversed):
1.004 / 1.009 / 1.005, i.e. the run-to-run noise is 0.5 %, every difference below is far above it.
GPU 2 (RTX PRO 6000), fp16 autocast, `attn`: attention in fp32 (q, k, v cast, exact MATH with the
block-diagonal mask); `q,k`: q_proj and k_proj (with their LoRA) in fp32; `block`: autocast off for
the whole block.

## Gradient error

| variant | fp32 in blocks | pack rel. L2 (cosine) | step rel. L2 (cosine) |
|---|---|---|---|
| V0 current training | none | 1.004 (0.654) | 1.04 (0.635) |
| V1 attn | 29-30 | 0.647 (0.801) | 0.628 (0.812) |
| V1q attn + q,k | 29-30 | 0.287 (0.958) | 0.295 (0.957) |
| V4 whole block | 29-30 | 0.266 (0.965) | 0.266 (0.965) |
| V2 attn | 26-30 | 0.681 (0.789) | 0.640 (0.807) |
| V2q attn + q,k | 26-30 | 0.267 (0.965) | 0.264 (0.965) |
| V3 attn | all | 0.648 (0.800) | 0.625 (0.813) |
| V5 attn | 29-31 | 0.609 (0.821) | 0.602 (0.826) |
| V5k q,k only (flash stays fp16) | 29-31 | 1.12 | 1.08 |
| **V5q attn + q,k** | **29-31** | **0.040 (0.999)** | **0.038 (0.999)** |
| V8q attn + q,k | 28-31 | 0.041 | 0.038 |
| V6q attn + q,k | 26-31 | 0.011 (1.000) | 0.012 (1.000) |
| V3q attn + q,k | all | 0.009 (1.000) | 0.009 (1.000) |
| V0 without loss scale (scale 1) | none | 69 (0.017) | 78 |
| V0 with loss scale 1024 | none | 1.012 (0.650) | 1.05 |

Single block (q,k + attention fp32 in block k only, all else V0), pack rel. L2: S24 1.17, S25 1.04,
S26 1.07, S27 1.05, S28 1.10, S29 0.63, S30 0.87, S31 0.94 (V0: 1.00).

Per parameter group, step mean, relative L2 (q_proj, k_proj, v_proj, fc1, fc2; blocks 0-25 / 26-28 /
29-30 / 31):

| variant | q | k | v | fc1 | fc2 |
|---|---|---|---|---|---|
| V0 | 1.08 / 1.59 / 1.58 / 0.93 | 1.13 / 1.35 / 1.48 / 1.18 | 0.97 / 1.10 / 0.52 / 0.41 | 1.06 / 1.40 / 0.39 / 0.13 | 0.99 / 0.75 / 0.27 / 0.11 |
| V1q | 0.18 / 0.22 / 0.64 / 0.84 | 0.19 / 0.34 / 0.40 / 1.34 | 0.17 / 0.19 / 0.17 / 0.35 | 0.18 / 0.22 / 0.29 / 0.02 | 0.16 / 0.16 / 0.23 / 0.02 |
| V5q | 0.02 / 0.03 / 0.06 / 0.04 | 0.03 / 0.19 / 0.06 / 0.05 | 0.02 / 0.03 / 0.02 / 0.02 | 0.02 / 0.03 / 0.02 / 0.01 | 0.02 / 0.02 / 0.01 / 0.01 |
| V6q | 0.01 / 0.01 / 0.02 / 0.02 | 0.01 / 0.01 / 0.02 / 0.02 | 0.01 / 0.01 / 0.01 / 0.01 | 0.01 / 0.01 / 0.01 / 0.00 | 0.01 / 0.01 / 0.01 / 0.00 |

## Cost

Median warm, cuda-synchronised forward + backward of one 1458-token pack (10 repeats, 3 warm-up,
one job on the card, GPU 2), peak allocated memory; the step estimate uses 3 packs per step and the
measured step of 0.83-0.92 s (P2 arm, selection 0.40-0.45 s of it).

| variant | ms / pack | vs V0 | peak GiB | est. step change |
|---|---|---|---|---|
| V0 | 140.2 | | 26.5 | |
| V1 (attn only) | 148.6 | +6.0 % | 27.0 | +25 ms |
| V1q | 151.1 | +7.8 % | 27.0 | +33 ms (+3.6 %) |
| V4 (blocks 29-30 fp32) | 163.4 | +16.6 % | 26.8 | +70 ms |
| V5k (q,k only; second run: V0 141.0, V5k 144.8) | 144.8 | +2.7 % | 26.4 | +11 ms |
| **V5q** | **157.8** | **+12.6 %** | **27.2** | **+53 ms (+5.8 %)** |
| V6q | 174.3 | +24 % | 27.9 | +102 ms (+11 %) |
| V3q (attention checkpointed to fit) | 373.6 | +166 % | 26.1 | not a candidate |

An fp32 MATH attention costs about 4.2 ms per block and pack (dense L x L mask, no varlen skip),
the fp32 q,k projections 1.3 ms; the timings ran alone on the card (load average 20-40 from other
users' jobs on the host).

## Learning check (300 steps, one seed each)

Rank-sweep recipe as in `train_precision_arm.py`, arm P2 (the Phi-2 default selection), seed 0,
`--train-attention V0` against `V5q` (2898 fp32 attention calls counted in the V5q run, i.e. the
dispatch was active; evaluation keeps the stock path). Held-out / GSM8K loss:

| step | V0 held-out | V5q held-out | diff | V0 GSM8K | V5q GSM8K | diff |
|---|---|---|---|---|---|---|
| 100 | 0.5696 | 0.5621 | -0.0076 | 0.7802 | 0.7760 | -0.0042 |
| 200 | 0.5515 | 0.5461 | -0.0054 | 0.7660 | 0.7544 | -0.0116 |
| 300 | 0.5473 | 0.5418 | -0.0055 | 0.7574 | 0.7483 | -0.0091 |

The earlier F seed-to-seed spread is 0.0023 / 0.0027 / 0.0021 (held-out) and 0.0046 / 0.0022 /
0.0015 (GSM8K) at 100 / 200 / 300, so all six differences point the same way and are 2-3x (held-out)
and 2-6x (GSM8K, except step 100) that spread. It is one seed pair, so "V5q is better" is indicated,
not established; the last-100 train loss is identical (0.5793 / 0.5792). Not extended. Step times of
the two runs (2.07 / 1.58 s) are not comparable: the card was shared with other jobs, use the
isolated timings above.

Second seed (seed 1, same recipe, `--train-attention V0` / `V5q`, 300 steps; raw data
`$COLM_ARTIFACT_ROOT/artifacts/CoLM/train-precision-seed1-20260929/`):

| step | V0 held-out | V5q held-out | diff | V0 GSM8K | V5q GSM8K | diff |
|---|---|---|---|---|---|---|
| 100 | 0.5713 | 0.5622 | -0.0091 | 0.7890 | 0.7759 | -0.0131 |
| 200 | 0.5551 | 0.5466 | -0.0085 | 0.7685 | 0.7559 | -0.0126 |
| 300 | 0.5493 | 0.5410 | -0.0083 | 0.7613 | 0.7502 | -0.0111 |

All twelve differences (two seeds x three steps x two sets) favour V5q; they are 2-4x the F
seed-to-seed spread. Two seeds are still not a significance test, but the direction, the size and
the mechanism (a gradient that is 0.04 instead of 1.0 from the exact one) agree, which is why the
option became a profile default.

## Verdict

- The error sits in the last three blocks (29, 30 and 31) and in the q/k path: fp32 attention alone
  (V1, V2, V3, V5) leaves 0.60-0.65, because autocast rounds q and k in the projections before the
  attention; q,k + attention in fp32 in blocks 29-31 (V5q) gives 0.04 (cosine 0.999), 26-31 0.011.
- The 1.0 error of the LoRA groups of blocks 0-25 is the corrupted backward signal of the last
  blocks (0.02-0.03 with V5q); MLPs are harmless (V4 = V1q); block 28 adds nothing.
- V5q costs +12.6 % of a pack's forward + backward, about +53 ms (+5.8 %) per step, +0.7 GiB.
- Learning (two seeds, 300 steps): V5q lower by 0.005-0.009 held-out and 0.004-0.013 GSM8K at every
  step, 2-4x the F spread, in all 12 comparisons: indicated, decided as the Phi-2 default.
- V0 error here is 1.0 (cosine 0.65) against 0.40-0.53 in `errors.md` (other packs, the trained
  adapter of `layer-signal-20260928`); without the fp16 loss scale the gradient is garbage (69).

## Library option `train_fp32_tail` (implemented)

- Option `train_fp32_tail: int = 0` (`TrainingArguments`, profile key `train_fp32_tail`): the last k
  blocks of the TRAINING forward run q_proj / k_proj (with their LoRA) and the attention in fp32.
  Profiles: phi 3, llama / default 0. A profile value applies only under mixed precision over fp32
  weights (`--precision fp32`, or bf16 weights, get 0); an explicit CLI / JSON value wins, and
  asking for it without mixed precision, with k outside [0, number of layers] or with
  q/k weights below fp32 fails at load. It is in `resolved_config.json` (`training` and `derived`),
  the step-timing metadata, and `colm-sweep create --dry-run` prints the resolved value per arm.
- Where: `colm/train/precision.py`, `TrainingPrecision(model, k)`.
  - `installed(key)` (a context manager around `Trainer.train`, key = the training attention
    implementation, e.g. `kernels-community/flash-attn2`) registers a dispatcher for that key in
    transformers' public `AttentionInterface` and wraps the `forward` of the q_proj / k_proj of the
    tail. `set_attn_implementation` is untouched: the selection's `sdpa` is another key, and the
    trainer's switching between the two works as before. The dispatcher sends a call to fp32 only for
    a tail block (`module.layer_idx`) and only when `cu_seq_lens_q` is present (packed rows): q, k, v
    to fp32, `sdpa` on the MATH backend with the block-diagonal causal mask built from
    `cu_seq_lens_q`; everything else goes to the wrapped function unchanged. Both are removed when
    `train()` returns, also on an error.
  - `running()` turns them on around the forward + backward of a training step
    (`CoresetTrainer._train_packs`, `_Trainer.training_step` for the baseline); outside it (selection,
    in-training and standalone evaluation, padded batches) nothing differs from k = 0.
  - Frozen weights: `frozen_base_low_precision` stores the frozen Linears of the fp16 layers in fp16;
    `store_frozen_linears(..., keep_qk_last=k)` keeps the q_proj / k_proj base weights and biases of the
    last k layers fp32 (the other Linears of those layers follow the existing `keep_last` rule: for the
    Phi-2 defaults the selection already keeps the last 3 layers fully fp32, so nothing extra is held;
    with k above that only q/k are added, 26 MB per layer more than in fp16). `check_tail` verifies the dtypes at load
    (a bf16-weight model or a stored q/k weight is an error, not a silent fp16 round trip).
- Cost with k = 3: +12 % of a pack's forward + backward, about +6 % of the step, +0.7 GiB.

Verification (GPU 2, RTX PRO 6000, library path; `measure_training_attention.py` variant `LIB3`,
the same 6 packs and protocol as above; raw data `$COLM_ARTIFACT_ROOT/artifacts/CoLM/train-fp32-tail-20260929/`):

| variant | pack rel. L2 (cosine) | step rel. L2 (cosine) | ms / pack | peak GiB |
|---|---|---|---|---|
| V0 (`train_fp32_tail=0`) | 1.011 (0.651) | 1.044 (0.633) | 141.0 | 26.5 |
| V5q (script prototype) | 0.0404 (0.999) | 0.0378 (0.999) | 157.3 | 27.2 |
| LIB3 (`train_fp32_tail=3`) | 0.0404 (0.999) | 0.0378 (0.999) | 157.6 | 27.2 |

The library path reproduces the prototype's gradient to the printed digits (and its timing to 0.2 %).

`colm.train.train` end to end (the rank-sweep r=128 / alpha=512 recipe through
`train_precision_arm.py --arm P2`, which passes no `train_fp32_tail`, so the Phi-2 profile's 3 applies;
GPU 2 alone, host load average 12-17):

- 30 steps, in-training evaluation at steps 15 and 30 (padded batches, no `cu_seq_lens_q`: the
  dispatcher was not involved and the run did not crash as the prototype did): trains; held-out /
  GSM8K loss 0.6811 / 0.9621 at step 15 and 0.6535 / 0.9023 at step 30; median step 0.834 s
  (selection 0.37 s), peak allocated 23.5 GB, peak reserved 30.8 GB. The log has "q/k projections of
  the last 3 layers fp32" (weights) and the dispatcher line; `resolved_config.json` records
  `train_fp32_tail: 3` (`training` and `derived`).
- Step time A/B, 100 steps of the same recipe and seed, `--train-fp32-tail 0` against the default 3
  (mean of steps 11-99, evaluation steps excluded): 0.804 s against 0.848 s, **+44 ms (+5.5 %)** per
  step; the training part of the step (step minus selection) 0.429 s against 0.472 s (+10 %);
  selection 0.375 s in both. Peak allocated memory over the same steps 21.5 GB against 23.1 GB
  (the peak follows the longest pack of the step; the pack timing above gives +0.7 GiB on one pack).
  One run per arm, single-step noise is in the means.
- 300 steps, seed 0, evaluation at 100 / 200 / 300 against the prototype's V5q run above (the
  first table; both runs are seed 0 with the same selection seed, but fp16 nondeterminism makes them
  different trajectories) and against V0:

| step | set | library (tail 3) | prototype V5q | library - V5q | V0 | library - V0 |
|---|---|---|---|---|---|---|
| 100 | held-out | 0.5602 | 0.5621 | -0.0019 | 0.5696 | -0.0094 |
| 200 | held-out | 0.5452 | 0.5461 | -0.0009 | 0.5515 | -0.0063 |
| 300 | held-out | 0.5406 | 0.5418 | -0.0012 | 0.5473 | -0.0067 |
| 100 | GSM8K | 0.7758 | 0.7760 | -0.0002 | 0.7802 | -0.0044 |
| 200 | GSM8K | 0.7559 | 0.7544 | +0.0015 | 0.7660 | -0.0101 |
| 300 | GSM8K | 0.7486 | 0.7483 | +0.0003 | 0.7574 | -0.0088 |

  The library run agrees with the prototype to at most 0.0019 (held-out) and 0.0015 (GSM8K), below
  the seed-to-seed spread (0.002-0.005), and sits below V0 by 0.006-0.009 held-out and 0.004-0.010
  GSM8K at every step, the same effect as in the two-seed comparison. Mean step 0.844 s over the
  run (evaluation steps excluded), training peak 23.05 GB, reserved 33.3 GB. Raw logs, `eval_loss.jsonl`
  and `trainer_state.json` of the three runs: `$COLM_ARTIFACT_ROOT/artifacts/CoLM/train-fp32-tail-20260929/`
  (checkpoints deleted).

CPU tests (`tests/test_train_precision.py`, bf16 autocast standing in for fp16): with no autocast and the
tail on every layer the dispatcher reproduces the stock fp32 logits (1e-5) and LoRA gradient (1e-4); with
autocast the tail-wrapped q/k projections return fp32 that equals the unwrapped module bit for bit and
only for the tail blocks; `train_fp32_tail=0` registers nothing; the dispatcher falls back without
`cu_seq_lens_q`; outside `running()` (evaluation, selection) the logits equal the stock path bit for
bit; the registry entry and the `forward` overrides are restored after `installed()`, also after an
error; the coreset selection forward makes no fp32 attention call while every training forward does;
option validation and profile defaults; frozen-weight dtypes.
