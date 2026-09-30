# Training attention precision (Phi-2)

2026-09-29, code at `4fffb2c` (no library change). Question: the LoRA gradients of the training
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

## Verdict

- The error sits in the last three blocks (29, 30 and 31) and in the q/k path: fp32 attention alone
  (V1, V2, V3, V5) leaves 0.60-0.65, because autocast rounds q and k in the projections before the
  attention; q,k + attention in fp32 in blocks 29-31 (V5q) gives 0.04 (cosine 0.999), 26-31 0.011.
- The 1.0 error of the LoRA groups of blocks 0-25 is the corrupted backward signal of the last
  blocks (0.02-0.03 with V5q); MLPs are harmless (V4 = V1q); block 28 adds nothing.
- V5q costs +12.6 % of a pack's forward + backward, about +53 ms (+5.8 %) per step, +0.7 GiB.
- Learning (one seed, 300 steps): V5q lower by 0.005-0.008 held-out and 0.004-0.012 GSM8K, 2-3x the
  F spread, consistently: indicated, not established; confirm with a second seed before a default.
- V0 error here is 1.0 (cosine 0.65) against 0.40-0.53 in `errors.md` (other packs, the trained
  adapter of `layer-signal-20260928`); without the fp16 loss scale the gradient is garbage (69).

## Proposal (not merged)

- Option `train_fp32_tail: int = 0` (`TrainingArguments`, profile key `train_fp32_tail`; phi
  profile 3 after a second seed of the learning check): the last k blocks of the TRAINING forward run
  q_proj / k_proj (with LoRA) and the attention in fp32.
- Where: a new `colm/train/precision.py`. It wraps the attention function of the model
  (`AttentionInterface`, the key of `config._attn_implementation`) with a dispatcher on
  `module.layer_idx` (blocks >= 32 - k: q, k, v to fp32, `sdpa` MATH with the block mask built from
  `cu_seq_lens_q`; the others: the hub kernel unchanged) and wraps `forward` of the q_proj / k_proj
  of those blocks in `torch.autocast(enabled=False)`. `scripts/diagnostics/measure_training_attention.py`
  (`install_dispatch`, `install_precision_hooks`, `block_mask`) is the prototype (about 60 lines);
  the selection forward (sdpa, own fp32 tail) is untouched.
- Cost with k = 3 (V5q): +12.6 % of a pack's forward + backward, +53 ms of a 0.9 s step (5.8 %), +0.7
  GiB peak. Cheaper variants exist only with less accuracy (V1q: +3.6 % of the step, error 0.29).
  A varlen fp32 attention (no dense mask, no MATH) would cut the 4.2 ms per block; not explored.
