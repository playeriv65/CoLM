# Selection precision (Phi-2, last-layer MeZO)

2026-09-29, code at `e0d8c28`. Question: how accurate must the MeZO selection forward be, does the
current default (P: fp16 decoder prefix, fp32 perturbed last layer, `docs/fp16-prefix.md`) hurt the
selection or the learning compared with full fp32 (F), and how does the upstream all-fp16 autocast
regime (H) behave. The study itself changed no default; its recommendation was then decided by the
user (see "Decision", 2026-09-29: the fp32 tail of 2 blocks is now the Phi-2 default).

Scripts: `scripts/precision_arms.py` (arms), `scripts/measure_selection_precision.py` (g_i),
`scripts/analyze_selection_precision.py` (tables, selection), `scripts/train_precision_arm.py` (arms F, P, P2, H, random) and
`scripts/summarize_precision_runs.py` (paired learning runs), `scripts/measure_hybrid_prefix.py`
(fp32-tail hybrid prefix), `scripts/measure_layer_sensitivity.py` and `scripts/measure_attention_ops.py`
(why the last blocks); CPU tests in
`tests/test_precision_arms.py`; the library option is `selection_prefix_fp32_tail`
(`tests/test_fp32_tail.py`). Raw data (npz, logs, run directories) are outside git under
`$COLM_ARTIFACT_ROOT/artifacts/CoLM/precision-20260929/`.

## Arms

| arm | prefix (layers 0-30) | perturbed last layer, head, loss |
|---|---|---|
| R | fp32 (the F prefix state, promoted to float64) | float64, exact directional derivative d/dt L_i(B + t z) by forward-mode autodiff |
| F | fp32 | fp32 (`selection_prefix_dtype=float32`) |
| P | fp16 autocast, promoted to fp32 | fp32 (`selection_prefix_dtype=float16`, `selection_prefix_fp32_tail=0`; the Phi-2 default until 2026-09-29) |
| H | fp16 autocast | fp16 autocast (the upstream regime; no library switch, patched in by the script) |

Suffix `r`: the pool packed in reverse order (a different pack composition: the packing-noise floor
for F, the run-to-run noise for P and H). `_e2`: eps 1e-2 instead of 1e-3. `Rfd3` / `Rfd2`:
float64 finite difference at eps 1e-3 / 1e-2 on the fp32 state.

Reference. R is not the whole model in float64: the prefix rounding is common to L(B + eps z) and
L(B - eps z), so an fp32 prefix state is enough for a float64 last layer, and the derivative is
exact (no finite-difference cancellation). Validated once (`--validate`, 8 examples, 2 directions)
against an all-float64 finite difference (eps 1e-3 and 1e-4): the largest relative difference of R
is 2.5e-3 (direction 0) and 5e-4 (direction 1) over 8 examples. That is one order of magnitude
below P's error and comparable with F's own fp32 error (below), so R cannot resolve F's error
better than ~0.3 %, and it resolves P and H fully. A float64 finite difference at eps 1e-3 on the
same state agrees with R to 1e-5 (median): the estimator itself is unbiased at this eps.

Data: the 100-step rank-128 / alpha-512 Phi-2 adapter and the 16 fixed MathInstruct pools of 32
examples (512 examples) of `docs/layer-signal.md`, TF32 off, epsilon 1e-3, five directions (the
recipe's ZO seed 534895718 and 734221-734224), 1536-token selection packs (the training default).
The script's F, P and H were checked against the library extractor (`MezoEfficient.extract`, H:
the function patched into it) on the first pack: max |difference| 6e-5 (F), 9e-5 (P), 1.5e-5 (H)
for median |g| of 0.27-0.30. Only the 100-step adapter state was measured (not the initial
lora_B = 0 state).

## Phase 1: g_i against R

512 examples x 5 directions = 2560 values per arm; median |g_R| = 0.1425. "Sign differs": the
number of values whose sign differs from R (the smallest third by |g_R| holds almost all of them).

| arm | Pearson | Spearman | sign differs | median rel. err. | p90 rel. err. | median abs. err. / median abs. g |
|---|---|---|---|---|---|---|
| F | 1.0000 | 1.0000 | 2 / 2560 | 0.0016 | 0.011 | 0.0015 |
| Fr (packing floor) | 1.0000 | 1.0000 | 2 / 2560 | 0.0017 | 0.012 | 0.0016 |
| P | 0.9849 | 0.9791 | 152 / 2560 | 0.185 | 1.24 | 0.186 |
| Pr (P, other packing) | 0.9844 | 0.9784 | 159 / 2560 | 0.185 | 1.27 | 0.192 |
| H | 0.8066 | 0.7617 | 544 / 2560 | 0.72 | 5.14 | 0.767 |
| Hr | 0.7877 | 0.7469 | 551 / 2560 | 0.787 | 5.01 | 0.767 |
| F, eps 1e-2 | 1.0000 | 1.0000 | 0 / 2560 | 0.0012 | 0.0072 | 0.0012 |
| P, eps 1e-2 | 0.9849 | 0.9791 | 153 / 2560 | 0.187 | 1.24 | 0.185 |
| H, eps 1e-2 | 0.9321 | 0.9124 | 309 / 2560 | 0.38 | 2.61 | 0.393 |
| Rfd3 (float64 FD, eps 1e-3) | 1.0000 | 1.0000 | 0 / 2560 | 1.2e-5 | 6.6e-5 | 1.1e-5 |

Between arms: P vs F Pearson 0.985, 154 sign differences, median rel. err. 0.19; Pr vs P Pearson
0.996 (83 sign differences: P's own run-to-run noise is about half of its error against R, the
rest is the systematic fp16 prefix error); Hr vs H Pearson 0.71 (643 sign differences: H is mostly
noise, only partly reproducible).

Reading. The fp32 estimate is accurate to ~0.2 % (median, fp32 rounding of L+ - L- at eps 1e-3),
independent of eps, with 2 sign flips of 2560. Under fp16 in the prefix alone the error is ~19 %
median (6 % sign flips); it does not change with eps (eps 1e-2: the same numbers), consistent
with a systematic error of the prefix state rather than rounding noise of the difference. Under fp16 in the suffix (H) the error is ~72 % at eps 1e-3
(21 % sign flips) and shrinks to 38 % at eps 1e-2 (12 % sign flips): a larger eps helps H (the
signal grows with eps, the fp16 rounding does not) but does not make it usable.

## Selection outcome

The library selector (default recipe: Adam transform, per-source coordinate mask, source-wise l1
facility location, `keep_sources` trained in full) on g_i z, 16 of 32 per pool, 16 pools in order,
five directions (5 x 16 = 80 cells; s.e. over cells ~0.02 but the cells of one direction share
pools). "Chain": each arm carries its own Adam moments across pools, as in training. "Forced": all
arms use R's moments (only g_i differs). On average 8.75 of the 16 are kept sources (trained in
full whatever g_i is) and 7.25 are facility-location picks from 23.25 candidates; the plain overlap
of 16 is inflated by the kept examples, so the overlap of the facility-location picks alone is the
informative column.

| arm vs R | chain overlap of 16 | chain: FL picks shared | forced: FL picks shared |
|---|---|---|---|
| F | 14.79 | 0.833 | 0.838 |
| Fr (F noise floor) | 14.55 | 0.800 | 0.817 |
| P | 13.28 | 0.624 | 0.598 |
| Pr | 13.18 | 0.610 | 0.593 |
| H | 11.96 | 0.443 | 0.448 |
| Hr | 12.20 | 0.476 | 0.460 |
| F, eps 1e-2 | 14.82 | 0.838 | 0.841 |
| P, eps 1e-2 | 13.20 | 0.614 | 0.603 |
| H, eps 1e-2 | 12.30 | 0.490 | 0.490 |
| random, same structure | 11.41 | 0.357 | 0.357 |
| random 16 of 32 (uniform) | 8.00 | - | - |

Direct: P vs F 13.29 of 16 (FL picks 0.626); Fr vs F 14.64 (0.812); Pr vs P 13.84 (0.702); Hr vs H
11.70 (0.407); H vs F 11.85 (0.428).

Reading. Even fp32 is not close to 1.0 (0.83; F and its reversed-packing repeat agree on 0.81):
the facility-location choice among near-tied candidates is decided at rounding level (E1). P moves
the FL picks from 0.83 to 0.62, i.e. 2.7 instead of 1.2 of the 7.25 picks differ from the exact
selection; that is a measurable loss but P is still far above random (0.36). H (0.44-0.49) is
only slightly above random (0.36): the upstream fp16 regime is weakly informative, not random-like,
and eps 1e-2 lifts it only to 0.49. The forced and chain columns agree, so the losses come from g_i
and not from diverging Adam states.

Upstream non-efficient path (fp16 autocast, batch 1, original code, 512 examples of the same kind,
the earlier study `logs/zo-precision-2026-09-28/`, arm A against a float64 finite difference):
Pearson 0.81, sign agreement 0.80, median relative error 0.70, and selection overlap of the
non-kept picks 0.50 against 0.31 for uniform random and 0.41 for a shuffled pipeline
(state S1, 16 pools). That autocast of the whole model is the same regime as H here and the
numbers agree (Pearson 0.81, median relative error 0.72). Not done there: the study was stopped
before its final report; its per-arm tables exist, an aggregate write-up does not.

## Phase 2: learning

300 steps instead of the sweep's 1024 (user's time budget), otherwise the rank sweep's r=128 /
alpha=512 arm (`scripts/train_precision_arm.py` builds the config from
`configs/rank_sweep/sweep.json`: holdout 1000, pool 32, 16 trained, 1536-token packs, lr 2e-5
linear, fp16 AMP, `selection_prefix_dtype` per arm), single GPU 2, W&B off. Held-out (1000
MathInstruct examples) and GSM8K-solution token-pooled loss at steps 100, 200, 300 (the rank
sweep's evaluation callback). One seed per arm plus a second F seed: a first look, not a
significance test. Seed 0 arms see the same pools and the same ZO direction, so F0, P0, H0 and the
random arms differ only in which 16 of the 32 are trained (F1 has another data order, another z and
another LoRA init).

| run | held-out @100 | @200 | @300 | GSM8K @100 | @200 | @300 | train loss, last 100 | step s | peak alloc. GB | loadavg |
|---|---|---|---|---|---|---|---|---|---|---|
| F seed 0 | 0.5683 | 0.5519 | 0.5472 | 0.7855 | 0.7655 | 0.7561 | 0.5854 | 1.452 | 30.5 | 7.7-9.0 |
| F seed 1 | 0.5706 | 0.5546 | 0.5493 | 0.7901 | 0.7677 | 0.7576 | 0.6244 | 1.518 (a) | 32.2 | 22.4-7.6 |
| P seed 0 | 0.5688 | 0.5519 | 0.5472 | 0.7834 | 0.7609 | 0.7552 | 0.5893 | 0.924 | 30.5 | 8.9-8.3 |
| H seed 0 | 0.5689 | 0.5525 | 0.5476 | 0.7864 | 0.7649 | 0.7565 | 0.5881 | 2.099 (b) | 29.9 | 9.7-22.5 |
| random 16 of 32 | 0.5706 | 0.5539 | 0.5491 | 0.7971 | 0.7748 | 0.7636 | 0.6293 | 1.451 (b) | 33.2 | 32.6-9.9 |
| random, kept sources always | 0.5691 | 0.5528 | 0.5488 | 0.7900 | 0.7632 | 0.7588 | 0.5877 | 0.562 (c) | 33.2 | 7.8-10.2 |

(a) the first ~3 minutes overlapped the validation job of Phase 1 on the same GPU; (b) ran
concurrently with the Phase 1 measurement on GPU 2, step time not comparable; (c) no selection
forward at all. Clean step times: F 1.45 s, P 0.92 s (the fp32 vs fp16 prefix, `docs/fp16-prefix.md`
measured 1.43 vs 0.89 s on 1024 steps); H was not timed alone. The train loss is the loss of the
selected examples, so it reflects which examples were picked (kept-source and short examples) as
much as the model and is not a quality measure across selectors; the held-out and GSM8K losses are.

Random baselines, exactly as set (script-local patches of `MezoEfficient.extract` and
`CoresetSelector.__call__` in `scripts/precision_arms.py`, no library change; the MeZO forward is
skipped): `random` chooses 16 of the 32 pool examples uniformly (numpy generator, seed 20260929) and
ignores `keep_sources`; `random, kept sources always` keeps the selector's structure: all
kept-source examples (8.75 of 16 on average) are trained and the other picks are drawn uniformly
inside each source with the selector's per-source quotas, so it differs from F only in the ranking
of the 7.25 facility-location picks. Training compute is the same (16 examples per step).

Differences (loss units), seed 0 unless stated. F seed spread |F0 - F1|: held-out 0.0023 / 0.0027 /
0.0021 and GSM8K 0.0046 / 0.0022 / 0.0015 at steps 100 / 200 / 300.

- P - F0: held-out +0.0005 / 0.0000 / 0.0000, GSM8K -0.0021 / -0.0046 / -0.0009.
- H - F0: held-out +0.0006 / +0.0006 / +0.0004, GSM8K +0.0009 / -0.0006 / +0.0004.
- random - F0: held-out +0.0023 / +0.0020 / +0.0019, GSM8K +0.0116 / +0.0093 / +0.0075.
- random with kept sources - F0: held-out +0.0008 / +0.0009 / +0.0016, GSM8K +0.0045 / -0.0023 /
  +0.0027.

Reading. First, how large the selection effect is at all: F beats the plain random baseline by
0.002 held-out (within the F seed spread) and by 0.008 GSM8K (well above it), but almost all of
that gap disappears when the random baseline is given the selector's structure (kept sources always
trained, per-source quotas): the remaining F-minus-random difference is 0.0016 held-out and 0.0027
GSM8K at step 300, inside the F seed spread. At 300 steps the facility-location ranking of the 7
picks is not distinguishable from a random ranking; what the selector buys here is the composition
(kept sources and quotas), which is not sensitive to precision. That bounds how much selection
precision can matter for learning at this horizon. Second, P and H differ from F0 by at most 0.0005
held-out and 0.0046 GSM8K (one step-100/200 point), inside the F seed spread; H is even marginally
worse only on held-out (+0.0004..0.0006). So the learning curves do not depend on the selection
precision at 300 steps for any of F, P, H, although the selections differ by 40-55 % of the
facility-location picks (previous section). 1024-step runs were not started: the 300-step curves do
not separate beyond the F seed spread. Caveat: one seed per arm plus one extra F seed, 300 steps, a
100-step adapter for the numeric part: a first look, not a significance test; a 1024-step or
multi-seed comparison could still show a small effect of the ranking.


## Hybrid prefix (fp32 tail)

Arm Pk: layers 0 ... 30-k under fp16 autocast, then the last k layers of the prefix (31-k ... 30)
in fp32 without autocast (a forward pre-hook on layer 31-k casts the hidden state, position
embeddings and masks to fp32 and disables autocast until the prefix stops); the perturbed last
layer, head and loss stay fp32. k = 0 is P, k = 31 is F. `scripts/measure_hybrid_prefix.py`, same
adapter, 16 pools and five directions as Phase 1 (R and F arrays reused). Endpoint check on one
pack: the prefix hidden state of k = 0 / k = 31 equals the P / F prefix bitwise (max difference
0.0); g differs by <= 1.2e-4 absolute between two evaluations of identical states (the fp32 noise
floor, median |g| ~ 0.1). Timing: per pack, warm, CUDA-synchronised, one job on GPU 2 (load average
7-12), on the examples the trainer forwards (`CoresetSelector.needed`, 5,750 tokens and 4.6 packs
per step, 8 pools); "selection fwd" = prefix + the two fp32 suffix replays of one direction. The
estimated step adds each k's measured selection-forward delta to the measured P step (0.924 s,
Phase 2, GPU alone); the F row reproduces the measured F step (1.44 vs 1.45 s).

| k (fp32 layers) | sign flips / 2560 | Pearson | median rel. err. | p90 rel. err. | FL picks shared with R (chain) | prefix ms / step | selection fwd ms / step | est. step ms |
|---|---|---|---|---|---|---|---|---|
| 0 (= P) | 152 | 0.9849 | 0.185 | 1.24 | 0.609 | 276 | 368 | 924 |
| 1 | 110 | 0.9934 | 0.113 | 0.81 | 0.650 | 293 | 385 | 941 |
| 2 | 19 | 0.9999 | 0.0115 | 0.098 | 0.788 | 309 | 401 | 957 |
| 4 | 15 | 0.9999 | 0.0109 | 0.092 | 0.805 | 343 | 435 | 991 |
| 8 | 2 | 1.0000 | 0.0031 | 0.022 | 0.791 | 410 | 503 | 1059 |
| 16 | 1 | 1.0000 | 0.0025 | 0.019 | 0.800 | 544 | 637 | 1193 |
| 31 (= F) | 2 | 1.0000 | 0.0016 | 0.011 | 0.821 | 793 | 886 | 1442 |

Median per pack, prefix / whole forward: k = 0 63 / 84 ms, k = 2 71 / 92 ms, k = 8 94 / 116 ms,
k = 31 182 / 206 ms. Each fp32 layer costs ~16.6 ms per step (k = 1: +17 ms). Against P
(P vs Pk), the FL picks shared are 0.58-0.61 for every k >= 1, i.e. all hybrids move away from P
by about as much as F does (P vs F: 0.61).

Verdict: the fp16 error sits in the last prefix layers. Two fp32 layers cut the median error from
18.5 % to 1.2 % and the sign flips from 152 to 19 and bring the FL picks shared with the exact
selection (0.79) to the fp32 packing-noise floor (F 0.83; F repeated in reverse packing 0.80), for
+33 ms per step (+3.6 %); k = 8 gets the error to 0.3 % for +135 ms (+15 %). No k gets the median
error to F's 0.16 % at a small cost, but none is needed to reach F's selection quality: k = 2 does.
Learning was not measured for Pk.

## Why the last two blocks

`scripts/measure_layer_sensitivity.py` and `scripts/measure_attention_ops.py`: 8 pools (256
examples) x 3 directions = 768 values per row, error against R, everything fp32 except the named
place (fp16 autocast, or values rounded to fp16 and cast back with the arithmetic in fp32). The
all-fp32 row is 0.14 % median error, the all-fp16 (P) row 20.3 % (44 sign flips).

Residual stream. Under the P setting the block outputs are fp32 (attention and MLP branch outputs
are fp16, and `fp16 + fp16 + fp32` promotes), so the residual stream is not stored in fp16; the fp16
error enters through the branches only.

One block in fp16, the rest fp32 (median rel. err. / sign flips of 768; F floor 0.14 % / 0):
blocks 0-24 and 28: 0.16-0.19 % / 0-1 (nothing); block 25: 0.28 % / 2; 26: 0.94 % / 5; 27: 0.32 % /
1; **29: 12.2 % / 26; 30: 14.8 % / 42**. The reverse (one block in fp32, the rest fp16): fp32 in any
single block j <= 28 leaves 18-21 % (nothing is fixed), fp32 in block 30 alone leaves 12.2 % (block
29 still fp16) and in block 29 alone 15.9 % (block 30 still fp16). Three consecutive fp32 blocks
help only when they cover both: window 28-30 gives 1.1 %, window 27-29 15.9 %, windows 0-2 ... 26-28
19-20 %. The two
errors add roughly in quadrature (relative L2 0.118 and 0.139 alone, 0.182 combined, 0.173 for all
of P). The k = 2 hybrid above (1.15 %) is exactly "blocks 29 and 30 in fp32"; what remains is the
0.3-0.9 % of blocks 25-27.

Which branch and which operation (fp32 prefix except the named item in fp16):

| block | attention branch | MLP branch | q,k outputs | v output | out. projection | LayerNorm output | sdpa q,k,v |
|---|---|---|---|---|---|---|---|
| 30 | 14.8 % / 40 | 0.37 % / 0 | 3.2 % / 10 | 0.14 % / 1 | 0.17 % / 0 | 12.9 % / 32 | 3.5 % / 11 |
| 29 | 12.0 % / 26 | 0.19 % / 0 | 4.6 % / 13 | 0.15 % / 0 | 0.15 % / 0 | 4.2 % / 12 | 4.7 % / 14 |
| 28 | 0.17 % / 0 | 0.18 % / 0 | 0.15 % / 0 | 0.15 % / 1 | 0.16 % / 0 | 0.15 % / 1 | 0.17 % / 0 |
| 20 | 0.18 % / 1 | 0.19 % / 0 | 0.16 % / 0 | 0.17 % / 1 | 0.17 % / 0 | 0.16 % / 0 | 0.15 % / 0 |

So it is the attention branch of blocks 29 and 30, and inside it the query/key -> logits ->
softmax path (rounding q and k, the LayerNorm output that feeds them, or running the kernel in
fp16); the value and output projections and every MLP are harmless. The error of the residual stream
against the fp32 run is 3-5e-4 (relative L2) through block 25, 3.5e-3 after block 26, 6.2e-2 after
block 29 and 9.1e-2 after block 30 (largest element difference 51).

Attention logits per block (first pack of the run, fp32; Δ = change when q and k are rounded to
fp16 before the product; TV = mean total-variation change of the softmax rows):

| block | fp16-only error (median) | max abs. logit | max abs. Δ logit | TV change of attention rows | max abs. residual element / median |
|---|---|---|---|---|---|
| 0 | 0.17 % | 42 | 0.010 | 0.0002 | 26 / 0.17 |
| 10 | 0.17 % | 27 | 0.015 | 0.0003 | 1406 / 0.62 |
| 14 | 0.17 % | 598 | 0.090 | 0.0004 | 1413 / 0.73 |
| 20 | 0.19 % | 717 | 0.105 | 0.0004 | 1416 / 0.89 |
| 25 | 0.28 % | 902 | 0.23 | 0.0006 | 1410 / 1.18 |
| 26 | 0.94 % | 5,547 | 0.97 | 0.0009 | 1397 / 1.26 |
| 27 | 0.32 % | 938 | 0.14 | 0.0007 | 1365 / 1.37 |
| 28 | 0.18 % | 203 | 0.039 | 0.0002 | 1281 / 1.53 |
| 29 | 12.2 % | 241,086 | 35.6 | 0.125 | 1049 / 1.61 |
| 30 | 14.8 % | 126,641 | 20.6 | 0.078 | 712 / 1.71 |

Input-noise sensitivity. A fp32 relative Gaussian perturbation of 1e-3 (elementwise, or scaled by
the token norm) of the hidden state at the input of block 31 / 30 / 29 changes g by a median 0.55 % /
0.25 % / 0.28 % (elementwise) and 0.64 % / 0.48 % / 0.60 % (token-scaled): an amplification of only
2.5-6.4, with at most 3 sign flips of 768. g_i is not a strong amplifier of noise in the residual
stream; fp16 storage of the residual (relative 5e-4) would cost ~0.3 %.

What the data supports: the fp16 error of the selection is one localised effect, not a
distributed rounding error: it comes from the attention logits of blocks 29 and 30, which are three
to five orders of magnitude larger than elsewhere (max |q k / sqrt(d)| 1.3e5-2.4e5 against <= 900
in blocks 0-25, 5.5e3 in block 26; q up to 644 and k up to 174). A relative rounding of 5e-4 of q
and k then moves a logit by up to 36 (rms 0.6-2.2) instead of <= 1 and changes the softmax rows by
8-12 % in total variation instead of <= 0.1 %; the per-block fp16 error rises with the logit size
(block 26 in between). The residual stream is fp32, and the noise test shows that a generic 1e-3
error of the hidden state is not amplified. The MLPs, v and the output projection, which see no
such magnitudes, are insensitive. Not the LayerNorm-plus-residual structure as such: the residual
stream carries massive activations (max element ~1400 against a median of 0.2-1.7, from block 1
on) in every block, yet blocks 1-24 are insensitive.

What is only a hypothesis: that the logits are large because of the massive activations (why
exactly blocks 29 and 30 develop q, k of that size was not investigated), and that the change in the
attention rows acts through a few near-tied keys (only the mean TV change was measured, not the
number of flipped argmaxes). The logit statistics of block 30 also differ between the first and
later identical fp32 forwards of a fresh process (max |logit| 7.7e4 against 1.3e5; bitwise
identical residual streams from the second forward on, checked block by block): that block is
numerically fragile even in fp32, the cause was not found, and its effect on g is at most within
the fp32 floor measured in Phase 1 (F vs F with reversed packing: 0.23 % median). The picture is
consistent but rests on one adapter state and 256 examples; the fp32-attention-only variant (q, k,
softmax of blocks 29 and 30 in fp32, everything else fp16) that this suggests was not run.

## Decision (2026-09-29: k = 2 chosen and made the default)

The user chose the hybrid prefix with an fp32 tail of **k = 2** blocks as the Phi-2 default
(`selection_prefix_fp32_tail=2`, next to `selection_prefix_dtype=float16`; `docs/fp16-prefix.md`).
Basis, from the tables above (512 examples x 5 directions against the exact reference R):

| k (fp32 prefix blocks) | median rel. err. of g_i | sign flips / 2560 | est. step ms |
|---|---|---|---|
| 0 (P, the former default) | 18.5 % | 152 | 924 |
| 1 | 11.3 % | 110 | 941 |
| **2 (new default)** | **1.15 %** | **19** | **957** |
| 4 | 1.09 % | 15 | 991 |
| 8 | 0.31 % | 2 | 1059 |
| 31 (F) | 0.16 % | 2 | 1442 |

The selection overlap of k = 2 with the exact selection is inside the fp32 noise floor, for +3.6 %
step time. The remaining cases and the findings that stay as recommended:

1. P (k = 0) has median error 19 % and 6 % sign flips, and its facility-location picks agree with
   the exact selection in 0.62 of the picks against 0.83 for F (noise floor 0.80); 300-step
   learning (held-out and GSM8K loss) of P and F differ by at most 0.0005 / 0.0046, inside the F
   seed-to-seed spread, and F is itself not distinguishable from a random ranking with the
   selector's structure at that horizon. Learning for k = 2 was not measured.
2. The error is localised in the q/k/softmax path of the attention of blocks 29 and 30
   (`Why the last two blocks`), so an fp32 attention-only tail there would be cheaper still (not
   implemented; not part of the default).
3. H (upstream fp16 autocast everywhere) is weakly informative (FL picks 0.44-0.49 against 0.36
   random, 21 % sign flips, run-to-run Pearson 0.71) but learns as well as F at 300 steps here; it
   is not a safe basis for selection studies. A larger eps helps H (error 72 % -> 38 %) but not P.
4. eps: fp32 suffix: 1e-3 is enough (0.16 % error, unchanged at 1e-2). The precision that matters
   is the prefix state entering the perturbed layer, not eps.

Unmeasured: learning at 1024 steps or with several seeds (a small ranking effect is not
excluded); learning for Pk (including the default k = 2) and for eps 1e-2; the initial lora_B = 0 state (only the 100-step
adapter was measured); H's step time on a clean GPU; other models (Llama/Qwen keep fp32 prefixes);
an aggregate write-up of the upstream non-efficient path (only the per-arm tables of the earlier
study exist).

## Reproduce

```bash
ROOT="$COLM_ARTIFACT_ROOT/artifacts/CoLM/layer-signal-20260928"
OUT="$COLM_ARTIFACT_ROOT/artifacts/CoLM/precision-DATE"
CUDA_VISIBLE_DEVICES=<gpu> python -u scripts/measure_selection_precision.py \
  --config configs/prefix_precision_phi2.json --pool-file "$ROOT/inputs/pools.pkl" \
  --adapter "$ROOT/inputs/adapter_model.safetensors" --out-dir "$OUT/measure"   # ~28 min
CUDA_VISIBLE_DEVICES="" python scripts/analyze_selection_precision.py "$OUT/measure/g.npz"
CUDA_VISIBLE_DEVICES=<gpu> python -u scripts/train_precision_arm.py --arm {F,P,H,random} \
  --seed 0 --steps 300 --eval-steps 100 200 300 --out-root "$OUT/train"
python scripts/summarize_precision_runs.py "$OUT"/train/*
CUDA_VISIBLE_DEVICES=<gpu> python -u scripts/measure_hybrid_prefix.py \
  --config configs/prefix_precision_phi2.json --pool-file "$ROOT/inputs/pools.pkl" \
  --adapter "$ROOT/inputs/adapter_model.safetensors" --phase1 "$OUT/measure/g.npz" \
  --out-dir "$OUT/hybrid"                                                       # ~5 min
python scripts/analyze_selection_precision.py "$OUT/hybrid/g.npz" --arms F,P0,P1,P2,P4,P8,P16,P31 \
  --pairs P1:P0,P2:P0,P4:P0,P8:P0,P16:P0,P31:P0,P31:F --out "$OUT/hybrid/analysis.json"
python scripts/measure_hybrid_prefix.py --table "$OUT/hybrid/analysis.json" "$OUT/hybrid/timing.json"
```

The library option (`selection_prefix_fp32_tail`, the shipped default k = 2) is checked through
`MezoEfficient` with `measure_selection_precision.py --library-tail 2` (adds arm `L`); result in
`docs/fp16-prefix.md`, section "Library check of the default".
