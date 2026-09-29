# FP16 decoder prefix with an FP32 MeZO suffix

2026-09-29, implementation based on `3244c97`. The
requested computation is exactly: run the first 31 Phi-2 decoder layers under
CUDA FP16 autocast, promote their captured floating inputs to FP32, then replay
the perturbed last layer, output head, and per-example cross entropy in FP32.
The base weights and LoRA parameters remain stored in FP32. Selection attention
remains stock SDPA; training attention remains stock FlashAttention-2. The
Phi-2 model profile sets `selection_prefix_dtype=float16` and requires FP32
model weights; `float32` retains the previous prefix computation. Command-line
and JSON settings override the profile, and the resolved configuration records
the chosen mode.

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

The user chose this precision split for the Phi-2 recipe. Selection differs
beyond the FP32 packing noise, which is disclosed above. The result demonstrates
stable execution and a speed gain, not equal
learning quality. A paired learning-curve comparison remains the next check.
