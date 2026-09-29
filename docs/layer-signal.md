# ZO loss difference versus the number of perturbed layers

2026-09-28 diagnostic on phi-2. This experiment asks whether expanding a fixed
last-layer `v_proj.lora_B` perturbation to the last 2, 4, 8, 16, or 32 layers
increases the per-example finite-difference signal. It does not alter the CoLM
selection recipe, which perturbs the last layer only.

## Protocol

- Base: `microsoft/phi-2`, fp32 weights, LoRA rank 128, alpha 512, target
  modules `q_proj`, `k_proj`, `v_proj`, `fc1`, `fc2`. The same adapter trained for
  100 steps is loaded in both runs. Evaluation mode disables dropout (configured
  LoRA dropout is 0.05). Attention is stock SDPA. TF32 is disabled.
- Eight fixed MathInstruct examples, 19-184 supervised tokens each; 3 random
  directions (seeds 734221-734223). The random tensor of each layer is reused
  in every cumulative layer count. Each layer's perturbation is `±1e-3 z` in
  its fp32 LoRA B parameter, and the loss is the mean over that example's
  supervised tokens. The comparison run uses fp16 autocast for the forward but
  retains fp32 parameter storage and fp32 cross-entropy input.
- Physical GPU 0 was empty before launch and after completion. Both runs
  completed normally in the `CoLM/CoLM-layer-signal` tmux window, with no
  continuous GPU sampler. Code base before this diagnostic: `3244c97`.
- The adapter SHA-256 is
  `81daf515ca1c6e4caba0377c942e112279ac98135b4afb7c315afae655a03971`;
  the saved sample pool SHA-256 is
  `befd7dd8150cbb5e549de543f246e8d8ee90d44c263a5f12bcdacafd050c0e6d`.
  The inputs, resolved configurations, per-pair losses, full logs, results,
  comparison JSON, and a copy of this script are stored under
  `$COLM_ARTIFACT_ROOT/artifacts/CoLM/layer-signal-20260928/`. Raw output is
  not tracked by Git.

## Results

All loss differences below are per-example `L(theta + eps*z) - L(theta - eps*z)`;
each row pools 8 examples × 3 directions. `Relative L2 error` compares the
fp16 difference vector to the fp32 one; `sign` is their sign agreement.

| Last layers | fp32 median absolute difference | fp32 RMS difference | fp32 RMS / sqrt(layers) | fp16 relative L2 error | fp16 sign |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000398 | 0.000645 | 0.000645 | 0.80 | 20/24 |
| 2 | 0.000747 | 0.001131 | 0.000800 | 1.06 | 20/24 |
| 4 | 0.000705 | 0.001482 | 0.000741 | 2.67 | 13/24 |
| 8 | 0.001280 | 0.001462 | 0.000517 | 3.38 | 11/24 |
| 16 | 0.002216 | 0.002715 | 0.000679 | 1.96 | 20/24 |
| 32 | 0.002422 | 0.004884 | 0.000863 | 0.92 | 17/24 |

The absolute fp32 signal is larger over 32 layers than over one (7.6x RMS),
supporting the proposed direction at this checkpoint. It is not monotonic in
every intermediate row or individual direction. The perturbation norm also
grows with the square root of the number of layers: after division by
`sqrt(layers)`, the RMS values are within a narrower 0.00052-0.00086 range.
Thus the observed growth does not show an intrinsic gain from earlier layers
at fixed total perturbation norm.

The larger fp16 loss difference is not necessarily better signal. Its error
relative to fp32 is still substantial at 32 layers, and it changes the sign of
7 of 24 per-example directional derivatives. Perturbing more layers does not
establish that fp16 can replace the current last-layer fp32 selection. A
separate fp16-prefix/fp32-suffix test remains needed; it is not represented by
these all-layer perturbations.

These are 8 examples at one adapter checkpoint, with three directions. There
is no learning-curve, selection-overlap, or performance claim from this short
diagnostic.

## Reproduce

Run `scripts/measure_layer_signal.py` with `configs/layer_signal_phi2.json`,
the saved `inputs/pools.pkl` and `inputs/adapter_model.safetensors`, and the
listed sample count, epsilon, direction seed, and layer counts. Use
`--autocast off` and `--autocast fp16` for the two arms. The script writes
`config.json`, `pairs.jsonl`, and `result.json` to its output directory.
