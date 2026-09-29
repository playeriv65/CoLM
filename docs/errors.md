# Errors of the upstream CoLM code and what was done about them

Audit of the original implementation (commit `a6257b0`) and of its port to transformers 5. Every
error listed as fixed is fixed. To prove the alignment with the upstream behaviour the refactored
code had a temporary `legacy` switch reproducing it (float64 CPU goldens of the tag
`pre-refactor`, `tests/test_equivalence.py`); the switch has been removed. The tag `pre-refactor`
is the upstream code (with the port), `legacy-bridge` the last commit that still has the switch,
its tests and the goldens.

Evidence numbers were measured on CPU (tiny Phi, real data) or on one RTX PRO 6000 (phi-2, 20
steps, GPU 2). Severity: **R** changes results, **M** changes memory / time only.

## Errors that changed the results (fixed)

| id | upstream behaviour | evidence | now |
|---|---|---|---|
| E2 (R) | The per-sample MeZO loss is divided by the padded width of the micro-batch minus one, so an example's feature depends on its micro-batch companions. | The same example: loss 4.256 alone, **1.252** next to a long example (token mean 4.626 in both). On phi-2 the features of the two pipelines correlate 0.78 (mean of 8 steps). | Mean over the example's own label tokens; independent of the pack (`tests/test_packing.py`). |
| E3 (R) | Every MeZO estimate calls `torch.manual_seed(zo_random_seed)` on the global RNG: the training RNG restarts from the same state at every step, on every rank. | RNG state at the first training micro-batch: identical hash in steps 0-3; identical on all ranks. | z from a private `torch.Generator`; the global RNG is untouched (`tests/test_trainers.py`). |
| E-drift (R, tiny) | Parameters are perturbed in place (+eps, -2 eps, +eps), leaving rounding error. | fp32, 327,680 values: 3.0e-8 max after 8 estimates, 3.1e-5 after 1024 steps (2e-4 relative). | Out-of-place (`torch.func.functional_call`); the parameters are never written. |
| E4a (R) | Prompt and completion are tokenised together; the prompt is masked by the length of its separate tokenisation. | **9.3%** of the examples (all PoT programs starting with `#`, `[`) have one token straddling the boundary: the first output token is never supervised and the training tokens differ from the inference prompt. | Prompt and completion tokenised separately and concatenated. |
| E4b (R) | Every example is cut at 512 tokens. | **8.6%** (22,640 of 262,039) are cut, 22,554 lose the EOS and the final answer (math50k_camel 40%, MATH CoT 17.8%); 315 have no label token at all (NaN loss at batch size 1). | No truncation, and **no truncation option exists** (`max_seq_length`, `model_max_length`, the SuperGLUE left-truncation and every `truncation=True` are deleted; `tests/test_no_truncation.py` greps for them). The only limit is the context window of the model (`max_position_embeddings`, 2048 for phi-2); examples above it (**31**) are dropped, counted per source in the log, in every data path (instruction/output, LESS prompt/completion and messages, SuperGLUE training, held-out and GSM8K eval loss); SuperGLUE evaluation fails fast (`PromptTooLong`) instead of cutting a prompt, and the accuracy evaluation has no prompt cut (`--max_new_tokens` only sets the generation length). Examples without a completion are dropped. |
| E5 (R, reporting) | The logged loss is divided by `small_batch_ratio`. | Logged / trained loss = 2.000. | The logged loss is the loss being minimised. |
| E7 (M/R) | The selected list `[kept..., class-0 picks, ...]` is cut into contiguous per-rank slices. | 2-rank run: rank 0 trains 75% kept-source examples, rank 1 37.5%. | Round-robin: every rank gets the same mixture. |
| E8 (R, eval) | The held-out set is drawn by row. | 20% of MathInstruct rows share their question with another row (CoT and PoT solutions, several sources): **21%** of a random held-out set have their question in the training part. | Whole groups of the same question are held out (`colm/data/holdout.py`). |
| E9 (R, eval) | Ground truths of numglue / simuleq / deepmind are not normalised like the predictions; `round(p) == gt` and a 4% tolerance count wrong answers; the article "a" is read as option A; bare stop words (`Question`, `Response`, ...) cut generations. | **139 of 1000** deepmind answers could never be matched by a perfect model (`1363.0`, `[3]`, ...). | Normalised ground truths (24 unreachable remain: several answers in one, algebra), numeric equality up to 1e-6, letters only as capitals, stop strings with their colon. |
| E12 (R, resume) | The Adam moments of the selection are not in the checkpoints; a resumed run restarts them with a step count that assumes they exist. | Derivation. | `CoresetSelector.state_dict()`; `load_state_dict()`. |
| E13 (R, noise) | `rep` and `length_loss_weighted` extract features with dropout on. `rep` also computes an unused loss over all positions. | Read. | Eval mode; no loss. |
| E14 (R) | Scalar features are stored in the AMP dtype; their squares overflow fp16 above 255. | Derivation. | float32. |
| E16 (R) | The SuperGLUE option / candidate is counted back from the padded width, so shorter examples lose option tokens. | `tests/test_superglue.py`. | Counted from the real length. |
| MG (R) | `masked_grad` scales the per-example gradient by 1 / (selected examples of the rank), while the real gradient is the mean over all ranks. | Off by the number of GPUs. | 1 / (selected per rank x ranks). |

Not an error, kept: the algorithm of the selection (E1). One fixed random direction z makes every
feature `g_i z`: after the Adam transform facility location is a 1-D k-medoids on `f(g_i)`, so the
selection changes with fp32 reordering (20-step teacher-forced runs of the upstream code agree with
themselves in 3 of 20 steps, mean overlap 13.5 of 16). The scalar `g_i` itself is accurate: on
phi-2, fp32 with eps 1e-3 is within 1e-3 (median) of the exact directional derivative, fp16
suffix forwards are noise (errors 0.05 to 12) - the selection forward must stay fp32.
The audit's E6 (weights of `weightedsubmodlib`) was a misreading: kept examples are scaled by the
ratio too, the total is consistent.

Upstream errors that the port had already fixed: `masked_grad` appended the parameter list at
every call (the feature grew each step); with more kept examples than the budget on several ranks
the selection crashed; the input tensors of all ranks were pickled as CUDA tensors, creating CUDA
contexts on foreign GPUs; the KV cache of the selection forward was copied.

## Memory and precision

| id | what | effect |
|---|---|---|
| M1 | The last layer computed the LM head over every position: fp32 logits `[4, T, 51200]` = 419 MB, a contiguous copy, the log-softmax (1.26 GB per +-eps call). | The head runs at the label positions only, on packed rows. |
| M2 | Padding: 36% of the selection tokens and 45% of the training tokens. | No padding: examples are packed into rows of at most `pack_tokens` (selection) / `train_max_tokens` (training) tokens (stock transformers attention reads the sequence boundaries from `position_ids` and the flash cumulative lengths; fp32 selection: `sdpa` with the block mask, fp16 training: `flash_attention_2`, variable length). |
| D1 | Peak memory was recorded on rank 0 only (`max_memory_allocated`, cumulative, reset by the evaluation on rank 0). Rank 0 also holds the gathered features and Adam temporaries (~0.28 GB x ranks). | `MemoryMeter`: the peaks of every rank, per phase (selection, train), allocated and reserved, gathered at every log and saved as `memory.json`. |
| - | Features are materialised as `[32, 327680]` fp32 (42 MB per rank) although they are 32 scalars times z. | Kept for now (optimisation work). |

phi-2, 1 GPU, 20 steps, allocated peak: legacy 26.4 GB (training) / 13.6 GB (selection); default
22.2 GB in the first steps, 32.3 GB over the run (the longest examples are no longer cut); worst
case, packs of 2048-token examples (`scripts/memory_worst_case.py`): **47.1 GB** for a training
forward of 2 examples, 12.8 GB for a selection forward of 4. Step time 2.1 s against 2.7 s.

fp16 attention: the fp32 packed selection forward equals the stock sdpa forward to 5e-6; the fp16
packed forward differs from the fp16 stock forward by 0.9% (a precision-class difference:
fp16 vs fp32 differs by 1.3%). Gradient error of the fp16 recipe against fp32 sdpa (eval mode,
4 examples): stock fp16 sdpa 1.13 relative (cosine 0.66), packed `varlen_attn` 0.71 (cosine 0.75).
Correction of the reference (2026-09-28, phi-2 + LoRA r=128, 16 real examples, GPU 0): the fp32
**memory-efficient SDPA backward is not a valid reference on sm_120.** Against the fp64 gradient
(exact MATH attention, packs of one example) fp32 MATH is 6.7e-4 off and fp32 EFFICIENT 0.31 off
(cosine 0.97); the fp32 EFFICIENT gradient also depends on the packing (0.30 between one pack of 8
examples and packs of one; MATH: 7.7e-4). The kernel is accurate on random q, k, v (7e-7 against
fp64 for head dim 64-128, length up to 512), so the error comes with the real activations of the
model; the forward is not affected (loss 1e-6, MeZO g_i within 1e-3 of the exact directional
derivative). Every fp32 gradient comparison therefore uses `sdpa_kernel(MATH)`. Against that
reference the fp16 training gradients are (relative error, cosine): stock fp16 sdpa, one example
per forward 1.17 (0.65); `colm_varlen` (torch `varlen_attn`, flash), one pack of 16 examples 0.44
(0.90), one example per forward 0.41 (0.94). The earlier numbers above (1.13 and 0.71) used the
EFFICIENT fp32 gradient as reference. The fp16 gradient error remains the pending precision
decision, not a packing effect.

Bug found on the way (with the former custom attention): under autocast the rotary embedding leaves
q and k in fp32 and v in fp16; the packed kernels read garbage (NaN, illegal memory access) until
q, k and v were cast to the autocast dtype. The stock flash path casts all three itself.

Stock attention (2026-09-28, task/stock-attn; phi-2 + LoRA r=128, packs of at most 1536 tokens of
real examples, 4 packs, fp16 autocast against the fp32 MATH gradient of packs of one example;
relative error, cosine, mean over the packs): `flash_attention_2` (hub kernel) 0.48 (0.89),
former `colm_varlen` 0.40 (0.92), `flex_attention` 0.36 (0.94), stock fp16 `sdpa` 0.53 (0.88);
per pack they range 0.19-0.80, and FA2 against `colm_varlen` differ by 0.52 (mean): the fp16
gradient of this recipe is decided by rounding, not by the kernel. At the operator level (random
fp16 q, k, v, 32 heads x 80, packed lengths 16 x 220 / 4 sequences / 2 x 2048, forward + backward)
the hub kernel and torch `varlen_attn` agree to 7e-6 (output) and 1e-5 - 3e-5 (dq, dk, dv), both
2.4e-4 / 3.5e-4 from the fp32 per-sequence result.

## Effect on the selection

The same pools (steps 0-7 of the upstream run, model at initialisation) through the legacy and the
fixed pipeline: correlation of the features 0.78, selected sets overlap 12.9 of 16 (random
selection of the non-kept part: 10.2, the upstream code against itself: 13.5).

## Alignment (commit `legacy-bridge`)

With `legacy=True`, against the tag `pre-refactor`: float64 CPU identity of the collated batches,
facility location (orders and weights), the selection stages (15 configurations, 5 steps each),
the features of every unit, and 3-step trajectories (selected indices, weights, losses, LoRA
weights to 1e-10, RNG state after selection) of every trainer. phi-2 on GPU: the selection of the
refactored code with `legacy=True` agrees with the original as well as the original agrees with
itself (3 of 20 identical steps, mean overlap 13.2-13.5 of 16 in both comparisons).
