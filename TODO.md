# TODO

Living task list for this research repo. Keep it short: move finished items to "Done" with the
commit / result pointer, delete them once they are recorded in docs. Optimisation details live in
`docs/optimization-backlog.md`; this file only tracks what is next.

## In progress

- [ ] Exact (semantics-preserving) step optimisations O1, O5, O6, O7 + host-overhead removal,
      re-timed against the 2868 ms baseline (`docs/optimization-backlog.md`). Branch `task/exact-opts`.

## Next

- [ ] Refactor to a modern, hack-free structure (scope to be agreed; see "Refactor" below).
- [ ] LoRA rank sweep: r ∈ {8, 16} vs the paper's 128 (design below, needs GPUs + decisions).
- [ ] Make evaluation report loss as well as accuracy (`math_eval` is accuracy-only today).

## Experiments

- [ ] **LoRA rank sweep.** Arms r = 8, 16, 128 (baseline) on the paper recipe
      (`configs/math_phi2.json`: 1024 steps, 64 of 128 selected per step on 4 GPUs).
      Open: α scaling (keep α/r = 4, i.e. α = 32 / 64 / 512, vs fixed α = 512), seeds (learning-curve
      experiment → ≥ 3), GPU count (the selection pool is per_device_bs × GAS × world_size, so 1 GPU
      does not reproduce the paper's 128-example pool). Note r changes the ZO feature too: it is
      `lora_B` of the last v_proj, 2560 × r values; `zo_dim` = 2560 is still ≤ 2560 × 8.
      Metrics: train loss curve, eval loss + accuracy on gsm8k / math / numglue / svamp / deepmind /
      simuleq, step time, peak memory.

## Refactor (hacks to remove)

- [ ] `custom_phi.py` re-implements the Phi forward split; Phi-only, layer index 31 default.
- [ ] Architecture-specific string checks (`"phi-2" in model_name` for LoRA targets / precision).
- [ ] MeZO perturbs `param.data` in place and re-seeds the global RNG (`torch.manual_seed`) on every
      estimate; use a dedicated `torch.Generator` and a functional / out-of-place perturbation
      (must keep the same z and the same training RNG stream, or be flagged as a change).
- [ ] Placeholder micro-batches so the HF loop's accumulation count / DDP sync line up.
- [ ] Logged loss divided by `small_batch_ratio` (kept for curve comparability).
- [ ] `keep_sources` as an underscore-separated string of source indices; sources as ints.
- [ ] fp32 base weights + fp16 AMP for phi-2, `embed_tokens` / `lm_head` `.float()` cast
      (a workaround for fp16 unscale errors).
- [ ] Many selection units / trainers (`rep`, `masked_grad`, `completion_length`, …) around one paper
      method; decide which are needed for the paper's tables before pruning.

## Decisions pending (user)

- D1–D4 in `docs/optimization-backlog.md` (F2 divisor, selection precision, dropout for activation
  reuse, 1-D facility location).

## Cleanup pending confirmation

- Smoke / timing outputs: `_worktrees/CoLM-original/out/default` (2.6G, local NVMe);
  `/mnt/data2/zelin4593/artifacts/CoLM/out/*-5steps-*`, `*-130steps-*`, `*-60steps-*`;
  the aborted fine timing log + jsonl and the first failed smoke log in `logs/`.

## Done

- 2026-09-28 Port to uv + transformers 5.17 + stock vLLM (`113005a`); unported baseline env on
  branch `task/original-env` (`96673d0`); per-phase step timing (`69a9235`, baseline 2868 ms/step).
