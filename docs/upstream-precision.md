# Precision map of the upstream code

What precision the upstream CoLM code (commit `a6257b0`; github.com/BigML-CS-UCLA/CoLM, 3 commits,
`hsgser` is the author's old handle) actually runs in, read from the code and measured on phi-2.
The paper (arXiv 2407.19580) never mentions fp16, bf16 or fp32: it says "one forward pass" and gives
LoRA r=128 / alpha=512, lr 2e-5, gradient accumulation 8, 4 x A40, 1K steps.

## Where each piece runs

| piece | precision | why |
|---|---|---|
| checkpoint on the hub | fp16 (5.2 GB) | |
| model in memory | **fp32** (11 GB) | the launch script passes `--torch_dtype none`, which becomes `None`, and the default load dtype is fp32 |
| training | fp16 AMP: fp32 master weights, fp16 autocast, GradScaler | `--fp16 True`; autocast casts the weights per matmul and keeps the fp16 copies for the backward (~5.6 GB) |
| trainable parameters | fp32 | the GradScaler needs fp32 trainable parameters: this is the reason for the `.float()` on `embed_tokens` / `lm_head` with the comment "Attempting to unscale FP16 gradients" |
| selection forward, efficient MeZO path (`EFF_MEZO=True`) | **fp32, by accident** | see below |
| selection forward, non-efficient path | fp16 autocast | goes through `compute_loss`, which calls the wrapped `model.forward` |

The efficient path calls `model.module.decomposer.forward_till_penultimate` and
`forward_final_layer` directly. accelerate wraps `model.forward` in autocast (native AMP, in
`accelerator.py`), so calling the decomposer bypasses that wrapper. transformers 4.43's
`compute_loss_context_manager` is a no-op on GPU (it only enters autocast for CPU AMP), so it adds
nothing either. The whole selection forward (the unperturbed prefix and the +-eps last layer) is
therefore fp32, but nothing in the code says it was chosen: it is a side effect of the call path.

## Measured consequence

`docs/selection-precision.md` (Phi-2, 512 examples x 5 directions, against the exact float64
derivative): the fp32 g_i is accurate (median relative error 0.16 %), an fp16 prefix gives 18.5 %
error with 6 % sign flips, and fp16 everywhere (the regime of the upstream non-efficient path) 72 %
with 21 % sign flips, which is weakly informative. The default here is the fp16 prefix with an fp32
tail of two blocks (1.15 %, `docs/fp16-prefix.md`).

## What cannot be decided from the repository

Which path produced the paper's numbers. The released launch script has `EFF_MEZO=True` but
`MAX_STEPS=5` (five steps, not the 1K of the paper), so it does not identify the path of the
reported runs. The efficient path selects with an fp32 forward, the non-efficient one with fp16
autocast, where the MeZO feature is only weakly informative (above); the numbers of the paper
cannot be attributed to either path from the repository.
