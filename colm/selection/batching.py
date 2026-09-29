"""How a selection pool is turned into forward passes: packed, padding-free.

Three steps for `CoresetTrainer`: the batches whose features are extracted, the examples
exchanged between ranks, and the (batch, weights) sub-batches that are trained.
"""

import torch
import torch.nn.functional as F

from colm.selection.packing import (
    Example,
    greedy_groups,
    label_positions,
    model_inputs,
    pack,
)


class PackedBatching:
    """Padding-free: examples are packed into rows of at most `pack_tokens` tokens.

    The loss of a step is the mean over all label tokens of the examples trained in the step
    (all sub-batches and all ranks), the gradient-accumulation semantics of transformers, times
    the weight of each example. It does not depend on how the examples are grouped, so the
    selected examples go through as few forwards as memory allows (`train_tokens`).
    """

    def __init__(self, args, mean_tokens: float, batched: bool):
        self.args, self.batched = args, batched
        self.mean_tokens = mean_tokens
        micro = args.micro_batch_size
        train_micro = max(1, int(micro * args.small_batch_ratio))
        # Tokens of a micro-batch of the padded recipe, without its padding.
        self.select_tokens = args.pack_tokens or int(micro * mean_tokens)
        # Training: the tokens that fit in memory (`set_train_tokens`, measured on the first
        # steps); until then, and if they cannot be measured, one micro-batch of the recipe
        # (CPU: no limit).
        self.train_tokens = args.pack_tokens or (
            int(train_micro * mean_tokens) if torch.cuda.is_available() else 2**62
        )
        self.derive_train_tokens = not args.pack_tokens and torch.cuda.is_available()

    def feature_batches(self, examples: list[Example]) -> list[dict]:
        """The packs whose features are extracted (`examples` in pool order)."""
        if not self.batched:
            return [pack([e]) for e in examples]
        groups = greedy_groups([len(e) for e in examples], self.select_tokens)
        return [pack([examples[i] for i in group]) for group in groups]

    def examples(self, inputs: dict) -> list[Example]:
        return inputs["examples"]

    def train_batches(self, examples: list[Example], weights: list[float]):
        """(pack, per-example weights) of the selected examples, in packs of `train_tokens`."""
        groups = greedy_groups([len(e) for e in examples], self.train_tokens)
        return [
            (pack([examples[i] for i in group]), torch.tensor([weights[i] for i in group]))
            for group in groups
        ]

    def total_labels(self, examples: list[Example]) -> int:
        return sum(e.num_labels for e in examples)

    def loss(self, trainer, model, batch, weights, total: int) -> torch.Tensor:
        """This sub-batch's share of the step loss: `world * sum(weighted token losses) / total`."""
        positions, targets, segment = label_positions(batch)
        logits = model(**model_inputs(batch), logits_to_keep=positions).logits[0]
        logits = logits.to(torch.promote_types(logits.dtype, torch.float32))  # fp16 -> fp32
        token = F.cross_entropy(logits, targets, reduction="none")
        weights = weights.to(token.device)
        # DDP averages the gradients of the ranks; this rank's share is scaled back up.
        return (token * weights[segment]).sum() * trainer.args.world_size / total
