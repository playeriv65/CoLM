"""How a selection pool is turned into forward passes: packed (default) or padded (`legacy`).

Both provide the same three steps to `CoresetTrainer`: the batches whose features are extracted,
the examples exchanged between ranks, and the (batch, weight) sub-batches that are trained.
"""

import torch
import torch.nn.functional as F

from colm.selection import legacy
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
    the weight of each example.
    """

    def __init__(self, args, mean_tokens: float, batched: bool):
        self.args, self.batched = args, batched
        micro = args.micro_batch_size
        train_micro = max(1, int(micro * args.small_batch_ratio))
        # Tokens of a micro-batch of the padded recipe, without its padding.
        self.select_tokens = args.pack_tokens or int(micro * mean_tokens)
        self.train_tokens = args.pack_tokens or int(train_micro * mean_tokens)

    def feature_batches(self, inputs: dict) -> list[dict]:
        examples = inputs["examples"]
        if not self.batched:
            return [pack([e]) for e in examples]
        groups = greedy_groups([len(e) for e in examples], self.select_tokens)
        return [pack([examples[i] for i in group]) for group in groups]

    def examples(self, inputs: dict) -> list[Example]:
        return inputs["examples"]

    def train_batches(self, examples: list[Example], weights: list[float], size: int):
        """(pack, per-example weights) of the selected examples; `size` is only used by `legacy`."""
        if not self.batched:  # SubsetTrainer: one example per forward, weighted
            groups = [[i] for i in range(len(examples))]
        else:
            groups = greedy_groups([len(e) for e in examples], self.train_tokens)
        return [
            (pack([examples[i] for i in group]), torch.tensor([weights[i] for i in group]))
            for group in groups
        ]

    def total_labels(self, examples: list[Example]) -> int:
        return sum(e.num_labels for e in examples)

    def loss(self, trainer, model, batch, weights, count: int, total: int) -> torch.Tensor:
        """This sub-batch's share of the step loss: `world * sum(weighted token losses) / total`."""
        positions, targets, segment = label_positions(batch)
        logits = model(**model_inputs(batch), logits_to_keep=positions).logits[0]
        token = F.cross_entropy(logits.float(), targets, reduction="none")
        weights = weights.to(token.device)
        # DDP averages the gradients of the ranks; this rank's share is scaled back up.
        return (token * weights[segment]).sum() * trainer.args.world_size / total


def build_batching(args, pad_token_id: int, mean_tokens: float | None, batched: bool):
    if args.legacy:
        return legacy.PaddedBatching(args, pad_token_id)
    return PackedBatching(args, mean_tokens, batched)
