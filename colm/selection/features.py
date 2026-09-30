"""Per-example features of the selection, one extractor per `data_selection_unit`, on packed batches.

`extract(pack)` takes a padding-free batch (`colm.selection.packing.pack`) and returns one feature
per example: `[n, d]` for vector units, `[n]` for scalar units. `batched` extractors accept
several examples per pack, the others get one. `mode` is the train / eval mode they need.
The loss of an example is the mean over its own label tokens (the loss training minimises), so a
feature never depends on the other examples of its pack.
"""

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from colm.selection.packing import label_counts, label_positions, model_inputs
from colm.selection.zo import LastLayerSplit, Perturbation, call
from colm.train.step_timing import StepTimer


def example_means(
    logits: torch.Tensor, targets: torch.Tensor, segment: torch.Tensor, counts: torch.Tensor
):
    """Mean cross-entropy over the label tokens of each example of a pack (`counts`: their number)."""
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    token = F.cross_entropy(logits, targets, reduction="none")
    sums = token.new_zeros(len(counts)).index_add_(0, segment, token)
    return sums / counts.clamp(min=1)


class Extractor:
    batched = False
    scalar = False
    mode = "eval"

    def __init__(self, args, model, zo_params, seed, timer: StepTimer | None = None):
        self.args, self.model, self.zo_params, self.seed = args, model, zo_params, seed
        self.timer = timer or StepTimer()

    def extract(self, pack: dict) -> torch.Tensor:
        raise NotImplementedError

    def expand(self, values: torch.Tensor) -> torch.Tensor:
        """The features of the selection from what `extract` returned (for all examples of the pool)."""
        return values

    def weight_prior(self) -> torch.Tensor:
        return torch.cat([p.flatten() for _, p in self.zo_params]).flatten()

    def _weight_grad(self) -> bool:
        return self.args.mezo_selection == "weight_grad"

    def _losses(self, pack, overrides=None) -> torch.Tensor:
        """Mean label-token loss of every example of the pack (full model, logits at labels only)."""
        positions, targets, segment = label_positions(pack)
        out = call(self.model, overrides, **model_inputs(pack), logits_to_keep=positions)
        return example_means(out.logits[0], targets, segment, label_counts(pack))


class Rep(Extractor):
    """Last-layer hidden state of the last token of each example."""

    def extract(self, pack):
        with torch.inference_mode():
            hidden = self.model(**model_inputs(pack), output_hidden_states=True).hidden_states[-1][
                0
            ]
        return hidden[pack["cu_seq_lens_q"][1:].long() - 1]


class CompletionLength(Extractor):
    scalar = True
    mode = None

    def extract(self, pack):
        return pack["colm_meta"]["completion_lengths"]


class LengthLoss(Extractor):
    """Token count times loss / 10 of one example."""

    scalar = True

    def extract(self, pack):
        with torch.no_grad():
            loss = self._losses(pack).item()
        value = pack["max_length_q"] * loss / 10
        value = 0.0 if math.isnan(value) or math.isinf(value) else max(value, 1e-8)
        return torch.tensor([value], dtype=torch.float32)


class MaskedGrad(Extractor):
    """Exact gradient of the loss of one example w.r.t. the ZO parameters."""

    mode = "train"

    def extract(self, pack):
        # The real gradient is the mean over the selected examples of all ranks.
        selected = int(self.args.per_device_train_batch_size * self.args.small_batch_ratio)
        loss = self._losses(pack)[0] / (selected * self.args.world_size)
        params = [p for _, p in self.zo_params]
        parts = []
        for grad, p in zip(torch.autograd.grad(loss, params), params, strict=True):
            update = grad.detach()
            if self._weight_grad() and not torch.all(p.data == 0):
                update = update * p.data
            parts.append(update.flatten())
        return torch.cat(parts).reshape(1, -1)


class Mezo(Extractor):
    """Two forwards of the whole model at theta +- eps z (one example per pack)."""

    def __init__(self, args, model, zo_params, seed, timer=None):
        super().__init__(args, model, zo_params, seed, timer)
        self.perturbation = Perturbation(zo_params, args.mezo_eps, seed)

    def extract(self, pack):
        def loss(overrides):
            with torch.inference_mode():
                return self._losses(pack, overrides)

        g = self.perturbation.projected_grad(loss)
        return self.perturbation.features(g, self._weight_grad())


class MezoEfficient(Extractor):
    """Batched MeZO: the decoder runs once up to its last layer, which is replayed at +- eps z.

    `extract` returns the projected gradient g_i of every example, one scalar: every feature is
    g_i z with the same z, so only the g_i are exchanged between ranks and `expand` builds the
    `[N, numel]` features where the selection runs.
    """

    batched = True

    def __init__(self, args, model, zo_params, seed, timer=None):
        super().__init__(args, model, zo_params, seed, timer)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.split = LastLayerSplit(base)
        self.perturbation = Perturbation(zo_params, args.mezo_eps, seed)
        self.names = {n: self.split.relative_name(n) for n in self.perturbation.names}
        self.prefix_dtype = args.selection_prefix_dtype
        self.fp32_tail = args.selection_prefix_fp32_tail
        prefix_layers = len(self.split.layers) - 1
        if self.prefix_dtype == "float16" and self._has_low_precision_weights():
            raise ValueError(
                "float16 selection prefix requires fp32 model weights (frozen Linear weights of "
                "the fp16 prefix layers may be stored in fp16: `colm/train/frozen_weights.py`)"
            )
        if not 0 <= self.fp32_tail <= prefix_layers:
            raise ValueError(
                f"selection_prefix_fp32_tail must be in [0, {prefix_layers}] "
                f"(the model has {prefix_layers} prefix layers), got {self.fp32_tail}"
            )
        if self.fp32_tail and self.prefix_dtype != "float16":
            raise ValueError(
                f"selection_prefix_fp32_tail={self.fp32_tail} needs selection_prefix_dtype=float16 "
                f"(got {self.prefix_dtype}); set selection_prefix_fp32_tail=0 for an fp32 prefix"
            )

    def _has_low_precision_weights(self) -> bool:
        """A weight below fp32 that the selection needs in fp32: any but the frozen Linear layers
        of the prefix layers that run under the fp16 autocast."""
        autocast_layers = self.split.layers[: max(0, len(self.split.layers) - 1 - self.fp32_tail)]
        stored_low = {
            id(p)
            for layer in autocast_layers
            for module in layer.modules()
            if isinstance(module, torch.nn.Linear)
            for p in module.parameters()
            if not p.requires_grad
        }
        return any(
            p.is_floating_point() and p.dtype != torch.float32 and id(p) not in stored_low
            for module in (self.split.decoder, self.split.head)
            for p in module.parameters()
        )

    def extract(self, pack):
        split, t = self.split, self.timer
        positions, targets, segment = label_positions(pack)
        counts = label_counts(pack)
        with t.fine("prefix"), torch.inference_mode():
            prefix_amp = self.prefix_dtype == "float16"
            if prefix_amp and not torch.cuda.is_available():
                raise RuntimeError("float16 selection prefix requires CUDA")
            context = torch.autocast("cuda", dtype=torch.float16) if prefix_amp else nullcontext()
            with context:
                state = split.prefix(tail=self.fp32_tail, device_type="cuda", **model_inputs(pack))
            if prefix_amp:
                state = state.float()

        def loss(overrides):
            overrides = {self.names[n]: v for n, v in overrides.items()}
            with t.fine("suffix"), torch.inference_mode():
                with t.fine("layer"):
                    hidden = split.hidden(state, overrides)[0, positions]
                with t.fine("head"):
                    logits = split.head(hidden)
                with t.fine("loss"):
                    return example_means(logits, targets, segment, counts)

        with t.fine("estimate"):
            return self.perturbation.projected_grad(loss)

    def expand(self, values):
        return self.perturbation.features(values, self._weight_grad())


EXTRACTORS = {
    "rep": Rep,
    "mezo": Mezo,
    "masked_grad": MaskedGrad,
    "completion_length": CompletionLength,
    "length_loss_weighted": LengthLoss,
}


def build_extractor(args, model, zo_params, seed, timer=None) -> Extractor:
    if args.efficient_mezo:
        return MezoEfficient(args, model, zo_params, seed, timer)
    return EXTRACTORS[args.data_selection_unit](args, model, zo_params, seed, timer)
