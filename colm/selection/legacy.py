"""Upstream (`legacy=True`) feature extraction on padded micro-batches, with its known errors.

Temporary bridge, see docs/errors.md; everything here is deleted with the `legacy` switch.
Errors reproduced: E2 (per-sample loss over the padded width), E3 and the in-place drift (global
RNG reseeded and parameters shifted in place, see `zo.Perturbation`), E13 (features of `rep` and
`length_loss_weighted` extracted with dropout on).
"""

import math

import torch

from colm.selection.zo import LastLayerSplit, Perturbation, call

MODEL_KEYS = ("input_ids", "attention_mask", "labels")


def model_inputs(batch: dict) -> dict:
    return {k: batch[k] for k in MODEL_KEYS if k in batch}


def per_sample_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean over the padded width of the micro-batch minus one (upstream error E2)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    batch, width = shift_labels.shape
    token_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), reduction="none"
    ).view(batch, width)
    return token_loss.mean(dim=1)


class Extractor:
    batched = False  # accepts micro-batches of more than one example
    scalar = False
    mode: str | None = "eval"

    def __init__(self, args, model, zo_params, seed):
        self.args, self.model, self.zo_params, self.seed = args, model, zo_params, seed

    def extract(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def weight_prior(self) -> torch.Tensor:
        return torch.cat([p.flatten() for _, p in self.zo_params]).flatten()

    def _weight_grad(self) -> bool:
        return self.args.mezo_selection == "weight_grad"


class Rep(Extractor):
    """Last-layer hidden state of the last real token."""

    mode = None  # E13: features were extracted in whatever mode the model was in

    def extract(self, batch):
        inputs = model_inputs(batch)
        with torch.inference_mode():
            inputs["labels"] = inputs["input_ids"]  # an unused loss over every position
            hidden = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
        rows = torch.arange(len(hidden), device=hidden.device)
        last = inputs["attention_mask"].sum(dim=1) - 1
        return hidden[rows, last].flatten(1)


class CompletionLength(Extractor):
    scalar = True
    mode = None

    def extract(self, batch):
        return batch["colm_meta"]["completion_lengths"]


class LengthLoss(Extractor):
    """Token count times loss / 10 of a single example."""

    scalar = True
    mode = None  # E13, as in Rep

    def extract(self, batch):
        length = max(batch["attention_mask"].sum(dim=1).max().item(), 1)
        with torch.no_grad():
            loss = self.model(**model_inputs(batch)).loss.view(-1)
        assert len(loss) == 1, "length_loss_weighted needs one example per micro-batch"
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss).item()
        value = length * loss / 10
        value = 0.0 if math.isnan(value) or math.isinf(value) else max(value, 1e-8)
        return torch.tensor([value], dtype=torch.float64)


class MaskedGrad(Extractor):
    """Exact gradient of the training loss of one example w.r.t. the ZO parameters."""

    mode = "train"

    def extract(self, batch):
        loss = self.model(**model_inputs(batch)).loss / int(
            self.args.per_device_train_batch_size * self.args.small_batch_ratio
        )
        params = [p for _, p in self.zo_params]
        grads = torch.autograd.grad(loss, params)
        parts = []
        for grad, p in zip(grads, params):
            update = grad.detach()
            if self._weight_grad() and not torch.all(p.data == 0):
                update = update * p.data
            parts.append(update.flatten())
        return torch.cat(parts).reshape(1, -1)


class Mezo(Extractor):
    """Two forwards of the whole model at theta +- eps z (one example per micro-batch)."""

    def __init__(self, args, model, zo_params, seed):
        super().__init__(args, model, zo_params, seed)
        self.perturbation = Perturbation(zo_params, args.mezo_eps, seed, legacy=True)

    def extract(self, batch):
        inputs = model_inputs(batch)

        def loss(overrides):
            with torch.inference_mode():
                return call(self.model, overrides, **inputs).loss.detach()

        g = self.perturbation.projected_grad(loss)
        return self.perturbation.features(g, self._weight_grad()).reshape(1, -1)


class MezoEfficient(Extractor):
    """Batched MeZO: the decoder runs once up to its last layer, which is replayed at +- eps z."""

    batched = True

    def __init__(self, args, model, zo_params, seed):
        super().__init__(args, model, zo_params, seed)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.split = LastLayerSplit(base)
        self.perturbation = Perturbation(zo_params, args.mezo_eps, seed, legacy=True)
        self.names = {n: self.split.relative_name(n) for n in self.perturbation.names}

    def extract(self, batch):
        pert, split = self.perturbation, self.split
        with torch.inference_mode():
            state = split.prefix(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

        def loss(overrides):
            if overrides is not None:
                overrides = {self.names[n]: v for n, v in overrides.items()}
            with torch.inference_mode():
                return per_sample_loss(split.logits(state, overrides), batch["labels"])

        g = pert.projected_grad(loss)
        return pert.features(g, self._weight_grad())


EXTRACTORS = {
    "rep": Rep,
    "mezo": Mezo,
    "masked_grad": MaskedGrad,
    "completion_length": CompletionLength,
    "length_loss_weighted": LengthLoss,
}


def build_extractor(args, model, zo_params, seed) -> Extractor:
    if args.efficient_mezo:
        return MezoEfficient(args, model, zo_params, seed)
    return EXTRACTORS[args.data_selection_unit](args, model, zo_params, seed)
