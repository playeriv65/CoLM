"""Coreset selection over a gathered pool of per-example features (rank 0 only).

Stages, in order: examples of `keep_sources` are set aside; features are transformed
(`mezo_transform`), replaced by the update Adam would make (`mezo_optim=adam`), reduced to `zo_dim`
coordinates per source (`mezo_topk`, vector features only), and facility location picks the
budget from every source. The selector keeps the Adam moments of the selected subset between steps.
"""

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from colm.selection.facility_location import get_orders_and_weights

logger = logging.getLogger(__name__)


@dataclass
class Selection:
    indices: list[int]  # into the gathered pool; kept-source examples first
    weights: list[float]
    candidates: list[int]  # pool positions that were candidates of facility location


class CoresetSelector:
    """`args` is the run's TrainingArguments (selection fields); see training_arguments.py."""

    def __init__(self, args, num_layers: int, moments=None, weight_prior=None):
        self.args = args
        self.num_layers = num_layers
        self.moments = moments  # () -> (m, v) | None; the real optimizer's Adam state
        self.weight_prior = weight_prior  # () -> flat parameter vector (mezo_selection=weight)
        self.prev_m = self.prev_v = None
        self.scalar = args.data_selection_unit in ("completion_length", "length_loss_weighted")

    # ----- checkpointing (the moments are part of the selection state) ------------------
    def state_dict(self) -> dict:
        return {"prev_m": self.prev_m, "prev_v": self.prev_v}

    def load_state_dict(self, state: dict) -> None:
        self.prev_m, self.prev_v = state["prev_m"], state["prev_v"]

    def __call__(self, feats: torch.Tensor, sources: list[int], total: int, step: int) -> Selection:
        args = self.args
        candidates = np.arange(len(feats))
        budget = total

        keep = []
        if args.keep_source_ids:
            is_kept = np.array([s in args.keep_source_ids for s in sources])
            keep = np.where(is_kept)[0].tolist()
            candidates = candidates[~is_kept]
            feats = feats[torch.from_numpy(~is_kept).to(feats.device)]
            budget -= len(keep)
            logger.info(
                f"Exclude {len(keep)} examples from selection. Select {budget} from the "
                f"remaining {len(candidates)} examples."
            )

        if feats.dim() == 1:  # scalar features are 1-D points
            feats = feats.unsqueeze(1)
        squared = torch.square(feats)
        feats = self._transform(feats)
        m_t = v_t = None
        if args.mezo_optim == "adam":
            feats, m_t, v_t = self._adam(feats, squared, step)

        source_list = None
        if args.source_wise_selection != "none":
            source_list = [sources[i] for i in candidates]
            logger.info(f"Source count: {sorted(Counter(source_list).items())}")

        if not self.scalar:
            if args.mezo_topk == "random":
                feats = feats[:, torch.randperm(feats.shape[1])[: args.zo_dim]]
            else:
                feats = self._mask(feats, source_list)

        if budget > 0:
            order, cluster = get_orders_and_weights(
                budget,
                feats,
                args.facility_similarity,
                y=source_list,
                per_class_start=args.num_per_class_start,
                strategy=args.source_wise_selection,
            )
            if args.mezo_optim == "adam" and not self.uses_optimizer_moments:
                # The selector's own moments: mean over the selected subset.
                self.prev_m = m_t[order].mean(dim=0).detach()
                self.prev_v = v_t[order].mean(dim=0).detach()
            indices = keep + candidates[order].tolist()
            weights = [1.0] * len(keep) + cluster.tolist()
        else:  # the kept examples fill the budget: train on the first `total` of them
            indices = keep[:total]
            weights = [1.0] * len(indices)
        assert len(indices) == total, f"selected {len(indices)} != budget {total}"
        if "weighted" in args.data_selection_method:
            weights = [w * args.small_batch_ratio for w in weights]
        else:
            weights = [1.0] * len(indices)
        return Selection([int(i) for i in indices], weights, candidates.tolist())

    @property
    def uses_optimizer_moments(self) -> bool:
        return self.moments is not None

    # ----- stages -------------------------------------------------------------------------
    def _transform(self, feats):
        if feats.dtype == torch.long:
            return feats
        args = self.args
        mean_norm = torch.norm(torch.mean(feats, dim=0), p=2)
        if args.mezo_transform == "self_normalize":
            feats = feats / torch.norm(feats, p=2, dim=1, keepdim=True)
        elif args.mezo_transform == "normalize":
            feats = feats / mean_norm
        elif args.mezo_transform == "clip_full":
            coef = args.max_grad_norm / mean_norm
            if coef < 1:
                feats = feats * coef
        elif args.mezo_transform == "clip_last":
            # The last layer's share of the norm is approximated by 1 / num_layers.
            coef = args.max_grad_norm / (mean_norm / self.num_layers)
            if coef < 1:
                feats = feats * coef
        return feats

    def _adam(self, feats, squared, step):
        """Replace per-example gradients by the Adam update they would produce."""
        args = self.args
        if self.uses_optimizer_moments:
            state = self.moments()
            prev_m, prev_v = (
                state
                if state is not None
                else (
                    torch.zeros_like(feats[0]),
                    torch.zeros_like(squared[0]),
                )
            )
        else:
            if self.prev_m is None:
                self.prev_m, self.prev_v = torch.zeros_like(feats[0]), torch.zeros_like(squared[0])
            prev_m, prev_v = self.prev_m, self.prev_v
        m_t = args.adam_beta1 * prev_m + (1 - args.adam_beta1) * feats
        v_t = args.adam_beta2 * prev_v + (1 - args.adam_beta2) * squared
        m_hat = m_t / (1 - args.adam_beta1 ** (step + 1))
        v_hat = v_t / (1 - args.adam_beta2 ** (step + 1))
        return m_hat / (torch.sqrt(v_hat) + args.adam_epsilon), m_t, v_t

    def _mask(self, feats, source_list):
        """Keep `zo_dim` coordinates per source, ranked by |mean feature| (or |weight|)."""
        args = self.args
        groups = (
            np.zeros(len(feats), dtype=np.int32) if source_list is None else np.array(source_list)
        )
        masked = torch.zeros((feats.shape[0], args.zo_dim), dtype=feats.dtype).to(feats.device)
        for source in np.unique(groups):
            rows = np.where(groups == source)[0]
            block = feats[rows]
            importance = torch.abs(torch.mean(block, dim=0))
            if args.mezo_selection == "weight":
                weights = self.weight_prior()
                if not torch.all(weights == 0):
                    importance = torch.abs(weights)
            masked[rows] = block[:, self._rank(importance)]
        return masked

    def _rank(self, importance):
        args = self.args
        k = args.zo_dim
        if args.mezo_topk == "smallest":
            return torch.argsort(importance)[:k]
        if args.mezo_topk == "largest":
            return torch.argsort(importance, descending=True)[:k]
        if args.mezo_topk == "sampling":
            p = importance.cpu().numpy().astype("float64")
            return np.random.choice(len(importance), size=k, replace=False, p=p / p.sum())
        if args.mezo_topk == "largest_smallest":
            return torch.cat(
                (
                    torch.argsort(importance)[: k // 2],
                    torch.argsort(importance, descending=True)[: k // 2],
                )
            )
        raise ValueError(f"unknown mezo_topk {args.mezo_topk}")
