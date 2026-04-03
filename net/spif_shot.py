"""Shot-preserving SPIFCE local matcher.

This variant keeps the original SPIFCE encoder, global cosine prototype branch,
local top-r partial matching score, and alpha fusion. The only redesign is that
local matching preserves the support-shot dimension until after per-shot local
scores are computed.

Aggregation modes:
- ``pooled``: reproduce the original SPIFCE local matcher exactly by collapsing
  all support shots before local matching;
- ``mean``: compute local top-r scores per support shot, then average over
  shots;
- ``softmax``: compute local top-r scores per support shot, then aggregate with
  query-adaptive softmax weights over shot-level local scores.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from net.spif import SPIFCE


class SPIFCEShot(SPIFCE):
    """SPIFCE with shot-preserving local top-r partial matching."""

    _VALID_SHOT_AGGREGATIONS = frozenset({"pooled", "mean", "softmax"})

    def __init__(
        self,
        *args,
        spif_shot_agg: str = "softmax",
        spif_shot_softmax_beta: float = 10.0,
        **kwargs,
    ) -> None:
        kwargs.setdefault("top_r", 4)
        super().__init__(*args, **kwargs)
        if spif_shot_agg not in self._VALID_SHOT_AGGREGATIONS:
            raise ValueError(
                f"spif_shot_agg must be one of {sorted(self._VALID_SHOT_AGGREGATIONS)}, got {spif_shot_agg!r}"
            )
        if float(spif_shot_softmax_beta) <= 0.0:
            raise ValueError("spif_shot_softmax_beta must be positive")

        self.spif_shot_agg = str(spif_shot_agg)
        self.spif_shot_softmax_beta = float(spif_shot_softmax_beta)

    def compute_pooled_local_scores(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> torch.Tensor:
        support_token_pool = self.build_support_token_pool(support_tokens)
        return self.compute_local_partial_scores(query_tokens, support_token_pool)

    def compute_per_shot_local_scores(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if support_tokens.dim() != 4:
            raise ValueError(
                "support_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        way_num, shot_num, token_num, dim = support_tokens.shape
        flat_support = support_tokens.reshape(way_num * shot_num, token_num, dim)

        query_tokens = F.normalize(query_tokens, p=2, dim=-1)
        flat_support = F.normalize(flat_support, p=2, dim=-1)
        similarity = torch.einsum("qld,wtd->qwlt", query_tokens, flat_support)
        top_r = min(self.top_r, similarity.shape[-1])
        top_vals = torch.topk(similarity, k=top_r, dim=-1).values
        per_shot_scores = top_vals.mean(dim=-1).mean(dim=-1)
        return per_shot_scores.reshape(query_tokens.shape[0], way_num, shot_num)

    def aggregate_shot_scores(
        self,
        per_shot_local_scores: torch.Tensor,
        mode: str,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if per_shot_local_scores.dim() != 3:
            raise ValueError(
                "per_shot_local_scores must have shape (NumQuery, Way, Shot), "
                f"got {tuple(per_shot_local_scores.shape)}"
            )
        if mode not in self._VALID_SHOT_AGGREGATIONS - {"pooled"}:
            raise ValueError(f"Unsupported shot aggregation mode: {mode}")

        if mode == "mean":
            shot_num = per_shot_local_scores.shape[-1]
            shot_weights = torch.full_like(per_shot_local_scores, 1.0 / float(max(shot_num, 1)))
            return per_shot_local_scores.mean(dim=-1), shot_weights

        shot_logits = float(beta) * per_shot_local_scores
        shot_weights = torch.softmax(shot_logits, dim=-1)
        class_scores = torch.sum(shot_weights * per_shot_local_scores, dim=-1)
        return class_scores, shot_weights

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)
        prototypes = self.build_support_prototypes(episode["support_global"])
        global_scores = self.compute_global_scores(episode["query_global"], prototypes)

        if self.spif_shot_agg == "pooled":
            local_scores = self.compute_pooled_local_scores(
                query_tokens=episode["query_tokens"],
                support_tokens=episode["support_tokens"],
            )
            per_shot_local_scores = None
            shot_weights = None
        else:
            per_shot_local_scores = self.compute_per_shot_local_scores(
                query_tokens=episode["query_tokens"],
                support_tokens=episode["support_tokens"],
            )
            local_scores, shot_weights = self.aggregate_shot_scores(
                per_shot_local_scores=per_shot_local_scores,
                mode=self.spif_shot_agg,
                beta=self.spif_shot_softmax_beta,
            )

        logits, alpha = self._fuse_scores(global_scores, local_scores)
        aux_loss = self._aux_loss(
            query_tokens=episode["query_tokens"],
            support_tokens=episode["support_tokens"],
            query_global=episode["query_global"],
            support_global=episode["support_global"],
            query_variant_global=episode["query_variant_global"],
            support_variant_global=episode["support_variant_global"],
            query_gate=episode["query_gate"],
            support_gate=episode["support_gate"],
        )
        output = {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": global_scores.detach(),
            "local_scores": local_scores.detach(),
            "alpha": alpha.detach(),
            "mean_gate": torch.cat(
                [
                    episode["query_gate"].reshape(-1, 1),
                    episode["support_gate"].reshape(-1, 1),
                ],
                dim=0,
            ).mean().detach(),
            "stable_global_embeddings": torch.cat(
                [
                    episode["query_global"],
                    episode["support_global"].reshape(-1, episode["support_global"].shape[-1]),
                ],
                dim=0,
            ).detach(),
            "variant_global_embeddings": torch.cat(
                [
                    episode["query_variant_global"],
                    episode["support_variant_global"].reshape(-1, episode["support_variant_global"].shape[-1]),
                ],
                dim=0,
            ).detach(),
        }
        if per_shot_local_scores is not None:
            output["per_shot_local_scores"] = per_shot_local_scores.detach()
        if shot_weights is not None:
            output["shot_aggregation_weights"] = shot_weights.detach()
        return output

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_outputs = []
        diagnostics = []
        for batch_idx in range(bsz):
            episode = self._forward_episode(query[batch_idx], support[batch_idx])
            batch_outputs.append(episode["logits"])
            diagnostics.append(episode)

        logits = torch.cat(batch_outputs, dim=0)
        if not return_aux:
            return logits

        payload = {
            "logits": logits,
            "global_scores": torch.cat([item["global_scores"] for item in diagnostics], dim=0),
            "local_scores": torch.cat([item["local_scores"] for item in diagnostics], dim=0),
            "alpha": diagnostics[0]["alpha"],
            "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }
        for key in ("per_shot_local_scores", "shot_aggregation_weights"):
            if key in diagnostics[0]:
                payload[key] = torch.cat([item[key] for item in diagnostics], dim=0)
        return payload
