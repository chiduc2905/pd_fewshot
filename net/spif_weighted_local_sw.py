"""Shared weighted local sliced Wasserstein episode logic for SPIF variants."""

from __future__ import annotations

import torch


class _SPIFWeightedLocalSWMixin:
    """Route SPIF local token matching through weighted paper-style SW."""

    @staticmethod
    def _build_support_weight_pool(support_gate: torch.Tensor) -> torch.Tensor:
        if support_gate.dim() != 4:
            raise ValueError(
                "support_gate must have shape (Way, Shot, Tokens, 1), "
                f"got {tuple(support_gate.shape)}"
            )
        way_num, shot_num, token_num, _ = support_gate.shape
        return support_gate.squeeze(-1).reshape(way_num, shot_num * token_num)

    def compute_local_partial_scores(
        self,
        query_tokens: torch.Tensor,
        support_token_pool: torch.Tensor,
        query_weights: torch.Tensor,
        support_weights: torch.Tensor,
    ) -> torch.Tensor:
        distances = self.local_transport_distance.pairwise_distance(
            query_tokens,
            support_token_pool,
            query_weights=query_weights,
            support_weights=support_weights,
            reduction="none",
        )
        return -self.local_transport_score_scale * distances

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)
        prototypes = self.build_support_prototypes(episode["support_global"])
        support_token_pool = self.build_support_token_pool(episode["support_tokens"])
        support_weight_pool = self._build_support_weight_pool(episode["support_gate"])
        global_scores = self.compute_global_scores(episode["query_global"], prototypes)
        local_scores = self.compute_local_partial_scores(
            query_tokens=episode["query_tokens"],
            support_token_pool=support_token_pool,
            query_weights=episode["query_gate"].squeeze(-1),
            support_weights=support_weight_pool,
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
        return {
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
