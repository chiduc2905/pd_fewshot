"""Shot-preserving local sliced Wasserstein logic for SPIF paper-SW variants.

Research statement:
Existing local Wasserstein matching collapses multiple support shots into a
single empirical token pool before transport, which destroys shot-level
structure and can enlarge intra-class support variance in multi-shot episodes.
We instead perform local transport per support shot and aggregate distances at
the shot level, optionally using query-adaptive soft weighting to suppress
unreliable support examples.
"""

from __future__ import annotations

from typing import Any

import torch


class _SPIFShotPreservingLocalSWMixin:
    """Hierarchical local SW: token-level transport inside shots, shot-level aggregation after."""

    _VALID_SHOT_AGGREGATIONS = {"pooled", "mean", "softmin"}

    @staticmethod
    def _validate_support_shot_structure(
        support_tokens: torch.Tensor,
        support_weights: torch.Tensor | None = None,
    ) -> None:
        if support_tokens.dim() != 4:
            raise ValueError(
                "support_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        if support_weights is not None and support_weights.shape != support_tokens.shape[:-1]:
            raise ValueError(
                "support_weights must have shape (Way, Shot, Tokens), "
                f"got {tuple(support_weights.shape)} for tokens {tuple(support_tokens.shape)}"
            )

    @staticmethod
    def _pool_support_shots_old_behavior(
        support_tokens: torch.Tensor,
        support_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reproduce the legacy local paper-SW behavior exactly: concat all shots before OT."""
        way_num, shot_num, token_num, dim = support_tokens.shape
        pooled_tokens = support_tokens.reshape(way_num, shot_num * token_num, dim)
        pooled_weights = None if support_weights is None else support_weights.reshape(way_num, shot_num * token_num)
        return pooled_tokens, pooled_weights

    @staticmethod
    def _flatten_support_shots(
        support_tokens: torch.Tensor,
        support_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Keep shots separate for transport, but flatten (Way, Shot) for vectorized pairwise SW."""
        way_num, shot_num, token_num, dim = support_tokens.shape
        flat_tokens = support_tokens.reshape(way_num * shot_num, token_num, dim)
        flat_weights = None if support_weights is None else support_weights.reshape(way_num * shot_num, token_num)
        return flat_tokens, flat_weights

    def compute_pooled_local_distances(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_weights: torch.Tensor | None = None,
        support_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_support_shot_structure(support_tokens, support_weights)
        pooled_tokens, pooled_weights = self._pool_support_shots_old_behavior(
            support_tokens=support_tokens,
            support_weights=support_weights,
        )
        return self.local_transport_distance.pairwise_distance(
            query_tokens,
            pooled_tokens,
            query_weights=query_weights,
            support_weights=pooled_weights,
            reduction="none",
        )

    def compute_per_shot_local_sw(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_weights: torch.Tensor | None = None,
        support_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return shot-preserving distances with shape (NumQuery, Way, Shot)."""
        self._validate_support_shot_structure(support_tokens, support_weights)
        way_num, shot_num = support_tokens.shape[:2]
        flat_tokens, flat_weights = self._flatten_support_shots(
            support_tokens=support_tokens,
            support_weights=support_weights,
        )
        per_shot = self.local_transport_distance.pairwise_distance(
            query_tokens,
            flat_tokens,
            query_weights=query_weights,
            support_weights=flat_weights,
            reduction="none",
        )
        return per_shot.reshape(query_tokens.shape[0], way_num, shot_num)

    def aggregate_shot_distances(
        self,
        per_shot_distances: torch.Tensor,
        mode: str,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if per_shot_distances.dim() != 3:
            raise ValueError(
                "per_shot_distances must have shape (NumQuery, Way, Shot), "
                f"got {tuple(per_shot_distances.shape)}"
            )
        if mode not in self._VALID_SHOT_AGGREGATIONS - {"pooled"}:
            raise ValueError(f"Unsupported shot aggregation mode: {mode}")

        if mode == "mean":
            shot_num = per_shot_distances.shape[-1]
            shot_weights = torch.full_like(per_shot_distances, 1.0 / float(max(shot_num, 1)))
            return per_shot_distances.mean(dim=-1), shot_weights

        shot_logits = -float(beta) * per_shot_distances
        shot_weights = torch.softmax(shot_logits, dim=-1)
        class_distances = torch.sum(shot_weights * per_shot_distances, dim=-1)
        return class_distances, shot_weights

    def compute_local_logits(
        self,
        class_distances: torch.Tensor,
    ) -> torch.Tensor:
        return -self.local_transport_score_scale * class_distances

    def compute_local_transport_outputs(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_weights: torch.Tensor | None = None,
        support_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = self.local_shot_agg
        if mode == "pooled":
            class_distances = self.compute_pooled_local_distances(
                query_tokens=query_tokens,
                support_tokens=support_tokens,
                query_weights=query_weights,
                support_weights=support_weights,
            )
            return {
                "class_distances": class_distances,
                "local_logits": self.compute_local_logits(class_distances),
            }

        per_shot_distances = self.compute_per_shot_local_sw(
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            query_weights=query_weights,
            support_weights=support_weights,
        )
        class_distances, shot_weights = self.aggregate_shot_distances(
            per_shot_distances=per_shot_distances,
            mode=mode,
            beta=self.local_shot_softmin_beta,
        )
        outputs = {
            "class_distances": class_distances,
            "local_logits": self.compute_local_logits(class_distances),
            "per_shot_distances": per_shot_distances,
            "shot_weights": shot_weights,
        }
        return outputs

    def _compute_local_margin_stats(
        self,
        local_logits: torch.Tensor,
        query_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if local_logits.dim() != 2:
            raise ValueError(f"local_logits must have shape (NumQuery, Way), got {tuple(local_logits.shape)}")
        if query_targets.dim() != 1 or query_targets.shape[0] != local_logits.shape[0]:
            raise ValueError(
                "query_targets must have shape (NumQuery,), "
                f"got {tuple(query_targets.shape)} for logits {tuple(local_logits.shape)}"
            )

        positive_logits = local_logits.gather(1, query_targets.unsqueeze(-1)).squeeze(-1)
        target_mask = torch.zeros_like(local_logits, dtype=torch.bool)
        target_mask.scatter_(1, query_targets.unsqueeze(-1), True)
        negative_logits = local_logits.masked_fill(target_mask, float("-inf"))
        hardest_negative_logits = negative_logits.max(dim=1).values
        mean_negative_logits = negative_logits.masked_fill(~torch.isfinite(negative_logits), 0.0).sum(dim=1) / (
            local_logits.shape[1] - 1
        )
        return {
            "positive_local_logit_mean": positive_logits.mean().detach(),
            "negative_local_logit_mean": mean_negative_logits.mean().detach(),
            "hardest_negative_local_logit_mean": hardest_negative_logits.mean().detach(),
            "local_margin_mean": (positive_logits - hardest_negative_logits).mean().detach(),
        }

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)
        prototypes = self.build_support_prototypes(episode["support_global"])
        global_scores = self.compute_global_scores(episode["query_global"], prototypes)

        local_outputs = self.compute_local_transport_outputs(
            query_tokens=episode["query_tokens"],
            support_tokens=episode["support_tokens"],
            query_weights=episode["query_gate"].squeeze(-1),
            support_weights=episode["support_gate"].squeeze(-1),
        )
        local_scores = local_outputs["local_logits"]
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

        outputs = {
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
            "local_class_distances": local_outputs["class_distances"].detach(),
        }

        if "per_shot_distances" in local_outputs:
            outputs["per_shot_local_distances"] = local_outputs["per_shot_distances"].detach()
            outputs["shot_aggregation_weights"] = local_outputs["shot_weights"].detach()

        if self.local_debug and query_targets is not None:
            outputs.update(self._compute_local_margin_stats(local_scores, query_targets))
        return outputs

    @staticmethod
    def _cat_optional(diagnostics: list[dict[str, Any]], key: str) -> torch.Tensor | None:
        values = [item[key] for item in diagnostics if key in item]
        if not values:
            return None
        return torch.cat(values, dim=0)

    @staticmethod
    def _mean_optional(diagnostics: list[dict[str, Any]], key: str) -> torch.Tensor | None:
        values = [item[key] for item in diagnostics if key in item]
        if not values:
            return None
        return torch.stack(values).mean()

