"""Shot-preserving SPIFAEB.

This variant keeps the original SPIFAEB encoder, global cosine prototype branch,
and adaptive evidence-budget local matcher, but preserves support-shot
structure until after local adaptive matching is computed.

Local matching regimes:
- ``pooled``: reproduce the original SPIFAEB behavior exactly by collapsing all
  support shots into one class token pool before adaptive matching;
- ``mean``: compute adaptive matching per support shot, then average scores over
  shots;
- ``softmax``: compute adaptive matching per support shot, then aggregate with
  query-adaptive softmax weights over shot-level local scores.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from net.spif_aeb import SPIFAEB


class SPIFAEBShot(SPIFAEB):
    """SPIFAEB with shot-preserving adaptive local evidence matching."""

    _VALID_SHOT_AGGREGATIONS = frozenset({"pooled", "mean", "softmax"})

    def __init__(
        self,
        *args,
        aeb_shot_agg: str = "softmax",
        aeb_shot_softmax_beta: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if aeb_shot_agg not in self._VALID_SHOT_AGGREGATIONS:
            raise ValueError(
                f"aeb_shot_agg must be one of {sorted(self._VALID_SHOT_AGGREGATIONS)}, got {aeb_shot_agg!r}"
            )
        if float(aeb_shot_softmax_beta) <= 0.0:
            raise ValueError("aeb_shot_softmax_beta must be positive")

        self.aeb_shot_agg = str(aeb_shot_agg)
        self.aeb_shot_softmax_beta = float(aeb_shot_softmax_beta)

    def _adaptive_scores_from_budget(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        rho: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_tokens = F.normalize(query_tokens, p=2, dim=-1)
        support_tokens = F.normalize(support_tokens, p=2, dim=-1)

        similarity = torch.einsum("qld,wtd->qwlt", query_tokens, support_tokens)
        if rho.shape != similarity.shape[:2]:
            raise ValueError(
                "rho must match the query/support leading dimensions: "
                f"rho={tuple(rho.shape)} similarity={tuple(similarity.shape)}"
            )

        row_mean = similarity.mean(dim=-1)
        row_var = similarity.var(dim=-1, unbiased=False)
        row_std = row_var.clamp_min(0.0).add(self.aeb_eps).sqrt()
        threshold = row_mean + self.aeb_beta * (1.0 - rho).unsqueeze(-1) * row_std

        # Larger predicted budgets lower the threshold and keep more evidence.
        # If an entire row is filtered out, fall back to softmax on raw
        # similarities so the local score stays finite.
        pre_weight = F.relu(similarity - threshold.unsqueeze(-1))
        weight_sum = pre_weight.sum(dim=-1, keepdim=True)
        zero_rows = weight_sum <= self.aeb_eps
        sparse_weights = pre_weight / weight_sum.clamp_min(self.aeb_eps)
        fallback_weights = F.softmax(similarity, dim=-1)
        weights = torch.where(zero_rows, fallback_weights, sparse_weights)

        local_scores = (weights * similarity).sum(dim=-1).mean(dim=-1)
        active_match_counts = (pre_weight > 0).sum(dim=-1).float().mean(dim=-1)
        fallback_row_fraction = zero_rows.squeeze(-1).float().mean(dim=-1)
        return {
            "local_scores": local_scores,
            "rho": rho,
            "active_match_counts": active_match_counts,
            "fallback_row_fraction": fallback_row_fraction,
        }

    def compute_pooled_local_outputs(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_global: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        support_token_pool = self.build_support_token_pool(support_tokens)
        rho = self.predict_budget(query_global, prototypes)
        return self._adaptive_scores_from_budget(query_tokens, support_token_pool, rho)

    def compute_per_shot_local_outputs(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_global: torch.Tensor,
        support_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if support_tokens.dim() != 4:
            raise ValueError(
                "support_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        if support_global.dim() != 3:
            raise ValueError(
                "support_global must have shape (Way, Shot, Dim), "
                f"got {tuple(support_global.shape)}"
            )

        way_num, shot_num, token_num, dim = support_tokens.shape
        flat_tokens = support_tokens.reshape(way_num * shot_num, token_num, dim)
        flat_globals = support_global.reshape(way_num * shot_num, support_global.shape[-1])
        per_shot_rho = self.predict_budget(query_global, flat_globals)
        flat_outputs = self._adaptive_scores_from_budget(query_tokens, flat_tokens, per_shot_rho)
        return {
            "per_shot_local_scores": flat_outputs["local_scores"].reshape(query_tokens.shape[0], way_num, shot_num),
            "per_shot_rho": flat_outputs["rho"].reshape(query_tokens.shape[0], way_num, shot_num),
            "per_shot_active_match_counts": flat_outputs["active_match_counts"].reshape(
                query_tokens.shape[0],
                way_num,
                shot_num,
            ),
            "per_shot_fallback_row_fraction": flat_outputs["fallback_row_fraction"].reshape(
                query_tokens.shape[0],
                way_num,
                shot_num,
            ),
        }

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

    def _aggregate_optional_stat(
        self,
        per_shot_stat: torch.Tensor,
        shot_weights: torch.Tensor,
    ) -> torch.Tensor:
        if per_shot_stat.shape != shot_weights.shape:
            raise ValueError(
                "per_shot_stat must match shot_weights: "
                f"stat={tuple(per_shot_stat.shape)} weights={tuple(shot_weights.shape)}"
            )
        return torch.sum(shot_weights * per_shot_stat, dim=-1)

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)
        prototypes = self.build_support_prototypes(episode["support_global"])
        global_scores = self.compute_global_scores(episode["query_global"], prototypes)

        if self.aeb_shot_agg == "pooled":
            local_outputs = self.compute_pooled_local_outputs(
                query_tokens=episode["query_tokens"],
                support_tokens=episode["support_tokens"],
                query_global=episode["query_global"],
                prototypes=prototypes,
            )
            local_scores = local_outputs["local_scores"]
            rho = local_outputs["rho"]
            active_match_counts = local_outputs["active_match_counts"]
            fallback_row_fraction = local_outputs["fallback_row_fraction"]
            shot_weights = None
            per_shot_local_scores = None
            per_shot_rho = None
            per_shot_active_match_counts = None
            per_shot_fallback_row_fraction = None
        else:
            per_shot_outputs = self.compute_per_shot_local_outputs(
                query_tokens=episode["query_tokens"],
                support_tokens=episode["support_tokens"],
                query_global=episode["query_global"],
                support_global=episode["support_global"],
            )
            per_shot_local_scores = per_shot_outputs["per_shot_local_scores"]
            local_scores, shot_weights = self.aggregate_shot_scores(
                per_shot_local_scores=per_shot_local_scores,
                mode=self.aeb_shot_agg,
                beta=self.aeb_shot_softmax_beta,
            )
            per_shot_rho = per_shot_outputs["per_shot_rho"]
            per_shot_active_match_counts = per_shot_outputs["per_shot_active_match_counts"]
            per_shot_fallback_row_fraction = per_shot_outputs["per_shot_fallback_row_fraction"]
            rho = self._aggregate_optional_stat(per_shot_rho, shot_weights)
            active_match_counts = self._aggregate_optional_stat(per_shot_active_match_counts, shot_weights)
            fallback_row_fraction = self._aggregate_optional_stat(per_shot_fallback_row_fraction, shot_weights)

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
            "rho": rho.detach(),
            "active_match_counts": active_match_counts.detach(),
            "fallback_row_fraction": fallback_row_fraction.detach(),
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
        if per_shot_rho is not None:
            output["per_shot_rho"] = per_shot_rho.detach()
        if per_shot_active_match_counts is not None:
            output["per_shot_active_match_counts"] = per_shot_active_match_counts.detach()
        if per_shot_fallback_row_fraction is not None:
            output["per_shot_fallback_row_fraction"] = per_shot_fallback_row_fraction.detach()
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
            "rho": torch.cat([item["rho"] for item in diagnostics], dim=0),
            "active_match_counts": torch.cat([item["active_match_counts"] for item in diagnostics], dim=0),
            "fallback_row_fraction": torch.cat([item["fallback_row_fraction"] for item in diagnostics], dim=0),
            "mean_budget": torch.stack([item["rho"].mean() for item in diagnostics]).mean(),
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

        for key in (
            "per_shot_local_scores",
            "shot_aggregation_weights",
            "per_shot_rho",
            "per_shot_active_match_counts",
            "per_shot_fallback_row_fraction",
        ):
            if key in diagnostics[0]:
                payload[key] = torch.cat([item[key] for item in diagnostics], dim=0)
        return payload
