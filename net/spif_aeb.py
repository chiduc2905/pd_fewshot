"""SPIFAEB: Stable Partial Invariance with Adaptive Evidence Budget.

This variant reuses the original SPIF encoder / episodic pipeline and replaces
only the fixed local top-r matcher with a stability-aware adaptive evidence
budget on top of the stable token similarity matrix.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.spif import _SPIFBase


class SPIFAEB(_SPIFBase):
    """SPIFCE-compatible fair variant with adaptive local evidence selection."""

    def __init__(
        self,
        *args,
        aeb_hidden: int | None = None,
        aeb_beta: float = 1.0,
        aeb_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        kwargs.setdefault("learnable_alpha", False)
        kwargs.setdefault("alpha_init", 0.7)
        kwargs.setdefault("consistency_weight", 0.0)
        kwargs.setdefault("decorr_weight", 0.0)
        kwargs.setdefault("sparse_weight", 0.0)
        super().__init__(*args, **kwargs)

        stable_dim = int(self.encoder_head.stable_dim)
        if aeb_hidden is None:
            aeb_hidden = max(16, stable_dim // 2)
        if int(aeb_hidden) <= 0:
            raise ValueError("aeb_hidden must be positive")
        if float(aeb_eps) <= 0.0:
            raise ValueError("aeb_eps must be positive")

        self.aeb_hidden = int(aeb_hidden)
        self.aeb_beta = float(aeb_beta)
        self.aeb_eps = float(aeb_eps)

        self.budget_predictor = nn.Sequential(
            nn.LayerNorm(4 * stable_dim),
            nn.Linear(4 * stable_dim, self.aeb_hidden),
            nn.GELU(),
            nn.Linear(self.aeb_hidden, 1),
            nn.Sigmoid(),
        )

    def predict_budget(self, query_global: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        query_global = F.normalize(query_global, p=2, dim=-1)
        prototypes = F.normalize(prototypes, p=2, dim=-1)

        query_expanded = query_global.unsqueeze(1).expand(-1, prototypes.shape[0], -1)
        prototype_expanded = prototypes.unsqueeze(0).expand(query_global.shape[0], -1, -1)
        budget_features = torch.cat(
            [
                query_expanded,
                prototype_expanded,
                (query_expanded - prototype_expanded).abs(),
                query_expanded * prototype_expanded,
            ],
            dim=-1,
        )
        return self.budget_predictor(budget_features).squeeze(-1)

    def compute_local_adaptive_scores(
        self,
        query_tokens: torch.Tensor,
        support_token_pool: torch.Tensor,
        query_global: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_tokens = F.normalize(query_tokens, p=2, dim=-1)
        support_token_pool = F.normalize(support_token_pool, p=2, dim=-1)

        similarity = torch.einsum("qld,wtd->qwlt", query_tokens, support_token_pool)
        rho = self.predict_budget(query_global, prototypes)

        row_mean = similarity.mean(dim=-1)
        row_var = similarity.var(dim=-1, unbiased=False)
        row_std = row_var.clamp_min(0.0).add(self.aeb_eps).sqrt()
        threshold = row_mean + self.aeb_beta * (1.0 - rho).unsqueeze(-1) * row_std

        # Rows with larger predicted budget use a lower threshold and retain
        # more stable token evidence. If everything is filtered, fall back to a
        # softmax over raw similarities for that row to avoid zero division.
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

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)
        prototypes = self.build_support_prototypes(episode["support_global"])
        support_token_pool = self.build_support_token_pool(episode["support_tokens"])
        global_scores = self.compute_global_scores(episode["query_global"], prototypes)
        local_outputs = self.compute_local_adaptive_scores(
            query_tokens=episode["query_tokens"],
            support_token_pool=support_token_pool,
            query_global=episode["query_global"],
            prototypes=prototypes,
        )
        logits, alpha = self._fuse_scores(global_scores, local_outputs["local_scores"])

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
            "local_scores": local_outputs["local_scores"].detach(),
            "rho": local_outputs["rho"].detach(),
            "active_match_counts": local_outputs["active_match_counts"].detach(),
            "fallback_row_fraction": local_outputs["fallback_row_fraction"].detach(),
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

        first = diagnostics[0]
        return {
            "logits": logits,
            "global_scores": torch.cat([item["global_scores"] for item in diagnostics], dim=0),
            "local_scores": torch.cat([item["local_scores"] for item in diagnostics], dim=0),
            "rho": torch.cat([item["rho"] for item in diagnostics], dim=0),
            "active_match_counts": torch.cat([item["active_match_counts"] for item in diagnostics], dim=0),
            "fallback_row_fraction": torch.cat([item["fallback_row_fraction"] for item in diagnostics], dim=0),
            "mean_budget": torch.stack([item["rho"].mean() for item in diagnostics]).mean(),
            "alpha": first["alpha"],
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
