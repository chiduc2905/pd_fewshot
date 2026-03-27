"""Reliability-aware set modules for the practical Hierarchical V3 model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DualConsensusReliabilityHead(nn.Module):
    """Estimate shot reliability from vector and token consensus, then refine class repr."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim * 5 + 2),
            nn.Linear(dim * 5 + 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.class_fusion = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.local_gate = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.class_norm = nn.LayerNorm(dim)

    def forward(
        self,
        shot_embeddings: torch.Tensor,
        support_tokens: torch.Tensor,
        base_class_repr: torch.Tensor,
        class_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if shot_embeddings.dim() != 3:
            raise ValueError(
                "shot_embeddings must have shape (Way, Shot, Dim), "
                f"got {tuple(shot_embeddings.shape)}"
            )
        if support_tokens.dim() != 4:
            raise ValueError(
                "support_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        if base_class_repr.dim() != 2:
            raise ValueError(
                "base_class_repr must have shape (Way, Dim), "
                f"got {tuple(base_class_repr.shape)}"
            )
        if class_summary.dim() != 2:
            raise ValueError(
                "class_summary must have shape (Way, Dim), "
                f"got {tuple(class_summary.shape)}"
            )

        base_expanded = base_class_repr.unsqueeze(1).expand(-1, shot_embeddings.shape[1], -1)
        summary_expanded = class_summary.unsqueeze(1).expand(-1, shot_embeddings.shape[1], -1)
        token_consensus = support_tokens.mean(dim=1, keepdim=True)

        vector_disagreement = (shot_embeddings - base_expanded).square().mean(dim=-1, keepdim=True)
        summary_disagreement = (shot_embeddings - summary_expanded).square().mean(dim=-1, keepdim=True)
        token_disagreement = (support_tokens - token_consensus).square().mean(dim=(-1, -2)).unsqueeze(-1)

        score_inputs = torch.cat(
            [
                shot_embeddings,
                base_expanded,
                summary_expanded,
                torch.abs(shot_embeddings - base_expanded),
                torch.abs(shot_embeddings - summary_expanded),
                vector_disagreement,
                token_disagreement,
            ],
            dim=-1,
        )
        shot_logits = self.score(score_inputs).squeeze(-1)
        shot_weights = torch.softmax(shot_logits, dim=1)

        weighted_shots = torch.sum(shot_weights.unsqueeze(-1) * shot_embeddings, dim=1)
        class_repr = self.class_norm(
            base_class_repr
            + self.class_fusion(torch.cat([base_class_repr, weighted_shots, class_summary], dim=-1))
        )

        reliability_entropy = -(shot_weights.clamp_min(1e-6) * shot_weights.clamp_min(1e-6).log()).sum(dim=1)
        if shot_embeddings.shape[1] > 1:
            reliability_entropy = reliability_entropy / math.log(float(shot_embeddings.shape[1]))
        reliability_features = torch.stack(
            [
                torch.sum(shot_weights * token_disagreement.squeeze(-1), dim=1),
                reliability_entropy,
            ],
            dim=-1,
        )
        local_gate = 0.75 + 0.5 * torch.sigmoid(self.local_gate(reliability_features)).squeeze(-1)
        return class_repr, shot_weights, local_gate


class ShotConditionedTokenAdapter(nn.Module):
    """Inject per-shot reliability and class context into support tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(dim * 3 + 1),
            nn.Linear(dim * 3 + 1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        support_tokens: torch.Tensor,
        shot_embeddings: torch.Tensor,
        class_repr: torch.Tensor,
        shot_weights: torch.Tensor,
    ) -> torch.Tensor:
        if support_tokens.dim() != 4:
            raise ValueError(
                "support_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        class_expanded = class_repr.unsqueeze(1).expand(-1, shot_embeddings.shape[1], -1)
        shot_context = self.adapter(
            torch.cat(
                [
                    shot_embeddings,
                    class_expanded,
                    torch.abs(shot_embeddings - class_expanded),
                    shot_weights.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        conditioned = support_tokens + shot_context.unsqueeze(2)
        return self.output_norm(conditioned)


class ReliabilityCoupledTokenSWHead(nn.Module):
    """Combine merged-token SW and shot-wise SW with shared support reliability."""

    def __init__(
        self,
        sw_distance: nn.Module,
        score_scale: float = 16.0,
        merged_weight: float = 1.0,
        shot_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.sw_distance = sw_distance
        self.score_scale = float(score_scale)
        self.merged_weight = float(merged_weight)
        self.shot_weight = float(shot_weight)

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        merged_tokens: torch.Tensor,
        shot_weights: torch.Tensor,
        local_gate: torch.Tensor,
    ) -> torch.Tensor:
        if query_tokens.dim() != 3:
            raise ValueError(
                "query_tokens must have shape (NQ, Tokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        if support_tokens.dim() != 4:
            raise ValueError(
                "support_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )
        if merged_tokens.dim() != 3:
            raise ValueError(
                "merged_tokens must have shape (Way, Tokens, Dim), "
                f"got {tuple(merged_tokens.shape)}"
            )

        nq = query_tokens.shape[0]
        way_num, shot_num, token_num, dim = support_tokens.shape

        query_merged = query_tokens.unsqueeze(1).expand(-1, way_num, -1, -1)
        merged_distance = self.sw_distance(
            query_merged,
            merged_tokens.unsqueeze(0).expand(nq, -1, -1, -1),
            reduction="none",
        )

        query_shot = query_tokens.unsqueeze(1).unsqueeze(2).expand(-1, way_num, shot_num, -1, -1)
        shot_distance = self.sw_distance(
            query_shot.reshape(nq * way_num * shot_num, token_num, dim),
            support_tokens.unsqueeze(0).expand(nq, -1, -1, -1, -1).reshape(
                nq * way_num * shot_num,
                token_num,
                dim,
            ),
            reduction="none",
        ).reshape(nq, way_num, shot_num)
        weighted_shot_distance = torch.sum(shot_distance * shot_weights.unsqueeze(0), dim=-1)

        total_distance = self.merged_weight * merged_distance + self.shot_weight * weighted_shot_distance
        return -self.score_scale * local_gate.unsqueeze(0) * total_distance
