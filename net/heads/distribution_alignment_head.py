"""Distribution-aware heads for hierarchical few-shot matching."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.metrics.sliced_wasserstein import SlicedWassersteinDistance, merge_support_tokens_by_class


class DistributionAlignmentSWHead(nn.Module):
    """SW alignment head for hierarchical token outputs."""

    def __init__(
        self,
        sw_distance: SlicedWassersteinDistance,
        merge_mode: str = "concat",
        score_scale: float = 8.0,
    ) -> None:
        super().__init__()
        self.sw_distance = sw_distance
        self.merge_mode = merge_mode
        self.score_scale = score_scale

    def forward(self, query_tokens: torch.Tensor, support_tokens: torch.Tensor) -> torch.Tensor:
        support_merged = merge_support_tokens_by_class(support_tokens, merge_mode=self.merge_mode)
        support_expanded = support_merged.unsqueeze(0).expand(query_tokens.shape[0], -1, -1, -1)
        return -self.score_scale * self.sw_distance(query_tokens, support_expanded, reduction="none")


class HierarchicalQueryMatcher(nn.Module):
    """Match query shot descriptors against hierarchical class memories."""

    def __init__(self, dim: int, temperature: float = 16.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_embeddings: torch.Tensor, class_memories: torch.Tensor) -> torch.Tensor:
        query_expanded = query_embeddings.unsqueeze(1).expand(-1, class_memories.shape[0], -1)
        class_expanded = class_memories.unsqueeze(0).expand(query_embeddings.shape[0], -1, -1)
        fused = self.fusion(torch.cat([query_expanded, class_expanded], dim=-1))
        fused = self.norm(fused + query_expanded)
        return self.temperature * torch.einsum(
            "nwd,wd->nw",
            F.normalize(fused, p=2, dim=-1),
            F.normalize(class_memories, p=2, dim=-1),
        )


class HierarchicalSWMetricHead(DistributionAlignmentSWHead):
    """SW metric head for hierarchical token distributions."""
