"""Query-conditioned transport metric scoring for SC-LFI v4.

Core formulas:
- query measure:
  `nu_q = sum_j rho_j^q delta_{u_j}`
- fixed class posterior:
  `muhat_c = sum_l a_{c,l} delta_{x_{c,l}}`
- query-conditioned diagonal transport metric:
  `C_{q,c}(x, y) = ||diag(s_{q,c})^{1/2} x - diag(s_{q,c})^{1/2} y||_2^2`
- score:
  `score_c(q) = -tau * D( nu_q, muhat_c ; C_{q,c} )`

Paper grounding:
- query-conditioned pairwise matching is inspired by local few-shot matching
  models, but the class distribution itself stays fixed.

Our adaptation / novelty:
- keep the class posterior predictive measure fixed;
- adapt only the transport geometry per query-class pair.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.modules.transport_distance_v2 import (
    WeightedEntropicOTDistanceV2,
    WeightedTransportScoringDistanceV2,
    normalize_measure_masses,
)


class QueryConditionedTransportScorerV4(nn.Module):
    """Score query-class pairs with a query-conditioned transport metric."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        *,
        hidden_dim: int | None = None,
        use_metric_conditioning: bool = True,
        distance_type: str = "weighted_entropic_ot",
        train_num_projections: int = 64,
        eval_num_projections: int = 128,
        sw_p: float = 2.0,
        normalize_inputs: bool = True,
        train_projection_mode: str = "resample",
        eval_projection_mode: str = "fixed",
        eval_num_repeats: int = 1,
        projection_seed: int = 7,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 60,
        sinkhorn_cost_power: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or context_dim <= 0:
            raise ValueError("latent_dim and context_dim must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        hidden_dim = int(hidden_dim or max(latent_dim, context_dim))
        self.latent_dim = int(latent_dim)
        self.use_metric_conditioning = bool(use_metric_conditioning)
        self.distance_type = str(distance_type).lower()
        self.eps = float(eps)

        self.query_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim * 2),
            nn.Linear(context_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.metric_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        if self.distance_type == "weighted_entropic_ot":
            self.distance = WeightedEntropicOTDistanceV2(
                sinkhorn_epsilon=sinkhorn_epsilon,
                max_iterations=sinkhorn_iterations,
                cost_power=sinkhorn_cost_power,
                normalize_inputs=normalize_inputs,
                reduction="mean",
                eps=eps,
            )
        elif self.distance_type == "weighted_sw":
            self.distance = WeightedTransportScoringDistanceV2(
                train_num_projections=train_num_projections,
                eval_num_projections=eval_num_projections,
                p=sw_p,
                normalize_inputs=normalize_inputs,
                train_projection_mode=train_projection_mode,
                eval_projection_mode=eval_projection_mode,
                eval_num_repeats=eval_num_repeats,
                projection_seed=projection_seed,
            )
        else:
            raise ValueError(f"Unsupported scoring distance type: {distance_type}")

    def _normalize_query_masses(self, query_masses: torch.Tensor) -> torch.Tensor:
        return normalize_measure_masses(
            query_masses,
            target_shape=query_masses.shape,
            device=query_masses.device,
            dtype=query_masses.dtype,
            eps=self.eps,
        )

    def _query_pooled(self, query_atoms: torch.Tensor, query_masses: torch.Tensor) -> torch.Tensor:
        query_masses = self._normalize_query_masses(query_masses)
        return (query_masses.unsqueeze(-1) * query_atoms).sum(dim=1)

    def compute_metric_scales(
        self,
        query_atoms: torch.Tensor,
        query_masses: torch.Tensor,
        class_summary: torch.Tensor,
        episode_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return positive per-query-class diagonal metric scales.

        Shapes:
        - query_atoms: `[NumQuery, QueryAtoms, LatentDim]`
        - query_masses: `[NumQuery, QueryAtoms]`
        - class_summary: `[Way, ContextDim]`
        - episode_context: `[Way, ContextDim]`
        - scales: `[NumQuery, Way, LatentDim]`
        """

        if query_atoms.dim() != 3:
            raise ValueError(f"query_atoms must have shape (NumQuery, QueryAtoms, LatentDim), got {tuple(query_atoms.shape)}")
        if query_masses.shape != query_atoms.shape[:-1]:
            raise ValueError("query_masses must match query_atoms without latent dim")
        if class_summary.dim() != 2 or episode_context.dim() != 2:
            raise ValueError("class_summary and episode_context must have shape (Way, ContextDim)")

        query_pooled = self._query_pooled(query_atoms, query_masses)
        if not self.use_metric_conditioning:
            return query_atoms.new_ones((query_atoms.shape[0], class_summary.shape[0], self.latent_dim))
        query_hidden = self.query_proj(query_pooled).unsqueeze(1)
        context_hidden = self.context_proj(torch.cat([class_summary, episode_context], dim=-1)).unsqueeze(0)
        fused = torch.cat(
            [
                query_hidden.expand(-1, context_hidden.shape[1], -1),
                context_hidden.expand(query_hidden.shape[0], -1, -1),
                query_hidden.expand(-1, context_hidden.shape[1], -1) * context_hidden.expand(query_hidden.shape[0], -1, -1),
                torch.abs(
                    query_hidden.expand(-1, context_hidden.shape[1], -1)
                    - context_hidden.expand(query_hidden.shape[0], -1, -1)
                ),
            ],
            dim=-1,
        )
        return F.softplus(self.metric_head(fused)) + self.eps

    def score(
        self,
        query_atoms: torch.Tensor,
        query_masses: torch.Tensor,
        class_atoms: torch.Tensor,
        class_masses: torch.Tensor,
        class_summary: torch.Tensor,
        episode_context: torch.Tensor,
        *,
        score_temperature: float,
    ) -> dict[str, torch.Tensor]:
        if score_temperature <= 0.0:
            raise ValueError("score_temperature must be positive")
        if class_atoms.dim() != 3:
            raise ValueError(f"class_atoms must have shape (Way, ClassAtoms, LatentDim), got {tuple(class_atoms.shape)}")
        if class_masses.shape != class_atoms.shape[:-1]:
            raise ValueError("class_masses must match class_atoms without latent dim")

        num_query = query_atoms.shape[0]
        way_num = class_atoms.shape[0]
        scales = self.compute_metric_scales(query_atoms, query_masses, class_summary, episode_context)
        sqrt_scales = torch.sqrt(scales).unsqueeze(2)

        query_mass_expanded = self._normalize_query_masses(query_masses).unsqueeze(1).expand(-1, way_num, -1)
        class_mass_expanded = normalize_measure_masses(
            class_masses.unsqueeze(0).expand(num_query, -1, -1),
            target_shape=(num_query, way_num, class_atoms.shape[1]),
            device=class_atoms.device,
            dtype=class_atoms.dtype,
            eps=self.eps,
        )

        query_expanded = query_atoms.unsqueeze(1).expand(-1, way_num, -1, -1) * sqrt_scales
        class_expanded = class_atoms.unsqueeze(0).expand(num_query, -1, -1, -1) * sqrt_scales
        distances = self.distance(
            query_expanded,
            class_expanded,
            source_masses=query_mass_expanded,
            target_masses=class_mass_expanded,
            reduction="none",
        )
        logits = -float(score_temperature) * distances
        return {
            "logits": logits,
            "pairwise_distances": distances,
            "metric_scales": scales,
        }
