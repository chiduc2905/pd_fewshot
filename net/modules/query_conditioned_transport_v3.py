"""Query-conditioned transport scoring for SC-LFI v3.

Core formulas:
- query measure:
  `nu_q = sum_j rho_j delta_{u_j}`
- query-conditioned class relevance:
  `omega_{c,l}(q) = softmax_l(psi(u_q^pool, x_{c,l}, h_c, t_ctx))`
- class-side reweighted posterior measure:
  `muhat_c^q = Reweight(muhat_c; omega(q, c))`
- score:
  `score_c(q) = -tau * D_score(nu_q, muhat_c^q)`

Paper grounding:
- query-conditioned class-side relevance is inspired by local matching models

Our adaptation / novelty:
- keep class atoms fixed as the support-conditioned posterior predictive measure
- only adapt class-side masses per query-class pair
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.modules.transport_distance_v2 import WeightedTransportScoringDistanceV2


class QueryConditionedTransportScorerV3(nn.Module):
    """Reweight posterior class masses per query-class pair and score transport fit."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        *,
        hidden_dim: int | None = None,
        relevance_temperature: float = 1.0,
        use_query_reweighting: bool = True,
        eps: float = 1e-8,
        score_train_num_projections: int = 64,
        score_eval_num_projections: int = 128,
        score_sw_p: float = 2.0,
        score_normalize_inputs: bool = True,
        score_train_projection_mode: str = "resample",
        score_eval_projection_mode: str = "fixed",
        score_eval_num_repeats: int = 1,
        score_projection_seed: int = 7,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or context_dim <= 0:
            raise ValueError("latent_dim and context_dim must be positive")
        if relevance_temperature <= 0.0:
            raise ValueError("relevance_temperature must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        hidden_dim = int(hidden_dim or max(latent_dim, context_dim))
        self.relevance_temperature = float(relevance_temperature)
        self.use_query_reweighting = bool(use_query_reweighting)
        self.eps = float(eps)

        self.query_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.atom_proj = nn.Sequential(
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
        self.relevance_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.score_distance = WeightedTransportScoringDistanceV2(
            train_num_projections=score_train_num_projections,
            eval_num_projections=score_eval_num_projections,
            p=score_sw_p,
            normalize_inputs=score_normalize_inputs,
            train_projection_mode=score_train_projection_mode,
            eval_projection_mode=score_eval_projection_mode,
            eval_num_repeats=score_eval_num_repeats,
            projection_seed=score_projection_seed,
        )

    def _normalize_query_masses(self, query_masses: torch.Tensor) -> torch.Tensor:
        masses = query_masses.clamp_min(0.0)
        masses = masses / masses.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return masses

    def _query_pooled(self, query_atoms: torch.Tensor, query_masses: torch.Tensor) -> torch.Tensor:
        query_masses = self._normalize_query_masses(query_masses)
        return (query_masses.unsqueeze(-1) * query_atoms).sum(dim=1)

    def compute_query_conditioned_class_masses(
        self,
        query_atoms: torch.Tensor,
        query_masses: torch.Tensor,
        class_atoms: torch.Tensor,
        class_masses: torch.Tensor,
        class_summary: torch.Tensor,
        episode_context: torch.Tensor,
        *,
        relevance_temperature: float | None = None,
        reweight_strength: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_atoms.dim() != 3 or class_atoms.dim() != 3:
            raise ValueError("query_atoms and class_atoms must have shape (NumQuery/Way, Tokens, LatentDim)")
        if query_masses.shape != query_atoms.shape[:-1]:
            raise ValueError("query_masses must match query_atoms without latent dim")
        if class_masses.shape != class_atoms.shape[:-1]:
            raise ValueError("class_masses must match class_atoms without latent dim")
        if class_summary.dim() != 2 or episode_context.dim() != 2:
            raise ValueError("class_summary and episode_context must have shape (Way, ContextDim)")

        num_query = query_atoms.shape[0]
        way_num, num_atoms, _ = class_atoms.shape

        temperature = float(relevance_temperature or self.relevance_temperature)
        if temperature <= 0.0:
            raise ValueError("relevance_temperature must be positive")
        strength = 1.0 if reweight_strength is None else float(reweight_strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("reweight_strength must be in [0, 1]")

        if not self.use_query_reweighting:
            base = class_masses.unsqueeze(0).expand(num_query, -1, -1)
            entropy = -(base * torch.log(base.clamp_min(self.eps))).sum(dim=-1)
            return base, entropy

        query_pooled = self._query_pooled(query_atoms, query_masses)
        query_hidden = self.query_proj(query_pooled).unsqueeze(1).unsqueeze(1).expand(-1, way_num, num_atoms, -1)
        atom_hidden = self.atom_proj(class_atoms).unsqueeze(0).expand(num_query, -1, -1, -1)
        context_hidden = self.context_proj(torch.cat([class_summary, episode_context], dim=-1))
        context_hidden = context_hidden.unsqueeze(0).unsqueeze(2).expand(num_query, -1, num_atoms, -1)

        fused = torch.cat(
            [
                atom_hidden,
                query_hidden,
                atom_hidden * query_hidden,
                torch.abs(atom_hidden - query_hidden) + context_hidden,
            ],
            dim=-1,
        )
        relevance_logits = self.relevance_head(fused).squeeze(-1)
        combined_logits = torch.log(class_masses.clamp_min(self.eps)).unsqueeze(0) + relevance_logits / temperature
        reweighted = torch.softmax(combined_logits, dim=-1)
        mixed = (1.0 - strength) * class_masses.unsqueeze(0) + strength * reweighted
        mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        entropy = -(mixed * torch.log(mixed.clamp_min(self.eps))).sum(dim=-1)
        return mixed, entropy

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
        relevance_temperature: float | None = None,
        reweight_strength: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if score_temperature <= 0.0:
            raise ValueError("score_temperature must be positive")
        reweighted_class_masses, relevance_entropy = self.compute_query_conditioned_class_masses(
            query_atoms,
            query_masses,
            class_atoms,
            class_masses,
            class_summary,
            episode_context,
            relevance_temperature=relevance_temperature,
            reweight_strength=reweight_strength,
        )
        num_query, way_num = reweighted_class_masses.shape[:2]
        query_expanded = query_atoms.unsqueeze(1).expand(-1, way_num, -1, -1)
        class_expanded = class_atoms.unsqueeze(0).expand(num_query, -1, -1, -1)
        query_mass_expanded = self._normalize_query_masses(query_masses).unsqueeze(1).expand(-1, way_num, -1)
        distances = self.score_distance(
            query_expanded,
            class_expanded,
            source_masses=query_mass_expanded,
            target_masses=reweighted_class_masses,
            reduction="none",
        )
        logits = -float(score_temperature) * distances
        return {
            "logits": logits,
            "pairwise_distances": distances,
            "query_conditioned_class_masses": reweighted_class_masses,
            "relevance_entropy": relevance_entropy,
        }
