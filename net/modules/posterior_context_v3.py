"""Posterior context builder for SC-LFI v3.

This module implements the class posterior base measure before any transport.

Core formulas:
- support evidence basis measure:
  `nu_c^basis = sum_i a_i^basis delta_{s_i}`
- support-conditioned prior measure:
  `pi_c = sum_r b_r delta_{p_r}`
- shot-aware shrinkage:
  `alpha_c = sigma(g_phi(h_c, t_ctx, uncert_c, log(1 + K)))`
- posterior base measure:
  `mu_c^0 = alpha_c * nu_c^basis + (1 - alpha_c) * pi_c`

Paper grounding:
- set-wise episode adaptation follows FEAT-style task adaptation in a light form
- compact support memory is consistent with set-based conditioning
- uncertainty-aware shrinkage is motivated by probabilistic few-shot modeling

Our adaptation / novelty:
- compress raw support evidence into a fixed-size invariant support basis
- predict a support-conditioned prior measure
- form an explicit posterior base measure before flow transport
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.modules.transport_distance_v2 import normalize_measure_masses
from net.ssm.set_invariant_pool import SetInvariantMemoryPool


class EpisodeSummaryAdapterV3(nn.Module):
    """Lightweight FEAT-style episode adapter over class summaries.

    Shapes:
    - class_summaries: `[Way, ContextDim]`
    - adapted summaries: `[Way, ContextDim]`
    - episode context: `[Way, ContextDim]`
    """

    def __init__(self, context_dim: int, num_heads: int = 4, hidden_multiplier: int = 2) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        hidden_dim = max(context_dim, context_dim * int(hidden_multiplier))
        attn_heads = num_heads if context_dim % num_heads == 0 else 1

        self.input_norm = nn.LayerNorm(context_dim)
        self.self_attn = nn.MultiheadAttention(context_dim, attn_heads, dropout=0.0, batch_first=True)
        self.attn_norm = nn.LayerNorm(context_dim)
        self.ffn_norm = nn.LayerNorm(context_dim)
        self.ffn = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )
        self.episode_proj = nn.Sequential(
            nn.LayerNorm(context_dim * 2),
            nn.Linear(context_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )
        self.episode_norm = nn.LayerNorm(context_dim)

    def forward(self, class_summaries: torch.Tensor) -> dict[str, torch.Tensor]:
        if class_summaries.dim() != 2:
            raise ValueError(
                f"class_summaries must have shape (Way, ContextDim), got {tuple(class_summaries.shape)}"
            )
        inputs = self.input_norm(class_summaries).unsqueeze(0)
        attended, _ = self.self_attn(inputs, inputs, inputs)
        adapted = self.attn_norm(inputs + attended)
        adapted = adapted + self.ffn(self.ffn_norm(adapted))
        adapted = adapted.squeeze(0)

        pooled_mean = adapted.mean(dim=0, keepdim=True).expand_as(adapted)
        pooled_max = adapted.max(dim=0, keepdim=True).values.expand_as(adapted)
        episode_context = self.episode_norm(self.episode_proj(torch.cat([pooled_mean, pooled_max], dim=-1)) + pooled_mean)
        return {
            "adapted_class_summary": adapted,
            "episode_context": episode_context,
        }


class PosteriorContextBuilderV3(nn.Module):
    """Build support-conditioned posterior base measures for each class.

    Tensor shapes:
    - support_latents: `[Way, SupportTokens, LatentDim]`
    - support_masses: `[Way, SupportTokens]`
    - support_basis atoms: `[Way, BasisAtoms, LatentDim]`
    - support_basis masses: `[Way, BasisAtoms]`
    - support_memory: `[Way, MemorySize, LatentDim]`
    - class_summary: `[Way, ContextDim]`
    - prior_atoms: `[Way, PriorAtoms, LatentDim]`
    - prior_masses: `[Way, PriorAtoms]`
    - base_atoms: `[Way, BaseAtoms, LatentDim]`
    - base_masses: `[Way, BaseAtoms]`
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        *,
        memory_size: int = 4,
        memory_num_heads: int = 4,
        memory_ffn_multiplier: int = 2,
        summary_hidden_dim: int | None = None,
        mass_feature_dim: int | None = None,
        prior_num_atoms: int = 4,
        episode_num_heads: int = 4,
        episode_hidden_multiplier: int = 2,
        alpha_hidden_dim: int | None = None,
        prior_scale: float = 0.5,
        alpha_shot_scale: float = 1.0,
        alpha_uncertainty_scale: float = 1.0,
        use_episode_adapter: bool = True,
        use_support_barycenter_only: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or context_dim <= 0:
            raise ValueError("latent_dim and context_dim must be positive")
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if prior_num_atoms < 0:
            raise ValueError("prior_num_atoms must be non-negative")
        if prior_scale < 0.0:
            raise ValueError("prior_scale must be non-negative")
        if alpha_shot_scale < 0.0 or alpha_uncertainty_scale < 0.0:
            raise ValueError("alpha scaling factors must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.prior_num_atoms = int(prior_num_atoms)
        self.prior_scale = float(prior_scale)
        self.alpha_shot_scale = float(alpha_shot_scale)
        self.alpha_uncertainty_scale = float(alpha_uncertainty_scale)
        self.use_episode_adapter = bool(use_episode_adapter)
        self.use_support_barycenter_only = bool(use_support_barycenter_only)
        self.eps = float(eps)

        mass_feature_dim = int(mass_feature_dim or latent_dim)
        summary_hidden_dim = int(summary_hidden_dim or max(latent_dim, context_dim))
        alpha_hidden_dim = int(alpha_hidden_dim or max(context_dim, latent_dim))

        self.mass_embed = nn.Sequential(
            nn.Linear(1, mass_feature_dim),
            nn.GELU(),
            nn.Linear(mass_feature_dim, latent_dim),
        )
        self.memory_pool = SetInvariantMemoryPool(
            dim=latent_dim,
            num_memory_tokens=int(memory_size),
            num_heads=int(memory_num_heads),
            ffn_multiplier=int(memory_ffn_multiplier),
        )
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(latent_dim * 4),
            nn.Linear(latent_dim * 4, summary_hidden_dim),
            nn.GELU(),
            nn.Linear(summary_hidden_dim, context_dim),
        )
        self.summary_norm = nn.LayerNorm(context_dim)
        self.memory_norm = nn.LayerNorm(latent_dim)
        self.episode_adapter = EpisodeSummaryAdapterV3(
            context_dim=context_dim,
            num_heads=episode_num_heads,
            hidden_multiplier=episode_hidden_multiplier,
        )
        prior_input_dim = context_dim * 2 + 5
        self.prior_atom_head = nn.Sequential(
            nn.LayerNorm(prior_input_dim),
            nn.Linear(prior_input_dim, alpha_hidden_dim),
            nn.GELU(),
            nn.Linear(alpha_hidden_dim, max(1, self.prior_num_atoms) * latent_dim),
        )
        self.prior_mass_head = nn.Sequential(
            nn.LayerNorm(latent_dim + context_dim * 2),
            nn.Linear(latent_dim + context_dim * 2, alpha_hidden_dim),
            nn.GELU(),
            nn.Linear(alpha_hidden_dim, 1),
        )
        self.support_basis_mass_head = nn.Sequential(
            nn.LayerNorm(latent_dim * 2 + context_dim * 2),
            nn.Linear(latent_dim * 2 + context_dim * 2, alpha_hidden_dim),
            nn.GELU(),
            nn.Linear(alpha_hidden_dim, 1),
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(context_dim * 2 + 4),
            nn.Linear(context_dim * 2 + 4, alpha_hidden_dim),
            nn.GELU(),
            nn.Linear(alpha_hidden_dim, 1),
        )

    def _weighted_stats(
        self,
        support_latents: torch.Tensor,
        support_masses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_masses = normalize_measure_masses(
            support_masses,
            target_shape=support_masses.shape,
            device=support_latents.device,
            dtype=support_latents.dtype,
            eps=self.eps,
        )
        weighted_mean = (normalized_masses.unsqueeze(-1) * support_latents).sum(dim=1)
        centered = support_latents - weighted_mean.unsqueeze(1)
        weighted_var = (normalized_masses.unsqueeze(-1) * centered.pow(2)).sum(dim=1)
        weighted_std = torch.sqrt(weighted_var + self.eps)
        mass_entropy = -(normalized_masses * torch.log(normalized_masses.clamp_min(self.eps))).sum(dim=-1, keepdim=True)
        max_mass = normalized_masses.max(dim=-1, keepdim=True).values
        dispersion = weighted_var.mean(dim=-1, keepdim=True)
        uncertainty_stats = torch.cat([dispersion, mass_entropy, max_mass, 1.0 - max_mass], dim=-1)
        return normalized_masses, weighted_mean, weighted_std, uncertainty_stats

    def _build_support_basis(
        self,
        support_memory: torch.Tensor,
        weighted_mean: torch.Tensor,
        weighted_std: torch.Tensor,
        class_summary: torch.Tensor,
        episode_context: torch.Tensor,
        *,
        shot_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_support_barycenter_only:
            support_atoms = weighted_mean.unsqueeze(1)
            support_atom_masses = weighted_mean.new_ones((weighted_mean.shape[0], 1))
            return support_atoms, support_atom_masses

        support_atoms = torch.cat([weighted_mean.unsqueeze(1), support_memory], dim=1)
        mean_offsets = support_atoms - weighted_mean.unsqueeze(1)
        basis_inputs = torch.cat(
            [
                support_atoms,
                mean_offsets / weighted_std.unsqueeze(1).clamp_min(self.eps),
                class_summary.unsqueeze(1).expand(-1, support_atoms.shape[1], -1),
                episode_context.unsqueeze(1).expand(-1, support_atoms.shape[1], -1),
            ],
            dim=-1,
        )
        basis_logits = self.support_basis_mass_head(basis_inputs).squeeze(-1)
        basis_logits[:, 0] = basis_logits[:, 0] + 0.15 * math.log1p(float(shot_num))
        support_atom_masses = normalize_measure_masses(
            torch.softmax(basis_logits, dim=-1),
            target_shape=basis_logits.shape,
            device=support_atoms.device,
            dtype=support_atoms.dtype,
            eps=self.eps,
        )
        return support_atoms, support_atom_masses

    @staticmethod
    def _alpha_bounds(shot_num: int) -> tuple[float, float]:
        if shot_num <= 1:
            return 0.35, 0.72
        if shot_num >= 5:
            return 0.55, 0.88
        blend = float(shot_num - 1) / 4.0
        alpha_min = 0.35 + blend * (0.55 - 0.35)
        alpha_max = 0.72 + blend * (0.88 - 0.72)
        return alpha_min, alpha_max

    def forward(
        self,
        support_latents: torch.Tensor,
        support_masses: torch.Tensor,
        *,
        shot_num: int,
    ) -> dict[str, torch.Tensor]:
        if support_latents.dim() != 3:
            raise ValueError(
                f"support_latents must have shape (Way, SupportTokens, LatentDim), got {tuple(support_latents.shape)}"
            )
        if support_masses.shape != support_latents.shape[:-1]:
            raise ValueError(
                "support_masses must match support_latents without the latent dim: "
                f"latents={tuple(support_latents.shape)} masses={tuple(support_masses.shape)}"
            )
        if shot_num <= 0:
            raise ValueError("shot_num must be positive")

        way_num = support_latents.shape[0]
        normalized_support_masses, weighted_mean, weighted_std, uncertainty_stats = self._weighted_stats(
            support_latents,
            support_masses,
        )
        mass_features = self.mass_embed(torch.log(normalized_support_masses.clamp_min(self.eps)).unsqueeze(-1))
        memory_inputs = support_latents + mass_features
        support_memory, pooled_summary = self.memory_pool(memory_inputs)
        support_memory = self.memory_norm(support_memory)

        raw_class_summary = self.summary_norm(
            self.summary_proj(torch.cat([weighted_mean, weighted_std, pooled_summary, support_memory.mean(dim=1)], dim=-1))
        )
        if self.use_episode_adapter:
            adapted = self.episode_adapter(raw_class_summary)
            class_summary = adapted["adapted_class_summary"]
            episode_context = adapted["episode_context"]
        else:
            class_summary = raw_class_summary
            episode_context = raw_class_summary.mean(dim=0, keepdim=True).expand_as(raw_class_summary)

        support_atoms, support_atom_masses = self._build_support_basis(
            support_memory,
            weighted_mean,
            weighted_std,
            class_summary,
            episode_context,
            shot_num=shot_num,
        )

        shot_feature = support_latents.new_full((way_num, 1), math.log1p(float(shot_num)))
        alpha_inputs = torch.cat([class_summary, episode_context, uncertainty_stats], dim=-1)
        dispersion = uncertainty_stats[:, :1]
        raw_alpha = 0.5 * self.alpha_head(alpha_inputs)
        raw_alpha = raw_alpha + self.alpha_shot_scale * shot_feature - self.alpha_uncertainty_scale * dispersion
        alpha_min, alpha_max = self._alpha_bounds(shot_num)
        alpha = alpha_min + (alpha_max - alpha_min) * torch.sigmoid(raw_alpha).squeeze(-1)

        if self.prior_num_atoms > 0:
            prior_context = torch.cat([class_summary, episode_context, uncertainty_stats, shot_feature], dim=-1)
            prior_offsets = self.prior_atom_head(prior_context).reshape(way_num, self.prior_num_atoms, self.latent_dim)
            scale = self.prior_scale * (1.0 + weighted_std.mean(dim=-1, keepdim=True)).unsqueeze(-1)
            prior_atoms = weighted_mean.unsqueeze(1) + torch.tanh(prior_offsets) * scale
            prior_context_expanded = torch.cat(
                [
                    prior_atoms,
                    class_summary.unsqueeze(1).expand(-1, self.prior_num_atoms, -1),
                    episode_context.unsqueeze(1).expand(-1, self.prior_num_atoms, -1),
                ],
                dim=-1,
            )
            prior_logits = self.prior_mass_head(prior_context_expanded).squeeze(-1)
            prior_masses = normalize_measure_masses(
                torch.softmax(prior_logits, dim=-1),
                target_shape=prior_logits.shape,
                device=prior_atoms.device,
                dtype=prior_atoms.dtype,
                eps=self.eps,
            )
        else:
            prior_atoms = support_latents.new_zeros((way_num, 0, self.latent_dim))
            prior_masses = support_latents.new_zeros((way_num, 0))

        if self.prior_num_atoms > 0:
            alpha_support = alpha.unsqueeze(-1) * support_atom_masses
            alpha_prior = (1.0 - alpha).unsqueeze(-1) * prior_masses
            base_atoms = torch.cat([support_atoms, prior_atoms], dim=1)
            base_masses = torch.cat([alpha_support, alpha_prior], dim=1)
        else:
            base_atoms = support_atoms
            base_masses = support_atom_masses
        base_masses = normalize_measure_masses(
            base_masses,
            target_shape=base_masses.shape,
            device=base_atoms.device,
            dtype=base_atoms.dtype,
            eps=self.eps,
        )

        return {
            "support_atoms": support_atoms,
            "support_masses": support_atom_masses,
            "support_memory": support_memory,
            "class_summary": class_summary,
            "episode_context": episode_context,
            "weighted_mean": weighted_mean,
            "weighted_std": weighted_std,
            "uncertainty_stats": uncertainty_stats,
            "prior_atoms": prior_atoms,
            "prior_masses": prior_masses,
            "alpha": alpha,
            "base_atoms": base_atoms,
            "base_masses": base_masses,
        }
