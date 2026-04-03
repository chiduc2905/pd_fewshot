"""Hierarchical posterior context builder for SC-LFI v4.

Core formulas:
- per-shot support basis:
  `nu_{c,k}^{basis} = sum_r omega_{c,k,r} delta_{b_{c,k,r}}`
- shot aggregation:
  `nu_c^{emp} = sum_k pi_{c,k} nu_{c,k}^{basis}`
- support-conditioned meta-prior:
  `pi_c = sum_r beta_{c,r}^{prior} delta_{p_{c,r}}`
- shot-aware shrinkage:
  `alpha_c = (K / (K + kappa)) * sigma(g_phi(h_c, t_ctx, uncert_c))`
- posterior base measure:
  `mu_c^0 = alpha_c * nu_c^{emp} + (1 - alpha_c) * pi_c`

Paper grounding:
- set-wise episode adaptation follows FEAT-style task adaptation;
- hierarchical shot preservation follows the few-shot lesson that each support
  image is an observation, not just another token in one flattened bag;
- shrinkage toward a meta-prior is aligned with probabilistic few-shot
  modeling and calibration ideas.

Our adaptation / novelty:
- preserve support at the shot level before class aggregation;
- construct a global meta-prior dictionary and attend to it per class;
- form an explicit posterior base measure over shot-basis atoms and prior atoms
  before any residual flow transport.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from net.modules.transport_distance_v2 import normalize_measure_masses
from net.ssm.set_invariant_pool import SetInvariantMemoryPool


class EpisodeSummaryAdapterV4(nn.Module):
    """Permutation-invariant episode adapter over class summaries.

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


class HierarchicalPosteriorContextBuilderV4(nn.Module):
    """Build shot-aware posterior base measures for SC-LFI v4.

    Tensor shapes:
    - support_latents: `[Way, Shot, Tokens, LatentDim]`
    - support_token_masses: `[Way, Shot, Tokens]`
    - shot_atoms: `[Way, Shot, ShotBasis, LatentDim]`
    - shot_basis_masses: `[Way, Shot, ShotBasis]`
    - shot_masses: `[Way, Shot]`
    - support_atoms: `[Way, Shot * ShotBasis, LatentDim]`
    - support_masses: `[Way, Shot * ShotBasis]`
    - class_summary: `[Way, ContextDim]`
    - episode_context: `[Way, ContextDim]`
    - class_memory: `[Way, ClassMemory, LatentDim]`
    - prior_atoms: `[Way, PriorAtoms, LatentDim]`
    - prior_masses: `[Way, PriorAtoms]`
    - base_atoms: `[Way, BaseAtoms, LatentDim]`
    - base_masses: `[Way, BaseAtoms]`

    Notes on fidelity:
    - shot preservation and the meta-prior are our few-shot-specific redesign;
    - shot basis construction uses invariant memory pooling as an engineering
      approximation to a compact local evidence basis.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        *,
        shot_memory_size: int = 2,
        class_memory_size: int = 4,
        memory_num_heads: int = 4,
        memory_ffn_multiplier: int = 2,
        summary_hidden_dim: int | None = None,
        prior_num_atoms: int = 4,
        global_prior_size: int = 16,
        episode_num_heads: int = 4,
        episode_hidden_multiplier: int = 2,
        alpha_hidden_dim: int | None = None,
        prior_scale: float = 0.25,
        shrinkage_kappa: float = 2.0,
        alpha_uncertainty_scale: float = 1.0,
        use_episode_adapter: bool = True,
        use_shot_barycenter_only: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or context_dim <= 0:
            raise ValueError("latent_dim and context_dim must be positive")
        if shot_memory_size < 0 or class_memory_size <= 0:
            raise ValueError("shot_memory_size must be non-negative and class_memory_size must be positive")
        if prior_num_atoms < 0:
            raise ValueError("prior_num_atoms must be non-negative")
        if global_prior_size <= 0:
            raise ValueError("global_prior_size must be positive")
        if prior_scale < 0.0:
            raise ValueError("prior_scale must be non-negative")
        if shrinkage_kappa <= 0.0:
            raise ValueError("shrinkage_kappa must be positive")
        if alpha_uncertainty_scale < 0.0:
            raise ValueError("alpha_uncertainty_scale must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.shot_basis_size = 1 if bool(use_shot_barycenter_only) else 1 + int(shot_memory_size)
        self.class_memory_size = int(class_memory_size)
        self.prior_num_atoms = int(prior_num_atoms)
        self.global_prior_size = int(global_prior_size)
        self.prior_scale = float(prior_scale)
        self.shrinkage_kappa = float(shrinkage_kappa)
        self.alpha_uncertainty_scale = float(alpha_uncertainty_scale)
        self.use_episode_adapter = bool(use_episode_adapter)
        self.use_shot_barycenter_only = bool(use_shot_barycenter_only)
        self.eps = float(eps)

        summary_hidden_dim = int(summary_hidden_dim or max(latent_dim, context_dim))
        alpha_hidden_dim = int(alpha_hidden_dim or max(latent_dim, context_dim))

        self.token_mass_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        if self.use_shot_barycenter_only:
            self.shot_token_pool = None
        else:
            self.shot_token_pool = SetInvariantMemoryPool(
                dim=latent_dim,
                num_memory_tokens=int(shot_memory_size),
                num_heads=int(memory_num_heads),
                ffn_multiplier=int(memory_ffn_multiplier),
            )
        self.shot_repr_proj = nn.Sequential(
            nn.LayerNorm(latent_dim * 4 + 4),
            nn.Linear(latent_dim * 4 + 4, summary_hidden_dim),
            nn.GELU(),
            nn.Linear(summary_hidden_dim, context_dim),
        )
        self.shot_repr_norm = nn.LayerNorm(context_dim)
        self.shot_basis_mass_head = nn.Sequential(
            nn.LayerNorm(latent_dim * 2 + context_dim + 4),
            nn.Linear(latent_dim * 2 + context_dim + 4, alpha_hidden_dim),
            nn.GELU(),
            nn.Linear(alpha_hidden_dim, 1),
        )
        self.class_summary_pool = SetInvariantMemoryPool(
            dim=context_dim,
            num_memory_tokens=int(class_memory_size),
            num_heads=int(episode_num_heads),
            ffn_multiplier=int(episode_hidden_multiplier),
        )
        self.shot_anchor_proj = nn.Sequential(
            nn.LayerNorm(latent_dim + context_dim),
            nn.Linear(latent_dim + context_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.class_memory_pool = SetInvariantMemoryPool(
            dim=latent_dim,
            num_memory_tokens=int(class_memory_size),
            num_heads=int(memory_num_heads),
            ffn_multiplier=int(memory_ffn_multiplier),
        )
        self.class_summary_norm = nn.LayerNorm(context_dim)
        self.class_memory_norm = nn.LayerNorm(latent_dim)
        self.episode_adapter = EpisodeSummaryAdapterV4(
            context_dim=context_dim,
            num_heads=episode_num_heads,
            hidden_multiplier=episode_hidden_multiplier,
        )
        self.shot_mass_head = nn.Sequential(
            nn.LayerNorm(context_dim * 3 + 4),
            nn.Linear(context_dim * 3 + 4, alpha_hidden_dim),
            nn.GELU(),
            nn.Linear(alpha_hidden_dim, 1),
        )

        self.global_prior_atoms = nn.Parameter(torch.randn(self.global_prior_size, latent_dim) * 0.02)
        self.global_prior_keys = nn.Parameter(torch.randn(self.global_prior_size, context_dim) * 0.02)
        prior_context_dim = context_dim * 2 + 5
        if self.prior_num_atoms > 0:
            self.prior_query_head = nn.Sequential(
                nn.LayerNorm(prior_context_dim),
                nn.Linear(prior_context_dim, alpha_hidden_dim),
                nn.GELU(),
                nn.Linear(alpha_hidden_dim, self.prior_num_atoms * context_dim),
            )
            self.prior_offset_head = nn.Sequential(
                nn.LayerNorm(prior_context_dim),
                nn.Linear(prior_context_dim, alpha_hidden_dim),
                nn.GELU(),
                nn.Linear(alpha_hidden_dim, self.prior_num_atoms * latent_dim),
            )
            self.prior_mass_head = nn.Sequential(
                nn.LayerNorm(latent_dim + context_dim * 2 + 4),
                nn.Linear(latent_dim + context_dim * 2 + 4, alpha_hidden_dim),
                nn.GELU(),
                nn.Linear(alpha_hidden_dim, 1),
            )
        else:
            self.prior_query_head = None
            self.prior_offset_head = None
            self.prior_mass_head = None

        self.alpha_head = nn.Sequential(
            nn.LayerNorm(context_dim * 2 + 5),
            nn.Linear(context_dim * 2 + 5, alpha_hidden_dim),
            nn.GELU(),
            nn.Linear(alpha_hidden_dim, 1),
        )

    def _weighted_token_stats(
        self,
        support_latents: torch.Tensor,
        support_token_masses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = normalize_measure_masses(
            support_token_masses,
            target_shape=support_token_masses.shape,
            device=support_latents.device,
            dtype=support_latents.dtype,
            eps=self.eps,
        )
        weighted_mean = (normalized.unsqueeze(-1) * support_latents).sum(dim=2)
        centered = support_latents - weighted_mean.unsqueeze(2)
        weighted_var = (normalized.unsqueeze(-1) * centered.pow(2)).sum(dim=2)
        weighted_std = torch.sqrt(weighted_var + self.eps)
        entropy = -(normalized * torch.log(normalized.clamp_min(self.eps))).sum(dim=-1, keepdim=True)
        max_mass = normalized.max(dim=-1, keepdim=True).values
        dispersion = weighted_var.mean(dim=-1, keepdim=True)
        uncertainty = torch.cat([dispersion, entropy, max_mass, 1.0 - max_mass], dim=-1)
        return normalized, weighted_mean, weighted_std, uncertainty

    def forward(
        self,
        support_latents: torch.Tensor,
        support_token_masses: torch.Tensor,
        *,
        shot_num: int,
    ) -> dict[str, torch.Tensor]:
        if support_latents.dim() != 4:
            raise ValueError(
                "support_latents must have shape (Way, Shot, Tokens, LatentDim), "
                f"got {tuple(support_latents.shape)}"
            )
        if support_token_masses.shape != support_latents.shape[:-1]:
            raise ValueError(
                "support_token_masses must match support_latents without latent dim: "
                f"latents={tuple(support_latents.shape)} masses={tuple(support_token_masses.shape)}"
            )
        if shot_num <= 0:
            raise ValueError("shot_num must be positive")

        way_num, support_shot_num, num_tokens, _ = support_latents.shape
        if support_shot_num != shot_num:
            raise ValueError(
                f"shot_num argument must match support tensor shot dim: arg={shot_num} tensor={support_shot_num}"
            )

        normalized_token_masses, shot_weighted_mean, shot_weighted_std, shot_uncertainty = self._weighted_token_stats(
            support_latents,
            support_token_masses,
        )
        flat_latents = support_latents.reshape(way_num * shot_num, num_tokens, self.latent_dim)
        flat_token_masses = normalized_token_masses.reshape(way_num * shot_num, num_tokens)
        flat_weighted_mean = shot_weighted_mean.reshape(way_num * shot_num, self.latent_dim)
        flat_weighted_std = shot_weighted_std.reshape(way_num * shot_num, self.latent_dim)
        flat_uncertainty = shot_uncertainty.reshape(way_num * shot_num, 4)

        if self.use_shot_barycenter_only:
            shot_memory = flat_latents.new_zeros((way_num * shot_num, 0, self.latent_dim))
            shot_token_summary = flat_weighted_mean
            shot_memory_summary = flat_weighted_mean
            shot_atoms = flat_weighted_mean.unsqueeze(1)
            shot_basis_masses = flat_latents.new_ones((way_num * shot_num, 1))
            shot_offsets = torch.zeros_like(shot_atoms)
        else:
            token_mass_features = self.token_mass_embed(torch.log(flat_token_masses.clamp_min(self.eps)).unsqueeze(-1))
            shot_memory, shot_token_summary = self.shot_token_pool(flat_latents + token_mass_features)
            shot_memory_summary = shot_memory.mean(dim=1)
            shot_atoms = torch.cat([flat_weighted_mean.unsqueeze(1), shot_memory], dim=1)
            shot_offsets = (shot_atoms - flat_weighted_mean.unsqueeze(1)) / flat_weighted_std.unsqueeze(1).clamp_min(self.eps)
        shot_repr = self.shot_repr_norm(
            self.shot_repr_proj(
                torch.cat(
                    [
                        flat_weighted_mean,
                        flat_weighted_std,
                        shot_token_summary,
                        shot_memory_summary,
                        flat_uncertainty,
                    ],
                    dim=-1,
                )
            )
        )
        if not self.use_shot_barycenter_only:
            shot_basis_logits = self.shot_basis_mass_head(
                torch.cat(
                    [
                        shot_atoms,
                        shot_offsets,
                        shot_repr.unsqueeze(1).expand(-1, self.shot_basis_size, -1),
                        flat_uncertainty.unsqueeze(1).expand(-1, self.shot_basis_size, -1),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            shot_basis_logits[:, 0] = shot_basis_logits[:, 0] + 0.1
            shot_basis_masses = normalize_measure_masses(
                torch.softmax(shot_basis_logits, dim=-1),
                target_shape=shot_basis_logits.shape,
                device=shot_atoms.device,
                dtype=shot_atoms.dtype,
                eps=self.eps,
            )

        shot_atoms = shot_atoms.reshape(way_num, shot_num, self.shot_basis_size, self.latent_dim)
        shot_basis_masses = shot_basis_masses.reshape(way_num, shot_num, self.shot_basis_size)
        shot_repr = shot_repr.reshape(way_num, shot_num, self.context_dim)
        shot_uncertainty = shot_uncertainty.reshape(way_num, shot_num, 4)

        _, raw_class_summary = self.class_summary_pool(shot_repr)
        raw_class_summary = self.class_summary_norm(raw_class_summary)
        if self.use_episode_adapter:
            adapted = self.episode_adapter(raw_class_summary)
            class_summary = adapted["adapted_class_summary"]
            episode_context = adapted["episode_context"]
        else:
            class_summary = raw_class_summary
            episode_context = raw_class_summary.mean(dim=0, keepdim=True).expand_as(raw_class_summary)

        shot_mass_logits = self.shot_mass_head(
            torch.cat(
                [
                    shot_repr,
                    class_summary.unsqueeze(1).expand(-1, shot_num, -1),
                    episode_context.unsqueeze(1).expand(-1, shot_num, -1),
                    shot_uncertainty,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        shot_masses = normalize_measure_masses(
            torch.softmax(shot_mass_logits, dim=-1),
            target_shape=shot_mass_logits.shape,
            device=shot_mass_logits.device,
            dtype=shot_mass_logits.dtype,
            eps=self.eps,
        )

        support_atoms = shot_atoms.reshape(way_num, shot_num * self.shot_basis_size, self.latent_dim)
        support_masses = (shot_masses.unsqueeze(-1) * shot_basis_masses).reshape(way_num, shot_num * self.shot_basis_size)
        support_masses = normalize_measure_masses(
            support_masses,
            target_shape=support_masses.shape,
            device=support_atoms.device,
            dtype=support_atoms.dtype,
            eps=self.eps,
        )

        shot_anchor_inputs = self.shot_anchor_proj(torch.cat([shot_weighted_mean, shot_repr], dim=-1))
        class_memory, _ = self.class_memory_pool(shot_anchor_inputs)
        class_memory = self.class_memory_norm(class_memory)

        class_uncertainty = (shot_masses.unsqueeze(-1) * shot_uncertainty).sum(dim=1)
        shot_entropy = -(shot_masses * torch.log(shot_masses.clamp_min(self.eps))).sum(dim=-1, keepdim=True)
        shot_feature = support_latents.new_full((way_num, 1), math.log1p(float(shot_num)))

        if self.prior_num_atoms > 0:
            prior_context = torch.cat([class_summary, episode_context, class_uncertainty, shot_feature], dim=-1)
            prior_queries = self.prior_query_head(prior_context).reshape(way_num, self.prior_num_atoms, self.context_dim)
            attention_logits = torch.einsum("wrc,mc->wrm", prior_queries, self.global_prior_keys) / math.sqrt(
                float(self.context_dim)
            )
            attention_weights = torch.softmax(attention_logits, dim=-1)
            prior_atoms = torch.einsum("wrm,md->wrd", attention_weights, self.global_prior_atoms)
            prior_offsets = torch.tanh(self.prior_offset_head(prior_context).reshape(way_num, self.prior_num_atoms, self.latent_dim))
            prior_scale = self.prior_scale * (1.0 + shot_weighted_std.mean(dim=(1, 2), keepdim=True))
            prior_atoms = prior_atoms + prior_offsets * prior_scale
            prior_masses = normalize_measure_masses(
                torch.softmax(
                    self.prior_mass_head(
                        torch.cat(
                            [
                                prior_atoms,
                                class_summary.unsqueeze(1).expand(-1, self.prior_num_atoms, -1),
                                episode_context.unsqueeze(1).expand(-1, self.prior_num_atoms, -1),
                                class_uncertainty.unsqueeze(1).expand(-1, self.prior_num_atoms, -1),
                            ],
                            dim=-1,
                        )
                    ).squeeze(-1),
                    dim=-1,
                ),
                target_shape=(way_num, self.prior_num_atoms),
                device=support_atoms.device,
                dtype=support_atoms.dtype,
                eps=self.eps,
            )
        else:
            prior_atoms = support_atoms.new_zeros((way_num, 0, self.latent_dim))
            prior_masses = support_masses.new_zeros((way_num, 0))

        alpha_gate = torch.sigmoid(
            self.alpha_head(torch.cat([class_summary, episode_context, class_uncertainty, shot_feature], dim=-1)).squeeze(-1)
            - self.alpha_uncertainty_scale * class_uncertainty[:, 0]
        )
        trust = float(shot_num) / (float(shot_num) + self.shrinkage_kappa)
        alpha = trust * alpha_gate

        if self.prior_num_atoms > 0:
            base_atoms = torch.cat([support_atoms, prior_atoms], dim=1)
            base_masses = torch.cat([alpha.unsqueeze(-1) * support_masses, (1.0 - alpha).unsqueeze(-1) * prior_masses], dim=1)
        else:
            base_atoms = support_atoms
            base_masses = support_masses
        base_masses = normalize_measure_masses(
            base_masses,
            target_shape=base_masses.shape,
            device=base_atoms.device,
            dtype=base_atoms.dtype,
            eps=self.eps,
        )

        return {
            "shot_atoms": shot_atoms,
            "shot_basis_masses": shot_basis_masses,
            "shot_masses": shot_masses,
            "support_atoms": support_atoms,
            "support_masses": support_masses,
            "class_summary": class_summary,
            "episode_context": episode_context,
            "class_memory": class_memory,
            "shot_weighted_mean": shot_weighted_mean,
            "shot_weighted_std": shot_weighted_std,
            "shot_uncertainty": shot_uncertainty,
            "class_uncertainty": class_uncertainty,
            "prior_atoms": prior_atoms,
            "prior_masses": prior_masses,
            "alpha": alpha,
            "base_atoms": base_atoms,
            "base_masses": base_masses,
        }
