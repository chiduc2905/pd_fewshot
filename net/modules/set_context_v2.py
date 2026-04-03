"""Support-conditioned class summary, memory, and anchor measure for SC-LFI v2."""

from __future__ import annotations

import torch
import torch.nn as nn

from net.ssm.set_invariant_pool import SetInvariantMemoryPool


class SupportConditionerV2(nn.Module):
    """Build a global class summary, compact support memory, and anchor measure.

    Formulas:
    - weighted class summary:
      `h_c = Phi_summary(E_c, a_c^sup)`
    - support memory tokens:
      `M_c = Phi_mem(E_c)`
    - anchor measure atoms for stable class scoring:
      `A_c = [bar(e)_c; M_c]`
    - anchor masses:
      `pi_c = softmax(W_anchor([A_c, h_c]))`

    Paper grounding:
    - permutation-invariant set conditioning is consistent with conditional
      distribution modeling and set-based few-shot inference.

    Our few-shot adaptation:
    - the conditioner returns both a global summary `h_c` and a compact support
      memory `M_c`, so the flow is not forced to depend on only one summary
      vector.

    Tensor shapes:
    - support_latents: `[Way, SupportTokens, LatentDim]`
    - support_masses: `[Way, SupportTokens]`
    - class_summary: `[Way, ContextDim]`
    - memory_tokens: `[Way, MemorySize, LatentDim]`
    - anchor_particles: `[Way, 1 + MemorySize, LatentDim]`
    - anchor_masses: `[Way, 1 + MemorySize]`
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        memory_size: int = 4,
        memory_num_heads: int = 4,
        memory_ffn_multiplier: int = 2,
        summary_hidden_dim: int | None = None,
        mass_feature_dim: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or context_dim <= 0:
            raise ValueError("latent_dim and context_dim must be positive")
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.eps = float(eps)
        mass_feature_dim = int(mass_feature_dim or latent_dim)
        summary_hidden_dim = int(summary_hidden_dim or max(latent_dim, context_dim))

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
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, summary_hidden_dim),
            nn.GELU(),
            nn.Linear(summary_hidden_dim, context_dim),
        )
        self.summary_norm = nn.LayerNorm(context_dim)
        self.memory_norm = nn.LayerNorm(latent_dim)
        self.anchor_context_proj = nn.Linear(context_dim, latent_dim)
        self.anchor_mass_head = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, summary_hidden_dim),
            nn.GELU(),
            nn.Linear(summary_hidden_dim, 1),
        )

    def _normalize_measure(self, masses: torch.Tensor) -> torch.Tensor:
        masses = masses.clamp_min(0.0)
        normalizer = masses.sum(dim=-1, keepdim=True)
        zero_rows = normalizer <= self.eps
        if zero_rows.any():
            uniform = torch.full_like(masses, 1.0 / float(masses.shape[-1]))
            masses = torch.where(zero_rows, uniform, masses)
            normalizer = masses.sum(dim=-1, keepdim=True)
        return masses / normalizer.clamp_min(self.eps)

    def forward(
        self,
        support_latents: torch.Tensor,
        support_masses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if support_latents.dim() != 3:
            raise ValueError(
                "support_latents must have shape (Way, SupportTokens, LatentDim), "
                f"got {tuple(support_latents.shape)}"
            )
        if support_masses.shape != support_latents.shape[:-1]:
            raise ValueError(
                "support_masses must match support_latents without the latent dim: "
                f"latents={tuple(support_latents.shape)} masses={tuple(support_masses.shape)}"
            )

        normalized_masses = support_masses / support_masses.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        weighted_mean = (normalized_masses.unsqueeze(-1) * support_latents).sum(dim=1)

        # Inject support reliability into the memory constructor while remaining
        # permutation invariant over support token order.
        mass_features = self.mass_embed(torch.log(normalized_masses.clamp_min(self.eps)).unsqueeze(-1))
        memory_inputs = support_latents + mass_features
        memory_tokens, pooled_summary = self.memory_pool(memory_inputs)
        class_summary = self.summary_norm(
            self.summary_proj(torch.cat([weighted_mean, pooled_summary], dim=-1))
        )
        memory_tokens = self.memory_norm(memory_tokens)

        # Anchor atoms include the prototype-like weighted mean and the compact
        # support memory tokens. This gives the scoring distribution a stable,
        # support-anchored component before any flow-generated particles.
        anchor_particles = torch.cat([weighted_mean.unsqueeze(1), memory_tokens], dim=1)
        anchor_context = self.anchor_context_proj(class_summary).unsqueeze(1).expand_as(anchor_particles)
        anchor_logits = self.anchor_mass_head(torch.cat([anchor_particles, anchor_context], dim=-1)).squeeze(-1)
        anchor_masses = self._normalize_measure(torch.softmax(anchor_logits, dim=-1))
        return {
            "class_summary": class_summary,
            "memory_tokens": memory_tokens,
            "weighted_mean": weighted_mean,
            "anchor_particles": anchor_particles,
            "anchor_masses": anchor_masses,
        }
