"""Latent evidence projector with learned token reliability for SC-LFI v2."""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentEvidenceProjectorV2(nn.Module):
    """Project backbone tokens into latent evidence and token reliability.

    Formula:
    - latent evidence: `e = Psi(z) in R^{d_l}`
    - mass logit: `r = W_mass(e) in R`
    - normalized token mass over a token set: `a = softmax(r)`

    Paper grounding:
    - latent-space state modeling is consistent with latent flow modeling work.

    Our few-shot adaptation:
    - learned token masses define weighted empirical evidence measures for
      support/query transport scoring.

    Tensor shapes:
    - input tokens: `[..., Tokens, InputDim]`
    - latent tokens: `[..., Tokens, LatentDim]`
    - mass logits: `[..., Tokens]`
    - normalized masses: `[..., Tokens]`
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int | None = None,
        mass_hidden_dim: int | None = None,
        mass_temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        if mass_temperature <= 0.0:
            raise ValueError("mass_temperature must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        hidden_dim = int(hidden_dim or max(input_dim, latent_dim))
        mass_hidden_dim = int(mass_hidden_dim or hidden_dim)
        self.mass_temperature = float(mass_temperature)
        self.eps = float(eps)

        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.mass_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, mass_hidden_dim),
            nn.GELU(),
            nn.Linear(mass_hidden_dim, 1),
        )

    def project(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() < 2:
            raise ValueError(f"tokens must have at least 2 dimensions, got {tuple(tokens.shape)}")
        return self.latent_norm(self.projector(tokens))

    def mass_logits(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        if latent_tokens.dim() < 2:
            raise ValueError(
                f"latent_tokens must have at least 2 dimensions, got {tuple(latent_tokens.shape)}"
            )
        return self.mass_head(latent_tokens).squeeze(-1) / self.mass_temperature

    def normalize_masses(self, mass_logits: torch.Tensor) -> torch.Tensor:
        if mass_logits.dim() < 1:
            raise ValueError(f"mass_logits must have at least 1 dimension, got {tuple(mass_logits.shape)}")
        masses = torch.softmax(mass_logits, dim=-1)
        masses = masses.clamp_min(0.0)
        normalizer = masses.sum(dim=-1, keepdim=True)
        zero_rows = normalizer <= self.eps
        if zero_rows.any():
            uniform = torch.full_like(masses, 1.0 / float(masses.shape[-1]))
            masses = torch.where(zero_rows, uniform, masses)
            normalizer = masses.sum(dim=-1, keepdim=True)
        return masses / normalizer.clamp_min(self.eps)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        latent_tokens = self.project(tokens)
        logits = self.mass_logits(latent_tokens)
        masses = self.normalize_masses(logits)
        return {
            "latent_tokens": latent_tokens,
            "mass_logits": logits,
            "masses": masses,
        }
