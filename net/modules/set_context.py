"""Permutation-invariant support context encoders for SC-LFI."""

from __future__ import annotations

import torch
import torch.nn as nn

from net.ssm.set_invariant_pool import SetInvariantMemoryPool


class DeepSetsContextEncoder(nn.Module):
    """Weighted DeepSets encoder for class-level support context.

    This support-conditioned context is our few-shot adaptation of using
    conditional covariates to parameterize a family of class distributions.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(input_dim, context_dim))
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(context_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
            squeeze_batch = True
        elif tokens.dim() != 3:
            raise ValueError(
                "tokens must have shape (Set, Dim) or (Batch, Set, Dim), "
                f"got {tuple(tokens.shape)}"
            )

        encoded_tokens = self.token_encoder(tokens)
        attention_logits = self.attention(encoded_tokens).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=-1).unsqueeze(-1)
        context = (attention_weights * encoded_tokens).sum(dim=1)
        context = self.output_norm(context)
        if squeeze_batch:
            return context.squeeze(0)
        return context


class LightweightSetTransformerContextEncoder(nn.Module):
    """Permutation-invariant context encoder using learned memory pooling."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        num_memory_tokens: int = 4,
        num_heads: int = 4,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        self.input_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, context_dim),
        )
        self.pool = SetInvariantMemoryPool(
            dim=context_dim,
            num_memory_tokens=int(num_memory_tokens),
            num_heads=int(num_heads),
            ffn_multiplier=int(ffn_multiplier),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.output_norm = nn.LayerNorm(context_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
            squeeze_batch = True
        elif tokens.dim() != 3:
            raise ValueError(
                "tokens must have shape (Set, Dim) or (Batch, Set, Dim), "
                f"got {tuple(tokens.shape)}"
            )

        adapted_tokens = self.input_adapter(tokens)
        _, summary = self.pool(adapted_tokens)
        context = self.output_norm(self.output_proj(summary) + summary)
        if squeeze_batch:
            return context.squeeze(0)
        return context


class SupportSetContextEncoder(nn.Module):
    """Factory wrapper for the support-set context encoder ablation hook."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        context_type: str = "deepsets",
        hidden_dim: int | None = None,
        num_memory_tokens: int = 4,
        num_heads: int = 4,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        context_type = str(context_type).lower()
        self.context_type = context_type
        if context_type == "deepsets":
            self.encoder = DeepSetsContextEncoder(
                input_dim=input_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
            )
        elif context_type == "lightweight_set_transformer":
            self.encoder = LightweightSetTransformerContextEncoder(
                input_dim=input_dim,
                context_dim=context_dim,
                num_memory_tokens=num_memory_tokens,
                num_heads=num_heads,
                ffn_multiplier=ffn_multiplier,
            )
        else:
            raise ValueError(f"Unsupported context_type: {context_type}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)
