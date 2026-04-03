"""SPIF-Mamba: adjusted SPIF variants with coarse-to-fine stable refinement.

The original SPIF baselines remain untouched in ``net/spif.py``.
This file defines the Mamba variants directly under the historical names
``SPIFMambaCE`` and ``SPIFMambaMAX``, but the architecture is the adjusted
design:

1. Stable / variant factorization
2. Raw coarse stable gate
3. Residual official-Mamba refinement on the coarse-filtered stable stream
4. Final stable gate
5. Global prototype + local partial matching heads

The goal is to preserve SPIF's low-variance inductive bias while giving Mamba
an explicit role as a stable-evidence refiner rather than a generic token
contextualizer before any filtering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.ssm.intra_image_mamba import build_2d_position_grid

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - optional in some local shells
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


def _require_mamba() -> None:
    if Mamba is None:
        raise ImportError(
            "SPIFMambaCE and SPIFMambaMAX require the official mamba-ssm package"
        ) from MAMBA_IMPORT_ERROR


def _make_projection_head(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Linear(out_dim, out_dim),
    )


def _make_gate_head(dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
    )


@dataclass
class SPIFMambaTokenOutputs:
    stable_tokens: torch.Tensor
    variant_tokens: torch.Tensor
    raw_gate: torch.Tensor
    coarse_stable_tokens: torch.Tensor
    refined_stable_tokens: torch.Tensor
    final_gate: torch.Tensor
    final_stable_tokens: torch.Tensor
    stable_global: torch.Tensor
    variant_global: torch.Tensor
    beta: torch.Tensor
    pre_refine_summary: torch.Tensor
    post_refine_summary: torch.Tensor


class StableVariantProjector(nn.Module):
    """Shallow factorization heads that split stable evidence from nuisance."""

    def __init__(self, input_dim: int, stable_dim: int, variant_dim: int) -> None:
        super().__init__()
        self.stable_dim = int(stable_dim)
        self.variant_dim = int(variant_dim)

        self.stable_head = _make_projection_head(input_dim, self.stable_dim)
        self.variant_head = _make_projection_head(input_dim, self.variant_dim)
        self.shared_head = _make_projection_head(input_dim, self.stable_dim)
        self.shared_variant_adapter = (
            nn.Identity()
            if self.variant_dim == self.stable_dim
            else nn.Linear(self.stable_dim, self.variant_dim)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        factorization_on: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if factorization_on:
            return self.stable_head(tokens), self.variant_head(tokens)
        shared = self.shared_head(tokens)
        return shared, self.shared_variant_adapter(shared)


class RawStableGate(nn.Module):
    """Coarse gate that removes obviously unstable tokens before propagation."""

    def __init__(self, stable_dim: int, gate_hidden: int) -> None:
        super().__init__()
        self.gate_head = _make_gate_head(stable_dim, gate_hidden)

    def forward(self, stable_tokens: torch.Tensor, gate_on: bool = True) -> torch.Tensor:
        if not gate_on:
            return torch.ones(
                stable_tokens.shape[0],
                stable_tokens.shape[1],
                1,
                device=stable_tokens.device,
                dtype=stable_tokens.dtype,
            )
        return self.gate_head(stable_tokens)


class FinalStableGate(nn.Module):
    """Final gate that selects refined stable evidence for classification."""

    def __init__(self, stable_dim: int, gate_hidden: int) -> None:
        super().__init__()
        self.gate_head = _make_gate_head(stable_dim, gate_hidden)

    def forward(self, refined_tokens: torch.Tensor, gate_on: bool = True) -> torch.Tensor:
        if not gate_on:
            return torch.ones(
                refined_tokens.shape[0],
                refined_tokens.shape[1],
                1,
                device=refined_tokens.device,
                dtype=refined_tokens.dtype,
            )
        return self.gate_head(refined_tokens)


class BidirectionalMambaContextBlock(nn.Module):
    """Produce a contextual refinement delta for the coarse stable stream."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        ffn_multiplier: int = 2,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _require_mamba()
        hidden_dim = max(dim, dim * ffn_multiplier)
        self.input_norm = nn.LayerNorm(dim)
        self.position_proj = nn.Sequential(
            nn.Linear(6, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.forward_mamba = Mamba(
            d_model=dim,
            d_state=state_dim,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba(
            d_model=dim,
            d_state=state_dim,
            d_conv=d_conv,
            expand=expand,
        )
        self.mix_proj = nn.Linear(dim * 3, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.delta_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens: torch.Tensor, spatial_hw: Tuple[int, int]) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(
                f"tokens must have shape (Batch, Tokens, Dim), got {tuple(tokens.shape)}"
            )

        height, width = spatial_hw
        if tokens.shape[1] != height * width:
            raise ValueError(
                "Token count does not match spatial_hw: "
                f"tokens={tokens.shape[1]} spatial_hw={spatial_hw}"
            )

        position = self.position_proj(
            build_2d_position_grid(height, width, tokens.device, tokens.dtype)
        ).expand(tokens.shape[0], -1, -1)
        scan_inputs = self.input_norm(tokens) + position

        forward_outputs = self.forward_mamba(scan_inputs)
        backward_outputs = torch.flip(
            self.backward_mamba(torch.flip(scan_inputs, dims=[1])),
            dims=[1],
        )
        delta = self.mix_proj(torch.cat([forward_outputs, backward_outputs, position], dim=-1))
        delta = self.delta_norm(self.dropout(delta))
        delta = delta + self.dropout(self.ffn(self.ffn_norm(delta)))
        return delta


class StableEvidenceMambaRefiner(nn.Module):
    """Residual official-Mamba refiner over already coarsely filtered evidence."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        depth: int = 1,
        ffn_multiplier: int = 2,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        beta_init: float = 0.1,
        learnable_beta: bool = True,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.learnable_beta = bool(learnable_beta)
        self.fixed_beta = float(max(0.0, beta_init))
        if self.learnable_beta:
            beta_init = min(max(float(beta_init), 1e-4), 1.0 - 1e-4)
            self.beta_logit = nn.Parameter(torch.tensor(math.log(beta_init / (1.0 - beta_init))))
        else:
            self.register_parameter("beta_logit", None)
        self.blocks = nn.ModuleList(
            [
                BidirectionalMambaContextBlock(
                    dim=dim,
                    state_dim=state_dim,
                    ffn_multiplier=ffn_multiplier,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def beta(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.blocks:
            return torch.tensor(0.0, device=device, dtype=dtype)
        if self.learnable_beta:
            return torch.sigmoid(self.beta_logit).to(device=device, dtype=dtype)
        return torch.tensor(self.fixed_beta, device=device, dtype=dtype)

    def forward(
        self,
        coarse_stable_tokens: torch.Tensor,
        spatial_hw: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.blocks:
            beta = self.beta(coarse_stable_tokens.device, coarse_stable_tokens.dtype)
            return coarse_stable_tokens, beta

        refined = coarse_stable_tokens
        for block in self.blocks:
            refined = refined + block(refined, spatial_hw=spatial_hw)

        beta = self.beta(coarse_stable_tokens.device, coarse_stable_tokens.dtype)
        refined = coarse_stable_tokens + beta * (refined - coarse_stable_tokens)
        return refined, beta


class SPIFMambaEncoder(nn.Module):
    """Adjusted encoder: coarse gate -> residual Mamba -> final gate."""

    def __init__(
        self,
        input_dim: int,
        stable_dim: int,
        variant_dim: int,
        gate_hidden: int,
        state_dim: int,
        mamba_depth: int = 1,
        mamba_ffn_multiplier: int = 2,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        beta_init: float = 0.1,
        learnable_beta: bool = True,
        use_raw_gate: bool = True,
        use_final_gate: bool = True,
        token_l2norm: bool = True,
    ) -> None:
        super().__init__()
        self.use_raw_gate = bool(use_raw_gate)
        self.use_final_gate = bool(use_final_gate)
        self.token_l2norm = bool(token_l2norm)

        self.projector = StableVariantProjector(
            input_dim=input_dim,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
        )
        self.raw_gate = RawStableGate(stable_dim=stable_dim, gate_hidden=gate_hidden)
        self.refiner = StableEvidenceMambaRefiner(
            dim=stable_dim,
            state_dim=state_dim,
            depth=mamba_depth,
            ffn_multiplier=mamba_ffn_multiplier,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            beta_init=beta_init,
            learnable_beta=learnable_beta,
        )
        self.final_gate = FinalStableGate(stable_dim=stable_dim, gate_hidden=gate_hidden)
        self.final_stable_token_norm = nn.LayerNorm(stable_dim)
        self.variant_token_norm = nn.LayerNorm(variant_dim)

    @staticmethod
    def pool_global(
        stable_tokens: torch.Tensor,
        gate: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        weighted_sum = (stable_tokens * gate).sum(dim=1)
        normalizer = gate.sum(dim=1).clamp_min(eps)
        return weighted_sum / normalizer

    def _normalize_token_branch(self, tokens: torch.Tensor, token_norm: nn.LayerNorm) -> torch.Tensor:
        tokens = token_norm(tokens)
        if self.token_l2norm:
            tokens = F.normalize(tokens, p=2, dim=-1)
        return tokens

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_hw: Tuple[int, int],
        factorization_on: bool = True,
    ) -> SPIFMambaTokenOutputs:
        stable_tokens, variant_tokens = self.projector(tokens, factorization_on=factorization_on)

        raw_gate = self.raw_gate(stable_tokens, gate_on=self.use_raw_gate)
        coarse_stable_tokens = stable_tokens * raw_gate
        refined_stable_tokens, beta = self.refiner(coarse_stable_tokens, spatial_hw=spatial_hw)

        final_gate = self.final_gate(refined_stable_tokens, gate_on=self.use_final_gate)
        final_stable_tokens = refined_stable_tokens * final_gate

        stable_global = self.pool_global(final_stable_tokens, final_gate)
        variant_global = variant_tokens.mean(dim=1)

        stable_global = F.normalize(stable_global, p=2, dim=-1)
        variant_global = F.normalize(variant_global, p=2, dim=-1)

        final_stable_tokens = self._normalize_token_branch(final_stable_tokens, self.final_stable_token_norm)
        variant_tokens = self._normalize_token_branch(variant_tokens, self.variant_token_norm)

        return SPIFMambaTokenOutputs(
            stable_tokens=stable_tokens,
            variant_tokens=variant_tokens,
            raw_gate=raw_gate,
            coarse_stable_tokens=coarse_stable_tokens,
            refined_stable_tokens=refined_stable_tokens,
            final_gate=final_gate,
            final_stable_tokens=final_stable_tokens,
            stable_global=stable_global,
            variant_global=variant_global,
            beta=beta,
            pre_refine_summary=coarse_stable_tokens.mean(dim=1),
            post_refine_summary=refined_stable_tokens.mean(dim=1),
        )


class _SPIFMambaBase(BaseConv64FewShotModel):
    """Shared adjusted SPIF-Mamba logic for CE-only and MAX variants."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        stable_dim: int = 64,
        variant_dim: int = 64,
        gate_hidden: int = 16,
        top_r: int = 3,
        alpha_init: float = 0.7,
        learnable_alpha: bool = False,
        factorization_on: bool = True,
        global_only: bool = False,
        local_only: bool = False,
        token_l2norm: bool = True,
        use_raw_gate: bool = True,
        use_final_gate: bool = True,
        beta_init: float = 0.1,
        learnable_beta: bool = True,
        consistency_weight: float = 0.1,
        decorr_weight: float = 0.01,
        sparse_raw_weight: float = 5e-4,
        sparse_final_weight: float = 5e-4,
        consistency_dropout: float = 0.1,
        mamba_state_dim: float = 16,
        mamba_depth: int = 1,
        mamba_ffn_multiplier: int = 2,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        gate_position: str = "after",
        backbone_name: str = "resnet12",
        image_size: int = 64,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        if global_only and local_only:
            raise ValueError("global_only and local_only cannot both be true")
        if top_r <= 0:
            raise ValueError("top_r must be positive")
        if gate_position not in {"after", "before"}:
            raise ValueError(f"Unsupported gate_position: {gate_position}")
        del gate_position  # Legacy CLI arg kept for backward compatibility.

        self.global_only = bool(global_only)
        self.local_only = bool(local_only)
        self.factorization_on = bool(factorization_on)
        self.use_raw_gate = bool(use_raw_gate)
        self.use_final_gate = bool(use_final_gate)
        self.top_r = int(top_r)
        self.learnable_alpha = bool(learnable_alpha)
        self.consistency_weight = float(consistency_weight)
        self.decorr_weight = float(decorr_weight)
        self.sparse_raw_weight = float(sparse_raw_weight)
        self.sparse_final_weight = float(sparse_final_weight)
        self.consistency_dropout = float(consistency_dropout)

        alpha_init = float(alpha_init)
        alpha_init = min(max(alpha_init, 1e-4), 1.0 - 1e-4)
        self.fixed_alpha = alpha_init
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init))))

        self.encoder_head = SPIFMambaEncoder(
            input_dim=hidden_dim,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            state_dim=mamba_state_dim,
            mamba_depth=mamba_depth,
            mamba_ffn_multiplier=mamba_ffn_multiplier,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_dropout=mamba_dropout,
            beta_init=beta_init,
            learnable_beta=learnable_beta,
            use_raw_gate=use_raw_gate,
            use_final_gate=use_final_gate,
            token_l2norm=token_l2norm,
        )
        self.variant_align = (
            nn.Identity()
            if int(variant_dim) == int(stable_dim)
            else nn.Linear(int(variant_dim), int(stable_dim), bias=False)
        )

    def _alpha(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.global_only:
            return torch.tensor(1.0, device=device, dtype=dtype)
        if self.local_only:
            return torch.tensor(0.0, device=device, dtype=dtype)
        if self.learnable_alpha:
            return torch.sigmoid(self.alpha_logit).to(device=device, dtype=dtype)
        return torch.tensor(self.fixed_alpha, device=device, dtype=dtype)

    def encode_tokens(self, images: torch.Tensor) -> SPIFMambaTokenOutputs:
        feature_map = self.encode(images)
        tokens = feature_map_to_tokens(feature_map)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        return self.encoder_head(
            tokens,
            spatial_hw=spatial_hw,
            factorization_on=self.factorization_on,
        )

    def _encode_episode(self, query: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        flat_support = support.reshape(way_num * shot_num, *support.shape[-3:])
        q_outputs = self.encode_tokens(query)
        s_outputs = self.encode_tokens(flat_support)

        return {
            "query_global": q_outputs.stable_global,
            "query_tokens": q_outputs.final_stable_tokens,
            "query_variant_global": q_outputs.variant_global,
            "query_raw_gate": q_outputs.raw_gate,
            "query_final_gate": q_outputs.final_gate,
            "query_beta": q_outputs.beta,
            "query_pre_refine_summary": q_outputs.pre_refine_summary,
            "query_post_refine_summary": q_outputs.post_refine_summary,
            "support_global": s_outputs.stable_global.reshape(way_num, shot_num, -1),
            "support_tokens": s_outputs.final_stable_tokens.reshape(
                way_num,
                shot_num,
                -1,
                s_outputs.final_stable_tokens.shape[-1],
            ),
            "support_variant_global": s_outputs.variant_global.reshape(way_num, shot_num, -1),
            "support_raw_gate": s_outputs.raw_gate.reshape(way_num, shot_num, -1, 1),
            "support_final_gate": s_outputs.final_gate.reshape(way_num, shot_num, -1, 1),
            "support_beta": s_outputs.beta,
            "support_pre_refine_summary": s_outputs.pre_refine_summary.reshape(way_num, shot_num, -1),
            "support_post_refine_summary": s_outputs.post_refine_summary.reshape(way_num, shot_num, -1),
        }

    @staticmethod
    def build_support_prototypes(support_global: torch.Tensor) -> torch.Tensor:
        prototypes = support_global.mean(dim=1)
        return F.normalize(prototypes, p=2, dim=-1)

    @staticmethod
    def build_support_token_pool(support_tokens: torch.Tensor) -> torch.Tensor:
        way_num, shot_num, token_num, dim = support_tokens.shape
        pooled = support_tokens.reshape(way_num, shot_num * token_num, dim)
        return F.normalize(pooled, p=2, dim=-1)

    @staticmethod
    def compute_global_scores(query_global: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        return torch.matmul(
            F.normalize(query_global, p=2, dim=-1),
            F.normalize(prototypes, p=2, dim=-1).transpose(0, 1),
        )

    def compute_local_partial_scores(
        self,
        query_tokens: torch.Tensor,
        support_token_pool: torch.Tensor,
    ) -> torch.Tensor:
        query_tokens = F.normalize(query_tokens, p=2, dim=-1)
        support_token_pool = F.normalize(support_token_pool, p=2, dim=-1)
        similarity = torch.einsum("qld,wtd->qwlt", query_tokens, support_token_pool)
        top_r = min(self.top_r, similarity.shape[-1])
        top_vals = torch.topk(similarity, k=top_r, dim=-1).values
        return top_vals.mean(dim=-1).mean(dim=-1)

    def _consistency_loss(self, stable_global: torch.Tensor) -> torch.Tensor:
        if self.consistency_dropout <= 0.0:
            return stable_global.new_zeros(())
        # The current episodic loader exposes one view per sample, so we use two
        # stochastic dropout views of the final stable embedding as a lightweight
        # consistency proxy rather than changing the data pipeline.
        view1 = F.dropout(stable_global, p=self.consistency_dropout, training=True)
        view2 = F.dropout(stable_global, p=self.consistency_dropout, training=True)
        view1 = F.normalize(view1, p=2, dim=-1)
        view2 = F.normalize(view2, p=2, dim=-1)
        return (1.0 - F.cosine_similarity(view1, view2, dim=-1)).mean()

    @staticmethod
    def _decorrelation_loss(stable_global: torch.Tensor, variant_global: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(
            F.normalize(stable_global, p=2, dim=-1),
            F.normalize(variant_global, p=2, dim=-1),
            dim=-1,
        ).pow(2).mean()

    @staticmethod
    def _sparse_gate_loss(gate: torch.Tensor) -> torch.Tensor:
        return gate.mean()

    def _aux_loss(
        self,
        query_global: torch.Tensor,
        support_global: torch.Tensor,
        query_variant_global: torch.Tensor,
        support_variant_global: torch.Tensor,
        query_raw_gate: torch.Tensor,
        support_raw_gate: torch.Tensor,
        query_final_gate: torch.Tensor,
        support_final_gate: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            return query_global.new_zeros(())

        loss = query_global.new_zeros(())

        if self.consistency_weight > 0.0:
            stable_global = torch.cat(
                [query_global, support_global.reshape(-1, support_global.shape[-1])],
                dim=0,
            )
            loss = loss + self.consistency_weight * self._consistency_loss(stable_global)

        if self.factorization_on and self.decorr_weight > 0.0:
            stable_global = torch.cat(
                [query_global, support_global.reshape(-1, support_global.shape[-1])],
                dim=0,
            )
            variant_global = torch.cat(
                [query_variant_global, support_variant_global.reshape(-1, support_variant_global.shape[-1])],
                dim=0,
            )
            variant_global = self.variant_align(variant_global)
            loss = loss + self.decorr_weight * self._decorrelation_loss(stable_global, variant_global)

        if self.use_raw_gate and self.sparse_raw_weight > 0.0:
            all_raw_gates = torch.cat(
                [query_raw_gate.reshape(-1, 1), support_raw_gate.reshape(-1, 1)],
                dim=0,
            )
            loss = loss + self.sparse_raw_weight * self._sparse_gate_loss(all_raw_gates)

        if self.use_final_gate and self.sparse_final_weight > 0.0:
            all_final_gates = torch.cat(
                [query_final_gate.reshape(-1, 1), support_final_gate.reshape(-1, 1)],
                dim=0,
            )
            loss = loss + self.sparse_final_weight * self._sparse_gate_loss(all_final_gates)

        return loss

    def _fuse_scores(
        self,
        global_scores: torch.Tensor,
        local_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._alpha(global_scores.device, global_scores.dtype)
        if self.global_only:
            return global_scores, alpha
        if self.local_only:
            return local_scores, alpha
        fused = alpha * global_scores + (1.0 - alpha) * local_scores
        return fused, alpha

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)
        prototypes = self.build_support_prototypes(episode["support_global"])
        support_token_pool = self.build_support_token_pool(episode["support_tokens"])
        global_scores = self.compute_global_scores(episode["query_global"], prototypes)
        local_scores = self.compute_local_partial_scores(episode["query_tokens"], support_token_pool)
        logits, alpha = self._fuse_scores(global_scores, local_scores)

        aux_loss = self._aux_loss(
            query_global=episode["query_global"],
            support_global=episode["support_global"],
            query_variant_global=episode["query_variant_global"],
            support_variant_global=episode["support_variant_global"],
            query_raw_gate=episode["query_raw_gate"],
            support_raw_gate=episode["support_raw_gate"],
            query_final_gate=episode["query_final_gate"],
            support_final_gate=episode["support_final_gate"],
        )

        all_raw_gates = torch.cat(
            [episode["query_raw_gate"].reshape(-1, 1), episode["support_raw_gate"].reshape(-1, 1)],
            dim=0,
        )
        all_final_gates = torch.cat(
            [episode["query_final_gate"].reshape(-1, 1), episode["support_final_gate"].reshape(-1, 1)],
            dim=0,
        )
        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": global_scores.detach(),
            "local_scores": local_scores.detach(),
            "alpha": alpha.detach(),
            "beta": episode["query_beta"].detach(),
            "raw_gate_mean": all_raw_gates.mean().detach(),
            "final_gate_mean": all_final_gates.mean().detach(),
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
            "pre_refine_summaries": torch.cat(
                [
                    episode["query_pre_refine_summary"],
                    episode["support_pre_refine_summary"].reshape(
                        -1,
                        episode["support_pre_refine_summary"].shape[-1],
                    ),
                ],
                dim=0,
            ).detach(),
            "post_refine_summaries": torch.cat(
                [
                    episode["query_post_refine_summary"],
                    episode["support_post_refine_summary"].reshape(
                        -1,
                        episode["support_post_refine_summary"].shape[-1],
                    ),
                ],
                dim=0,
            ).detach(),
        }


class SPIFMambaCE(_SPIFMambaBase):
    """Fair adjusted SPIF-Mamba variant trained only with episodic CE."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("learnable_alpha", False)
        kwargs.setdefault("alpha_init", 0.7)
        kwargs.setdefault("consistency_weight", 0.0)
        kwargs.setdefault("decorr_weight", 0.0)
        kwargs.setdefault("sparse_raw_weight", 0.0)
        kwargs.setdefault("sparse_final_weight", 0.0)
        super().__init__(*args, **kwargs)

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
            "alpha": first["alpha"],
            "beta": first["beta"],
            "raw_gate_mean": torch.stack([item["raw_gate_mean"] for item in diagnostics]).mean(),
            "final_gate_mean": torch.stack([item["final_gate_mean"] for item in diagnostics]).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "pre_refine_summaries": torch.cat(
                [item["pre_refine_summaries"] for item in diagnostics],
                dim=0,
            ),
            "post_refine_summaries": torch.cat(
                [item["post_refine_summaries"] for item in diagnostics],
                dim=0,
            ),
        }


class SPIFMambaMAX(_SPIFMambaBase):
    """Stronger adjusted SPIF-Mamba variant with lightweight auxiliary losses."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("learnable_alpha", True)
        kwargs.setdefault("alpha_init", 0.7)
        super().__init__(*args, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_outputs = []
        aux_losses = []
        diagnostics = []
        for batch_idx in range(bsz):
            episode = self._forward_episode(query[batch_idx], support[batch_idx])
            batch_outputs.append(episode["logits"])
            aux_losses.append(episode["aux_loss"])
            diagnostics.append(episode)

        logits = torch.cat(batch_outputs, dim=0)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())

        if not self.training and not return_aux:
            return logits

        first = diagnostics[0]
        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": torch.cat([item["global_scores"] for item in diagnostics], dim=0),
            "local_scores": torch.cat([item["local_scores"] for item in diagnostics], dim=0),
            "alpha": first["alpha"],
            "beta": first["beta"],
            "raw_gate_mean": torch.stack([item["raw_gate_mean"] for item in diagnostics]).mean(),
            "final_gate_mean": torch.stack([item["final_gate_mean"] for item in diagnostics]).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "pre_refine_summaries": torch.cat(
                [item["pre_refine_summaries"] for item in diagnostics],
                dim=0,
            ),
            "post_refine_summaries": torch.cat(
                [item["post_refine_summaries"] for item in diagnostics],
                dim=0,
            ),
        }
