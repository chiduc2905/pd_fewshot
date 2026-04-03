"""SPIF: Stable Partial Invariance Few-shot Networks.

This file implements the architecture described in ``new_model.md``:

- SPIFCE: fair version using only episodic cross-entropy
- SPIFMAX: same core architecture with lightweight auxiliary regularizers

The design is intentionally conservative:
- no support-query cross-attention
- no reconstruction classifier
- no episode transformer / Mamba
- no shot routing

Instead, the model factorizes each token map into stable/variant branches,
learns a lightweight stable evidence gate, and classifies with a low-variance
few-shot head that combines global stable prototypes with exact local top-r
partial matching on stable tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens


def _make_projection_head(in_dim: int, out_dim: int) -> nn.Sequential:
    """Small stable projection head mandated by the model spec."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Linear(out_dim, out_dim),
    )


@dataclass
class SPIFTokenOutputs:
    stable_tokens: torch.Tensor
    variant_tokens: torch.Tensor
    gate: torch.Tensor
    stable_global: torch.Tensor
    variant_global: torch.Tensor


class SPIFEncoder(nn.Module):
    """Token factorization + gate + pooled global stable embedding."""

    def __init__(
        self,
        input_dim: int,
        stable_dim: int,
        variant_dim: int,
        gate_hidden: int,
        token_l2norm: bool = True,
    ) -> None:
        super().__init__()
        self.stable_dim = int(stable_dim)
        self.variant_dim = int(variant_dim)
        self.token_l2norm = bool(token_l2norm)

        self.stable_head = _make_projection_head(input_dim, self.stable_dim)
        self.variant_head = _make_projection_head(input_dim, self.variant_dim)
        self.shared_head = _make_projection_head(input_dim, self.stable_dim)
        self.shared_variant_adapter = (
            nn.Identity()
            if self.variant_dim == self.stable_dim
            else nn.Linear(self.stable_dim, self.variant_dim)
        )

        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.stable_dim),
            nn.Linear(self.stable_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        self.stable_token_norm = nn.LayerNorm(self.stable_dim)
        self.variant_token_norm = nn.LayerNorm(self.variant_dim)

    def factorize_tokens(
        self,
        tokens: torch.Tensor,
        factorization_on: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if factorization_on:
            return self.stable_head(tokens), self.variant_head(tokens)
        shared = self.shared_head(tokens)
        return shared, self.shared_variant_adapter(shared)

    def compute_gate(
        self,
        stable_tokens: torch.Tensor,
        gate_on: bool = True,
    ) -> torch.Tensor:
        if not gate_on:
            return torch.ones(
                stable_tokens.shape[0],
                stable_tokens.shape[1],
                1,
                device=stable_tokens.device,
                dtype=stable_tokens.dtype,
            )
        return self.gate_head(stable_tokens)

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
        factorization_on: bool = True,
        gate_on: bool = True,
    ) -> SPIFTokenOutputs:
        stable_tokens, variant_tokens = self.factorize_tokens(tokens, factorization_on=factorization_on)
        gate = self.compute_gate(stable_tokens, gate_on=gate_on)

        # pool_global multiplies by gate internally, so pass raw stable_tokens
        stable_global = self.pool_global(stable_tokens, gate)
        variant_global = variant_tokens.mean(dim=1)

        stable_global = F.normalize(stable_global, p=2, dim=-1)
        variant_global = F.normalize(variant_global, p=2, dim=-1)

        # Gate stable tokens ONCE for local branch after global pooling
        gated_stable = stable_tokens * gate
        stable_tokens = self._normalize_token_branch(gated_stable, self.stable_token_norm)
        variant_tokens = self._normalize_token_branch(variant_tokens, self.variant_token_norm)

        return SPIFTokenOutputs(
            stable_tokens=stable_tokens,
            variant_tokens=variant_tokens,
            gate=gate,
            stable_global=stable_global,
            variant_global=variant_global,
        )


class _SPIFBase(BaseConv64FewShotModel):
    """Shared SPIF logic for fair CE-only and regularized variants."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        stable_dim: int = 64,
        variant_dim: int = 64,
        gate_hidden: int = 16,
        top_r: int = 4,
        alpha_init: float = 0.7,
        learnable_alpha: bool = False,
        gate_on: bool = True,
        factorization_on: bool = True,
        global_only: bool = False,
        local_only: bool = False,
        token_l2norm: bool = True,
        consistency_weight: float = 0.1,
        decorr_weight: float = 0.01,
        sparse_weight: float = 0.001,
        consistency_dropout: float = 0.1,
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

        self.global_only = bool(global_only)
        self.local_only = bool(local_only)
        self.gate_on = bool(gate_on)
        self.factorization_on = bool(factorization_on)
        self.top_r = int(top_r)
        self.learnable_alpha = bool(learnable_alpha)
        self.consistency_weight = float(consistency_weight)
        self.decorr_weight = float(decorr_weight)
        self.sparse_weight = float(sparse_weight)
        self.consistency_dropout = float(consistency_dropout)

        alpha_init = float(alpha_init)
        alpha_init = min(max(alpha_init, 1e-4), 1.0 - 1e-4)
        self.fixed_alpha = alpha_init
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init))))

        self.encoder_head = SPIFEncoder(
            input_dim=hidden_dim,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
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

    def encode_tokens(self, images: torch.Tensor) -> SPIFTokenOutputs:
        tokens = feature_map_to_tokens(self.encode(images))
        return self.encoder_head(
            tokens,
            factorization_on=self.factorization_on,
            gate_on=self.gate_on,
        )

    def _encode_episode(self, query: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        flat_support = support.reshape(way_num * shot_num, *support.shape[-3:])
        q_outputs = self.encode_tokens(query)
        s_outputs = self.encode_tokens(flat_support)

        return {
            "query_global": q_outputs.stable_global,
            "query_tokens": q_outputs.stable_tokens,
            "query_variant_global": q_outputs.variant_global,
            "query_gate": q_outputs.gate,
            "query_gated_tokens": q_outputs.stable_tokens,
            "support_global": s_outputs.stable_global.reshape(way_num, shot_num, -1),
            "support_tokens": s_outputs.stable_tokens.reshape(way_num, shot_num, -1, s_outputs.stable_tokens.shape[-1]),
            "support_variant_global": s_outputs.variant_global.reshape(way_num, shot_num, -1),
            "support_gate": s_outputs.gate.reshape(way_num, shot_num, -1, 1),
            "support_gated_tokens": s_outputs.stable_tokens.reshape(
                way_num,
                shot_num,
                -1,
                s_outputs.stable_tokens.shape[-1],
            ),
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
        # query_global and prototypes are already L2-normalized upstream
        return torch.matmul(
            query_global,
            prototypes.transpose(0, 1),
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

    def _consistency_loss(self, stable_tokens: torch.Tensor) -> torch.Tensor:
        if self.consistency_dropout <= 0.0:
            return stable_tokens.new_zeros(())
        view1 = F.dropout(stable_tokens, p=self.consistency_dropout, training=True).mean(dim=1)
        view2 = F.dropout(stable_tokens, p=self.consistency_dropout, training=True).mean(dim=1)
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
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_global: torch.Tensor,
        support_global: torch.Tensor,
        query_variant_global: torch.Tensor,
        support_variant_global: torch.Tensor,
        query_gate: torch.Tensor,
        support_gate: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            return query_tokens.new_zeros(())

        loss = query_tokens.new_zeros(())
        if self.consistency_weight > 0.0:
            all_stable_tokens = torch.cat(
                [
                    query_tokens,
                    support_tokens.reshape(-1, support_tokens.shape[-2], support_tokens.shape[-1]),
                ],
                dim=0,
            )
            loss = loss + self.consistency_weight * self._consistency_loss(all_stable_tokens)

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

        if self.gate_on and self.sparse_weight > 0.0:
            all_gates = torch.cat(
                [query_gate.reshape(-1, 1), support_gate.reshape(-1, 1)],
                dim=0,
            )
            loss = loss + self.sparse_weight * self._sparse_gate_loss(all_gates)
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
            query_tokens=episode["query_tokens"],
            support_tokens=episode["support_tokens"],
            query_global=episode["query_global"],
            support_global=episode["support_global"],
            query_variant_global=episode["query_variant_global"],
            support_variant_global=episode["support_variant_global"],
            query_gate=episode["query_gate"],
            support_gate=episode["support_gate"],
        )
        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": global_scores.detach(),
            "local_scores": local_scores.detach(),
            "alpha": alpha.detach(),
            "mean_gate": torch.cat(
                [
                    episode["query_gate"].reshape(-1, 1),
                    episode["support_gate"].reshape(-1, 1),
                ],
                dim=0,
            ).mean().detach(),
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
        }


class SPIFCE(_SPIFBase):
    """Fair SPIF variant: episodic CE only."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("learnable_alpha", False)
        kwargs.setdefault("alpha_init", 0.7)
        kwargs.setdefault("consistency_weight", 0.0)
        kwargs.setdefault("decorr_weight", 0.0)
        kwargs.setdefault("sparse_weight", 0.0)
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
            "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }


class SPIFMAX(_SPIFBase):
    """Regularized SPIF variant with lightweight auxiliary losses."""

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
            "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }
