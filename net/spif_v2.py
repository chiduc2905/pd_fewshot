"""SPIF-v2: conservative SPIF extension with competitive evidence selection.

This file is intentionally separate from ``net/spif.py`` so the original SPIF
baselines remain untouched. SPIF-v2 keeps the same few-shot philosophy:
- stable / variant factorization;
- a strong global prototype path for low-variance behavior;
- lightweight local evidence matching.

The architectural changes relative to SPIF are:
1. shared bottleneck before stable / variant splitting;
2. competitive token gate over the token dimension;
3. mutual local partial matching;
4. global-anchored residual local correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


def _prob_to_logit(prob: float, eps: float = 1e-4) -> float:
    clipped = min(max(float(prob), eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


@dataclass
class SPIFv2TokenOutputs:
    """Per-image decomposition outputs.

    Shapes:
    - shared_tokens: `(B, L, D_shared or C)`
    - stable_tokens: `(B, L, Ds)`
    - variant_tokens: `(B, L, Dv)`
    - gated_stable_tokens: `(B, L, Ds)`
    - gate: `(B, L, 1)`
    - stable_global: `(B, Ds)`
    - stable_global_raw: `(B, Ds)`
    - variant_global: `(B, Dv)`
    - gate_entropy: `(B,)`
    """

    shared_tokens: torch.Tensor
    stable_tokens: torch.Tensor
    variant_tokens: torch.Tensor
    gated_stable_tokens: torch.Tensor
    gate: torch.Tensor
    stable_global: torch.Tensor
    stable_global_raw: torch.Tensor
    variant_global: torch.Tensor
    gate_entropy: torch.Tensor


class SharedStableVariantProjector(nn.Module):
    """Shared bottleneck followed by stable / variant split.

    The stable branch is intentionally narrower by default so the model has to
    compress class-stable evidence into a tighter subspace, while the variant
    branch keeps more capacity for nuisance information.
    """

    def __init__(
        self,
        input_dim: int,
        shared_dim: int,
        stable_dim: int,
        variant_dim: int,
        use_shared_bottleneck: bool = True,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(input_dim, shared_dim))
        self.use_shared_bottleneck = bool(use_shared_bottleneck)

        if self.use_shared_bottleneck:
            self.shared_projector = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, shared_dim),
            )
            split_input_dim = shared_dim
        else:
            self.shared_projector = nn.Identity()
            split_input_dim = input_dim

        self.stable_projector = nn.Sequential(
            nn.LayerNorm(split_input_dim),
            nn.Linear(split_input_dim, stable_dim),
        )
        self.variant_projector = nn.Sequential(
            nn.LayerNorm(split_input_dim),
            nn.Linear(split_input_dim, variant_dim),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_tokens = self.shared_projector(tokens)
        stable_tokens = self.stable_projector(shared_tokens)
        variant_tokens = self.variant_projector(shared_tokens)
        return shared_tokens, stable_tokens, variant_tokens


class CompetitiveTokenGate(nn.Module):
    """Token gate with competition over the token dimension.

    Default SPIF-v2 uses sparsemax so evidence mass is limited and can become
    sparse. Softmax is available as a dense competitive alternative, while
    sigmoid is kept only as an ablation against the original independent gate.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int | None = None,
        gate_type: str = "sparsemax",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.gate_type = str(gate_type).lower()
        if self.gate_type not in {"softmax", "sparsemax", "sigmoid"}:
            raise ValueError(f"Unsupported gate_type: {gate_type}")
        hidden_dim = int(hidden_dim or (token_dim if token_dim < 32 else max(token_dim // 4, 8)))
        self.scorer = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.eps = float(eps)

    @staticmethod
    def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        shifted = logits - logits.max(dim=dim, keepdim=True).values
        sorted_logits, _ = torch.sort(shifted, dim=dim, descending=True)
        cumsum = sorted_logits.cumsum(dim) - 1.0

        view_shape = [1] * shifted.dim()
        view_shape[dim] = shifted.size(dim)
        range_values = torch.arange(
            1,
            shifted.size(dim) + 1,
            device=shifted.device,
            dtype=shifted.dtype,
        ).view(view_shape)

        support = range_values * sorted_logits > cumsum
        k = support.sum(dim=dim, keepdim=True).clamp_min(1)
        tau = cumsum.gather(dim, k - 1) / k.to(shifted.dtype)
        return torch.clamp(shifted - tau, min=0.0)

    def _normalize(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "softmax":
            return torch.softmax(logits, dim=1)
        if self.gate_type == "sparsemax":
            return self._sparsemax(logits, dim=1)
        return torch.sigmoid(logits)

    def _distribution(self, gate: torch.Tensor) -> torch.Tensor:
        return gate / gate.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def _entropy(self, distribution: torch.Tensor) -> torch.Tensor:
        token_count = max(int(distribution.shape[1]), 2)
        entropy = -(distribution.clamp_min(self.eps) * distribution.clamp_min(self.eps).log()).sum(dim=1)
        return entropy / math.log(token_count)

    def forward(self, stable_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_logits = self.scorer(stable_tokens).squeeze(-1)
        gate = self._normalize(token_logits)
        gate_distribution = self._distribution(gate)
        gate_entropy = self._entropy(gate_distribution)
        return gate.unsqueeze(-1), gate_entropy


class MutualLocalMatcher(nn.Module):
    """Mutual partial token matcher.

    Query-to-support top-r keeps the original SPIF local partial evidence idea.
    Support-to-query top-r adds reverse consistency so one-way shortcut matches
    are less likely to dominate.
    """

    def __init__(self, top_r: int = 3, mutual: bool = True) -> None:
        super().__init__()
        self.top_r = int(top_r)
        self.mutual = bool(mutual)

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_tokens.dim() != 3:
            raise ValueError(
                "query_tokens must have shape (NumQuery, Tokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        if support_tokens.dim() != 4:
            raise ValueError(
                "support_tokens must have shape (Way, Shot, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )

        way_num, shot_num, token_num, dim = support_tokens.shape
        support_pool = support_tokens.reshape(way_num, shot_num * token_num, dim)
        query_tokens = _safe_normalize(query_tokens, dim=-1)
        support_pool = _safe_normalize(support_pool, dim=-1)
        similarity = torch.einsum("qld,wtd->qwlt", query_tokens, support_pool)

        top_r_q = min(max(1, self.top_r), similarity.shape[-1])
        s_q2s = similarity.topk(k=top_r_q, dim=-1).values.mean(dim=-1).mean(dim=-1)
        if not self.mutual:
            return s_q2s, s_q2s, s_q2s

        top_r_s = min(max(1, self.top_r), similarity.shape[-2])
        s_s2q = similarity.transpose(-1, -2).topk(k=top_r_s, dim=-1).values.mean(dim=-1).mean(dim=-1)
        s_local = 0.5 * (s_q2s + s_s2q)
        return s_local, s_q2s, s_s2q


class GlobalAnchoredResidualFusion(nn.Module):
    """Bound local evidence so it only corrects the global anchor."""

    def __init__(
        self,
        fusion_mode: str = "residual",
        lambda_local_init: float = 0.2,
        learnable_lambda_local: bool = False,
        alpha_init: float = 0.7,
        learnable_alpha: bool = False,
    ) -> None:
        super().__init__()
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {"residual", "linear"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        lambda_logit = torch.tensor(_prob_to_logit(lambda_local_init), dtype=torch.float32)
        if bool(learnable_lambda_local):
            self.lambda_local_logit = nn.Parameter(lambda_logit)
        else:
            self.register_buffer("lambda_local_logit", lambda_logit)

        alpha_logit = torch.tensor(_prob_to_logit(alpha_init), dtype=torch.float32)
        if bool(learnable_alpha):
            self.alpha_logit = nn.Parameter(alpha_logit)
        else:
            self.register_buffer("alpha_logit", alpha_logit)

    def lambda_local(self) -> torch.Tensor:
        return torch.sigmoid(self.lambda_local_logit)

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        global_scores: torch.Tensor,
        local_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lambda_local = self.lambda_local().to(device=global_scores.device, dtype=global_scores.dtype)
        alpha = self.alpha().to(device=global_scores.device, dtype=global_scores.dtype)
        if self.fusion_mode == "residual":
            delta_local = torch.tanh(local_scores - global_scores)
            fused = global_scores + lambda_local * delta_local
            return fused, delta_local, lambda_local, alpha

        fused = alpha * global_scores + (1.0 - alpha) * local_scores
        delta_local = local_scores - global_scores
        return fused, delta_local, lambda_local, alpha


class SPIFv2Encoder(nn.Module):
    """Shared bottleneck factorization + competitive stable evidence selection."""

    def __init__(
        self,
        input_dim: int,
        shared_dim: int,
        stable_dim: int,
        variant_dim: int,
        gate_hidden: int | None = None,
        gate_type: str = "sparsemax",
        use_shared_bottleneck: bool = True,
        token_l2norm: bool = True,
    ) -> None:
        super().__init__()
        self.projector = SharedStableVariantProjector(
            input_dim=input_dim,
            shared_dim=shared_dim,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            use_shared_bottleneck=use_shared_bottleneck,
        )
        self.gate = CompetitiveTokenGate(
            token_dim=stable_dim,
            hidden_dim=gate_hidden,
            gate_type=gate_type,
        )
        self.stable_norm = nn.LayerNorm(stable_dim)
        self.variant_norm = nn.LayerNorm(variant_dim)
        self.token_l2norm = bool(token_l2norm)

    def _normalize_branch(self, tokens: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        tokens = norm(tokens)
        if self.token_l2norm:
            tokens = _safe_normalize(tokens, dim=-1)
        return tokens

    def forward(self, tokens: torch.Tensor) -> SPIFv2TokenOutputs:
        shared_tokens, stable_tokens, variant_tokens = self.projector(tokens)
        gate, gate_entropy = self.gate(stable_tokens)
        gated_stable_tokens = gate * stable_tokens

        gate_mass = gate.sum(dim=1).clamp_min(1e-6)
        stable_global_raw = gated_stable_tokens.sum(dim=1) / gate_mass
        stable_global = _safe_normalize(stable_global_raw, dim=-1)
        variant_global = _safe_normalize(variant_tokens.mean(dim=1), dim=-1)

        gated_stable_tokens = self._normalize_branch(gated_stable_tokens, self.stable_norm)
        variant_tokens = self._normalize_branch(variant_tokens, self.variant_norm)

        return SPIFv2TokenOutputs(
            shared_tokens=shared_tokens,
            stable_tokens=stable_tokens,
            variant_tokens=variant_tokens,
            gated_stable_tokens=gated_stable_tokens,
            gate=gate,
            stable_global=stable_global,
            stable_global_raw=stable_global_raw,
            variant_global=variant_global,
            gate_entropy=gate_entropy,
        )


class _SPIFv2Base(BaseConv64FewShotModel):
    """Shared SPIF-v2 logic for the fair and regularized variants."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        shared_dim: int = 64,
        stable_dim: int = 32,
        variant_dim: int = 64,
        gate_hidden: int = 16,
        top_r: int = 3,
        gate_type: str = "sparsemax",
        use_shared_bottleneck: bool = True,
        mutual_local: bool = True,
        fusion_mode: str = "residual",
        lambda_local_init: float = 0.2,
        learnable_lambda_local: bool = False,
        alpha_init: float = 0.7,
        learnable_alpha: bool = False,
        global_only: bool = False,
        local_only: bool = False,
        token_l2norm: bool = True,
        consistency_weight: float = 0.0,
        decorr_weight: float = 0.0,
        gate_reg_weight: float = 0.0,
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
        self.top_r = int(top_r)
        self.factorization_on = bool(use_shared_bottleneck)
        self.consistency_weight = float(consistency_weight)
        self.decorr_weight = float(decorr_weight)
        self.gate_reg_weight = float(gate_reg_weight)
        self.consistency_dropout = float(consistency_dropout)

        self.encoder_head = SPIFv2Encoder(
            input_dim=hidden_dim,
            shared_dim=int(shared_dim),
            stable_dim=int(stable_dim),
            variant_dim=int(variant_dim),
            gate_hidden=int(gate_hidden),
            gate_type=gate_type,
            use_shared_bottleneck=use_shared_bottleneck,
            token_l2norm=token_l2norm,
        )
        self.local_matcher = MutualLocalMatcher(top_r=top_r, mutual=mutual_local)
        self.fusion = GlobalAnchoredResidualFusion(
            fusion_mode=fusion_mode,
            lambda_local_init=lambda_local_init,
            learnable_lambda_local=learnable_lambda_local,
            alpha_init=alpha_init,
            learnable_alpha=learnable_alpha,
        )
        self.variant_align = (
            nn.Identity()
            if int(variant_dim) == int(stable_dim)
            else nn.Linear(int(variant_dim), int(stable_dim), bias=False)
        )

    def encode_tokens(self, images: torch.Tensor) -> SPIFv2TokenOutputs:
        tokens = feature_map_to_tokens(self.encode(images))
        return self.encoder_head(tokens)

    @staticmethod
    def _reshape_support_outputs(outputs: SPIFv2TokenOutputs, way_num: int, shot_num: int) -> dict[str, torch.Tensor]:
        return {
            "support_global": outputs.stable_global.reshape(way_num, shot_num, -1),
            "support_tokens": outputs.gated_stable_tokens.reshape(
                way_num,
                shot_num,
                outputs.gated_stable_tokens.shape[1],
                outputs.gated_stable_tokens.shape[2],
            ),
            "support_variant_global": outputs.variant_global.reshape(way_num, shot_num, -1),
            "support_gate": outputs.gate.reshape(way_num, shot_num, -1, 1),
            "support_gate_entropy": outputs.gate_entropy.reshape(way_num, shot_num),
        }

    def _encode_episode(self, query: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        flat_support = support.reshape(way_num * shot_num, *support.shape[-3:])
        q_outputs = self.encode_tokens(query)
        s_outputs = self.encode_tokens(flat_support)

        episode = {
            "query_global": q_outputs.stable_global,
            "query_tokens": q_outputs.gated_stable_tokens,
            "query_variant_global": q_outputs.variant_global,
            "query_gate": q_outputs.gate,
            "query_gate_entropy": q_outputs.gate_entropy,
        }
        episode.update(self._reshape_support_outputs(s_outputs, way_num, shot_num))
        return episode

    @staticmethod
    def build_support_prototypes(support_global: torch.Tensor) -> torch.Tensor:
        return _safe_normalize(support_global.mean(dim=1), dim=-1)

    @staticmethod
    def compute_global_scores(query_global: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        return torch.matmul(
            _safe_normalize(query_global, dim=-1),
            _safe_normalize(prototypes, dim=-1).transpose(0, 1),
        )

    def compute_local_scores(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.local_matcher(query_tokens, support_tokens)

    def _consistency_loss(self, stable_global: torch.Tensor) -> torch.Tensor:
        if self.consistency_dropout <= 0.0:
            return stable_global.new_zeros(())
        # The current pulse_fewshot pipeline provides one view per image, so the
        # MAX variant forms two stochastic views from the final stable embedding.
        view1 = _safe_normalize(F.dropout(stable_global, p=self.consistency_dropout, training=True), dim=-1)
        view2 = _safe_normalize(F.dropout(stable_global, p=self.consistency_dropout, training=True), dim=-1)
        return (1.0 - F.cosine_similarity(view1, view2, dim=-1)).mean()

    @staticmethod
    def _decorrelation_loss(stable_global: torch.Tensor, variant_global: torch.Tensor) -> torch.Tensor:
        # Keep the original SPIF-style sample-wise cosine decorrelation because
        # it is bounded and lower-variance than batch-level cross-covariance in
        # small episodic few-shot training.
        return F.cosine_similarity(
            _safe_normalize(stable_global, dim=-1),
            _safe_normalize(variant_global, dim=-1),
            dim=-1,
        ).pow(2).mean()

    @staticmethod
    def _gate_reg_loss(gate_entropy: torch.Tensor) -> torch.Tensor:
        # Minimizing normalized entropy encourages selective, non-uniform gates.
        return gate_entropy.mean()

    def _aux_loss(
        self,
        query_global: torch.Tensor,
        support_global: torch.Tensor,
        query_variant_global: torch.Tensor,
        support_variant_global: torch.Tensor,
        query_gate_entropy: torch.Tensor,
        support_gate_entropy: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            return query_global.new_zeros(())

        loss = query_global.new_zeros(())
        if self.consistency_weight > 0.0:
            stable_global = torch.cat(
                [
                    query_global,
                    support_global.reshape(-1, support_global.shape[-1]),
                ],
                dim=0,
            )
            loss = loss + self.consistency_weight * self._consistency_loss(stable_global)

        if self.decorr_weight > 0.0:
            stable_global = torch.cat(
                [
                    query_global,
                    support_global.reshape(-1, support_global.shape[-1]),
                ],
                dim=0,
            )
            variant_global = torch.cat(
                [
                    query_variant_global,
                    support_variant_global.reshape(-1, support_variant_global.shape[-1]),
                ],
                dim=0,
            )
            variant_global = self.variant_align(variant_global)
            loss = loss + self.decorr_weight * self._decorrelation_loss(stable_global, variant_global)

        if self.gate_reg_weight > 0.0:
            gate_entropy = torch.cat(
                [
                    query_gate_entropy.reshape(-1),
                    support_gate_entropy.reshape(-1),
                ],
                dim=0,
            )
            loss = loss + self.gate_reg_weight * self._gate_reg_loss(gate_entropy)

        return loss

    def _fuse_scores(
        self,
        global_scores: torch.Tensor,
        local_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.global_only:
            lambda_local = self.fusion.lambda_local().to(device=global_scores.device, dtype=global_scores.dtype)
            alpha = self.fusion.alpha().to(device=global_scores.device, dtype=global_scores.dtype)
            return global_scores, torch.zeros_like(global_scores), lambda_local, alpha
        if self.local_only:
            lambda_local = self.fusion.lambda_local().to(device=global_scores.device, dtype=global_scores.dtype)
            alpha = self.fusion.alpha().to(device=global_scores.device, dtype=global_scores.dtype)
            return local_scores, local_scores - global_scores, lambda_local, alpha
        return self.fusion(global_scores, local_scores)

    def _forward_episode(self, query: torch.Tensor, support: torch.Tensor) -> Dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)
        prototypes = self.build_support_prototypes(episode["support_global"])
        global_scores = self.compute_global_scores(episode["query_global"], prototypes)
        local_scores, q2s_scores, s2q_scores = self.compute_local_scores(
            episode["query_tokens"],
            episode["support_tokens"],
        )
        logits, delta_local, lambda_local, alpha = self._fuse_scores(global_scores, local_scores)

        aux_loss = self._aux_loss(
            query_global=episode["query_global"],
            support_global=episode["support_global"],
            query_variant_global=episode["query_variant_global"],
            support_variant_global=episode["support_variant_global"],
            query_gate_entropy=episode["query_gate_entropy"],
            support_gate_entropy=episode["support_gate_entropy"],
        )

        all_gate_values = torch.cat(
            [
                episode["query_gate"].reshape(-1, 1),
                episode["support_gate"].reshape(-1, 1),
            ],
            dim=0,
        )
        all_gate_entropy = torch.cat(
            [
                episode["query_gate_entropy"].reshape(-1),
                episode["support_gate_entropy"].reshape(-1),
            ],
            dim=0,
        )
        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_scores": global_scores.detach(),
            "local_scores": local_scores.detach(),
            "q2s_scores": q2s_scores.detach(),
            "s2q_scores": s2q_scores.detach(),
            "delta_local": delta_local.detach(),
            "lambda_local": lambda_local.detach(),
            "alpha": alpha.detach(),
            "gate_mean": all_gate_values.mean().detach(),
            "gate_entropy": all_gate_entropy.mean().detach(),
            "gate_sparsity": (1.0 - all_gate_entropy.mean()).detach(),
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


class SPIFv2CE(_SPIFv2Base):
    """Fair SPIF-v2 variant: architecture-only, episodic CE only."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("learnable_lambda_local", False)
        kwargs.setdefault("learnable_alpha", False)
        kwargs.setdefault("consistency_weight", 0.0)
        kwargs.setdefault("decorr_weight", 0.0)
        kwargs.setdefault("gate_reg_weight", 0.0)
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
            "q2s_scores": torch.cat([item["q2s_scores"] for item in diagnostics], dim=0),
            "s2q_scores": torch.cat([item["s2q_scores"] for item in diagnostics], dim=0),
            "delta_local": torch.cat([item["delta_local"] for item in diagnostics], dim=0),
            "lambda_local": first["lambda_local"],
            "alpha": first["alpha"],
            "gate_mean": torch.stack([item["gate_mean"] for item in diagnostics]).mean(),
            "gate_entropy": torch.stack([item["gate_entropy"] for item in diagnostics]).mean(),
            "gate_sparsity": torch.stack([item["gate_sparsity"] for item in diagnostics]).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }


class SPIFv2MAX(_SPIFv2Base):
    """Stronger SPIF-v2 variant with lightweight regularization losses."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("learnable_lambda_local", True)
        kwargs.setdefault("learnable_alpha", False)
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
            "q2s_scores": torch.cat([item["q2s_scores"] for item in diagnostics], dim=0),
            "s2q_scores": torch.cat([item["s2q_scores"] for item in diagnostics], dim=0),
            "delta_local": torch.cat([item["delta_local"] for item in diagnostics], dim=0),
            "lambda_local": first["lambda_local"],
            "alpha": first["alpha"],
            "gate_mean": torch.stack([item["gate_mean"] for item in diagnostics]).mean(),
            "gate_entropy": torch.stack([item["gate_entropy"] for item in diagnostics]).mean(),
            "gate_sparsity": torch.stack([item["gate_sparsity"] for item in diagnostics]).mean(),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }
