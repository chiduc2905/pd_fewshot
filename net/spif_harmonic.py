"""SPIF-HMamba: stronger SPIF dual-branch model with a Mamba global branch.

The older SPIF-Harmonic attempt is intentionally replaced here with a much
cleaner design:

1. Keep the original SPIF stable / variant factorization and stable evidence
   gate.
2. Keep the local branch as paper-style sliced Wasserstein over gated stable
   tokens.
3. Strengthen only the global branch with a residual Mamba refiner followed by
   gated attentive token pooling, instead of simple GAP / cosine over pooled
   tokens.
4. Fuse global and local logits with an adaptive residual gate, not with
   support-query routing or distillation-style coupling.

Two variants are exposed:
- SPIFHarmCE: fused episodic CE only.
- SPIFHarmMAX: fused CE + branch CE + lightweight SPIF/Mamba regularization.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.metrics.sliced_wasserstein_weighted import WeightedPaperSlicedWassersteinDistance
from net.spif import SPIFEncoder
from net.spif_mamba import FinalStableGate, Mamba as OFFICIAL_MAMBA, StableEvidenceMambaRefiner
from net.ssm.intra_image_mamba import IntraImageMambaEncoder


def _normalized_entropy(weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = weights.clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1)
    normalizer = math.log(max(int(weights.shape[-1]), 2))
    return entropy / normalizer


def _top2_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] < 2:
        return torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


class MambaRefinedGlobalHead(nn.Module):
    """Global branch: gated stable tokens -> residual Mamba -> attentive pooling."""

    def __init__(
        self,
        dim: int,
        gate_hidden: int,
        state_dim: int = 16,
        mamba_depth: int = 1,
        mamba_ffn_multiplier: int = 2,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        beta_init: float = 0.1,
        learnable_beta: bool = True,
        use_final_gate: bool = True,
        score_scale: float = 16.0,
    ) -> None:
        super().__init__()
        if score_scale <= 0.0:
            raise ValueError("score_scale must be positive")

        self.use_final_gate = bool(use_final_gate)
        self.score_scale = float(score_scale)

        self.refiner = ResidualMambaRefiner(
            dim=dim,
            state_dim=state_dim,
            depth=mamba_depth,
            mamba_ffn_multiplier=mamba_ffn_multiplier,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_dropout=mamba_dropout,
            beta_init=beta_init,
            learnable_beta=learnable_beta,
        )
        self.final_gate = FinalStableGate(stable_dim=dim, gate_hidden=gate_hidden)
        self.pool_input_norm = nn.LayerNorm(dim)
        self.pool_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, stable_tokens: torch.Tensor, spatial_hw: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        if stable_tokens.dim() != 3:
            raise ValueError(
                "stable_tokens must have shape (Batch, Tokens, Dim), "
                f"got {tuple(stable_tokens.shape)}"
            )

        refined_tokens, beta = self.refiner(stable_tokens, spatial_hw=spatial_hw)
        final_gate = self.final_gate(refined_tokens, gate_on=self.use_final_gate)

        pooled_tokens = self.pool_input_norm(refined_tokens)
        attn_logits = self.pool_score(pooled_tokens).squeeze(-1)
        if self.use_final_gate:
            attn_logits = attn_logits + final_gate.squeeze(-1).clamp_min(1e-6).log()

        pool_weights = torch.softmax(attn_logits, dim=-1)
        embeddings = (pooled_tokens * pool_weights.unsqueeze(-1)).sum(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return {
            "embeddings": embeddings,
            "refined_tokens": F.normalize(pooled_tokens, p=2, dim=-1),
            "final_gate": final_gate,
            "pool_weights": pool_weights,
            "pool_entropy": _normalized_entropy(pool_weights),
            "beta": beta,
        }

    def compute_scores(self, query_global: torch.Tensor, support_global: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(support_global.mean(dim=1), p=2, dim=-1)
        query_global = F.normalize(query_global, p=2, dim=-1)
        return self.score_scale * torch.einsum("qd,wd->qw", query_global, prototypes)


class ResidualMambaRefiner(nn.Module):
    """Use official Mamba when available, otherwise fallback to the internal SSM."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        depth: int,
        mamba_ffn_multiplier: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mamba_dropout: float,
        beta_init: float,
        learnable_beta: bool,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")

        self.use_official_mamba = OFFICIAL_MAMBA is not None and depth > 0
        self.use_fallback_ssm = OFFICIAL_MAMBA is None and depth > 0
        self.learnable_beta = bool(learnable_beta)
        self.fixed_beta = float(max(0.0, beta_init))

        if self.learnable_beta and depth > 0:
            beta_init = min(max(float(beta_init), 1e-4), 1.0 - 1e-4)
            self.beta_logit = nn.Parameter(torch.tensor(math.log(beta_init / (1.0 - beta_init))))
        else:
            self.register_parameter("beta_logit", None)

        self.official_refiner = None
        if self.use_official_mamba:
            self.official_refiner = StableEvidenceMambaRefiner(
                dim=dim,
                state_dim=state_dim,
                depth=depth,
                ffn_multiplier=mamba_ffn_multiplier,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=mamba_dropout,
                beta_init=beta_init,
                learnable_beta=learnable_beta,
            )

        self.fallback_refiner = None
        if self.use_fallback_ssm:
            del mamba_d_conv, mamba_expand, mamba_dropout
            self.fallback_refiner = IntraImageMambaEncoder(
                dim=dim,
                state_dim=state_dim,
                depth=depth,
                ffn_multiplier=mamba_ffn_multiplier,
            )

    def beta(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.official_refiner is not None:
            return self.official_refiner.beta(device, dtype)
        if self.fallback_refiner is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        if self.learnable_beta and self.beta_logit is not None:
            return torch.sigmoid(self.beta_logit).to(device=device, dtype=dtype)
        return torch.tensor(self.fixed_beta, device=device, dtype=dtype)

    def forward(self, tokens: torch.Tensor, spatial_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.official_refiner is not None:
            return self.official_refiner(tokens, spatial_hw=spatial_hw)

        beta = self.beta(tokens.device, tokens.dtype)
        if self.fallback_refiner is None:
            return tokens, beta

        refined = self.fallback_refiner(tokens, spatial_hw=spatial_hw)
        refined = tokens + beta * (refined - tokens)
        return refined, beta


class LocalPaperSWHead(nn.Module):
    """Local branch: weighted paper-style SW over gated stable token sets."""

    def __init__(
        self,
        train_num_projections: int = 128,
        eval_num_projections: int = 512,
        p: float = 2.0,
        normalize_inputs: bool = False,
        train_projection_mode: str = "resample",
        eval_projection_mode: str = "fixed",
        eval_num_repeats: int = 1,
        score_scale: float = 8.0,
        projection_seed: int = 7,
    ) -> None:
        super().__init__()
        if score_scale <= 0.0:
            raise ValueError("score_scale must be positive")

        self.score_scale = float(score_scale)
        self.distance = WeightedPaperSlicedWassersteinDistance(
            train_num_projections=int(train_num_projections),
            eval_num_projections=int(eval_num_projections),
            p=float(p),
            reduction="none",
            normalize_inputs=bool(normalize_inputs),
            train_projection_mode=str(train_projection_mode),
            eval_projection_mode=str(eval_projection_mode),
            eval_num_repeats=int(eval_num_repeats),
            projection_seed=int(projection_seed),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        query_weights: torch.Tensor | None = None,
        support_weights: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
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
        support_weight_pool = None
        if support_weights is not None:
            if support_weights.dim() != 3:
                raise ValueError(
                    "support_weights must have shape (Way, Shot, Tokens), "
                    f"got {tuple(support_weights.shape)}"
                )
            support_weight_pool = support_weights.reshape(way_num, shot_num * token_num)
        distances = self.distance.pairwise_distance(
            query_tokens,
            support_pool,
            query_weights=query_weights,
            support_weights=support_weight_pool,
            reduction="none",
        )
        logits = -self.score_scale * distances
        return {
            "logits": logits,
            "distances": distances,
        }


class AdaptiveResidualFusion(nn.Module):
    """Adaptive residual fusion with the global branch as the anchor."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        hidden_dim = max(int(hidden_dim), 16)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        global_logits: torch.Tensor,
        local_logits: torch.Tensor,
        global_only: bool = False,
        local_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if global_only:
            return {
                "logits": global_logits,
                "fusion_gate": torch.zeros_like(global_logits),
                "delta_local": torch.zeros_like(global_logits),
            }
        if local_only:
            return {
                "logits": local_logits,
                "fusion_gate": torch.ones_like(local_logits),
                "delta_local": local_logits - global_logits,
            }

        delta = local_logits - global_logits
        global_margin = _top2_margin(global_logits).unsqueeze(-1).expand_as(global_logits)
        local_margin = _top2_margin(local_logits).unsqueeze(-1).expand_as(local_logits)
        gate_inputs = torch.stack(
            [
                global_logits,
                local_logits,
                delta,
                torch.abs(delta),
                global_margin,
                local_margin,
            ],
            dim=-1,
        )
        fusion_gate = torch.sigmoid(self.gate_mlp(gate_inputs).squeeze(-1))
        logits = global_logits + fusion_gate * torch.tanh(delta)
        return {
            "logits": logits,
            "fusion_gate": fusion_gate,
            "delta_local": delta,
        }


class _SPIFHarmonicBase(BaseConv64FewShotModel):
    """Shared SPIF-HMamba logic for CE and MAX variants."""

    requires_query_targets = True

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        stable_dim: int = 64,
        variant_dim: int = 64,
        gate_hidden: int = 16,
        gate_on: bool = True,
        factorization_on: bool = True,
        global_only: bool = False,
        local_only: bool = False,
        token_l2norm: bool = True,
        global_scale: float = 16.0,
        fusion_hidden_dim: int = 32,
        mamba_state_dim: int = 16,
        mamba_depth: int = 1,
        mamba_ffn_multiplier: int = 2,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        beta_init: float = 0.1,
        learnable_beta: bool = True,
        use_final_gate: bool = True,
        local_papersw_train_num_projections: int = 128,
        local_papersw_eval_num_projections: int = 512,
        local_papersw_p: float = 2.0,
        local_papersw_normalize_inputs: bool = False,
        local_papersw_train_projection_mode: str = "resample",
        local_papersw_eval_projection_mode: str = "fixed",
        local_papersw_eval_num_repeats: int = 1,
        local_papersw_score_scale: float = 8.0,
        local_papersw_projection_seed: int = 7,
        factor_consistency_weight: float = 0.0,
        factor_decorr_weight: float = 0.0,
        factor_sparse_weight: float = 0.0,
        factor_consistency_dropout: float = 0.1,
        final_sparse_weight: float = 0.0,
        global_ce_weight: float = 0.0,
        local_ce_weight: float = 0.0,
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

        self.global_only = bool(global_only)
        self.local_only = bool(local_only)
        self.gate_on = bool(gate_on)
        self.factorization_on = bool(factorization_on)
        self.factor_consistency_weight = float(factor_consistency_weight)
        self.factor_decorr_weight = float(factor_decorr_weight)
        self.factor_sparse_weight = float(factor_sparse_weight)
        self.factor_consistency_dropout = float(factor_consistency_dropout)
        self.final_sparse_weight = float(final_sparse_weight)
        self.global_ce_weight = float(global_ce_weight)
        self.local_ce_weight = float(local_ce_weight)

        self.encoder_head = SPIFEncoder(
            input_dim=hidden_dim,
            stable_dim=stable_dim,
            variant_dim=variant_dim,
            gate_hidden=gate_hidden,
            token_l2norm=token_l2norm,
        )
        self.global_head = MambaRefinedGlobalHead(
            dim=int(stable_dim),
            gate_hidden=int(gate_hidden),
            state_dim=int(mamba_state_dim),
            mamba_depth=int(mamba_depth),
            mamba_ffn_multiplier=int(mamba_ffn_multiplier),
            mamba_d_conv=int(mamba_d_conv),
            mamba_expand=int(mamba_expand),
            mamba_dropout=float(mamba_dropout),
            beta_init=float(beta_init),
            learnable_beta=bool(learnable_beta),
            use_final_gate=bool(use_final_gate),
            score_scale=float(global_scale),
        )
        self.local_head = LocalPaperSWHead(
            train_num_projections=int(local_papersw_train_num_projections),
            eval_num_projections=int(local_papersw_eval_num_projections),
            p=float(local_papersw_p),
            normalize_inputs=bool(local_papersw_normalize_inputs),
            train_projection_mode=str(local_papersw_train_projection_mode),
            eval_projection_mode=str(local_papersw_eval_projection_mode),
            eval_num_repeats=int(local_papersw_eval_num_repeats),
            score_scale=float(local_papersw_score_scale),
            projection_seed=int(local_papersw_projection_seed),
        )
        self.fusion_head = AdaptiveResidualFusion(hidden_dim=int(fusion_hidden_dim))
        self.variant_align = (
            nn.Identity()
            if int(variant_dim) == int(stable_dim)
            else nn.Linear(int(variant_dim), int(stable_dim), bias=False)
        )

    def _encode_images(self, images: torch.Tensor):
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        outputs = self.encoder_head(
            tokens,
            factorization_on=self.factorization_on,
            gate_on=self.gate_on,
        )
        return outputs, spatial_hw

    def _encode_episode(self, query: torch.Tensor, support: torch.Tensor) -> Dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        flat_support = support.reshape(way_num * shot_num, *support.shape[-3:])

        q_outputs, q_spatial_hw = self._encode_images(query)
        s_outputs, s_spatial_hw = self._encode_images(flat_support)

        return {
            "query_tokens": q_outputs.stable_tokens,
            "query_variant_global": q_outputs.variant_global,
            "query_gate": q_outputs.gate,
            "query_spatial_hw": q_spatial_hw,
            "support_tokens": s_outputs.stable_tokens.reshape(
                way_num,
                shot_num,
                -1,
                s_outputs.stable_tokens.shape[-1],
            ),
            "support_variant_global": s_outputs.variant_global.reshape(way_num, shot_num, -1),
            "support_gate": s_outputs.gate.reshape(way_num, shot_num, -1, 1),
            "support_spatial_hw": s_spatial_hw,
        }

    def _factor_consistency_loss(self, stable_tokens: torch.Tensor) -> torch.Tensor:
        if self.factor_consistency_dropout <= 0.0:
            return stable_tokens.new_zeros(())
        view1 = F.dropout(stable_tokens, p=self.factor_consistency_dropout, training=True).mean(dim=1)
        view2 = F.dropout(stable_tokens, p=self.factor_consistency_dropout, training=True).mean(dim=1)
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

    def _factorization_aux_loss(
        self,
        episode: Dict[str, torch.Tensor],
        query_global_embed: torch.Tensor,
        support_global_embed: torch.Tensor,
        query_final_gate: torch.Tensor,
        support_final_gate: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            return query_global_embed.new_zeros(())

        loss = query_global_embed.new_zeros(())
        if self.factor_consistency_weight > 0.0:
            all_stable_tokens = torch.cat(
                [
                    episode["query_tokens"],
                    episode["support_tokens"].reshape(
                        -1,
                        episode["support_tokens"].shape[-2],
                        episode["support_tokens"].shape[-1],
                    ),
                ],
                dim=0,
            )
            loss = loss + self.factor_consistency_weight * self._factor_consistency_loss(all_stable_tokens)

        if self.factorization_on and self.factor_decorr_weight > 0.0:
            stable_global = torch.cat(
                [query_global_embed, support_global_embed.reshape(-1, support_global_embed.shape[-1])],
                dim=0,
            )
            variant_global = torch.cat(
                [
                    episode["query_variant_global"],
                    episode["support_variant_global"].reshape(-1, episode["support_variant_global"].shape[-1]),
                ],
                dim=0,
            )
            variant_global = self.variant_align(variant_global)
            loss = loss + self.factor_decorr_weight * self._decorrelation_loss(stable_global, variant_global)

        if self.gate_on and self.factor_sparse_weight > 0.0:
            all_gates = torch.cat(
                [episode["query_gate"].reshape(-1, 1), episode["support_gate"].reshape(-1, 1)],
                dim=0,
            )
            loss = loss + self.factor_sparse_weight * self._sparse_gate_loss(all_gates)

        if self.final_sparse_weight > 0.0:
            all_final_gates = torch.cat(
                [query_final_gate.reshape(-1, 1), support_final_gate.reshape(-1, 1)],
                dim=0,
            )
            loss = loss + self.final_sparse_weight * self._sparse_gate_loss(all_final_gates)

        return loss

    def _branch_aux_loss(
        self,
        global_logits: torch.Tensor,
        local_logits: torch.Tensor,
        query_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.training or query_targets is None:
            return global_logits.new_zeros(())

        loss = global_logits.new_zeros(())
        if self.global_ce_weight > 0.0:
            loss = loss + self.global_ce_weight * F.cross_entropy(global_logits, query_targets)
        if self.local_ce_weight > 0.0:
            loss = loss + self.local_ce_weight * F.cross_entropy(local_logits, query_targets)
        return loss

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        episode = self._encode_episode(query, support)
        way_num, shot_num = support.shape[:2]

        query_global_out = self.global_head(episode["query_tokens"], spatial_hw=episode["query_spatial_hw"])
        support_global_out = self.global_head(
            episode["support_tokens"].reshape(
                way_num * shot_num,
                episode["support_tokens"].shape[-2],
                episode["support_tokens"].shape[-1],
            ),
            spatial_hw=episode["support_spatial_hw"],
        )

        support_global_embeddings = support_global_out["embeddings"].reshape(way_num, shot_num, -1)
        global_logits = self.global_head.compute_scores(
            query_global_out["embeddings"],
            support_global_embeddings,
        )
        local_out = self.local_head(
            episode["query_tokens"],
            episode["support_tokens"],
            query_weights=episode["query_gate"].squeeze(-1),
            support_weights=episode["support_gate"].squeeze(-1),
        )
        local_logits = local_out["logits"]
        fusion_out = self.fusion_head(
            global_logits=global_logits,
            local_logits=local_logits,
            global_only=self.global_only,
            local_only=self.local_only,
        )

        aux_loss = self._factorization_aux_loss(
            episode=episode,
            query_global_embed=query_global_out["embeddings"],
            support_global_embed=support_global_embeddings,
            query_final_gate=query_global_out["final_gate"],
            support_final_gate=support_global_out["final_gate"].reshape(
                way_num,
                shot_num,
                -1,
                1,
            ),
        ) + self._branch_aux_loss(
            global_logits=global_logits,
            local_logits=local_logits,
            query_targets=query_targets,
        )

        all_final_gates = torch.cat(
            [
                query_global_out["final_gate"].reshape(-1, 1),
                support_global_out["final_gate"].reshape(-1, 1),
            ],
            dim=0,
        )
        all_pool_entropy = torch.cat(
            [query_global_out["pool_entropy"], support_global_out["pool_entropy"]],
            dim=0,
        )

        return {
            "logits": fusion_out["logits"],
            "aux_loss": aux_loss,
            "global_logits": global_logits.detach(),
            "local_logits": local_logits.detach(),
            "fusion_gate": fusion_out["fusion_gate"].detach(),
            "delta_local": fusion_out["delta_local"].detach(),
            "beta": query_global_out["beta"].detach(),
            "final_gate_mean": all_final_gates.mean().detach(),
            "pool_entropy_mean": all_pool_entropy.mean().detach(),
            "mean_gate": torch.cat(
                [episode["query_gate"].reshape(-1, 1), episode["support_gate"].reshape(-1, 1)],
                dim=0,
            ).mean().detach(),
            "local_distances": local_out["distances"].detach(),
            "stable_global_embeddings": torch.cat(
                [
                    query_global_out["embeddings"],
                    support_global_embeddings.reshape(-1, support_global_embeddings.shape[-1]),
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

    @staticmethod
    def _reshape_query_targets(query_targets: Optional[torch.Tensor], batch_size: int, query_num: int):
        if query_targets is None:
            return None
        return query_targets.view(batch_size, query_num)


class SPIFHarmCE(_SPIFHarmonicBase):
    """Fair CE-only variant."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("factor_consistency_weight", 0.0)
        kwargs.setdefault("factor_decorr_weight", 0.0)
        kwargs.setdefault("factor_sparse_weight", 0.0)
        kwargs.setdefault("final_sparse_weight", 0.0)
        kwargs.setdefault("global_ce_weight", 0.0)
        kwargs.setdefault("local_ce_weight", 0.0)
        super().__init__(*args, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
        query_targets: Optional[torch.Tensor] = None,
        support_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        del support_targets
        bsz, nq, _, _, _, _ = self.validate_episode_inputs(query, support)
        query_targets = self._reshape_query_targets(query_targets, batch_size=bsz, query_num=nq)

        diagnostics = []
        for batch_idx in range(bsz):
            diagnostics.append(
                self._forward_episode(
                    query[batch_idx],
                    support[batch_idx],
                    query_targets=None if query_targets is None else query_targets[batch_idx],
                )
            )

        logits = torch.cat([item["logits"] for item in diagnostics], dim=0)
        if not return_aux:
            return logits

        return {
            "logits": logits,
            "global_logits": torch.cat([item["global_logits"] for item in diagnostics], dim=0),
            "local_logits": torch.cat([item["local_logits"] for item in diagnostics], dim=0),
            "fusion_gate": torch.cat([item["fusion_gate"] for item in diagnostics], dim=0),
            "delta_local": torch.cat([item["delta_local"] for item in diagnostics], dim=0),
            "beta": diagnostics[0]["beta"],
            "final_gate_mean": torch.stack([item["final_gate_mean"] for item in diagnostics]).mean(),
            "pool_entropy_mean": torch.stack([item["pool_entropy_mean"] for item in diagnostics]).mean(),
            "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
            "local_distances": torch.cat([item["local_distances"] for item in diagnostics], dim=0),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }


class SPIFHarmMAX(_SPIFHarmonicBase):
    """MAX variant with branch CE and lightweight gate / factor regularization."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("factor_consistency_weight", 0.05)
        kwargs.setdefault("factor_decorr_weight", 0.01)
        kwargs.setdefault("factor_sparse_weight", 5e-4)
        kwargs.setdefault("final_sparse_weight", 5e-4)
        kwargs.setdefault("global_ce_weight", 0.25)
        kwargs.setdefault("local_ce_weight", 0.25)
        super().__init__(*args, **kwargs)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
        query_targets: Optional[torch.Tensor] = None,
        support_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        del support_targets
        bsz, nq, _, _, _, _ = self.validate_episode_inputs(query, support)
        query_targets = self._reshape_query_targets(query_targets, batch_size=bsz, query_num=nq)

        diagnostics = []
        for batch_idx in range(bsz):
            diagnostics.append(
                self._forward_episode(
                    query[batch_idx],
                    support[batch_idx],
                    query_targets=None if query_targets is None else query_targets[batch_idx],
                )
            )

        logits = torch.cat([item["logits"] for item in diagnostics], dim=0)
        aux_loss = torch.stack([item["aux_loss"] for item in diagnostics]).mean()

        if not self.training and not return_aux:
            return logits

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "global_logits": torch.cat([item["global_logits"] for item in diagnostics], dim=0),
            "local_logits": torch.cat([item["local_logits"] for item in diagnostics], dim=0),
            "fusion_gate": torch.cat([item["fusion_gate"] for item in diagnostics], dim=0),
            "delta_local": torch.cat([item["delta_local"] for item in diagnostics], dim=0),
            "beta": diagnostics[0]["beta"],
            "final_gate_mean": torch.stack([item["final_gate_mean"] for item in diagnostics]).mean(),
            "pool_entropy_mean": torch.stack([item["pool_entropy_mean"] for item in diagnostics]).mean(),
            "mean_gate": torch.stack([item["mean_gate"] for item in diagnostics]).mean(),
            "local_distances": torch.cat([item["local_distances"] for item in diagnostics], dim=0),
            "stable_global_embeddings": torch.cat(
                [item["stable_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
            "variant_global_embeddings": torch.cat(
                [item["variant_global_embeddings"] for item in diagnostics],
                dim=0,
            ),
        }
