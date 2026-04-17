"""Hyperbolic Relational Optimal Transport for few-shot classification."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens, merge_support_tokens
from net.hyperbolic.poincare_ops import (
    PoincareBall,
    get_ball,
    hyperbolic_distance_matrix,
    resolve_hyperbolic_backend,
    safe_project_to_ball,
)
from net.modules.episode_adaptive_mass import EpisodeAdaptiveMass, summarize_hyperbolic_tokens
from net.modules.unbalanced_ot import (
    compute_transport_cost,
    compute_transported_mass,
    resolve_ot_backend,
    sinkhorn_balanced_log,
    sinkhorn_balanced_pot,
    sinkhorn_unbalanced_log,
    sinkhorn_unbalanced_pot,
)


class HROTFSLResult(dict):
    """Dict-like container that still exposes `.shape` through logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse softplus expects a positive value")
    return math.log(math.expm1(value))


class HROTFSL(BaseConv64FewShotModel):
    """Vectorized HROT few-shot model with ablation variants A-P."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 640,
        token_dim: int = 128,
        backbone_name: str = "resnet12",
        image_size: int = 84,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        variant: str = "E",
        eam_hidden_dim: int = 256,
        curvature_init: float = 1.0,
        projection_scale: float = 0.1,
        token_temperature: float = 0.1,
        score_scale: float = 16.0,
        tau_q: float = 0.5,
        tau_c: float = 0.5,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 60,
        sinkhorn_tolerance: float = 1e-5,
        fixed_mass: float = 0.8,
        min_mass: float = 0.1,
        mass_bonus_init: float = 1.0,
        transport_cost_threshold_init: float | None = None,
        lambda_rho: float = 0.01,
        rho_target: float = 0.8,
        lambda_rho_rank: float = 0.05,
        rho_rank_margin: float = 0.05,
        rho_rank_temperature: float = 0.05,
        lambda_curvature: float = 0.0,
        min_curvature: float = 0.05,
        normalize_euclidean_tokens: bool = True,
        normalize_rho: bool = False,
        eval_use_float64: bool = True,
        hyperbolic_backend: str = "auto",
        ot_backend: str = "native",
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        variant = str(variant).upper()
        if variant not in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"}:
            raise ValueError(f"Unsupported HROT variant: {variant}")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if projection_scale <= 0.0:
            raise ValueError("projection_scale must be positive")
        if token_temperature <= 0.0:
            raise ValueError("token_temperature must be positive")
        if score_scale <= 0.0:
            raise ValueError("score_scale must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if not 0.0 < fixed_mass <= 1.0:
            raise ValueError("fixed_mass must be in (0, 1]")
        if lambda_rho_rank < 0.0:
            raise ValueError("lambda_rho_rank must be non-negative")
        if rho_rank_margin < 0.0:
            raise ValueError("rho_rank_margin must be non-negative")
        if rho_rank_temperature <= 0.0:
            raise ValueError("rho_rank_temperature must be positive")

        self.variant = variant
        self.uses_hyperbolic_geometry = variant in {"C", "D", "E"}
        self.uses_unbalanced_transport = variant in {"B", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"}
        self.uses_learned_mass = variant in {"E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q"}
        self.uses_shot_decomposed_transport = variant in {"G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"}
        self.uses_geodesic_eam = variant in {"H", "K", "L", "M", "N", "O", "P", "Q"}
        self.uses_euclidean_geometric_eam = variant == "I"
        self.uses_reduced_geodesic_eam = variant == "L"
        self.uses_hybrid_ablation_eam = variant == "M"
        self.uses_transport_aware_eam = variant == "O"
        self.uses_cost_threshold_score = variant in {"H", "I", "J", "L", "M", "N", "O", "P", "Q"}
        self.uses_hybrid_mass_reward = variant == "M"
        self.uses_rho_rank_loss = variant == "N"
        self.uses_hyperbolic_token_attention = variant == "P"
        self.uses_noise_calibrated_transport = variant == "Q"
        self.normalize_euclidean_tokens = bool(normalize_euclidean_tokens)
        self.normalize_rho = bool(normalize_rho)
        self.eval_use_float64 = bool(eval_use_float64)
        self.hyperbolic_backend = resolve_hyperbolic_backend(hyperbolic_backend)
        self.ot_backend = resolve_ot_backend(ot_backend)
        self.projection_scale = float(projection_scale)
        self.score_scale = float(score_scale)
        self.tau_q = float(tau_q)
        self.tau_c = float(tau_c)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.fixed_mass = float(fixed_mass)
        self.min_mass = float(min_mass)
        self.lambda_rho = float(lambda_rho)
        self.rho_target = float(rho_target)
        self.lambda_rho_rank = float(lambda_rho_rank)
        self.rho_rank_margin = float(rho_rank_margin)
        self.rho_rank_temperature = float(rho_rank_temperature)
        self.lambda_curvature = float(lambda_curvature)
        self.min_curvature = float(min_curvature)
        self.eps = float(eps)
        self.default_token_temperature = float(token_temperature)

        self.token_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, token_dim, bias=False),
        )
        if self.hyperbolic_backend == "geoopt":
            self.manifold = get_ball(float(curvature_init), backend="geoopt", learnable=True)
            self.raw_curvature = None
        else:
            self.manifold = None
            self.raw_curvature = nn.Parameter(torch.tensor(_inverse_softplus(float(curvature_init)), dtype=torch.float32))
        eam_input_dim = None
        if self.uses_transport_aware_eam:
            eam_input_dim = 8
        elif self.uses_reduced_geodesic_eam:
            eam_input_dim = 3
        elif self.uses_geodesic_eam or self.uses_euclidean_geometric_eam:
            eam_input_dim = 4
        self.eam = EpisodeAdaptiveMass(
            embed_dim=token_dim,
            hidden_dim=eam_hidden_dim,
            min_mass=min_mass,
            default_mass=fixed_mass,
            input_dim=eam_input_dim,
        )
        if self.uses_hybrid_ablation_eam:
            self.euclidean_eam = EpisodeAdaptiveMass(
                embed_dim=token_dim,
                hidden_dim=eam_hidden_dim,
                min_mass=min_mass,
                default_mass=fixed_mass,
                input_dim=4,
            )
            self.reduced_geodesic_eam = EpisodeAdaptiveMass(
                embed_dim=token_dim,
                hidden_dim=eam_hidden_dim,
                min_mass=min_mass,
                default_mass=fixed_mass,
                input_dim=3,
            )
        else:
            self.euclidean_eam = None
            self.reduced_geodesic_eam = None
        self.raw_token_temperature = (
            nn.Parameter(torch.tensor(_inverse_softplus(float(token_temperature)), dtype=torch.float32))
            if self.uses_hyperbolic_token_attention or self.uses_noise_calibrated_transport
            else None
        )
        if self.uses_noise_calibrated_transport:
            self.query_reliability_weights = nn.Parameter(
                torch.tensor([1.0, -1.0, -0.5, -0.5], dtype=torch.float32)
            )
            self.support_reliability_weights = nn.Parameter(
                torch.tensor([1.0, -1.0, -0.5, -0.5], dtype=torch.float32)
            )
            self.raw_token_reliability_mix = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
            self.raw_support_consensus_mix = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
            self.raw_consensus_temperature = nn.Parameter(
                torch.tensor(_inverse_softplus(float(token_temperature)), dtype=torch.float32)
            )
            self.raw_noise_sink_cost = nn.Parameter(torch.tensor(_inverse_softplus(1.0), dtype=torch.float32))
            self.raw_shot_pool_temperature = nn.Parameter(torch.tensor(_inverse_softplus(1.0), dtype=torch.float32))
            self.raw_shot_pool_mix = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        else:
            self.query_reliability_weights = None
            self.support_reliability_weights = None
            self.raw_token_reliability_mix = None
            self.raw_support_consensus_mix = None
            self.raw_consensus_temperature = None
            self.raw_noise_sink_cost = None
            self.raw_shot_pool_temperature = None
            self.raw_shot_pool_mix = None
        if self.uses_cost_threshold_score:
            threshold_init = (
                float(transport_cost_threshold_init)
                if transport_cost_threshold_init is not None
                else float(mass_bonus_init) / self.score_scale
            )
            if threshold_init <= 0.0:
                raise ValueError("transport_cost_threshold_init must be positive for cost-threshold scoring")
            self.raw_transport_cost_threshold = nn.Parameter(
                torch.tensor(_inverse_softplus(threshold_init), dtype=torch.float32)
            )
            self.mass_bonus = (
                nn.Parameter(torch.tensor(float(mass_bonus_init), dtype=torch.float32))
                if self.uses_hybrid_mass_reward
                else None
            )
        else:
            self.raw_transport_cost_threshold = None
            self.mass_bonus = nn.Parameter(torch.tensor(float(mass_bonus_init), dtype=torch.float32))

    @property
    def curvature(self) -> torch.Tensor:
        if self.manifold is not None:
            return self.manifold.c.clamp_min(self.eps)
        return F.softplus(self.raw_curvature) + self.eps

    @property
    def curvature_parameter(self) -> torch.nn.Parameter:
        if self.manifold is not None:
            return self.manifold.manifold.isp_c
        return self.raw_curvature

    def _build_ball(self, reference: torch.Tensor) -> PoincareBall:
        if self.manifold is not None:
            return self.manifold
        return get_ball(
            self.curvature.to(device=reference.device, dtype=reference.dtype),
            backend="native",
        )

    @property
    def transport_cost_threshold(self) -> torch.Tensor:
        if self.raw_transport_cost_threshold is not None:
            return F.softplus(self.raw_transport_cost_threshold).clamp_min(self.eps)
        return self.mass_bonus / self.score_scale

    @property
    def token_temperature(self) -> torch.Tensor:
        if self.raw_token_temperature is not None:
            return F.softplus(self.raw_token_temperature).clamp_min(self.eps)
        return self.curvature.new_tensor(self.default_token_temperature)

    @property
    def consensus_temperature(self) -> torch.Tensor:
        if self.raw_consensus_temperature is not None:
            return F.softplus(self.raw_consensus_temperature).clamp_min(self.eps)
        return self.token_temperature

    @property
    def token_reliability_mix(self) -> torch.Tensor:
        if self.raw_token_reliability_mix is None:
            return self.curvature.new_tensor(0.0)
        return torch.sigmoid(self.raw_token_reliability_mix)

    @property
    def support_consensus_mix(self) -> torch.Tensor:
        if self.raw_support_consensus_mix is None:
            return self.curvature.new_tensor(0.0)
        return torch.sigmoid(self.raw_support_consensus_mix)

    @property
    def noise_sink_cost(self) -> torch.Tensor:
        if self.raw_noise_sink_cost is None:
            return self.curvature.new_tensor(1.0)
        return F.softplus(self.raw_noise_sink_cost).clamp_min(self.eps)

    @property
    def shot_pool_temperature(self) -> torch.Tensor:
        if self.raw_shot_pool_temperature is None:
            return self.curvature.new_tensor(1.0)
        return F.softplus(self.raw_shot_pool_temperature).clamp_min(self.eps)

    @property
    def shot_pool_mix(self) -> torch.Tensor:
        if self.raw_shot_pool_mix is None:
            return self.curvature.new_tensor(0.0)
        return torch.sigmoid(self.raw_shot_pool_mix)

    def _mass_reward_weight(self, reference: torch.Tensor) -> torch.Tensor:
        if self.raw_transport_cost_threshold is not None:
            threshold = self.transport_cost_threshold.to(device=reference.device, dtype=reference.dtype)
            reward = self.score_scale * threshold
            if self.uses_hybrid_mass_reward:
                reward = reward + self.mass_bonus.to(device=reference.device, dtype=reference.dtype)
            return reward
        return self.mass_bonus.to(device=reference.device, dtype=reference.dtype)

    def _normalize_rho_budget(self, rho: torch.Tensor) -> torch.Tensor:
        if not self.normalize_rho:
            return rho
        if rho.dim() == 2:
            reduce_dims = (1,)
        elif rho.dim() == 3:
            reduce_dims = (1, 2)
        else:
            raise ValueError(f"rho normalization expects 2D or 3D rho, got {tuple(rho.shape)}")
        rho_mean = rho.mean(dim=reduce_dims, keepdim=True).clamp_min(self.eps)
        rho = (rho / rho_mean) * self.fixed_mass
        return rho.clamp(min=self.min_mass, max=1.0)

    def _reshape_query_targets(
        self,
        query_targets: torch.Tensor | None,
        *,
        batch_size: int,
        num_query: int,
    ) -> torch.Tensor | None:
        if query_targets is None:
            return None
        if query_targets.dim() == 1:
            if query_targets.shape[0] != batch_size * num_query:
                raise ValueError(
                    "Flat query_targets must have length batch_size * num_query, "
                    f"got {query_targets.shape[0]} vs {batch_size * num_query}"
                )
            return query_targets.reshape(batch_size, num_query)
        if query_targets.dim() == 2 and tuple(query_targets.shape) == (batch_size, num_query):
            return query_targets
        raise ValueError(
            "query_targets must be shaped either (Batch * NumQuery,) or (Batch, NumQuery), "
            f"got {tuple(query_targets.shape)}"
        )

    def _encode_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected = self.token_projector(tokens)
        euclidean_tokens = (
            F.normalize(projected, p=2, dim=-1, eps=self.eps)
            if self.normalize_euclidean_tokens
            else projected
        )
        ball = self._build_ball(projected)
        hyperbolic_tokens = safe_project_to_ball(projected * self.projection_scale, ball)
        return euclidean_tokens, hyperbolic_tokens, spatial_hw

    def _euclidean_cost(self, query_tokens: torch.Tensor, class_tokens: torch.Tensor) -> torch.Tensor:
        query_norm = query_tokens.pow(2).sum(dim=-1)
        class_norm = class_tokens.pow(2).sum(dim=-1)
        dot = torch.einsum("qtd,wkd->qwtk", query_tokens, class_tokens)
        cost = query_norm[:, None, :, None] + class_norm[None, :, None, :] - 2.0 * dot
        return cost.clamp_min(0.0)

    def _cosine_cost(self, query_tokens: torch.Tensor, class_tokens: torch.Tensor) -> torch.Tensor:
        query_norm = F.normalize(query_tokens, p=2, dim=-1, eps=self.eps)
        class_norm = F.normalize(class_tokens, p=2, dim=-1, eps=self.eps)
        similarity = torch.einsum("qtd,wkd->qwtk", query_norm, class_norm)
        return (1.0 - similarity).clamp_min(0.0)

    def _hyperbolic_cost(
        self,
        query_tokens: torch.Tensor,
        class_tokens: torch.Tensor,
    ) -> torch.Tensor:
        calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens.dtype
        query_cast = query_tokens.to(dtype=calc_dtype)
        class_cast = class_tokens.to(dtype=calc_dtype)
        ball = self._build_ball(query_cast)
        cost = hyperbolic_distance_matrix(query_cast.unsqueeze(1), class_cast.unsqueeze(0), ball)
        return cost.to(dtype=query_tokens.dtype)

    def _build_pairwise_rho(
        self,
        query_tokens_hyp: torch.Tensor,
        class_tokens_hyp: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_unbalanced_transport:
            return query_tokens_hyp.new_ones(query_tokens_hyp.shape[0], class_tokens_hyp.shape[0])
        if not self.uses_learned_mass:
            return query_tokens_hyp.new_full((query_tokens_hyp.shape[0], class_tokens_hyp.shape[0]), self.fixed_mass)

        calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens_hyp.dtype
        query_cast = query_tokens_hyp.to(dtype=calc_dtype)
        class_cast = class_tokens_hyp.to(dtype=calc_dtype)
        ball = self._build_ball(query_cast)
        query_stats = summarize_hyperbolic_tokens(query_cast, ball)
        class_stats = summarize_hyperbolic_tokens(class_cast, ball)
        rho = self.eam.forward_from_stats(query_stats, class_stats)
        rho = self._normalize_rho_budget(rho)
        return rho.to(dtype=query_tokens_hyp.dtype)

    def _build_geodesic_rho_per_shot(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_unbalanced_transport:
            return query_tokens_hyp.new_ones(query_tokens_hyp.shape[0], *support_tokens_hyp.shape[:2])
        if not self.uses_learned_mass:
            return query_tokens_hyp.new_full(
                (query_tokens_hyp.shape[0], *support_tokens_hyp.shape[:2]),
                self.fixed_mass,
            )

        features = self._build_geodesic_eam_features(query_tokens_hyp, support_tokens_hyp)
        rho = self.eam.forward_features(features)
        rho = self._normalize_rho_budget(rho)
        return rho.to(dtype=query_tokens_hyp.dtype)

    def _build_geodesic_eam_features(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
    ) -> torch.Tensor:
        calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens_hyp.dtype
        query_cast = query_tokens_hyp.to(dtype=calc_dtype)
        support_cast = support_tokens_hyp.to(dtype=calc_dtype)
        way_num, shot_num = support_cast.shape[:2]
        ball = self._build_ball(query_cast)

        query_stats = summarize_hyperbolic_tokens(query_cast, ball)
        flat_support = support_cast.reshape(way_num * shot_num, support_cast.shape[-2], support_cast.shape[-1])
        shot_stats = summarize_hyperbolic_tokens(flat_support, ball)
        shot_means = shot_stats.mean_hyp.reshape(way_num, shot_num, -1)
        shot_variance = shot_stats.variance.reshape(way_num, shot_num)
        class_support = support_cast.reshape(way_num, shot_num * support_cast.shape[-2], support_cast.shape[-1])
        class_stats = summarize_hyperbolic_tokens(class_support, ball)

        mean_distance = ball.dist(
            query_stats.mean_hyp[:, None, None, :],
            shot_means[None, :, :, :],
        )
        shot_spread = ball.dist(
            shot_means,
            class_stats.mean_hyp[:, None, :],
        )[None, :, :].expand_as(mean_distance)
        query_variance = query_stats.variance[:, None, None].expand_as(mean_distance)
        support_variance = shot_variance[None, :, :].expand_as(mean_distance)
        if self.uses_reduced_geodesic_eam:
            return torch.stack([mean_distance, query_variance, support_variance], dim=-1)
        return torch.stack([mean_distance, shot_spread, query_variance, support_variance], dim=-1)

    def _build_euclidean_rho_per_shot(
        self,
        query_tokens_euc: torch.Tensor,
        support_tokens_euc: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_unbalanced_transport:
            return query_tokens_euc.new_ones(query_tokens_euc.shape[0], *support_tokens_euc.shape[:2])
        if not self.uses_learned_mass:
            return query_tokens_euc.new_full(
                (query_tokens_euc.shape[0], *support_tokens_euc.shape[:2]),
                self.fixed_mass,
            )

        features = self._build_euclidean_eam_features(query_tokens_euc, support_tokens_euc)
        rho = self.eam.forward_features(features)
        rho = self._normalize_rho_budget(rho)
        return rho.to(dtype=query_tokens_euc.dtype)

    def _build_hybrid_rho_per_shot(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
        query_tokens_euc: torch.Tensor,
        support_tokens_euc: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_unbalanced_transport:
            return query_tokens_euc.new_ones(query_tokens_euc.shape[0], *support_tokens_euc.shape[:2])
        if not self.uses_learned_mass:
            return query_tokens_euc.new_full(
                (query_tokens_euc.shape[0], *support_tokens_euc.shape[:2]),
                self.fixed_mass,
            )
        if self.euclidean_eam is None or self.reduced_geodesic_eam is None:
            raise RuntimeError("Hybrid HROT variant requires Euclidean and reduced-geodesic EAM heads")

        geodesic_features = self._build_geodesic_eam_features(query_tokens_hyp, support_tokens_hyp)
        euclidean_features = self._build_euclidean_eam_features(query_tokens_euc, support_tokens_euc)
        reduced_geodesic_features = geodesic_features[..., [0, 2, 3]]
        rho_geodesic = self.eam.forward_features(geodesic_features)
        rho_euclidean = self.euclidean_eam.forward_features(euclidean_features)
        rho_reduced = self.reduced_geodesic_eam.forward_features(reduced_geodesic_features)
        rho_fixed = torch.full_like(rho_geodesic, self.fixed_mass)
        rho = torch.stack([rho_geodesic, rho_euclidean, rho_reduced, rho_fixed], dim=0).mean(dim=0)
        rho = self._normalize_rho_budget(rho)
        return rho.to(dtype=query_tokens_euc.dtype)

    def _build_euclidean_eam_features(
        self,
        query_tokens_euc: torch.Tensor,
        support_tokens_euc: torch.Tensor,
    ) -> torch.Tensor:
        query_mean = query_tokens_euc.mean(dim=-2)
        way_num, shot_num = support_tokens_euc.shape[:2]
        flat_support = support_tokens_euc.reshape(way_num * shot_num, support_tokens_euc.shape[-2], support_tokens_euc.shape[-1])
        shot_mean = flat_support.mean(dim=-2).reshape(way_num, shot_num, -1)
        class_support = support_tokens_euc.reshape(way_num, shot_num * support_tokens_euc.shape[-2], support_tokens_euc.shape[-1])
        class_mean = class_support.mean(dim=-2)

        query_variance = (query_tokens_euc - query_mean[:, None, :]).pow(2).sum(dim=-1).mean(dim=-1)
        flat_shot_mean = shot_mean.reshape(way_num * shot_num, -1)
        shot_variance = (flat_support - flat_shot_mean[:, None, :]).pow(2).sum(dim=-1).mean(dim=-1)
        shot_variance = shot_variance.reshape(way_num, shot_num)

        mean_distance = torch.linalg.vector_norm(
            query_mean[:, None, None, :] - shot_mean[None, :, :, :],
            dim=-1,
        )
        shot_spread = torch.linalg.vector_norm(
            shot_mean - class_mean[:, None, :],
            dim=-1,
        )[None, :, :].expand_as(mean_distance)
        query_variance = query_variance[:, None, None].expand_as(mean_distance)
        support_variance = shot_variance[None, :, :].expand_as(mean_distance)
        return torch.stack([mean_distance, shot_spread, query_variance, support_variance], dim=-1)

    def _build_transport_probe_features(
        self,
        flat_cost: torch.Tensor,
        *,
        way_num: int,
        shot_num: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        num_query, num_pairs, query_tokens, support_tokens = flat_cost.shape
        if num_pairs != way_num * shot_num:
            raise ValueError(
                "flat_cost pair dimension must equal way_num * shot_num, "
                f"got {num_pairs} vs {way_num * shot_num}"
            )

        fixed_rho = flat_cost.new_full((num_query, num_pairs), self.fixed_mass)
        with torch.no_grad():
            probe_plan, probe_cost, probe_mass = self._transport_match(flat_cost.detach(), fixed_rho)
            plan_mass = probe_plan.sum(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
            plan_prob = probe_plan / plan_mass
            probe_entropy = -(plan_prob * plan_prob.clamp_min(self.eps).log()).sum(dim=(-1, -2))
            entropy_scale = math.log(float(max(2, query_tokens * support_tokens)))
            probe_entropy = probe_entropy / entropy_scale
            min_token_cost = flat_cost.detach().amin(dim=(-1, -2))

        features = torch.stack(
            [
                probe_cost,
                probe_mass,
                probe_entropy,
                min_token_cost,
            ],
            dim=-1,
        ).reshape(num_query, way_num, shot_num, 4)
        payload = {
            "transport_probe_cost": probe_cost.reshape(num_query, way_num, shot_num),
            "transport_probe_mass": probe_mass.reshape(num_query, way_num, shot_num),
            "transport_probe_entropy": probe_entropy.reshape(num_query, way_num, shot_num),
            "transport_probe_min_cost": min_token_cost.reshape(num_query, way_num, shot_num),
        }
        return features.to(dtype=flat_cost.dtype), payload

    def _build_transport_aware_geodesic_rho_per_shot(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
        flat_cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        geodesic_features = self._build_geodesic_eam_features(query_tokens_hyp, support_tokens_hyp)
        way_num, shot_num = support_tokens_hyp.shape[:2]
        transport_features, probe_payload = self._build_transport_probe_features(
            flat_cost,
            way_num=way_num,
            shot_num=shot_num,
        )
        features = torch.cat(
            [
                geodesic_features,
                transport_features.to(device=geodesic_features.device, dtype=geodesic_features.dtype),
            ],
            dim=-1,
        )
        rho = self.eam.forward_features(features)
        rho = self._normalize_rho_budget(rho)
        return rho.to(dtype=query_tokens_hyp.dtype), features, probe_payload

    def _standardize_over_tokens(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-2, keepdim=True)
        std = values.std(dim=-2, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (values - mean) / std

    def _build_probe_token_features(
        self,
        flat_cost: torch.Tensor,
        flat_rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            probe_plan, probe_cost, probe_mass = self._transport_match(flat_cost.detach(), flat_rho.detach())
            plan_mass = probe_plan.sum(dim=(-1, -2), keepdim=True).clamp_min(self.eps)

            row_mass = probe_plan.sum(dim=-1)
            row_cost = (probe_plan * flat_cost.detach()).sum(dim=-1) / row_mass.clamp_min(self.eps)
            row_prob = probe_plan / row_mass.unsqueeze(-1).clamp_min(self.eps)
            row_entropy = -(row_prob * row_prob.clamp_min(self.eps).log()).sum(dim=-1)
            row_entropy = row_entropy / math.log(float(max(2, flat_cost.shape[-1])))
            row_min_cost = flat_cost.detach().amin(dim=-1)
            row_mass_share = row_mass / plan_mass.squeeze(-1) * float(flat_cost.shape[-2])

            col_mass = probe_plan.sum(dim=-2)
            col_cost = (probe_plan * flat_cost.detach()).sum(dim=-2) / col_mass.clamp_min(self.eps)
            col_prob = probe_plan / col_mass.unsqueeze(-2).clamp_min(self.eps)
            col_entropy = -(col_prob * col_prob.clamp_min(self.eps).log()).sum(dim=-2)
            col_entropy = col_entropy / math.log(float(max(2, flat_cost.shape[-2])))
            col_min_cost = flat_cost.detach().amin(dim=-2)
            col_mass_share = col_mass / plan_mass.squeeze(-2) * float(flat_cost.shape[-1])

            query_features = torch.stack(
                [
                    row_mass_share.clamp_min(self.eps).log(),
                    row_cost,
                    row_entropy,
                    row_min_cost,
                ],
                dim=-1,
            )
            support_features = torch.stack(
                [
                    col_mass_share.clamp_min(self.eps).log(),
                    col_cost,
                    col_entropy,
                    col_min_cost,
                ],
                dim=-1,
            )

            query_features = self._standardize_over_tokens(query_features)
            support_features = self._standardize_over_tokens(support_features)
            plan_prob = probe_plan / plan_mass
            probe_entropy = -(plan_prob * plan_prob.clamp_min(self.eps).log()).sum(dim=(-1, -2))
            probe_entropy = probe_entropy / math.log(float(max(2, flat_cost.shape[-2] * flat_cost.shape[-1])))
            probe_min_cost = flat_cost.detach().amin(dim=(-1, -2))

        payload = {
            "transport_probe_cost": probe_cost,
            "transport_probe_mass": probe_mass,
            "transport_probe_entropy": probe_entropy,
            "transport_probe_min_cost": probe_min_cost,
        }
        return query_features.to(dtype=flat_cost.dtype), support_features.to(dtype=flat_cost.dtype), payload

    def _build_support_consensus_scores(self, support_tokens_euc: torch.Tensor) -> torch.Tensor:
        way_num, shot_num, token_num, _ = support_tokens_euc.shape
        if shot_num <= 1:
            return support_tokens_euc.new_zeros(way_num, shot_num, token_num)

        pair_cost = (
            support_tokens_euc[:, :, None, :, None, :] - support_tokens_euc[:, None, :, None, :, :]
        ).pow(2).sum(dim=-1)
        nearest_other = pair_cost.amin(dim=-1)
        shot_mask = ~torch.eye(shot_num, device=support_tokens_euc.device, dtype=torch.bool)
        nearest_other = nearest_other.masked_fill(~shot_mask[None, :, :, None], 0.0)
        mean_nearest = nearest_other.sum(dim=2) / float(shot_num - 1)
        temperature = self.consensus_temperature.to(device=support_tokens_euc.device, dtype=support_tokens_euc.dtype)
        scores = -mean_nearest / temperature
        return self._standardize_over_tokens(scores.unsqueeze(-1)).squeeze(-1)

    def _compute_noise_calibrated_token_marginals(
        self,
        flat_cost: torch.Tensor,
        flat_rho: torch.Tensor,
        support_tokens_euc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.query_reliability_weights is None or self.support_reliability_weights is None:
            raise RuntimeError("Noise-calibrated transport requires reliability weights")

        way_num, shot_num = support_tokens_euc.shape[:2]
        query_features, support_features, payload = self._build_probe_token_features(flat_cost, flat_rho)
        query_logits = torch.einsum(
            "qstf,f->qst",
            query_features,
            self.query_reliability_weights.to(device=flat_cost.device, dtype=flat_cost.dtype),
        )
        support_logits = torch.einsum(
            "qstf,f->qst",
            support_features,
            self.support_reliability_weights.to(device=flat_cost.device, dtype=flat_cost.dtype),
        )

        support_consensus = self._build_support_consensus_scores(support_tokens_euc)
        flat_consensus = support_consensus.reshape(way_num * shot_num, support_consensus.shape[-1])
        flat_consensus = flat_consensus.unsqueeze(0).expand(flat_cost.shape[0], -1, -1)
        consensus_mix = self.support_consensus_mix.to(device=flat_cost.device, dtype=flat_cost.dtype)
        support_logits = support_logits + consensus_mix * flat_consensus

        temperature = self.token_temperature.to(device=flat_cost.device, dtype=flat_cost.dtype)
        query_attn = torch.softmax(query_logits / temperature, dim=-1)
        support_attn = torch.softmax(support_logits / temperature, dim=-1)
        query_uniform = flat_cost.new_full(query_attn.shape, 1.0 / float(query_attn.shape[-1]))
        support_uniform = flat_cost.new_full(support_attn.shape, 1.0 / float(support_attn.shape[-1]))
        reliability_mix = self.token_reliability_mix.to(device=flat_cost.device, dtype=flat_cost.dtype)
        query_weights = (1.0 - reliability_mix) * query_uniform + reliability_mix * query_attn
        support_weights = (1.0 - reliability_mix) * support_uniform + reliability_mix * support_attn

        query_mass = query_weights * flat_rho.unsqueeze(-1).to(dtype=query_weights.dtype)
        support_mass = support_weights * flat_rho.unsqueeze(-1).to(dtype=support_weights.dtype)
        payload.update(
            {
                "probe_query_reliability": query_weights,
                "probe_support_reliability": support_weights,
                "support_consensus": flat_consensus,
            }
        )
        return query_mass, support_mass, payload

    def _append_noise_sink(
        self,
        flat_cost: torch.Tensor,
        query_mass: torch.Tensor,
        support_mass: torch.Tensor,
        flat_rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sink_cost = self.noise_sink_cost.to(device=flat_cost.device, dtype=flat_cost.dtype)
        cost_with_sink = sink_cost.expand(
            flat_cost.shape[0],
            flat_cost.shape[1],
            flat_cost.shape[2] + 1,
            flat_cost.shape[3] + 1,
        ).clone()
        cost_with_sink[..., :-1, :-1] = flat_cost
        cost_with_sink[..., -1, -1] = 0.0

        sink_mass = (1.0 - flat_rho).clamp_min(self.eps).to(device=flat_cost.device, dtype=flat_cost.dtype)
        query_mass_with_sink = torch.cat([query_mass, sink_mass.unsqueeze(-1)], dim=-1)
        support_mass_with_sink = torch.cat([support_mass, sink_mass.unsqueeze(-1)], dim=-1)
        return cost_with_sink, query_mass_with_sink, support_mass_with_sink

    def _pool_shot_scores(
        self,
        shot_logits: torch.Tensor,
        shot_transport_cost: torch.Tensor,
        shot_transport_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shot_num = shot_logits.shape[-1]
        if shot_num == 1:
            weights = torch.ones_like(shot_logits)
        else:
            evidence = shot_logits / float(self.score_scale)
            evidence = (evidence - evidence.mean(dim=-1, keepdim=True)) / evidence.std(
                dim=-1,
                keepdim=True,
                unbiased=False,
            ).clamp_min(self.eps)
            temperature = self.shot_pool_temperature.to(device=shot_logits.device, dtype=shot_logits.dtype)
            attentive = torch.softmax(evidence / temperature, dim=-1)
            uniform = torch.full_like(attentive, 1.0 / float(shot_num))
            mix = self.shot_pool_mix.to(device=shot_logits.device, dtype=shot_logits.dtype)
            weights = (1.0 - mix) * uniform + mix * attentive

        logits = (weights * shot_logits).sum(dim=-1)
        transport_cost = (weights * shot_transport_cost).sum(dim=-1)
        transport_mass = (weights * shot_transport_mass).sum(dim=-1)
        return logits, transport_cost, transport_mass, weights

    def _compute_hyperbolic_token_marginals(
        self,
        query_tokens_hyp: torch.Tensor,
        support_tokens_hyp: torch.Tensor,
        rho: torch.Tensor,
        ball: PoincareBall | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_tokens_hyp.dim() != 3 or support_tokens_hyp.dim() != 3:
            raise ValueError(
                "Hyperbolic token marginals expect query/support tokens shaped "
                "(NumQuery, Tokens, Dim) and (NumPairs, Tokens, Dim)"
            )
        num_query = query_tokens_hyp.shape[0]
        num_pairs = support_tokens_hyp.shape[0]
        if tuple(rho.shape) != (num_query, num_pairs):
            raise ValueError(
                "rho must have shape (NumQuery, NumPairs), "
                f"got {tuple(rho.shape)} vs {(num_query, num_pairs)}"
            )

        calc_dtype = torch.float64 if (self.eval_use_float64 and not self.training) else query_tokens_hyp.dtype
        query_cast = query_tokens_hyp.to(dtype=calc_dtype)
        support_cast = support_tokens_hyp.to(dtype=calc_dtype)
        ball = self._build_ball(query_cast) if ball is None else ball
        query_cast = ball.project(query_cast)
        support_cast = ball.project(support_cast)

        query_stats = summarize_hyperbolic_tokens(query_cast, ball)
        support_stats = summarize_hyperbolic_tokens(support_cast, ball)

        query_to_support = ball.dist(
            query_cast[:, None, :, :],
            support_stats.mean_hyp[None, :, None, :],
        )
        support_to_query = ball.dist(
            support_cast[None, :, :, :],
            query_stats.mean_hyp[:, None, None, :],
        )
        temperature = self.token_temperature.to(device=query_to_support.device, dtype=query_to_support.dtype)
        query_weights = torch.softmax(-query_to_support / temperature, dim=-1)
        support_weights = torch.softmax(-support_to_query / temperature, dim=-1)

        rho_cast = rho.to(device=query_weights.device, dtype=query_weights.dtype)
        query_mass = query_weights * rho_cast[..., None]
        support_mass = support_weights * rho_cast[..., None]
        return query_mass.to(dtype=query_tokens_hyp.dtype), support_mass.to(dtype=support_tokens_hyp.dtype)

    def _transport_match(
        self,
        cost: torch.Tensor,
        rho: torch.Tensor,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_query, num_way, query_tokens, class_tokens = cost.shape
        if a is None:
            base_a = cost.new_full((num_query, query_tokens), 1.0 / float(query_tokens))
            a = base_a.unsqueeze(1).expand(-1, num_way, -1)
            if self.uses_unbalanced_transport:
                a = a * rho.unsqueeze(-1)
        else:
            if tuple(a.shape) != (num_query, num_way, query_tokens):
                raise ValueError(
                    "a must have shape (NumQuery, NumPairs, QueryTokens), "
                    f"got {tuple(a.shape)}"
                )
            a = a.to(device=cost.device, dtype=cost.dtype)

        if b is None:
            base_b = cost.new_full((num_way, class_tokens), 1.0 / float(class_tokens))
            b = base_b.unsqueeze(0).expand(num_query, -1, -1)
            if self.uses_unbalanced_transport:
                b = b * rho.unsqueeze(-1)
        else:
            if tuple(b.shape) != (num_query, num_way, class_tokens):
                raise ValueError(
                    "b must have shape (NumQuery, NumPairs, SupportTokens), "
                    f"got {tuple(b.shape)}"
                )
            b = b.to(device=cost.device, dtype=cost.dtype)

        pair_cost = cost.reshape(num_query * num_way, query_tokens, class_tokens)
        pair_a = a.reshape(num_query * num_way, query_tokens)
        pair_b = b.reshape(num_query * num_way, class_tokens)

        if self.ot_backend == "pot":
            if self.uses_unbalanced_transport:
                pair_plan = sinkhorn_unbalanced_pot(
                    pair_cost,
                    pair_a,
                    pair_b,
                    tau_q=self.tau_q,
                    tau_c=self.tau_c,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )
            else:
                pair_plan = sinkhorn_balanced_pot(
                    pair_cost,
                    pair_a,
                    pair_b,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )
        else:
            if self.uses_unbalanced_transport:
                pair_plan = sinkhorn_unbalanced_log(
                    pair_cost,
                    pair_a,
                    pair_b,
                    tau_q=self.tau_q,
                    tau_c=self.tau_c,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )
            else:
                pair_plan = sinkhorn_balanced_log(
                    pair_cost,
                    pair_a,
                    pair_b,
                    eps=self.sinkhorn_epsilon,
                    max_iter=self.sinkhorn_iterations,
                    tol=self.sinkhorn_tolerance,
                )

        pair_transport_cost = compute_transport_cost(pair_plan, pair_cost)
        pair_transport_mass = compute_transported_mass(pair_plan)
        plan = pair_plan.reshape(num_query, num_way, query_tokens, class_tokens)
        transport_cost = pair_transport_cost.reshape(num_query, num_way)
        transport_mass = pair_transport_mass.reshape(num_query, num_way)
        return plan, transport_cost, transport_mass

    def _rho_rank_loss(
        self,
        rho: torch.Tensor,
        geodesic_distance: torch.Tensor | None,
    ) -> torch.Tensor:
        zero = rho.new_zeros(())
        if (
            not self.training
            or not self.uses_rho_rank_loss
            or self.lambda_rho_rank <= 0.0
            or geodesic_distance is None
            or rho.dim() != 3
        ):
            return zero

        distance = geodesic_distance.to(device=rho.device, dtype=rho.dtype)
        if distance.shape != rho.shape:
            raise ValueError(f"geodesic_distance must match rho shape, got {tuple(distance.shape)} vs {tuple(rho.shape)}")

        rho_flat = rho.reshape(rho.shape[0], -1)
        distance_flat = distance.detach().reshape(distance.shape[0], -1)
        distance_scale = distance_flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        distance_norm = (distance_flat - distance_flat.mean(dim=1, keepdim=True)) / distance_scale

        distance_gap = distance_norm.unsqueeze(1) - distance_norm.unsqueeze(2)
        order_weight = distance_gap.clamp_min(0.0)
        rho_gap = rho_flat.unsqueeze(2) - rho_flat.unsqueeze(1)
        penalty = F.softplus((self.rho_rank_margin - rho_gap) / self.rho_rank_temperature)
        penalty = penalty * self.rho_rank_temperature
        per_query_loss = (penalty * order_weight).sum(dim=(1, 2)) / order_weight.sum(dim=(1, 2)).clamp_min(self.eps)
        return per_query_loss.mean()

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        needs_payload: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        way_num, shot_num = support.shape[:2]
        query_euclidean, query_hyperbolic, query_hw = self._encode_images(query)
        support_euclidean, support_hyperbolic, support_hw = self._encode_images(
            support.reshape(way_num * shot_num, *support.shape[-3:])
        )
        if query_hw != support_hw:
            raise ValueError(f"Query/support token grids must match, got {query_hw} vs {support_hw}")

        support_euclidean = support_euclidean.reshape(
            way_num,
            shot_num,
            support_euclidean.shape[-2],
            support_euclidean.shape[-1],
        )
        support_hyperbolic = support_hyperbolic.reshape(
            way_num,
            shot_num,
            support_hyperbolic.shape[-2],
            support_hyperbolic.shape[-1],
        )
        class_euclidean = merge_support_tokens(support_euclidean, merge_mode="concat")
        class_hyperbolic = merge_support_tokens(support_hyperbolic, merge_mode="concat")

        shot_transport_cost = None
        shot_transport_mass = None
        shot_rho = None
        shot_geodesic_distance = None
        transport_probe_payload = None
        if self.uses_shot_decomposed_transport:
            flat_support_euclidean = support_euclidean.reshape(
                way_num * shot_num,
                support_euclidean.shape[-2],
                support_euclidean.shape[-1],
            )
            flat_support_hyperbolic = support_hyperbolic.reshape(
                way_num * shot_num,
                support_hyperbolic.shape[-2],
                support_hyperbolic.shape[-1],
            )
            if self.uses_noise_calibrated_transport:
                flat_cost = self._euclidean_cost(query_euclidean, flat_support_euclidean)
                geodesic_features = self._build_geodesic_eam_features(query_hyperbolic, support_hyperbolic)
                shot_geodesic_distance = geodesic_features[..., 0].to(dtype=query_hyperbolic.dtype)
                shot_rho = self.eam.forward_features(geodesic_features)
                shot_rho = self._normalize_rho_budget(shot_rho).to(dtype=query_hyperbolic.dtype)
                flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)

                query_token_mass, support_token_mass, transport_probe_payload = (
                    self._compute_noise_calibrated_token_marginals(
                        flat_cost,
                        flat_rho,
                        support_euclidean,
                    )
                )
                cost_with_sink, query_mass_with_sink, support_mass_with_sink = self._append_noise_sink(
                    flat_cost,
                    query_token_mass,
                    support_token_mass,
                    flat_rho,
                )
                flat_plan_with_sink, _, _ = self._transport_match(
                    cost_with_sink,
                    flat_rho,
                    a=query_mass_with_sink,
                    b=support_mass_with_sink,
                )
                flat_plan = flat_plan_with_sink[..., :-1, :-1]
                flat_transport_cost = compute_transport_cost(flat_plan, flat_cost)
                flat_transport_mass = compute_transported_mass(flat_plan)

                shot_transport_cost = flat_transport_cost.reshape(query.shape[0], way_num, shot_num)
                shot_transport_mass = flat_transport_mass.reshape(query.shape[0], way_num, shot_num)
                shot_logits = -self.score_scale * shot_transport_cost
                if self.uses_unbalanced_transport:
                    shot_logits = shot_logits + self._mass_reward_weight(shot_logits) * shot_transport_mass

                logits, transport_cost, transport_mass, shot_pool_weights = self._pool_shot_scores(
                    shot_logits,
                    shot_transport_cost,
                    shot_transport_mass,
                )
                rho = shot_rho
                cost = flat_cost.reshape(query.shape[0], way_num, shot_num, flat_cost.shape[-2], flat_cost.shape[-1])
                plan = flat_plan.reshape(query.shape[0], way_num, shot_num, flat_plan.shape[-2], flat_plan.shape[-1])
                transport_probe_payload.update(
                    {
                        "transport_probe_cost": transport_probe_payload["transport_probe_cost"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "transport_probe_mass": transport_probe_payload["transport_probe_mass"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "transport_probe_entropy": transport_probe_payload["transport_probe_entropy"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "transport_probe_min_cost": transport_probe_payload["transport_probe_min_cost"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "query_token_mass": query_token_mass.reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            query_token_mass.shape[-1],
                        ),
                        "support_token_mass": support_token_mass.reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            support_token_mass.shape[-1],
                        ),
                        "probe_query_reliability": transport_probe_payload["probe_query_reliability"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            query_token_mass.shape[-1],
                        ),
                        "probe_support_reliability": transport_probe_payload["probe_support_reliability"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            support_token_mass.shape[-1],
                        ),
                        "support_consensus": transport_probe_payload["support_consensus"].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                            support_token_mass.shape[-1],
                        ),
                        "shot_logits": shot_logits,
                        "shot_pool_weights": shot_pool_weights,
                        "noise_sink_query_mass": flat_plan_with_sink[..., :-1, -1].sum(dim=-1).reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "noise_sink_support_mass": flat_plan_with_sink[..., -1, :-1].sum(dim=-1).reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "noise_sink_self_mass": flat_plan_with_sink[..., -1, -1].reshape(
                            query.shape[0],
                            way_num,
                            shot_num,
                        ),
                        "token_temperature": self.token_temperature.detach().to(
                            device=logits.device,
                            dtype=logits.dtype,
                        ),
                        "token_reliability_mix": self.token_reliability_mix.detach().to(
                            device=logits.device,
                            dtype=logits.dtype,
                        ),
                        "support_consensus_mix": self.support_consensus_mix.detach().to(
                            device=logits.device,
                            dtype=logits.dtype,
                        ),
                        "noise_sink_cost": self.noise_sink_cost.detach().to(device=logits.device, dtype=logits.dtype),
                        "shot_pool_temperature": self.shot_pool_temperature.detach().to(
                            device=logits.device,
                            dtype=logits.dtype,
                        ),
                        "shot_pool_mix": self.shot_pool_mix.detach().to(device=logits.device, dtype=logits.dtype),
                    }
                )
            elif self.uses_hyperbolic_token_attention:
                flat_cost = self._euclidean_cost(query_euclidean, flat_support_euclidean)
                geodesic_features = self._build_geodesic_eam_features(query_hyperbolic, support_hyperbolic)
                shot_geodesic_distance = geodesic_features[..., 0].to(dtype=query_hyperbolic.dtype)
                shot_rho = self.eam.forward_features(geodesic_features)
                shot_rho = self._normalize_rho_budget(shot_rho).to(dtype=query_hyperbolic.dtype)
                flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                query_token_mass, support_token_mass = self._compute_hyperbolic_token_marginals(
                    query_hyperbolic,
                    flat_support_hyperbolic,
                    flat_rho,
                )
                flat_plan, flat_transport_cost, flat_transport_mass = self._transport_match(
                    flat_cost,
                    flat_rho,
                    a=query_token_mass,
                    b=support_token_mass,
                )

                shot_transport_cost = flat_transport_cost.reshape(query.shape[0], way_num, shot_num)
                shot_transport_mass = flat_transport_mass.reshape(query.shape[0], way_num, shot_num)
                shot_logits = -self.score_scale * shot_transport_cost
                if self.uses_unbalanced_transport:
                    shot_logits = shot_logits + self._mass_reward_weight(shot_logits) * shot_transport_mass

                logits = shot_logits.mean(dim=-1)
                transport_cost = shot_transport_cost.mean(dim=-1)
                transport_mass = shot_transport_mass.mean(dim=-1)
                rho = shot_rho
                cost = flat_cost.reshape(query.shape[0], way_num, shot_num, flat_cost.shape[-2], flat_cost.shape[-1])
                plan = flat_plan.reshape(query.shape[0], way_num, shot_num, flat_plan.shape[-2], flat_plan.shape[-1])
                transport_probe_payload = {
                    "query_token_mass": query_token_mass.reshape(
                        query.shape[0],
                        way_num,
                        shot_num,
                        query_token_mass.shape[-1],
                    ),
                    "support_token_mass": support_token_mass.reshape(
                        query.shape[0],
                        way_num,
                        shot_num,
                        support_token_mass.shape[-1],
                    ),
                    "shot_logits": shot_logits,
                    "token_temperature": self.token_temperature.detach().to(device=logits.device, dtype=logits.dtype),
                }
            else:
                flat_cost = self._euclidean_cost(query_euclidean, flat_support_euclidean)
                if self.uses_hybrid_ablation_eam:
                    shot_rho = self._build_hybrid_rho_per_shot(
                        query_hyperbolic,
                        support_hyperbolic,
                        query_euclidean,
                        support_euclidean,
                    )
                    flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                elif self.uses_euclidean_geometric_eam:
                    shot_rho = self._build_euclidean_rho_per_shot(query_euclidean, support_euclidean)
                    flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                elif self.uses_transport_aware_eam:
                    shot_rho, transport_aware_features, transport_probe_payload = (
                        self._build_transport_aware_geodesic_rho_per_shot(
                            query_hyperbolic,
                            support_hyperbolic,
                            flat_cost,
                        )
                    )
                    shot_geodesic_distance = transport_aware_features[..., 0].to(dtype=query_hyperbolic.dtype)
                    flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                elif self.uses_geodesic_eam:
                    if self.uses_rho_rank_loss:
                        geodesic_features = self._build_geodesic_eam_features(query_hyperbolic, support_hyperbolic)
                        shot_geodesic_distance = geodesic_features[..., 0].to(dtype=query_hyperbolic.dtype)
                        shot_rho = self.eam.forward_features(geodesic_features)
                        shot_rho = self._normalize_rho_budget(shot_rho).to(dtype=query_hyperbolic.dtype)
                    else:
                        shot_rho = self._build_geodesic_rho_per_shot(query_hyperbolic, support_hyperbolic)
                    flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
                else:
                    flat_rho = self._build_pairwise_rho(query_hyperbolic, flat_support_hyperbolic)
                flat_plan, flat_transport_cost, flat_transport_mass = self._transport_match(flat_cost, flat_rho)

                shot_transport_cost = flat_transport_cost.reshape(query.shape[0], way_num, shot_num)
                shot_transport_mass = flat_transport_mass.reshape(query.shape[0], way_num, shot_num)
                if shot_rho is None:
                    shot_rho = flat_rho.reshape(query.shape[0], way_num, shot_num)
                shot_logits = -self.score_scale * shot_transport_cost
                if self.uses_unbalanced_transport:
                    shot_logits = shot_logits + self._mass_reward_weight(shot_logits) * shot_transport_mass

                logits = shot_logits.mean(dim=-1)
                transport_cost = shot_transport_cost.mean(dim=-1)
                transport_mass = shot_transport_mass.mean(dim=-1)
                rho = shot_rho
                cost = flat_cost.reshape(query.shape[0], way_num, shot_num, flat_cost.shape[-2], flat_cost.shape[-1])
                plan = flat_plan.reshape(query.shape[0], way_num, shot_num, flat_plan.shape[-2], flat_plan.shape[-1])
        else:
            if self.uses_hyperbolic_geometry:
                cost = self._hyperbolic_cost(query_hyperbolic, class_hyperbolic)
            else:
                cost = self._euclidean_cost(query_euclidean, class_euclidean)

            rho = self._build_pairwise_rho(query_hyperbolic, class_hyperbolic)
            plan, transport_cost, transport_mass = self._transport_match(cost, rho)

            logits = -self.score_scale * transport_cost
            if self.uses_unbalanced_transport:
                logits = logits + self._mass_reward_weight(logits) * transport_mass

        rho_regularization = logits.new_zeros(())
        if self.uses_learned_mass and self.lambda_rho > 0.0:
            rho_regularization = (rho - self.rho_target).pow(2).mean()

        rho_rank_loss = self._rho_rank_loss(rho, shot_geodesic_distance)

        curvature_regularization = logits.new_zeros(())
        if self.lambda_curvature > 0.0:
            curvature_regularization = (self.min_curvature - self.curvature).clamp_min(0.0).pow(2)

        aux_loss = (
            self.lambda_rho * rho_regularization
            + self.lambda_rho_rank * rho_rank_loss
            + self.lambda_curvature * curvature_regularization
        )

        if not needs_payload:
            return logits

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "class_scores": logits,
            "total_distance": transport_cost,
            "transport_cost": transport_cost,
            "transported_mass": transport_mass,
            "rho": rho,
            "rho_regularization": rho_regularization,
            "rho_rank_loss": rho_rank_loss,
            "curvature_regularization": curvature_regularization,
            "curvature": self.curvature.detach().to(dtype=logits.dtype),
            "mass_bonus": self._mass_reward_weight(logits).detach().to(dtype=logits.dtype),
            "transport_cost_threshold": self.transport_cost_threshold.detach().to(
                device=logits.device,
                dtype=logits.dtype,
            ),
            "hyperbolic_backend": logits.new_tensor(0.0 if self.hyperbolic_backend == "native" else 1.0),
            "ot_backend": logits.new_tensor(0.0 if self.ot_backend == "native" else 1.0),
        }
        if self.uses_shot_decomposed_transport:
            outputs.update(
                {
                    "shot_transport_cost": shot_transport_cost,
                    "shot_transported_mass": shot_transport_mass,
                    "shot_rho": shot_rho,
                }
            )
            if transport_probe_payload is not None:
                outputs.update(transport_probe_payload)
        if return_aux:
            payload_support_euclidean = support_euclidean if self.uses_shot_decomposed_transport else class_euclidean
            payload_support_hyperbolic = support_hyperbolic if self.uses_shot_decomposed_transport else class_hyperbolic
            outputs.update(
                {
                    "transport_plan": plan,
                    "cost_matrix": cost,
                    "query_euclidean_tokens": query_euclidean,
                    "support_euclidean_tokens": payload_support_euclidean,
                    "query_hyperbolic_tokens": query_hyperbolic,
                    "support_hyperbolic_tokens": payload_support_hyperbolic,
                }
            )
        return outputs

    @staticmethod
    def _stack_outputs(batch_outputs: list[dict[str, torch.Tensor]]) -> HROTFSLResult:
        stacked: dict[str, Any] = {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "aux_loss": torch.stack([item["aux_loss"] for item in batch_outputs]).mean(),
            "class_scores": torch.cat([item["class_scores"] for item in batch_outputs], dim=0),
            "total_distance": torch.cat([item["total_distance"] for item in batch_outputs], dim=0),
            "transport_cost": torch.cat([item["transport_cost"] for item in batch_outputs], dim=0),
            "transported_mass": torch.cat([item["transported_mass"] for item in batch_outputs], dim=0),
            "rho": torch.cat([item["rho"] for item in batch_outputs], dim=0),
            "rho_regularization": torch.stack([item["rho_regularization"] for item in batch_outputs]).mean(),
            "rho_rank_loss": torch.stack([item["rho_rank_loss"] for item in batch_outputs]).mean(),
            "curvature_regularization": torch.stack([item["curvature_regularization"] for item in batch_outputs]).mean(),
            "curvature": torch.stack([item["curvature"] for item in batch_outputs]).mean(),
            "mass_bonus": torch.stack([item["mass_bonus"] for item in batch_outputs]).mean(),
            "transport_cost_threshold": torch.stack(
                [item["transport_cost_threshold"] for item in batch_outputs]
            ).mean(),
            "hyperbolic_backend": torch.stack([item["hyperbolic_backend"] for item in batch_outputs]).mean(),
            "ot_backend": torch.stack([item["ot_backend"] for item in batch_outputs]).mean(),
        }
        if "shot_transport_cost" in batch_outputs[0]:
            stacked["shot_transport_cost"] = torch.cat([item["shot_transport_cost"] for item in batch_outputs], dim=0)
            stacked["shot_transported_mass"] = torch.cat(
                [item["shot_transported_mass"] for item in batch_outputs],
                dim=0,
            )
            stacked["shot_rho"] = torch.cat([item["shot_rho"] for item in batch_outputs], dim=0)
        if "transport_probe_cost" in batch_outputs[0]:
            stacked["transport_probe_cost"] = torch.cat(
                [item["transport_probe_cost"] for item in batch_outputs],
                dim=0,
            )
            stacked["transport_probe_mass"] = torch.cat(
                [item["transport_probe_mass"] for item in batch_outputs],
                dim=0,
            )
            stacked["transport_probe_entropy"] = torch.cat(
                [item["transport_probe_entropy"] for item in batch_outputs],
                dim=0,
            )
            stacked["transport_probe_min_cost"] = torch.cat(
                [item["transport_probe_min_cost"] for item in batch_outputs],
                dim=0,
            )
        for key in (
            "query_token_mass",
            "support_token_mass",
            "probe_query_reliability",
            "probe_support_reliability",
            "support_consensus",
            "shot_logits",
            "shot_pool_weights",
            "noise_sink_query_mass",
            "noise_sink_support_mass",
            "noise_sink_self_mass",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        for key in (
            "token_temperature",
            "token_reliability_mix",
            "support_consensus_mix",
            "noise_sink_cost",
            "shot_pool_temperature",
            "shot_pool_mix",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        if "transport_plan" in batch_outputs[0]:
            stacked["transport_plan"] = torch.cat([item["transport_plan"] for item in batch_outputs], dim=0)
            stacked["cost_matrix"] = torch.cat([item["cost_matrix"] for item in batch_outputs], dim=0)
            stacked["query_euclidean_tokens"] = torch.cat(
                [item["query_euclidean_tokens"] for item in batch_outputs],
                dim=0,
            )
            stacked["support_euclidean_tokens"] = torch.stack(
                [item["support_euclidean_tokens"] for item in batch_outputs],
                dim=0,
            )
            stacked["query_hyperbolic_tokens"] = torch.cat(
                [item["query_hyperbolic_tokens"] for item in batch_outputs],
                dim=0,
            )
            stacked["support_hyperbolic_tokens"] = torch.stack(
                [item["support_hyperbolic_tokens"] for item in batch_outputs],
                dim=0,
            )
        return HROTFSLResult(stacked)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del support_targets
        batch_size, num_query, _, _, _, _ = self.validate_episode_inputs(query, support)
        self._reshape_query_targets(query_targets, batch_size=batch_size, num_query=num_query)

        needs_payload = bool(return_aux or self.training)
        batch_outputs = []
        batch_logits = []

        for batch_idx in range(batch_size):
            episode_outputs = self._forward_episode(
                query=query[batch_idx],
                support=support[batch_idx],
                needs_payload=needs_payload,
                return_aux=return_aux,
            )
            if needs_payload:
                batch_outputs.append(episode_outputs)
                batch_logits.append(episode_outputs["logits"])
            else:
                batch_logits.append(episode_outputs)

        logits = torch.cat(batch_logits, dim=0)
        if not needs_payload:
            return logits

        stacked = self._stack_outputs(batch_outputs)
        stacked["logits"] = logits
        if return_aux:
            return stacked
        return HROTFSLResult({"logits": logits, "aux_loss": stacked["aux_loss"]})
