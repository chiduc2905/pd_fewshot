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
    """Vectorized HROT few-shot model with ablation variants A-H."""

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
        score_scale: float = 16.0,
        tau_q: float = 0.5,
        tau_c: float = 0.5,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 60,
        sinkhorn_tolerance: float = 1e-5,
        fixed_mass: float = 0.8,
        min_mass: float = 0.1,
        mass_bonus_init: float = 1.0,
        lambda_rho: float = 0.01,
        rho_target: float = 0.8,
        lambda_curvature: float = 0.0,
        min_curvature: float = 0.05,
        normalize_euclidean_tokens: bool = True,
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
        if variant not in {"A", "B", "C", "D", "E", "F", "G", "H"}:
            raise ValueError(f"Unsupported HROT variant: {variant}")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if projection_scale <= 0.0:
            raise ValueError("projection_scale must be positive")
        if score_scale <= 0.0:
            raise ValueError("score_scale must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if not 0.0 < fixed_mass <= 1.0:
            raise ValueError("fixed_mass must be in (0, 1]")

        self.variant = variant
        self.uses_hyperbolic_geometry = variant in {"C", "D", "E"}
        self.uses_unbalanced_transport = variant in {"B", "D", "E", "F", "G", "H"}
        self.uses_learned_mass = variant in {"E", "F", "G", "H"}
        self.uses_shot_decomposed_transport = variant in {"G", "H"}
        self.uses_geodesic_eam = variant == "H"
        self.uses_mass_normalized_score = variant == "H"
        self.normalize_euclidean_tokens = bool(normalize_euclidean_tokens)
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
        self.lambda_rho = float(lambda_rho)
        self.rho_target = float(rho_target)
        self.lambda_curvature = float(lambda_curvature)
        self.min_curvature = float(min_curvature)
        self.eps = float(eps)

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
        self.eam = EpisodeAdaptiveMass(
            embed_dim=token_dim,
            hidden_dim=eam_hidden_dim,
            min_mass=min_mass,
            default_mass=fixed_mass,
            input_dim=4 if self.uses_geodesic_eam else None,
        )
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

        mean_distance = ball.dist(
            query_stats.mean_hyp[:, None, None, :],
            shot_means[None, :, :, :],
        )
        shot_spread = torch.zeros_like(mean_distance)
        query_variance = query_stats.variance[:, None, None].expand_as(mean_distance)
        support_variance = shot_variance[None, :, :].expand_as(mean_distance)
        features = torch.stack([mean_distance, shot_spread, query_variance, support_variance], dim=-1)
        rho = self.eam.forward_features(features)
        return rho.to(dtype=query_tokens_hyp.dtype)

    def _transport_match(
        self,
        cost: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_query, num_way, query_tokens, class_tokens = cost.shape
        base_a = cost.new_full((num_query, query_tokens), 1.0 / float(query_tokens))
        base_b = cost.new_full((num_way, class_tokens), 1.0 / float(class_tokens))
        a = base_a.unsqueeze(1).expand(-1, num_way, -1)
        b = base_b.unsqueeze(0).expand(num_query, -1, -1)

        if self.uses_unbalanced_transport:
            a = a * rho.unsqueeze(-1)
            b = b * rho.unsqueeze(-1)

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
            flat_cost = self._euclidean_cost(query_euclidean, flat_support_euclidean)
            if self.uses_geodesic_eam:
                shot_rho = self._build_geodesic_rho_per_shot(query_hyperbolic, support_hyperbolic)
                flat_rho = shot_rho.reshape(query.shape[0], way_num * shot_num)
            else:
                flat_rho = self._build_pairwise_rho(query_hyperbolic, flat_support_hyperbolic)
            flat_plan, flat_transport_cost, flat_transport_mass = self._transport_match(flat_cost, flat_rho)

            shot_transport_cost = flat_transport_cost.reshape(query.shape[0], way_num, shot_num)
            shot_transport_mass = flat_transport_mass.reshape(query.shape[0], way_num, shot_num)
            if shot_rho is None:
                shot_rho = flat_rho.reshape(query.shape[0], way_num, shot_num)
            if self.uses_mass_normalized_score:
                normalized_shot_cost = shot_transport_cost / shot_transport_mass.clamp_min(self.eps)
                shot_logits = -self.score_scale * normalized_shot_cost
            else:
                normalized_shot_cost = None
                shot_logits = -self.score_scale * shot_transport_cost
            if self.uses_unbalanced_transport and not self.uses_mass_normalized_score:
                shot_logits = shot_logits + self.mass_bonus.to(dtype=shot_logits.dtype) * shot_transport_mass

            logits = shot_logits.mean(dim=-1)
            transport_cost = (
                normalized_shot_cost.mean(dim=-1)
                if normalized_shot_cost is not None
                else shot_transport_cost.mean(dim=-1)
            )
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
                logits = logits + self.mass_bonus.to(dtype=logits.dtype) * transport_mass

        rho_regularization = logits.new_zeros(())
        if self.uses_learned_mass and self.lambda_rho > 0.0:
            rho_regularization = (rho - self.rho_target).pow(2).mean()

        curvature_regularization = logits.new_zeros(())
        if self.lambda_curvature > 0.0:
            curvature_regularization = (self.min_curvature - self.curvature).clamp_min(0.0).pow(2)

        aux_loss = self.lambda_rho * rho_regularization + self.lambda_curvature * curvature_regularization

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
            "curvature_regularization": curvature_regularization,
            "curvature": self.curvature.detach().to(dtype=logits.dtype),
            "mass_bonus": self.mass_bonus.detach().to(dtype=logits.dtype),
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
            "curvature_regularization": torch.stack([item["curvature_regularization"] for item in batch_outputs]).mean(),
            "curvature": torch.stack([item["curvature"] for item in batch_outputs]).mean(),
            "mass_bonus": torch.stack([item["mass_bonus"] for item in batch_outputs]).mean(),
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
