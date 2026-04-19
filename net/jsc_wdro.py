"""Joint-Support Competitive WDRO for few-shot classification."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.modules.unbalanced_ot import (
    barycenter_balanced_pot,
    barycenter_unbalanced_pot,
    compute_transport_cost,
    compute_transported_mass,
    resolve_ot_backend,
    sinkhorn_balanced_pot,
    sinkhorn_unbalanced_pot,
)


class JSCWDROResult(dict):
    """Dict-like container that exposes `.shape` through logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse softplus expects a positive value")
    return math.log(math.expm1(value))


def _normalize_weights(weights: torch.Tensor, eps: float) -> torch.Tensor:
    weights = weights.clamp_min(0.0)
    total = weights.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(weights, 1.0 / float(weights.shape[-1]))
    normalized = weights / total.clamp_min(eps)
    return torch.where(total > eps, normalized, uniform)


class JSCWDRO(BaseConv64FewShotModel):
    """Barycentric WDRO classifier with adaptive Wasserstein uncertainty.

    The class-level nominal distribution is a POT fixed-support entropic
    Wasserstein/UOT barycenter over the union of all support tokens for a class.
    Scores use POT Sinkhorn solvers and follow the closed-form WDRO distance
    `score(q, c) = -[W_p(mu_q, nu_c) - epsilon_c]_+`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 640,
        token_dim: int = 128,
        backbone_name: str = "resnet12",
        image_size: int = 84,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 80,
        sinkhorn_tolerance: float = 1e-5,
        barycenter_iterations: int = 40,
        barycenter_tolerance: float = 1e-5,
        barycenter_transport: str = "unbalanced",
        barycenter_tau: float = 0.5,
        barycenter_method: str = "sinkhorn",
        tau_q: float = 0.5,
        tau_c: float = 0.5,
        query_transport: str = "balanced",
        score_scale: float = 16.0,
        epsilon_alpha_init: float = 0.05,
        epsilon_beta_init: float = 0.25,
        epsilon_floor_init: float = 1e-4,
        learn_epsilon: bool = True,
        epsilon_dimension: int | None = None,
        epsilon_reg_weight: float = 0.0,
        normalize_tokens: bool = True,
        cost_power: float = 2.0,
        normalize_unbalanced_cost: bool = True,
        use_competitive_diagnostics: bool = True,
        competitive_temperature: float = 0.1,
        profile: bool = False,
        ot_backend: str = "pot",
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
            resnet12_drop_rate=resnet12_drop_rate,
            resnet12_dropblock_size=resnet12_dropblock_size,
        )
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if sinkhorn_epsilon <= 0.0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if barycenter_iterations <= 0:
            raise ValueError("barycenter_iterations must be positive")
        if tau_q <= 0.0 or tau_c <= 0.0:
            raise ValueError("tau_q and tau_c must be positive")
        if barycenter_tau <= 0.0:
            raise ValueError("barycenter_tau must be positive")
        if score_scale <= 0.0:
            raise ValueError("score_scale must be positive")
        if epsilon_alpha_init <= 0.0 or epsilon_beta_init <= 0.0 or epsilon_floor_init <= 0.0:
            raise ValueError("epsilon initial values must be positive")
        if cost_power <= 0.0:
            raise ValueError("cost_power must be positive")
        query_transport = str(query_transport).lower()
        if query_transport not in {"balanced", "unbalanced"}:
            raise ValueError(f"Unsupported query_transport: {query_transport}")
        barycenter_transport = str(barycenter_transport).lower()
        if barycenter_transport not in {"balanced", "unbalanced"}:
            raise ValueError(f"Unsupported barycenter_transport: {barycenter_transport}")
        if competitive_temperature <= 0.0:
            raise ValueError("competitive_temperature must be positive")
        ot_backend = "pot" if str(ot_backend).lower() == "auto" else str(ot_backend).lower()
        if ot_backend != "pot":
            raise ValueError("JSC-WDRO uses POT solvers; set ot_backend='pot'")

        self.token_dim = int(token_dim)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.barycenter_iterations = int(barycenter_iterations)
        self.barycenter_tolerance = float(barycenter_tolerance)
        self.barycenter_transport = barycenter_transport
        self.barycenter_tau = float(barycenter_tau)
        self.barycenter_method = str(barycenter_method)
        self.tau_q = float(tau_q)
        self.tau_c = float(tau_c)
        self.query_transport = query_transport
        self.score_scale = float(score_scale)
        self.learn_epsilon = bool(learn_epsilon)
        self.epsilon_dimension = None if epsilon_dimension is None else int(epsilon_dimension)
        self.epsilon_reg_weight = float(epsilon_reg_weight)
        self.normalize_tokens = bool(normalize_tokens)
        self.cost_power = float(cost_power)
        self.normalize_unbalanced_cost = bool(normalize_unbalanced_cost)
        self.use_competitive_diagnostics = bool(use_competitive_diagnostics)
        self.competitive_temperature = float(competitive_temperature)
        self.profile = bool(profile)
        self.ot_backend = resolve_ot_backend(ot_backend)
        self.eps = float(eps)

        self.token_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, token_dim, bias=False),
        )

        raw_alpha = torch.tensor(_inverse_softplus(float(epsilon_alpha_init)), dtype=torch.float32)
        raw_beta = torch.tensor(_inverse_softplus(float(epsilon_beta_init)), dtype=torch.float32)
        raw_floor = torch.tensor(_inverse_softplus(float(epsilon_floor_init)), dtype=torch.float32)
        if self.learn_epsilon:
            self.raw_epsilon_alpha = nn.Parameter(raw_alpha)
            self.raw_epsilon_beta = nn.Parameter(raw_beta)
            self.raw_epsilon_floor = nn.Parameter(raw_floor)
        else:
            self.register_buffer("raw_epsilon_alpha", raw_alpha)
            self.register_buffer("raw_epsilon_beta", raw_beta)
            self.register_buffer("raw_epsilon_floor", raw_floor)

    @property
    def epsilon_alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_epsilon_alpha).clamp_min(self.eps)

    @property
    def epsilon_beta(self) -> torch.Tensor:
        return F.softplus(self.raw_epsilon_beta).clamp_min(self.eps)

    @property
    def epsilon_floor(self) -> torch.Tensor:
        return F.softplus(self.raw_epsilon_floor).clamp_min(self.eps)

    def _encode_images(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feature_map = self.encode(images)
        spatial_hw = (feature_map.shape[-2], feature_map.shape[-1])
        tokens = feature_map_to_tokens(feature_map)
        projected = self.token_projector(tokens)
        if self.normalize_tokens:
            projected = F.normalize(projected, p=2, dim=-1, eps=self.eps)
        return projected, spatial_hw

    def _as_support_token_sets(self, supports: torch.Tensor) -> torch.Tensor:
        if supports.dim() == 2:
            return supports.unsqueeze(1)
        if supports.dim() == 3:
            return supports
        raise ValueError(
            "supports must have shape (Shot, Dim) or (Shot, Tokens, Dim), "
            f"got {tuple(supports.shape)}"
        )

    def _as_query_batch(self, query: torch.Tensor) -> torch.Tensor:
        if query.dim() == 2:
            return query.unsqueeze(1)
        if query.dim() == 3:
            return query
        raise ValueError(
            "query must have shape (NumQuery, Dim) or (NumQuery, Tokens, Dim), "
            f"got {tuple(query.shape)}"
        )

    def _prepare_measure_weights(
        self,
        supports: torch.Tensor,
        support_weights: torch.Tensor | None = None,
        shot_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shot_num, token_num = supports.shape[:2]
        if support_weights is None:
            token_weights = supports.new_full((shot_num, token_num), 1.0 / float(token_num))
        else:
            support_weights = support_weights.to(device=supports.device, dtype=supports.dtype)
            if support_weights.dim() == 1 and support_weights.shape[0] == shot_num:
                if shot_weights is not None:
                    raise ValueError("support_weights with shape (Shot,) conflicts with explicit shot_weights")
                shot_weights = support_weights
                token_weights = supports.new_full((shot_num, token_num), 1.0 / float(token_num))
            elif tuple(support_weights.shape) == (shot_num, token_num):
                token_weights = _normalize_weights(support_weights, self.eps)
            else:
                raise ValueError(
                    "support_weights must have shape (Shot,) or (Shot, Tokens), "
                    f"got {tuple(support_weights.shape)}"
                )

        if shot_weights is None:
            shot_weights = supports.new_full((shot_num,), 1.0 / float(shot_num))
        else:
            shot_weights = shot_weights.to(device=supports.device, dtype=supports.dtype)
            if tuple(shot_weights.shape) != (shot_num,):
                raise ValueError(f"shot_weights must have shape ({shot_num},), got {tuple(shot_weights.shape)}")
            shot_weights = _normalize_weights(shot_weights, self.eps)
        return token_weights, shot_weights

    def _build_sparse_common_histograms(self, token_weights: torch.Tensor) -> torch.Tensor:
        shot_num, token_num = token_weights.shape
        histograms = token_weights.new_zeros(shot_num, shot_num * token_num)
        for shot_idx in range(shot_num):
            start = shot_idx * token_num
            histograms[shot_idx, start : start + token_num] = token_weights[shot_idx]
        return histograms

    def _pairwise_cost(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        lhs_sq = lhs.pow(2).sum(dim=-1)
        rhs_sq = rhs.pow(2).sum(dim=-1)
        dot = torch.matmul(lhs, rhs.transpose(-1, -2))
        sq_cost = (lhs_sq[..., :, None] + rhs_sq[..., None, :] - 2.0 * dot).clamp_min(0.0)
        if abs(self.cost_power - 2.0) < 1e-12:
            return sq_cost
        distance = torch.sqrt(sq_cost.clamp_min(0.0))
        if abs(self.cost_power - 1.0) < 1e-12:
            return distance
        return distance.pow(self.cost_power)

    def _entropic_barycenter_weights(
        self,
        histograms: torch.Tensor,
        support_cost: torch.Tensor,
        shot_weights: torch.Tensor,
    ) -> torch.Tensor:
        if histograms.shape[0] == 1:
            return _normalize_weights(histograms[0], self.eps)

        histograms = histograms.clamp_min(self.eps)
        weights = _normalize_weights(shot_weights, self.eps)
        if self.barycenter_transport == "unbalanced":
            barycenter = barycenter_unbalanced_pot(
                histograms,
                support_cost,
                weights,
                eps=self.sinkhorn_epsilon,
                reg_m=self.barycenter_tau,
                max_iter=self.barycenter_iterations,
                tol=self.barycenter_tolerance,
                method=self.barycenter_method,
            )
        else:
            barycenter = barycenter_balanced_pot(
                histograms,
                support_cost,
                weights,
                eps=self.sinkhorn_epsilon,
                max_iter=self.barycenter_iterations,
                tol=self.barycenter_tolerance,
                method=self.barycenter_method,
            )
        return _normalize_weights(barycenter.to(device=histograms.device, dtype=histograms.dtype), self.eps)

    def compute_barycenter(
        self,
        supports: torch.Tensor,
        support_weights: torch.Tensor | None = None,
        shot_weights: torch.Tensor | None = None,
        reg: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a fixed-support entropic Wasserstein barycenter.

        Args:
            supports: `(Shot, Dim)` or `(Shot, Tokens, Dim)` support measure.
            support_weights: optional `(Shot,)` shot weights or `(Shot, Tokens)`
                token masses.
            shot_weights: optional explicit barycenter mixing weights.
            reg: optional temporary Sinkhorn regularization.

        Returns:
            `(barycenter_tokens, barycenter_weights)` where tokens are the
            union support grid and weights sum to one.
        """
        old_reg = self.sinkhorn_epsilon
        if reg is not None:
            if reg <= 0.0:
                raise ValueError("reg must be positive")
            self.sinkhorn_epsilon = float(reg)
        try:
            support_tokens = self._as_support_token_sets(supports)
            token_weights, class_shot_weights = self._prepare_measure_weights(
                support_tokens,
                support_weights=support_weights,
                shot_weights=shot_weights,
            )
            barycenter_tokens = support_tokens.reshape(-1, support_tokens.shape[-1])
            histograms = self._build_sparse_common_histograms(token_weights)
            support_cost = self._pairwise_cost(barycenter_tokens, barycenter_tokens)
            barycenter_weights = self._entropic_barycenter_weights(histograms, support_cost, class_shot_weights)
            return barycenter_tokens, barycenter_weights
        finally:
            self.sinkhorn_epsilon = old_reg

    def _prepare_query_weights(self, query_tokens: torch.Tensor, query_weights: torch.Tensor | None) -> torch.Tensor:
        query_num, token_num = query_tokens.shape[:2]
        if query_weights is None:
            return query_tokens.new_full((query_num, token_num), 1.0 / float(token_num))
        query_weights = query_weights.to(device=query_tokens.device, dtype=query_tokens.dtype)
        if query_weights.dim() == 1 and query_weights.shape[0] == token_num:
            query_weights = query_weights.unsqueeze(0).expand(query_num, -1)
        if tuple(query_weights.shape) != (query_num, token_num):
            raise ValueError(
                "query_weights must have shape (Tokens,) or (NumQuery, Tokens), "
                f"got {tuple(query_weights.shape)}"
            )
        return _normalize_weights(query_weights, self.eps)

    def _compute_wot_details(
        self,
        query_tokens: torch.Tensor,
        query_weights: torch.Tensor | None,
        class_tokens: torch.Tensor,
        class_weights: torch.Tensor,
        *,
        unbalanced: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_tokens.dim() == 2:
            query_tokens = query_tokens.unsqueeze(0)
            squeeze_output = True
        elif query_tokens.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError(
                "query_tokens must have shape (Tokens, Dim) or (NumQuery, Tokens, Dim), "
                f"got {tuple(query_tokens.shape)}"
            )
        if class_tokens.dim() != 2:
            raise ValueError(f"class_tokens must have shape (Tokens, Dim), got {tuple(class_tokens.shape)}")

        query_mass = self._prepare_query_weights(query_tokens, query_weights)
        class_mass = _normalize_weights(class_weights.to(device=query_tokens.device, dtype=query_tokens.dtype), self.eps)
        class_mass = class_mass.unsqueeze(0).expand(query_tokens.shape[0], -1)
        cost = self._pairwise_cost(query_tokens, class_tokens)

        if unbalanced:
            plan = sinkhorn_unbalanced_pot(
                cost,
                query_mass,
                class_mass,
                tau_q=self.tau_q,
                tau_c=self.tau_c,
                eps=self.sinkhorn_epsilon,
                max_iter=self.sinkhorn_iterations,
                tol=self.sinkhorn_tolerance,
            )
        else:
            plan = sinkhorn_balanced_pot(
                cost,
                query_mass,
                class_mass,
                eps=self.sinkhorn_epsilon,
                max_iter=self.sinkhorn_iterations,
                tol=self.sinkhorn_tolerance,
            )

        transported_cost = compute_transport_cost(plan, cost)
        transported_mass = compute_transported_mass(plan)
        if unbalanced and self.normalize_unbalanced_cost:
            transported_cost = transported_cost / transported_mass.clamp_min(self.eps)
        distance = transported_cost.clamp_min(0.0).pow(1.0 / self.cost_power)

        if squeeze_output:
            return distance.squeeze(0), plan.squeeze(0), cost.squeeze(0), transported_mass.squeeze(0)
        return distance, plan, cost, transported_mass

    def compute_wot(
        self,
        query: torch.Tensor,
        query_weights: torch.Tensor | None,
        barycenter_tokens: torch.Tensor,
        barycenter_weights: torch.Tensor,
        reg: float | None = None,
    ) -> torch.Tensor:
        """Compute query-to-class entropic Wasserstein distance."""
        old_reg = self.sinkhorn_epsilon
        if reg is not None:
            if reg <= 0.0:
                raise ValueError("reg must be positive")
            self.sinkhorn_epsilon = float(reg)
        try:
            distance, _, _, _ = self._compute_wot_details(
                query,
                query_weights,
                barycenter_tokens,
                barycenter_weights,
                unbalanced=self.query_transport == "unbalanced",
            )
            return distance
        finally:
            self.sinkhorn_epsilon = old_reg

    def estimate_epsilon(
        self,
        supports: torch.Tensor,
        barycenter_tokens: torch.Tensor,
        barycenter_weights: torch.Tensor,
        support_weights: torch.Tensor | None = None,
        shot_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate class WDRO radius from sample size and support dispersion."""
        support_tokens = self._as_support_token_sets(supports)
        token_weights, class_shot_weights = self._prepare_measure_weights(
            support_tokens,
            support_weights=support_weights,
            shot_weights=shot_weights,
        )
        shot_distances = []
        for shot_idx in range(support_tokens.shape[0]):
            distance, _, _, _ = self._compute_wot_details(
                barycenter_tokens,
                barycenter_weights,
                support_tokens[shot_idx],
                token_weights[shot_idx],
                unbalanced=False,
            )
            shot_distances.append(distance)
        shot_distances_tensor = torch.stack(shot_distances, dim=0)
        dispersion = (class_shot_weights * shot_distances_tensor).sum()

        dim = float(max(self.epsilon_dimension or support_tokens.shape[-1], 2))
        sample_term = support_tokens.new_tensor(float(support_tokens.shape[0]) ** (-1.0 / dim))
        alpha = self.epsilon_alpha.to(device=support_tokens.device, dtype=support_tokens.dtype)
        beta = self.epsilon_beta.to(device=support_tokens.device, dtype=support_tokens.dtype)
        floor = self.epsilon_floor.to(device=support_tokens.device, dtype=support_tokens.dtype)
        epsilon = alpha * sample_term + beta * torch.sqrt(dispersion.clamp_min(0.0)) + floor
        return epsilon, dispersion, shot_distances_tensor

    def compute_competitive_assignment(
        self,
        query_tokens: torch.Tensor,
        barycenter_tokens: torch.Tensor,
        barycenter_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Soft class-competitive fallback allocation for diagnostics.

        This is not the full multi-marginal solver. It gives each query token a
        shared class budget using the barycenter kernel evidence, which is a
        cheap fallback for inspecting whether tokens are claimed by many
        classes.
        """
        if query_tokens.dim() != 3:
            raise ValueError(f"query_tokens must have shape (NumQuery, Tokens, Dim), got {tuple(query_tokens.shape)}")
        logits = []
        for class_idx in range(barycenter_tokens.shape[0]):
            cost = self._pairwise_cost(query_tokens, barycenter_tokens[class_idx])
            log_weights = torch.log(barycenter_weights[class_idx].clamp_min(self.eps))
            token_logit = torch.logsumexp(log_weights[None, None, :] - cost / self.sinkhorn_epsilon, dim=-1)
            logits.append(token_logit)
        token_class_logits = torch.stack(logits, dim=-1)
        return torch.softmax(token_class_logits / self.competitive_temperature, dim=-1)

    def _forward_feature_episode(
        self,
        supports: torch.Tensor,
        query: torch.Tensor,
        *,
        support_weights: torch.Tensor | None = None,
        query_weights: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        if supports.dim() == 3:
            supports = supports.unsqueeze(2)
        if supports.dim() != 4:
            raise ValueError(
                "supports must have shape (Way, Shot, Dim) or (Way, Shot, Tokens, Dim), "
                f"got {tuple(supports.shape)}"
            )
        query_tokens = self._as_query_batch(query)
        way_num = supports.shape[0]

        bary_start = time.perf_counter()
        barycenter_tokens = []
        barycenter_weights = []
        epsilons = []
        dispersions = []
        shot_distance_items = []
        for class_idx in range(way_num):
            class_support_weights = None if support_weights is None else support_weights[class_idx]
            class_tokens, class_weights = self.compute_barycenter(
                supports[class_idx],
                support_weights=class_support_weights,
            )
            epsilon, dispersion, shot_distances = self.estimate_epsilon(
                supports[class_idx],
                class_tokens,
                class_weights,
                support_weights=class_support_weights,
            )
            barycenter_tokens.append(class_tokens)
            barycenter_weights.append(class_weights)
            epsilons.append(epsilon)
            dispersions.append(dispersion)
            shot_distance_items.append(shot_distances)
        bary_ms = (time.perf_counter() - bary_start) * 1000.0

        bary_tokens_tensor = torch.stack(barycenter_tokens, dim=0)
        bary_weights_tensor = torch.stack(barycenter_weights, dim=0)
        epsilon_tensor = torch.stack(epsilons, dim=0)
        dispersion_tensor = torch.stack(dispersions, dim=0)
        shot_distance_tensor = torch.stack(shot_distance_items, dim=0)

        ot_start = time.perf_counter()
        distances = []
        plans = []
        costs = []
        masses = []
        for class_idx in range(way_num):
            distance, plan, cost, transported_mass = self._compute_wot_details(
                query_tokens,
                query_weights,
                bary_tokens_tensor[class_idx],
                bary_weights_tensor[class_idx],
                unbalanced=self.query_transport == "unbalanced",
            )
            distances.append(distance)
            plans.append(plan)
            costs.append(cost)
            masses.append(transported_mass)
        ot_ms = (time.perf_counter() - ot_start) * 1000.0

        distance_tensor = torch.stack(distances, dim=-1)
        transport_plan = torch.stack(plans, dim=1)
        cost_matrix = torch.stack(costs, dim=1)
        transported_mass = torch.stack(masses, dim=-1)

        score_start = time.perf_counter()
        robust_distance = (distance_tensor - epsilon_tensor[None, :]).clamp_min(0.0)
        logits = -self.score_scale * robust_distance
        aux_loss = logits.new_zeros(())
        if self.epsilon_reg_weight > 0.0:
            aux_loss = aux_loss + self.epsilon_reg_weight * epsilon_tensor.mean()
        score_ms = (time.perf_counter() - score_start) * 1000.0

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "class_scores": logits,
            "total_distance": robust_distance,
            "robust_distance": robust_distance,
            "transport_cost": distance_tensor,
            "query_class_distance": distance_tensor,
            "transported_mass": transported_mass,
            "epsilon": epsilon_tensor,
            "support_dispersion": dispersion_tensor,
            "shot_barycenter_distance": shot_distance_tensor,
        }
        if return_aux:
            outputs.update(
                {
                    "barycenter_tokens": bary_tokens_tensor,
                    "barycenter_weights": bary_weights_tensor,
                    "transport_plan": transport_plan,
                    "cost_matrix": cost_matrix,
                }
            )
            if self.use_competitive_diagnostics:
                outputs["competitive_assignment"] = self.compute_competitive_assignment(
                    query_tokens,
                    bary_tokens_tensor,
                    bary_weights_tensor,
                )
        if self.profile or return_aux:
            outputs.update(
                {
                    "profile_barycenter_ms": logits.new_tensor(bary_ms),
                    "profile_ot_ms": logits.new_tensor(ot_ms),
                    "profile_scoring_ms": logits.new_tensor(score_ms),
                    "profile_total_ms": logits.new_tensor(bary_ms + ot_ms + score_ms),
                }
            )
        return outputs

    def forward_episode_features(
        self,
        supports: torch.Tensor,
        query: torch.Tensor,
        *,
        support_weights: torch.Tensor | None = None,
        query_weights: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        outputs = self._forward_feature_episode(
            supports,
            query,
            support_weights=support_weights,
            query_weights=query_weights,
            return_aux=return_aux,
        )
        return outputs if return_aux else outputs["logits"]

    @staticmethod
    def _stack_outputs(batch_outputs: list[dict[str, torch.Tensor]]) -> JSCWDROResult:
        stacked: dict[str, Any] = {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "aux_loss": torch.stack([item["aux_loss"] for item in batch_outputs]).mean(),
            "class_scores": torch.cat([item["class_scores"] for item in batch_outputs], dim=0),
            "total_distance": torch.cat([item["total_distance"] for item in batch_outputs], dim=0),
            "robust_distance": torch.cat([item["robust_distance"] for item in batch_outputs], dim=0),
            "transport_cost": torch.cat([item["transport_cost"] for item in batch_outputs], dim=0),
            "query_class_distance": torch.cat([item["query_class_distance"] for item in batch_outputs], dim=0),
            "transported_mass": torch.cat([item["transported_mass"] for item in batch_outputs], dim=0),
            "epsilon": torch.stack([item["epsilon"] for item in batch_outputs], dim=0),
            "support_dispersion": torch.stack([item["support_dispersion"] for item in batch_outputs], dim=0),
            "shot_barycenter_distance": torch.stack(
                [item["shot_barycenter_distance"] for item in batch_outputs],
                dim=0,
            ),
        }
        optional_cat_keys = ("transport_plan", "cost_matrix", "competitive_assignment")
        for key in optional_cat_keys:
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        optional_stack_keys = ("barycenter_tokens", "barycenter_weights")
        for key in optional_stack_keys:
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs], dim=0)
        for key in ("profile_barycenter_ms", "profile_ot_ms", "profile_scoring_ms", "profile_total_ms"):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        return JSCWDROResult(stacked)

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
            way_num, shot_num = support[batch_idx].shape[:2]
            query_tokens, query_hw = self._encode_images(query[batch_idx])
            support_tokens, support_hw = self._encode_images(
                support[batch_idx].reshape(way_num * shot_num, *support.shape[-3:])
            )
            if query_hw != support_hw:
                raise ValueError(f"Query/support token grids must match, got {query_hw} vs {support_hw}")
            support_tokens = support_tokens.reshape(
                way_num,
                shot_num,
                support_tokens.shape[-2],
                support_tokens.shape[-1],
            )
            episode_outputs = self._forward_feature_episode(
                supports=support_tokens,
                query=query_tokens,
                return_aux=return_aux,
            )
            if needs_payload:
                batch_outputs.append(episode_outputs)
                batch_logits.append(episode_outputs["logits"])
            else:
                batch_logits.append(episode_outputs["logits"])

        logits = torch.cat(batch_logits, dim=0)
        if not needs_payload:
            return logits

        stacked = self._stack_outputs(batch_outputs)
        stacked["logits"] = logits
        if return_aux:
            return stacked
        return JSCWDROResult({"logits": logits, "aux_loss": stacked["aux_loss"]})


def train_one_episode(
    model: JSCWDRO,
    optimizer: torch.optim.Optimizer,
    supports: torch.Tensor,
    query: torch.Tensor,
    true_label: torch.Tensor,
    *,
    support_weights: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train JSC-WDRO on one feature-space episode."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model._forward_feature_episode(
        supports,
        query,
        support_weights=support_weights,
        query_weights=query_weights,
        return_aux=True,
    )
    targets = true_label.to(device=outputs["logits"].device, dtype=torch.long)
    if targets.dim() == 0:
        targets = targets.expand(outputs["logits"].shape[0])
    loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    loss.backward()
    optimizer.step()
    return loss.detach(), outputs


@torch.no_grad()
def predict_one_episode(
    model: JSCWDRO,
    supports: torch.Tensor,
    query: torch.Tensor,
    *,
    support_weights: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predict class indices for one feature-space episode."""
    model.eval()
    logits = model.forward_episode_features(
        supports,
        query,
        support_weights=support_weights,
        query_weights=query_weights,
        return_aux=False,
    )
    return logits.argmax(dim=-1)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.to(device=logits.device, dtype=torch.long)
    return (logits.argmax(dim=-1) == targets).float().mean()


def expected_calibration_error(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    """Expected calibration error computed by confidence bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    probs = torch.softmax(logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)
    targets = targets.to(device=logits.device, dtype=torch.long)
    correct = (predictions == targets).float()
    ece = logits.new_zeros(())
    boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=logits.device, dtype=logits.dtype)
    for bin_idx in range(n_bins):
        lower = boundaries[bin_idx]
        upper = boundaries[bin_idx + 1]
        if bin_idx == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            bin_confidence = confidence[mask].mean()
            bin_accuracy = correct[mask].mean()
            ece = ece + mask.float().mean() * (bin_confidence - bin_accuracy).abs()
    return ece
