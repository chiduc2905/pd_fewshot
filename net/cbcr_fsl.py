"""CBCR-FSL: exact class-barycentric competitive robust transport model."""

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
    compute_transport_cost,
    compute_transported_mass,
    compute_unbalanced_transport_objective,
    resolve_ot_backend,
    sinkhorn_balanced_log,
    sinkhorn_balanced_pot,
    sinkhorn_unbalanced_log,
    sinkhorn_unbalanced_pot,
)


class CBCRFSLResult(dict):
    """Dict-like container that exposes `.shape` via logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


def _normalize_weights(weights: torch.Tensor, eps: float) -> torch.Tensor:
    weights = weights.clamp_min(0.0)
    total = weights.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(weights, 1.0 / float(weights.shape[-1]))
    normalized = weights / total.clamp_min(eps)
    return torch.where(total > eps, normalized, uniform)


class CBCRFSL(BaseConv64FewShotModel):
    """Closed-form CBCR-FSL implementation from the design document.

    Pipeline:
    1. backbone spatial tokens
    2. class-internal cross-reference token masses on supports
    3. fixed-support Wasserstein barycenter on the union of support tokens
    4. dynamic class uncertainty radius from shot-to-barycenter OT distance
    5. competitive token-to-class allocation on the query
    6. class-specific unbalanced OT score with rejectable mass
    7. robust logit `-max(0, D_c(q) - epsilon_c)`
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
        barycenter_method: str = "sinkhorn",
        alpha: float = 0.3,
        beta: float = 0.1,
        tau: float = 0.5,
        rho: float = 1.0,
        normalize_tokens: bool = True,
        cost_power: float = 2.0,
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
        if alpha <= 0.0 or beta <= 0.0 or tau <= 0.0 or rho <= 0.0:
            raise ValueError("alpha, beta, tau, and rho must be positive")
        if cost_power <= 0.0:
            raise ValueError("cost_power must be positive")
        ot_backend = "native" if str(ot_backend).lower() == "auto" else str(ot_backend).lower()
        if ot_backend not in {"native", "pot"}:
            raise ValueError("CBCR-FSL ot_backend must be 'native', 'pot', or 'auto'")

        self.token_dim = int(token_dim)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.barycenter_iterations = int(barycenter_iterations)
        self.barycenter_tolerance = float(barycenter_tolerance)
        self.barycenter_method = str(barycenter_method)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.rho = float(rho)
        self.normalize_tokens = bool(normalize_tokens)
        self.cost_power = float(cost_power)
        self.profile = bool(profile)
        self.ot_backend = resolve_ot_backend(ot_backend)
        self.eps = float(eps)

        self.token_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, token_dim, bias=False),
        )

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

    def compute_support_token_weights(self, support_tokens: torch.Tensor) -> torch.Tensor:
        """Cross-reference masses computed inside one support class.

        The closed document defines a support-side cross-image weighting rule.
        For 1-shot there is no cross-shot reference, so we use uniform weights.
        """
        support_tokens = self._as_support_token_sets(support_tokens)
        shot_num, token_num = support_tokens.shape[:2]
        if shot_num == 1:
            return support_tokens.new_full((1, token_num), 1.0 / float(token_num))

        normalized = F.normalize(support_tokens, p=2, dim=-1, eps=self.eps)
        token_weights = []
        for shot_idx in range(shot_num):
            reference_score = normalized.new_zeros(token_num)
            for other_idx in range(shot_num):
                if other_idx == shot_idx:
                    continue
                cosine = torch.matmul(normalized[shot_idx], normalized[other_idx].transpose(0, 1))
                reference_score = reference_score + cosine.max(dim=-1).values
            logits = reference_score / float(shot_num)
            token_weights.append(torch.softmax(logits, dim=-1))
        return torch.stack(token_weights, dim=0)

    def _build_sparse_common_histograms(self, token_weights: torch.Tensor) -> torch.Tensor:
        shot_num, token_num = token_weights.shape
        histograms = token_weights.new_zeros(shot_num, shot_num * token_num)
        for shot_idx in range(shot_num):
            start = shot_idx * token_num
            histograms[shot_idx, start : start + token_num] = token_weights[shot_idx]
        return histograms

    def compute_barycenter(
        self,
        support_tokens: torch.Tensor,
        support_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support_tokens = self._as_support_token_sets(support_tokens)
        support_weights = _normalize_weights(
            support_weights.to(device=support_tokens.device, dtype=support_tokens.dtype),
            self.eps,
        )
        if tuple(support_weights.shape) != tuple(support_tokens.shape[:2]):
            raise ValueError(
                "support_weights must have shape (Shot, Tokens), "
                f"got {tuple(support_weights.shape)} for supports {tuple(support_tokens.shape)}"
            )

        if support_tokens.shape[0] == 1:
            return support_tokens[0], support_weights[0]

        barycenter_tokens = support_tokens.reshape(-1, support_tokens.shape[-1])
        histograms = self._build_sparse_common_histograms(support_weights).clamp_min(self.eps)
        support_cost = self._pairwise_cost(barycenter_tokens, barycenter_tokens)
        shot_weights = support_tokens.new_full((support_tokens.shape[0],), 1.0 / float(support_tokens.shape[0]))
        barycenter_weights = barycenter_balanced_pot(
            histograms,
            support_cost,
            shot_weights,
            eps=self.sinkhorn_epsilon,
            max_iter=self.barycenter_iterations,
            tol=self.barycenter_tolerance,
            method=self.barycenter_method,
        )
        barycenter_weights = _normalize_weights(
            barycenter_weights.to(device=support_tokens.device, dtype=support_tokens.dtype),
            self.eps,
        )
        return barycenter_tokens, barycenter_weights

    def _prepare_query_uniform_weights(self, query_tokens: torch.Tensor) -> torch.Tensor:
        query_num, token_num = query_tokens.shape[:2]
        return query_tokens.new_full((query_num, token_num), 1.0 / float(token_num))

    def _expand_class_weights(
        self,
        class_weights: torch.Tensor,
        *,
        query_num: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        class_weights = _normalize_weights(class_weights.to(device=device, dtype=dtype), self.eps)
        if class_weights.dim() == 1:
            return class_weights.unsqueeze(0).expand(query_num, -1)
        if class_weights.dim() == 2 and class_weights.shape[0] == query_num:
            return _normalize_weights(class_weights, self.eps)
        raise ValueError(
            "class_weights must have shape (ClassTokens,) or (NumQuery, ClassTokens), "
            f"got {tuple(class_weights.shape)} for NumQuery={query_num}"
        )

    def _compute_balanced_transport_details(
        self,
        source_tokens: torch.Tensor,
        source_weights: torch.Tensor,
        target_tokens: torch.Tensor,
        target_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_tokens.dim() == 2:
            source_tokens = source_tokens.unsqueeze(0)
            source_weights = source_weights.unsqueeze(0)
            squeeze_output = True
        elif source_tokens.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError(
                "source_tokens must have shape (Tokens, Dim) or (NumQuery, Tokens, Dim), "
                f"got {tuple(source_tokens.shape)}"
            )
        if target_tokens.dim() != 2:
            raise ValueError(f"target_tokens must have shape (Tokens, Dim), got {tuple(target_tokens.shape)}")

        source_mass = _normalize_weights(
            source_weights.to(device=source_tokens.device, dtype=source_tokens.dtype),
            self.eps,
        )
        target_mass = self._expand_class_weights(
            target_weights,
            query_num=source_tokens.shape[0],
            device=source_tokens.device,
            dtype=source_tokens.dtype,
        )
        cost = self._pairwise_cost(source_tokens, target_tokens)
        solver = sinkhorn_balanced_pot if self.ot_backend == "pot" else sinkhorn_balanced_log
        plan = solver(
            cost,
            source_mass,
            target_mass,
            eps=self.sinkhorn_epsilon,
            max_iter=self.sinkhorn_iterations,
            tol=self.sinkhorn_tolerance,
        )
        transport_cost = compute_transport_cost(plan, cost)
        transported_mass = compute_transported_mass(plan)
        if squeeze_output:
            return transport_cost.squeeze(0), plan.squeeze(0), cost.squeeze(0), transported_mass.squeeze(0)
        return transport_cost, plan, cost, transported_mass

    def _compute_unbalanced_transport_details(
        self,
        query_tokens: torch.Tensor,
        query_weights: torch.Tensor,
        class_tokens: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_tokens.dim() == 2:
            query_tokens = query_tokens.unsqueeze(0)
            query_weights = query_weights.unsqueeze(0)
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

        query_mass = query_weights.to(device=query_tokens.device, dtype=query_tokens.dtype).clamp_min(0.0)
        class_mass = self._expand_class_weights(
            class_weights,
            query_num=query_tokens.shape[0],
            device=query_tokens.device,
            dtype=query_tokens.dtype,
        )
        cost = self._pairwise_cost(query_tokens, class_tokens)
        solver = sinkhorn_unbalanced_pot if self.ot_backend == "pot" else sinkhorn_unbalanced_log
        plan = solver(
            cost,
            query_mass,
            class_mass,
            tau_q=self.rho,
            tau_c=self.rho,
            eps=self.sinkhorn_epsilon,
            max_iter=self.sinkhorn_iterations,
            tol=self.sinkhorn_tolerance,
        )
        objective = compute_unbalanced_transport_objective(
            plan,
            cost,
            query_mass,
            class_mass,
            tau_q=self.rho,
            tau_c=self.rho,
            eps=self.eps,
        )
        transported_mass = compute_transported_mass(plan)
        if squeeze_output:
            return objective.squeeze(0), plan.squeeze(0), cost.squeeze(0), transported_mass.squeeze(0)
        return objective, plan, cost, transported_mass

    def estimate_epsilon(
        self,
        support_tokens: torch.Tensor,
        barycenter_tokens: torch.Tensor,
        barycenter_weights: torch.Tensor,
        support_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        support_tokens = self._as_support_token_sets(support_tokens)
        support_weights = _normalize_weights(
            support_weights.to(device=support_tokens.device, dtype=support_tokens.dtype),
            self.eps,
        )
        shot_distances = []
        for shot_idx in range(support_tokens.shape[0]):
            distance, _, _, _ = self._compute_balanced_transport_details(
                support_tokens[shot_idx],
                support_weights[shot_idx],
                barycenter_tokens,
                barycenter_weights,
            )
            shot_distances.append(distance)
        shot_distance_tensor = torch.stack(shot_distances, dim=0)
        dispersion = shot_distance_tensor.mean()
        floor = support_tokens.new_tensor(self.beta / math.sqrt(float(support_tokens.shape[0])))
        epsilon = torch.maximum(support_tokens.new_tensor(self.alpha) * dispersion, floor)
        return epsilon, dispersion, shot_distance_tensor

    def compute_competitive_allocation(
        self,
        query_tokens: torch.Tensor,
        barycenter_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_tokens.dim() != 3:
            raise ValueError(f"query_tokens must have shape (NumQuery, Tokens, Dim), got {tuple(query_tokens.shape)}")
        nearest_distances = []
        for class_idx in range(barycenter_tokens.shape[0]):
            cost = self._pairwise_cost(query_tokens, barycenter_tokens[class_idx])
            nearest_distances.append(cost.min(dim=-1).values)
        raw_distance = torch.stack(nearest_distances, dim=-1)
        assignment = torch.softmax(-raw_distance / self.tau, dim=-1)
        return assignment, raw_distance

    def _forward_feature_episode(
        self,
        supports: torch.Tensor,
        query: torch.Tensor,
        *,
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
        support_weight_items = []
        epsilons = []
        dispersions = []
        shot_distance_items = []
        for class_idx in range(way_num):
            class_support_weights = self.compute_support_token_weights(supports[class_idx])
            class_tokens, class_weights = self.compute_barycenter(
                supports[class_idx],
                class_support_weights,
            )
            epsilon, dispersion, shot_distances = self.estimate_epsilon(
                supports[class_idx],
                class_tokens,
                class_weights,
                class_support_weights,
            )
            barycenter_tokens.append(class_tokens)
            barycenter_weights.append(class_weights)
            support_weight_items.append(class_support_weights)
            epsilons.append(epsilon)
            dispersions.append(dispersion)
            shot_distance_items.append(shot_distances)
        bary_ms = (time.perf_counter() - bary_start) * 1000.0

        bary_tokens_tensor = torch.stack(barycenter_tokens, dim=0)
        bary_weights_tensor = torch.stack(barycenter_weights, dim=0)
        support_weight_tensor = torch.stack(support_weight_items, dim=0)
        epsilon_tensor = torch.stack(epsilons, dim=0)
        dispersion_tensor = torch.stack(dispersions, dim=0)
        shot_distance_tensor = torch.stack(shot_distance_items, dim=0)

        competition_start = time.perf_counter()
        base_query_weights = self._prepare_query_uniform_weights(query_tokens)
        competitive_assignment, competitive_distance = self.compute_competitive_allocation(
            query_tokens,
            bary_tokens_tensor,
        )
        competition_ms = (time.perf_counter() - competition_start) * 1000.0

        ot_start = time.perf_counter()
        distances = []
        plans = []
        costs = []
        masses = []
        competitive_query_weight_items = []
        for class_idx in range(way_num):
            class_query_weights = base_query_weights * competitive_assignment[..., class_idx]
            distance, plan, cost, transported_mass = self._compute_unbalanced_transport_details(
                query_tokens,
                class_query_weights,
                bary_tokens_tensor[class_idx],
                bary_weights_tensor[class_idx],
            )
            distances.append(distance)
            plans.append(plan)
            costs.append(cost)
            masses.append(transported_mass)
            competitive_query_weight_items.append(class_query_weights)
        ot_ms = (time.perf_counter() - ot_start) * 1000.0

        distance_tensor = torch.stack(distances, dim=-1)
        transport_plan = torch.stack(plans, dim=1)
        cost_matrix = torch.stack(costs, dim=1)
        transported_mass = torch.stack(masses, dim=-1)
        competitive_query_weights = torch.stack(competitive_query_weight_items, dim=-1)

        score_start = time.perf_counter()
        robust_distance = (distance_tensor - epsilon_tensor.unsqueeze(0)).clamp_min(0.0)
        logits = -robust_distance
        aux_loss = logits.new_zeros(())
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
                    "support_token_weights": support_weight_tensor,
                    "transport_plan": transport_plan,
                    "cost_matrix": cost_matrix,
                    "competitive_assignment": competitive_assignment,
                    "competitive_distance": competitive_distance,
                    "competitive_query_weights": competitive_query_weights,
                    "query_token_weights": base_query_weights,
                }
            )
        if self.profile or return_aux:
            outputs.update(
                {
                    "profile_barycenter_ms": logits.new_tensor(bary_ms),
                    "profile_competition_ms": logits.new_tensor(competition_ms),
                    "profile_ot_ms": logits.new_tensor(ot_ms),
                    "profile_scoring_ms": logits.new_tensor(score_ms),
                    "profile_total_ms": logits.new_tensor(bary_ms + competition_ms + ot_ms + score_ms),
                }
            )
        return outputs

    def forward_episode_features(
        self,
        supports: torch.Tensor,
        query: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        outputs = self._forward_feature_episode(
            supports,
            query,
            return_aux=return_aux,
        )
        return outputs if return_aux else outputs["logits"]

    @staticmethod
    def _stack_outputs(batch_outputs: list[dict[str, torch.Tensor]]) -> CBCRFSLResult:
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
        optional_cat_keys = (
            "transport_plan",
            "cost_matrix",
            "competitive_assignment",
            "competitive_distance",
            "competitive_query_weights",
            "query_token_weights",
        )
        for key in optional_cat_keys:
            if key in batch_outputs[0]:
                stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
        optional_stack_keys = ("barycenter_tokens", "barycenter_weights", "support_token_weights")
        for key in optional_stack_keys:
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs], dim=0)
        for key in (
            "profile_barycenter_ms",
            "profile_competition_ms",
            "profile_ot_ms",
            "profile_scoring_ms",
            "profile_total_ms",
        ):
            if key in batch_outputs[0]:
                stacked[key] = torch.stack([item[key] for item in batch_outputs]).mean()
        return CBCRFSLResult(stacked)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del query_targets
        del support_targets
        batch_size, num_query, _, _, _, _ = self.validate_episode_inputs(query, support)

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

        logits = torch.cat(batch_logits, dim=0)
        if not needs_payload:
            return logits

        stacked = self._stack_outputs(batch_outputs)
        stacked["logits"] = logits
        if return_aux:
            return stacked
        return CBCRFSLResult({"logits": logits, "aux_loss": stacked["aux_loss"]})
