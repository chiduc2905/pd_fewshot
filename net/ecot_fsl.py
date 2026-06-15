"""Episode-competitive semi-relaxed optimal transport for few-shot learning.

The transport solver is a batched PyTorch implementation of generalized
Sinkhorn scaling with an exact source marginal and a KL-relaxed target
marginal. The update follows the semi-relaxed case documented by PythonOT:
``reg_m=(inf, rho)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoders.smnet_conv64f_encoder import build_resnet12_family_encoder


@dataclass(frozen=True)
class SemiRelaxedSinkhornStats:
    source_marginal_l1: torch.Tensor
    target_kl: torch.Tensor
    fixed_point_residual: torch.Tensor
    linear_cost: torch.Tensor
    plan_kl: torch.Tensor
    objective: torch.Tensor


def _normalize_histogram(histogram: torch.Tensor, eps: float) -> torch.Tensor:
    if histogram.dim() != 2:
        raise ValueError("histogram must have shape [batch, num_bins]")
    if torch.any(histogram < 0):
        raise ValueError("histogram entries must be non-negative")
    return histogram.clamp_min(eps) / histogram.sum(dim=-1, keepdim=True).clamp_min(eps)


def _generalized_kl(value: torch.Tensor, reference: torch.Tensor, eps: float) -> torch.Tensor:
    value_safe = value.clamp_min(eps)
    reference_safe = reference.clamp_min(eps)
    return (
        value_safe * (value_safe.log() - reference_safe.log())
        - value_safe
        + reference_safe
    ).sum(dim=-1)


def semi_relaxed_sinkhorn_log(
    cost: torch.Tensor,
    source_mass: torch.Tensor,
    target_prior: torch.Tensor,
    *,
    epsilon: float,
    target_relaxation: float,
    max_iterations: int,
    tolerance: float,
    numerical_eps: float = 1e-8,
) -> tuple[torch.Tensor, SemiRelaxedSinkhornStats]:
    r"""Solve KL-regularized semi-relaxed OT in the log domain.

    The solved objective is

    .. math::
        \min_{P \ge 0,\;P\mathbf 1=a}
        \langle P,C\rangle
        + \varepsilon\,\mathrm{KL}(P\Vert a b^\top)
        + \rho\,\mathrm{KL}(P^\top\mathbf 1\Vert b).

    ``source_mass`` is therefore exact, while ``target_prior`` is a soft
    capacity prior. This is the ``reg_m=(inf, rho)`` generalized Sinkhorn
    special case.
    """

    if cost.dim() != 3:
        raise ValueError("cost must have shape [batch, num_source, num_target]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if target_relaxation <= 0:
        raise ValueError("target_relaxation must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if not torch.isfinite(cost).all():
        raise FloatingPointError("cost contains non-finite values")

    batch_size, num_source, num_target = cost.shape
    if source_mass.shape != (batch_size, num_source):
        raise ValueError("source_mass shape does not match cost")
    if target_prior.shape != (batch_size, num_target):
        raise ValueError("target_prior shape does not match cost")

    source_mass = _normalize_histogram(source_mass, numerical_eps)
    target_prior = _normalize_histogram(target_prior, numerical_eps)
    log_a = source_mass.log()
    log_b = target_prior.log()

    # PythonOT's KL-reference kernel is exp(-C / epsilon) * (a outer b).
    log_reference = log_a.unsqueeze(-1) + log_b.unsqueeze(-2)
    log_kernel = log_reference - cost / float(epsilon)
    target_exponent = float(target_relaxation) / float(target_relaxation + epsilon)

    log_v = torch.zeros_like(log_b)
    fixed_point_residual = cost.new_full((batch_size,), float("inf"))
    for _ in range(max_iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        log_ktu = torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        next_log_v = target_exponent * (log_b - log_ktu)
        fixed_point_residual = (next_log_v - log_v).abs().amax(dim=-1)
        log_v = next_log_v
        if tolerance > 0 and float(fixed_point_residual.detach().amax().item()) <= tolerance:
            break

    # Re-project the exact source marginal after the final relaxed-target step.
    log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    plan = log_plan.exp()

    source_observed = plan.sum(dim=-1)
    target_observed = plan.sum(dim=-2)
    source_marginal_l1 = (source_observed - source_mass).abs().sum(dim=-1)
    target_kl = _generalized_kl(target_observed, target_prior, numerical_eps)

    log_ktu = torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
    fixed_log_v = target_exponent * (log_b - log_ktu)
    fixed_point_residual = (fixed_log_v - log_v).abs().amax(dim=-1)

    linear_cost = (plan * cost).sum(dim=(-1, -2))
    plan_kl = (
        plan.clamp_min(numerical_eps)
        * (log_plan - log_reference)
        - plan
        + log_reference.exp()
    ).sum(dim=(-1, -2))
    objective = linear_cost + float(epsilon) * plan_kl + float(target_relaxation) * target_kl

    if not torch.isfinite(plan).all():
        raise FloatingPointError("semi-relaxed Sinkhorn produced a non-finite plan")

    stats = SemiRelaxedSinkhornStats(
        source_marginal_l1=source_marginal_l1,
        target_kl=target_kl,
        fixed_point_residual=fixed_point_residual,
        linear_cost=linear_cost,
        plan_kl=plan_kl,
        objective=objective,
    )
    return plan, stats


class EpisodeCompetitiveOT(nn.Module):
    """Joint N-way local-descriptor transport classifier."""

    def __init__(
        self,
        *,
        image_size: int = 84,
        fewshot_backbone: str = "resnet12",
        epsilon: float = 0.05,
        target_relaxation: float = 0.10,
        sinkhorn_iterations: int = 60,
        sinkhorn_tolerance: float = 1e-6,
        logit_scale: float = 1.0,
        claim_margin: float = 0.05,
        numerical_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if str(fewshot_backbone).lower() != "resnet12":
            raise ValueError("ECOT-FSL currently requires fewshot_backbone='resnet12'")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if target_relaxation <= 0:
            raise ValueError("target_relaxation must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if sinkhorn_tolerance < 0:
            raise ValueError("sinkhorn_tolerance must be non-negative")
        if logit_scale <= 0:
            raise ValueError("logit_scale must be positive")
        if claim_margin < 0:
            raise ValueError("claim_margin must be non-negative")

        self.encoder = build_resnet12_family_encoder(
            image_size=image_size,
            backbone_name="resnet12",
            pool_output=False,
            variant="fewshot",
            drop_rate=0.0,
        )
        self.feat_dim = int(self.encoder.out_channels)
        self.epsilon = float(epsilon)
        self.target_relaxation = float(target_relaxation)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.logit_scale = float(logit_scale)
        self.claim_margin = float(claim_margin)
        self.numerical_eps = float(numerical_eps)

    @staticmethod
    def _flatten_normalized_tokens(features: torch.Tensor) -> torch.Tensor:
        tokens = features.flatten(start_dim=-2).transpose(-1, -2)
        return F.normalize(tokens, p=2, dim=-1)

    def _build_joint_cost(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine cost and similarity for all episode classes jointly."""

        batch_queries, num_query_tokens, feature_dim = query_tokens.shape
        if support_tokens.dim() != 5:
            raise ValueError("support_tokens must have shape [batch, way, shot, token, dim]")
        batch_size, way_num, shot_num, num_support_tokens, support_dim = support_tokens.shape
        if support_dim != feature_dim:
            raise ValueError("query and support feature dimensions differ")
        if batch_queries % batch_size != 0:
            raise ValueError("query batch cannot be aligned with support episodes")

        queries_per_episode = batch_queries // batch_size
        support_flat = support_tokens.reshape(
            batch_size,
            way_num * shot_num * num_support_tokens,
            feature_dim,
        )
        support_flat = (
            support_flat.unsqueeze(1)
            .expand(batch_size, queries_per_episode, -1, -1)
            .reshape(batch_queries, way_num * shot_num * num_support_tokens, feature_dim)
        )
        similarity = torch.bmm(query_tokens, support_flat.transpose(1, 2)).clamp(-1.0, 1.0)
        return 1.0 - similarity, similarity

    def _compute_diagnostics(
        self,
        *,
        plan: torch.Tensor,
        similarity: torch.Tensor,
        class_mass: torch.Tensor,
        source_mass: torch.Tensor,
        stats: SemiRelaxedSinkhornStats,
        way_num: int,
        shot_num: int,
        support_token_num: int,
        query_targets: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch_queries, query_token_num, _ = plan.shape
        structured_plan = plan.reshape(
            batch_queries,
            query_token_num,
            way_num,
            shot_num,
            support_token_num,
        )
        token_class_mass = structured_plan.sum(dim=(-1, -2))
        token_class_prob = token_class_mass / source_mass.unsqueeze(-1).clamp_min(self.numerical_eps)

        class_entropy = -(
            class_mass.clamp_min(self.numerical_eps)
            * class_mass.clamp_min(self.numerical_eps).log()
        ).sum(dim=-1)
        token_entropy = -(
            token_class_prob.clamp_min(self.numerical_eps)
            * token_class_prob.clamp_min(self.numerical_eps).log()
        ).sum(dim=-1)
        entropy_normalizer = max(log(float(way_num)), self.numerical_eps)

        similarity_by_class = similarity.reshape(
            batch_queries,
            query_token_num,
            way_num,
            shot_num,
            support_token_num,
        )
        independent_claim = similarity_by_class.amax(dim=(-1, -2))
        best_claim = independent_claim.amax(dim=-1, keepdim=True)
        claim_count = (independent_claim >= best_claim - self.claim_margin).sum(dim=-1)

        diagnostics = {
            "ecot/source_marginal_l1": stats.source_marginal_l1.detach().mean(),
            "ecot/target_kl": stats.target_kl.detach().mean(),
            "ecot/fixed_point_residual": stats.fixed_point_residual.detach().mean(),
            "ecot/linear_cost": stats.linear_cost.detach().mean(),
            "ecot/plan_kl": stats.plan_kl.detach().mean(),
            "ecot/objective": stats.objective.detach().mean(),
            "ecot/class_mass_entropy": (class_entropy / entropy_normalizer).detach().mean(),
            "ecot/class_mass_peak": class_mass.detach().amax(dim=-1).mean(),
            "ecot/effective_class_count": class_entropy.detach().exp().mean(),
            "ecot/token_class_entropy": (token_entropy / entropy_normalizer).detach().mean(),
            "ecot/token_claim_collision": (claim_count > 1).float().detach().mean(),
            "ecot/token_claim_count": claim_count.float().detach().mean(),
        }

        if query_targets is not None:
            targets = query_targets.reshape(-1).to(device=class_mass.device, dtype=torch.long)
            if targets.numel() != batch_queries:
                raise ValueError("query_targets size does not match the number of queries")
            true_mass = class_mass.gather(1, targets.unsqueeze(1)).squeeze(1)
            if way_num > 1:
                target_mask = F.one_hot(targets, num_classes=way_num).bool()
                rival_mass = class_mass.masked_fill(target_mask, -1.0).amax(dim=-1)
            else:
                rival_mass = torch.zeros_like(true_mass)
            diagnostics.update(
                {
                    "ecot/true_mass": true_mass.detach().mean(),
                    "ecot/best_negative_mass": rival_mass.detach().mean(),
                    "ecot/true_rival_mass_gap": (true_mass - rival_mass).detach().mean(),
                    "ecot/true_rival_log_mass_margin": (
                        true_mass.clamp_min(self.numerical_eps).log()
                        - rival_mass.clamp_min(self.numerical_eps).log()
                    ).detach().mean(),
                    "ecot/mass_prediction_accuracy": (
                        class_mass.argmax(dim=-1).eq(targets).float().detach().mean()
                    ),
                }
            )
        return diagnostics

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        *,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del support_targets  # Class order is encoded by the support tensor axis.

        if query.dim() != 5:
            raise ValueError("query must have shape [batch, num_query, channels, height, width]")
        if support.dim() != 6:
            raise ValueError("support must have shape [batch, way, shot, channels, height, width]")

        batch_size, num_queries, channels, height, width = query.shape
        support_batch, way_num, shot_num, support_channels, support_height, support_width = support.shape
        if (support_batch, support_channels, support_height, support_width) != (
            batch_size,
            channels,
            height,
            width,
        ):
            raise ValueError("query and support episode shapes are incompatible")

        query_features = self.encoder.forward_features(
            query.reshape(batch_size * num_queries, channels, height, width)
        )
        support_features = self.encoder.forward_features(
            support.reshape(batch_size * way_num * shot_num, channels, height, width)
        )
        feature_height, feature_width = query_features.shape[-2:]
        if support_features.shape[-2:] != (feature_height, feature_width):
            raise ValueError("query and support feature maps have different spatial sizes")

        query_tokens = self._flatten_normalized_tokens(query_features)
        support_tokens = self._flatten_normalized_tokens(support_features).reshape(
            batch_size,
            way_num,
            shot_num,
            feature_height * feature_width,
            self.feat_dim,
        )
        cost, similarity = self._build_joint_cost(query_tokens, support_tokens)

        batch_queries, query_token_num, target_token_num = cost.shape
        source_mass = cost.new_full(
            (batch_queries, query_token_num),
            1.0 / float(query_token_num),
        )
        target_prior = cost.new_full(
            (batch_queries, target_token_num),
            1.0 / float(target_token_num),
        )
        plan, stats = semi_relaxed_sinkhorn_log(
            cost,
            source_mass,
            target_prior,
            epsilon=self.epsilon,
            target_relaxation=self.target_relaxation,
            max_iterations=self.sinkhorn_iterations,
            tolerance=self.sinkhorn_tolerance,
            numerical_eps=self.numerical_eps,
        )

        class_mass = plan.reshape(
            batch_queries,
            query_token_num,
            way_num,
            shot_num,
            feature_height * feature_width,
        ).sum(dim=(1, 3, 4))
        class_mass = class_mass / class_mass.sum(dim=-1, keepdim=True).clamp_min(self.numerical_eps)
        logits = self.logit_scale * class_mass.clamp_min(self.numerical_eps).log()

        if not return_aux:
            return logits
        diagnostics = self._compute_diagnostics(
            plan=plan,
            similarity=similarity,
            class_mass=class_mass,
            source_mass=source_mass,
            stats=stats,
            way_num=way_num,
            shot_num=shot_num,
            support_token_num=feature_height * feature_width,
            query_targets=query_targets,
        )
        return {
            "logits": logits,
            "ecot_class_mass": class_mass.detach(),
            "ecot_token_class_assignment": (
                plan.reshape(
                    batch_queries,
                    query_token_num,
                    way_num,
                    shot_num,
                    feature_height * feature_width,
                )
                .sum(dim=(-1, -2))
                .div(source_mass.unsqueeze(-1).clamp_min(self.numerical_eps))
                .detach()
            ),
            **diagnostics,
        }


ECOTFSL = EpisodeCompetitiveOT
