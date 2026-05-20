"""PARE-FSL: Partial Adaptive Relational Evidence for few-shot scalograms.

Unlike multi-mass-bank SPOT variants, PARE uses:
  - Episode-gated shrinkage marginals with sum(a)=sum(b)=1 (no fixed rho budget).
  - One entropic partial OT solve per (query, class, shot).
  - A learned transported-mass fraction alpha(q,c,k) in [alpha_min, alpha_max].
  - Partial discrepancy score -scale * C / M with shot pooling by match quality.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel, feature_map_to_tokens
from net.modules.egsm_marginal import EpisodeGatedShrinkageMarginal
from net.modules.partial_ot import (
    compute_partial_transport_cost,
    compute_partial_transported_mass,
    solve_partial_transport,
)


class PAREFSLResult(dict):
    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


class PairPartialMassHead(nn.Module):
    """Predict partial transport fraction alpha per (query, class, shot) pair."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        alpha_min: float = 0.30,
        alpha_max: float = 0.70,
        alpha_prior: float = 0.50,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 < float(alpha_min) < float(alpha_max) <= 1.0:
            raise ValueError("require 0 < alpha_min < alpha_max <= 1")
        if not float(alpha_min) <= float(alpha_prior) <= float(alpha_max):
            raise ValueError("alpha_prior must lie in [alpha_min, alpha_max]")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_prior = float(alpha_prior)
        self.eps = float(eps)
        self.mlp = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight, gain=0.1)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        cost: torch.Tensor,
        *,
        egsm_kappa: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha, pair_features) with shapes (NumQuery, Way, Shot)."""
        if cost.dim() != 5:
            raise ValueError(f"cost must be (Q, W, K, Lq, Ls), got {tuple(cost.shape)}")
        num_query, way_num, shot_num, _, _ = cost.shape
        min_cost = cost.amin(dim=(-1, -2))
        mean_cost = cost.mean(dim=(-1, -2))
        std_cost = cost.std(dim=(-1, -2), unbiased=False).clamp_min(self.eps)

        class_best = cost.amin(dim=2).amin(dim=(-1, -2))
        shot_gap = (min_cost - class_best.unsqueeze(2)).clamp_min(0.0)

        features = torch.stack(
            [
                torch.log1p(min_cost.clamp_min(0.0)),
                torch.log1p(mean_cost.clamp_min(0.0)),
                torch.log1p(std_cost),
                torch.log1p(shot_gap.clamp_min(0.0)),
            ],
            dim=-1,
        )
        if egsm_kappa is not None:
            if tuple(egsm_kappa.shape) != (num_query,):
                raise ValueError(f"egsm_kappa must be (NumQuery,), got {tuple(egsm_kappa.shape)}")
            kappa_feat = egsm_kappa.view(num_query, 1, 1).expand(num_query, way_num, shot_num)
            features = torch.cat([features, kappa_feat.unsqueeze(-1)], dim=-1)
        else:
            features = torch.cat(
                [features, torch.zeros_like(min_cost).unsqueeze(-1)],
                dim=-1,
            )

        if egsm_kappa is not None:
            query_margin = cost.new_zeros(num_query, way_num, shot_num)
            if way_num >= 2:
                best_per_class = cost.amin(dim=2).amin(dim=-1)
                top2, _ = best_per_class.topk(2, dim=1, largest=False)
                margin = (top2[:, 1, :] - top2[:, 0, :]).clamp_min(0.0)
                query_margin = margin.mean(dim=-1, keepdim=True).unsqueeze(-1).expand(
                    num_query,
                    way_num,
                    shot_num,
                )
            features = torch.cat(
                [features, torch.log1p(query_margin.clamp_min(0.0)).unsqueeze(-1)],
                dim=-1,
            )
        else:
            features = torch.cat(
                [features, torch.zeros_like(min_cost).unsqueeze(-1)],
                dim=-1,
            )

        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        dtype = self.mlp[0].weight.dtype
        logit = self.mlp(features.to(dtype=dtype)).squeeze(-1).to(dtype=cost.dtype, device=cost.device)
        span = self.alpha_max - self.alpha_min
        alpha = self.alpha_min + span * torch.sigmoid(logit)
        return alpha, features


class PAREFSL(BaseConv64FewShotModel):
    """Shot-wise partial OT with EGSM marginals and learned transport fraction."""

    VALID_MARGINAL_MODES = frozenset({"egsm", "uniform"})
    VALID_SHOT_POOLING = frozenset({"mean", "logsumexp", "discrepancy"})

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 640,
        token_dim: int = 128,
        backbone_name: str = "resnet12",
        image_size: int = 84,
        resnet12_drop_rate: float = 0.0,
        resnet12_dropblock_size: int = 5,
        sinkhorn_epsilon: float = 0.04,
        sinkhorn_iterations: int = 80,
        sinkhorn_tolerance: float = 1e-6,
        score_scale: float = 16.0,
        marginal_mode: str = "egsm",
        egsm_hidden_dim: int = 32,
        egsm_kappa_min: float = 0.05,
        egsm_kappa_max: float = 0.35,
        egsm_candidate_tau_q: float = 1.0,
        egsm_candidate_tau_b: float = 1.0,
        enable_learned_alpha: bool = True,
        alpha_min: float = 0.30,
        alpha_max: float = 0.70,
        alpha_prior: float = 0.50,
        fixed_alpha: float | None = None,
        alpha_reg_lambda: float = 0.01,
        alpha_head_hidden_dim: int = 32,
        shot_pooling: str = "discrepancy",
        shot_temperature_init: float = 1.0,
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
        if int(token_dim) <= 0:
            raise ValueError("token_dim must be positive")
        if float(sinkhorn_epsilon) <= 0.0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if int(sinkhorn_iterations) <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if float(sinkhorn_tolerance) < 0.0:
            raise ValueError("sinkhorn_tolerance must be non-negative")
        if float(score_scale) <= 0.0:
            raise ValueError("score_scale must be positive")
        if float(shot_temperature_init) <= 0.0:
            raise ValueError("shot_temperature_init must be positive")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        if float(alpha_reg_lambda) < 0.0:
            raise ValueError("alpha_reg_lambda must be non-negative")
        if fixed_alpha is not None and not 0.0 < float(fixed_alpha) <= 1.0:
            raise ValueError("fixed_alpha must be in (0, 1] when set")

        marginal_mode = str(marginal_mode).lower()
        if marginal_mode not in self.VALID_MARGINAL_MODES:
            raise ValueError(f"marginal_mode must be one of {sorted(self.VALID_MARGINAL_MODES)}")
        shot_pooling = str(shot_pooling).lower()
        if shot_pooling not in self.VALID_SHOT_POOLING:
            raise ValueError(f"shot_pooling must be one of {sorted(self.VALID_SHOT_POOLING)}")

        self.token_dim = int(token_dim)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.score_scale = float(score_scale)
        self.marginal_mode = marginal_mode
        self.enable_learned_alpha = bool(enable_learned_alpha) and fixed_alpha is None
        self.fixed_alpha = None if fixed_alpha is None else float(fixed_alpha)
        self.alpha_prior = float(alpha_prior)
        self.alpha_reg_lambda = float(alpha_reg_lambda)
        self.shot_pooling = shot_pooling
        self.eps = float(eps)

        self.token_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.token_dim, bias=False),
        )
        self.egsm = (
            EpisodeGatedShrinkageMarginal(
                hidden_dim=int(egsm_hidden_dim),
                candidate_tau_q=float(egsm_candidate_tau_q),
                candidate_tau_b=float(egsm_candidate_tau_b),
                kappa_min=float(egsm_kappa_min),
                kappa_max=float(egsm_kappa_max),
                enable_adaptive_rho=False,
                eps=self.eps,
            )
            if marginal_mode == "egsm"
            else None
        )
        alpha_feature_dim = 6
        self.alpha_head = (
            PairPartialMassHead(
                input_dim=alpha_feature_dim,
                hidden_dim=int(alpha_head_hidden_dim),
                alpha_min=float(alpha_min),
                alpha_max=float(alpha_max),
                alpha_prior=float(alpha_prior),
                eps=self.eps,
            )
            if self.enable_learned_alpha
            else None
        )
        self.raw_shot_temperature = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(shot_temperature_init))), dtype=torch.float32)
        )

    @property
    def shot_temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_shot_temperature).clamp_min(self.eps)

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        tokens = feature_map_to_tokens(self.encode(images))
        tokens = self.token_projector(tokens)
        return F.normalize(tokens, p=2, dim=-1, eps=self.eps)

    def _encode_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.dim() != 4:
            raise ValueError(f"query must be (NumQuery, C, H, W), got {tuple(query.shape)}")
        if support.dim() != 5:
            raise ValueError(f"support must be (Way, Shot, C, H, W), got {tuple(support.shape)}")
        way_num, shot_num = support.shape[:2]
        images = torch.cat([query, support.reshape(way_num * shot_num, *support.shape[2:])], dim=0)
        tokens = self._encode_images(images)
        query_tokens = tokens[: query.shape[0]]
        support_tokens = tokens[query.shape[0] :].reshape(way_num, shot_num, tokens.shape[-2], tokens.shape[-1])
        return query_tokens, support_tokens

    @staticmethod
    def _pairwise_squared_distance(
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> torch.Tensor:
        diff = query_tokens[:, None, None, :, None, :] - support_tokens[None, :, :, None, :, :]
        return diff.square().sum(dim=-1).clamp_min(0.0)

    def _uniform_marginals(self, cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_query, way_num, shot_num, query_len, support_len = cost.shape
        num_pairs = way_num * shot_num
        flat_cost = cost.reshape(num_query, num_pairs, query_len, support_len)
        a = flat_cost.new_full((num_query, num_pairs, query_len), 1.0 / float(query_len))
        b = flat_cost.new_full((num_query, num_pairs, support_len), 1.0 / float(support_len))
        return a, b

    def _build_marginals(
        self,
        cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        num_query, way_num, shot_num, query_len, support_len = cost.shape
        num_pairs = way_num * shot_num
        flat_cost = cost.reshape(num_query, num_pairs, query_len, support_len)
        if self.marginal_mode == "uniform":
            return *self._uniform_marginals(cost), {}

        if self.egsm is None:
            raise RuntimeError("EGSM marginal module is not initialized")
        query_mass, support_mass, aux = self.egsm(
            flat_cost,
            way_num=way_num,
            shot_num=shot_num,
            rho=1.0,
        )
        query_mass = query_mass / query_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        support_mass = support_mass / support_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return query_mass, support_mass, aux

    def _resolve_alpha(
        self,
        cost: torch.Tensor,
        egsm_aux: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_query, way_num, shot_num, _, _ = cost.shape
        if self.fixed_alpha is not None:
            alpha = cost.new_full((num_query, way_num, shot_num), self.fixed_alpha)
            return alpha, None
        if not self.enable_learned_alpha or self.alpha_head is None:
            alpha = cost.new_full((num_query, way_num, shot_num), self.alpha_prior)
            return alpha, None
        kappa = egsm_aux.get("egsm_kappa") if egsm_aux else None
        return self.alpha_head(cost, egsm_kappa=kappa)

    def _pool_shot_logits(self, shot_logits: torch.Tensor, shot_discrepancy: torch.Tensor) -> torch.Tensor:
        if self.shot_pooling == "mean":
            return shot_logits.mean(dim=-1)
        if self.shot_pooling == "logsumexp":
            tau = self.shot_temperature.to(device=shot_logits.device, dtype=shot_logits.dtype)
            scaled = shot_logits / tau.clamp_min(self.eps)
            return tau * (torch.logsumexp(scaled, dim=-1) - math.log(float(shot_logits.shape[-1])))
        tau = self.shot_temperature.to(device=shot_discrepancy.device, dtype=shot_discrepancy.dtype)
        weights = torch.softmax(-shot_discrepancy / tau.clamp_min(self.eps), dim=-1)
        return (weights * shot_logits).sum(dim=-1)

    def compute_partial_match(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> PAREFSLResult:
        cost = self._pairwise_squared_distance(query_tokens, support_tokens)
        num_query, way_num, shot_num, query_len, support_len = cost.shape
        num_pairs = num_query * way_num * shot_num

        query_mass, support_mass, egsm_aux = self._build_marginals(cost)
        alpha, alpha_features = self._resolve_alpha(cost, egsm_aux)

        flat_cost = cost.reshape(num_pairs, query_len, support_len)
        flat_a = query_mass.reshape(num_pairs, query_len)
        flat_b = support_mass.reshape(num_pairs, support_len)
        flat_alpha = alpha.reshape(num_pairs)

        flat_plan = solve_partial_transport(
            flat_cost,
            flat_a,
            flat_b,
            transport_mass_ratio=flat_alpha,
            backend="native",
            reg=self.sinkhorn_epsilon,
            max_iter=self.sinkhorn_iterations,
            tol=self.sinkhorn_tolerance,
            eps=self.eps,
        )
        flat_transport_cost = compute_partial_transport_cost(flat_plan, flat_cost)
        flat_transport_mass = compute_partial_transported_mass(flat_plan)

        transport_cost = flat_transport_cost.reshape(num_query, way_num, shot_num)
        transport_mass = flat_transport_mass.reshape(num_query, way_num, shot_num)
        partial_discrepancy = transport_cost / transport_mass.clamp_min(self.eps)
        shot_logits = -self.score_scale * partial_discrepancy
        logits = self._pool_shot_logits(shot_logits, partial_discrepancy)

        aux_loss = cost.new_zeros(())
        if self.alpha_reg_lambda > 0.0 and self.enable_learned_alpha:
            aux_loss = aux_loss + self.alpha_reg_lambda * (alpha - self.alpha_prior).pow(2).mean()
        if "egsm_aux_loss" in egsm_aux:
            aux_loss = aux_loss + egsm_aux["egsm_aux_loss"]

        return PAREFSLResult(
            {
                "logits": logits,
                "class_scores": logits,
                "aux_loss": aux_loss,
                "transport_cost": transport_cost,
                "transported_mass": transport_mass,
                "partial_discrepancy": partial_discrepancy,
                "partial_alpha": alpha,
                "shot_logits": shot_logits,
                "total_distance": self._pool_shot_logits(partial_discrepancy, partial_discrepancy),
                "shot_temperature": self.shot_temperature.detach().to(device=logits.device, dtype=logits.dtype),
                "transport_plan": flat_plan.reshape(num_query, way_num, shot_num, query_len, support_len).detach(),
                "cost_matrix": cost.detach(),
                **{key: value.detach() if torch.is_tensor(value) else value for key, value in egsm_aux.items()},
            }
        )

    def _forward_episode(self, query: torch.Tensor, support: torch.Tensor) -> PAREFSLResult:
        query_tokens, support_tokens = self._encode_episode(query, support)
        outputs = self.compute_partial_match(query_tokens, support_tokens)
        outputs["query_tokens"] = query_tokens.detach()
        outputs["support_tokens"] = support_tokens.detach()
        return outputs

    @staticmethod
    def _stack_outputs(batch_outputs: list[PAREFSLResult]) -> PAREFSLResult:
        stacked: dict[str, Any] = {
            "logits": torch.cat([item["logits"] for item in batch_outputs], dim=0),
            "aux_loss": torch.stack([item["aux_loss"] for item in batch_outputs]).mean(),
            "class_scores": torch.cat([item["class_scores"] for item in batch_outputs], dim=0),
            "transport_cost": torch.cat([item["transport_cost"] for item in batch_outputs], dim=0),
            "transported_mass": torch.cat([item["transported_mass"] for item in batch_outputs], dim=0),
            "partial_discrepancy": torch.cat([item["partial_discrepancy"] for item in batch_outputs], dim=0),
            "partial_alpha": torch.cat([item["partial_alpha"] for item in batch_outputs], dim=0),
            "shot_logits": torch.cat([item["shot_logits"] for item in batch_outputs], dim=0),
            "total_distance": torch.cat([item["total_distance"] for item in batch_outputs], dim=0),
            "shot_temperature": batch_outputs[0]["shot_temperature"],
        }
        for key in ("transport_plan", "cost_matrix", "query_tokens", "support_tokens", "egsm_kappa"):
            if key in batch_outputs[0]:
                value = batch_outputs[0][key]
                if torch.is_tensor(value):
                    stacked[key] = torch.cat([item[key] for item in batch_outputs], dim=0)
                else:
                    stacked[key] = value
        return PAREFSLResult(stacked)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
    ) -> PAREFSLResult:
        del return_aux
        bsz, _, _, _, _, _ = self.validate_episode_inputs(query, support)
        batch_outputs = [self._forward_episode(query[idx], support[idx]) for idx in range(bsz)]
        return self._stack_outputs(batch_outputs)
