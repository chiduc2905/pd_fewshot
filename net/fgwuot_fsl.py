"""Fused Gromov-Wasserstein Unbalanced OT few-shot classifier."""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fewshot_common import BaseConv64FewShotModel
from net.modules.fgw_uot_solver import (
    fgw_uot_solve,
    normalize_intra_dist,
    pairwise_sq_l2,
    sinkhorn_uot_log,
)


MASS_MODES = {"uniform", "reliability"}
SUPPORT_MODES = {"concat", "shotwise"}
SHOT_AGGREGATIONS = {"mean", "softmin"}
SCORE_MODES = {"negative_distance", "radius_margin", "robust_radius"}


class FGWUOTFSLResult(dict):
    """Dict-like output that exposes `.shape` through logits."""

    @property
    def shape(self):
        logits = self.get("logits")
        return None if logits is None else logits.shape


class RhoHead(nn.Module):
    """Predict per-(query, class) transport budget rho in (0, 1)."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        C_feat: torch.Tensor,
        D_q: torch.Tensor,
        D_s: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        stats = torch.stack(
            [
                C_feat.mean(dim=(-1, -2)),
                C_feat.std(dim=(-1, -2)).clamp_min(eps),
                D_q.mean(dim=(-1, -2)),
                D_s.mean(dim=(-1, -2)),
            ],
            dim=-1,
        )
        return self.net(stats).squeeze(-1)


class FGWUOTFewShot(BaseConv64FewShotModel):
    """Few-shot classifier using Fused Gromov-Wasserstein Unbalanced OT."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        token_dim: int = 128,
        backbone_name: str = "conv64f",
        image_size: int = 84,
        tau: float = 0.5,
        eps_sinkhorn: float = 0.1,
        fgw_iters: int = 8,
        sinkhorn_iters: int = 60,
        sinkhorn_tol: float = 1e-5,
        alpha_init: float = 0.5,
        score_scale_init: float = 16.0,
        rho_head_hidden: int = 32,
        lambda_rho: float = 0.01,
        rho_target: float = 0.8,
        normalize_tokens: bool = True,
        mass_mode: str = "reliability",
        reliability_mix: float = 0.65,
        reliability_temperature: float = 0.25,
        support_mode: str = "shotwise",
        shot_aggregation: str = "softmin",
        shot_softmin_beta: float = 8.0,
        structure_prior_weight: float = 0.12,
        score_mode: str = "radius_margin",
        radius_alpha: float = 0.5,
        radius_floor: float = 0.02,
        eps: float = 1e-8,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            image_size=image_size,
        )
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if eps_sinkhorn <= 0.0:
            raise ValueError("eps_sinkhorn must be positive")
        if fgw_iters < 0 or sinkhorn_iters <= 0:
            raise ValueError("fgw_iters must be non-negative and sinkhorn_iters must be positive")
        mass_mode = str(mass_mode).lower()
        support_mode = str(support_mode).lower()
        shot_aggregation = str(shot_aggregation).lower()
        score_mode = str(score_mode).lower()
        if mass_mode not in MASS_MODES:
            raise ValueError(f"mass_mode must be one of {sorted(MASS_MODES)}, got {mass_mode}")
        if support_mode not in SUPPORT_MODES:
            raise ValueError(f"support_mode must be one of {sorted(SUPPORT_MODES)}, got {support_mode}")
        if shot_aggregation not in SHOT_AGGREGATIONS:
            raise ValueError(
                f"shot_aggregation must be one of {sorted(SHOT_AGGREGATIONS)}, got {shot_aggregation}"
            )
        if score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {sorted(SCORE_MODES)}, got {score_mode}")
        if not 0.0 <= reliability_mix <= 1.0:
            raise ValueError("reliability_mix must be in [0, 1]")
        if reliability_temperature <= 0.0:
            raise ValueError("reliability_temperature must be positive")
        if shot_softmin_beta <= 0.0:
            raise ValueError("shot_softmin_beta must be positive")
        if not 0.0 <= structure_prior_weight <= 1.0:
            raise ValueError("structure_prior_weight must be in [0, 1]")
        if radius_alpha < 0.0 or radius_floor < 0.0:
            raise ValueError("radius_alpha and radius_floor must be non-negative")

        self.token_dim = int(token_dim)
        self.tau = float(tau)
        self.eps_sinkhorn = float(eps_sinkhorn)
        self.fgw_iters = int(fgw_iters)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.lambda_rho = float(lambda_rho)
        self.rho_target = float(rho_target)
        self.normalize_tokens = bool(normalize_tokens)
        self.mass_mode = mass_mode
        self.reliability_mix = float(reliability_mix)
        self.reliability_temperature = float(reliability_temperature)
        self.support_mode = support_mode
        self.shot_aggregation = shot_aggregation
        self.shot_softmin_beta = float(shot_softmin_beta)
        self.structure_prior_weight = float(structure_prior_weight)
        self.score_mode = score_mode
        self.radius_alpha = float(radius_alpha)
        self.radius_floor = float(radius_floor)
        self.eps = float(eps)

        if self.token_dim != hidden_dim:
            self.token_projector: nn.Module = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.token_dim, bias=False),
            )
        else:
            self.token_projector = nn.Identity()

        alpha_init = min(max(float(alpha_init), self.eps), 1.0 - self.eps)
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init))))
        self.raw_score_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(float(score_scale_init), self.eps))))
        )
        self.rho_head = RhoHead(hidden_dim=int(rho_head_hidden))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    @property
    def score_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_score_scale)

    def _encode_token_map(self, images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        feat = self.encode(images)
        if feat.dim() != 4:
            raise ValueError(f"encoded feature map must be 4D, got {tuple(feat.shape)}")
        height, width = int(feat.shape[-2]), int(feat.shape[-1])
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_projector(tokens)
        if self.normalize_tokens:
            tokens = F.normalize(tokens, dim=-1)
        return tokens, height, width

    def _encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        tokens, _, _ = self._encode_token_map(images)
        return tokens

    @staticmethod
    def _position_tokens(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([yy, xx], dim=-1).reshape(1, height * width, 2)

    def _grid_distance(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        repeat: int = 1,
    ) -> torch.Tensor:
        grid = self._position_tokens(height, width, device, dtype)
        if repeat > 1:
            grid = grid.repeat(1, int(repeat), 1)
        return normalize_intra_dist(pairwise_sq_l2(grid, grid), self.eps)

    def _structure_distance(
        self,
        tokens: torch.Tensor,
        grid_distance: torch.Tensor,
    ) -> torch.Tensor:
        feature_distance = normalize_intra_dist(pairwise_sq_l2(tokens, tokens), self.eps)
        if self.structure_prior_weight <= 0.0:
            return feature_distance
        grid_distance = grid_distance.to(device=tokens.device, dtype=tokens.dtype)
        while grid_distance.dim() < feature_distance.dim():
            grid_distance = grid_distance.expand(feature_distance.shape[0], -1, -1)
        weight = float(self.structure_prior_weight)
        return normalize_intra_dist((1.0 - weight) * feature_distance + weight * grid_distance, self.eps)

    @staticmethod
    def _normalize_profile(profile: torch.Tensor, eps: float) -> torch.Tensor:
        profile = profile.clamp_min(eps)
        return profile / profile.sum(dim=-1, keepdim=True).clamp_min(eps)

    def _transport_marginal_logs(
        self,
        C_feat: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_q = C_feat.shape[-2]
        token_s = C_feat.shape[-1]
        log_rho = rho.clamp_min(self.eps).log()
        uniform_q = C_feat.new_full((C_feat.shape[0], token_q), 1.0 / float(token_q))
        uniform_s = C_feat.new_full((C_feat.shape[0], token_s), 1.0 / float(token_s))

        if self.mass_mode == "uniform" or self.reliability_mix <= 0.0:
            return (
                log_rho.unsqueeze(-1) + uniform_q.clamp_min(self.eps).log(),
                log_rho.unsqueeze(-1) + uniform_s.clamp_min(self.eps).log(),
                uniform_q,
                uniform_s,
            )

        with torch.no_grad():
            probe = sinkhorn_uot_log(
                C_feat.detach(),
                log_rho.detach().unsqueeze(-1) + uniform_q.clamp_min(self.eps).log(),
                log_rho.detach().unsqueeze(-1) + uniform_s.clamp_min(self.eps).log(),
                tau=self.tau,
                eps=self.eps_sinkhorn,
                max_iter=max(8, min(self.sinkhorn_iters, 30)),
                tol=self.sinkhorn_tol,
            )
            row_mass = probe.sum(dim=-1)
            col_mass = probe.sum(dim=-2)
            row_prob = probe / row_mass.unsqueeze(-1).clamp_min(self.eps)
            col_prob = probe.transpose(-1, -2) / col_mass.unsqueeze(-1).clamp_min(self.eps)
            row_entropy = -(row_prob * row_prob.clamp_min(self.eps).log()).sum(dim=-1)
            col_entropy = -(col_prob * col_prob.clamp_min(self.eps).log()).sum(dim=-1)
            row_conf = 1.0 - row_entropy / math.log(max(token_s, 2))
            col_conf = 1.0 - col_entropy / math.log(max(token_q, 2))
            row_cost = (probe * C_feat.detach()).sum(dim=-1) / row_mass.clamp_min(self.eps)
            col_cost = (probe * C_feat.detach()).sum(dim=-2) / col_mass.clamp_min(self.eps)
            temp = float(self.reliability_temperature)
            query_reliability = row_mass * row_conf.clamp_min(0.0) * torch.exp(-row_cost / temp)
            support_reliability = col_mass * col_conf.clamp_min(0.0) * torch.exp(-col_cost / temp)
            query_reliability = self._normalize_profile(query_reliability, self.eps)
            support_reliability = self._normalize_profile(support_reliability, self.eps)

        mix = float(self.reliability_mix)
        query_profile = self._normalize_profile((1.0 - mix) * uniform_q + mix * query_reliability, self.eps)
        support_profile = self._normalize_profile((1.0 - mix) * uniform_s + mix * support_reliability, self.eps)
        return (
            log_rho.unsqueeze(-1) + query_profile.clamp_min(self.eps).log(),
            log_rho.unsqueeze(-1) + support_profile.clamp_min(self.eps).log(),
            query_profile,
            support_profile,
        )

    def _solve_pair_batch(
        self,
        Q_b: torch.Tensor,
        S_b: torch.Tensor,
        grid_q: torch.Tensor,
        grid_s: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        C_feat = pairwise_sq_l2(Q_b, S_b)
        D_q = self._structure_distance(Q_b, grid_q)
        D_s = self._structure_distance(S_b, grid_s)

        rho = self.rho_head(C_feat, D_q, D_s, self.eps).clamp(self.eps, 1.0 - self.eps)
        log_a, log_b, query_mass, support_mass = self._transport_marginal_logs(C_feat, rho)
        P, C_final = fgw_uot_solve(
            C_feat,
            D_q,
            D_s,
            log_a,
            log_b,
            alpha=self.alpha,
            tau=self.tau,
            eps=self.eps_sinkhorn,
            fgw_iters=self.fgw_iters,
            sinkhorn_iters=self.sinkhorn_iters,
            tol=self.sinkhorn_tol,
        )
        transport_cost = (P * C_final).sum(dim=(-1, -2))
        shot_distance = transport_cost / rho.clamp_min(self.eps)
        return {
            "C_feat": C_feat,
            "D_q": D_q,
            "D_s": D_s,
            "rho": rho,
            "query_mass": query_mass,
            "support_mass": support_mass,
            "transport_plan": P,
            "C_final": C_final,
            "transport_cost": transport_cost,
            "shot_distance": shot_distance,
        }

    def _aggregate_shots(self, shot_distance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if shot_distance.shape[-1] == 1:
            weights = torch.ones_like(shot_distance)
            return shot_distance.squeeze(-1), weights
        if self.shot_aggregation == "mean":
            weights = torch.full_like(shot_distance, 1.0 / float(shot_distance.shape[-1]))
        else:
            weights = torch.softmax(-float(self.shot_softmin_beta) * shot_distance, dim=-1)
        return (weights * shot_distance).sum(dim=-1), weights

    def _radius_from_shots(self, shot_distance: torch.Tensor) -> torch.Tensor:
        shot_count = int(shot_distance.shape[-1])
        if shot_count > 1:
            dispersion = shot_distance.std(dim=-1, unbiased=False)
        else:
            dispersion = torch.zeros_like(shot_distance.squeeze(-1))
        floor = float(self.radius_floor) / math.sqrt(float(max(shot_count, 1)))
        return float(self.radius_alpha) * dispersion + floor

    def _score_from_distance(
        self,
        class_distance: torch.Tensor,
        radius: torch.Tensor,
    ) -> torch.Tensor:
        if self.score_mode == "negative_distance":
            base_score = -class_distance
        elif self.score_mode == "radius_margin":
            base_score = -(class_distance - radius)
        else:
            base_score = -F.relu(class_distance - radius)
        return self.score_scale * base_score

    def _forward_episode(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: Optional[torch.Tensor] = None,
        support_targets: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, Any]:
        del query_targets, support_targets
        if query.dim() != 4:
            raise ValueError(f"query must have shape (NumQuery, C, H, W), got {tuple(query.shape)}")
        if support.dim() != 5:
            raise ValueError(f"support must have shape (Way, Shot, C, H, W), got {tuple(support.shape)}")

        way_num = support.shape[0]
        shot_num = support.shape[1]
        num_query = query.shape[0]

        Q, query_grid_h, query_grid_w = self._encode_token_map(query)
        S_all, support_grid_h, support_grid_w = self._encode_token_map(
            support.reshape(way_num * shot_num, *support.shape[2:])
        )

        Tq = Q.shape[1]
        Ts = S_all.shape[1]
        token_dim = Q.shape[2]
        grid_q = self._grid_distance(query_grid_h, query_grid_w, Q.device, Q.dtype)

        if self.support_mode == "concat":
            joint_support_tokens = shot_num * Ts
            S_class = S_all.reshape(way_num, joint_support_tokens, token_dim)
            pair_count = num_query * way_num
            Q_b = Q.unsqueeze(1).expand(-1, way_num, -1, -1).reshape(pair_count, Tq, token_dim)
            S_b = S_class.unsqueeze(0).expand(num_query, -1, -1, -1).reshape(
                pair_count,
                joint_support_tokens,
                token_dim,
            )
            grid_s = self._grid_distance(
                support_grid_h,
                support_grid_w,
                S_all.device,
                S_all.dtype,
                repeat=shot_num,
            )
            pair_out = self._solve_pair_batch(Q_b, S_b, grid_q, grid_s)
            flat_shape = (num_query, way_num, 1)
            pair_axis_shape = (num_query, way_num)
            token_s_for_aux = joint_support_tokens
        else:
            S_shot = S_all.reshape(way_num, shot_num, Ts, token_dim)
            pair_count = num_query * way_num * shot_num
            Q_b = (
                Q.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, way_num, shot_num, -1, -1)
                .reshape(pair_count, Tq, token_dim)
            )
            S_b = (
                S_shot.unsqueeze(0)
                .expand(num_query, -1, -1, -1, -1)
                .reshape(pair_count, Ts, token_dim)
            )
            grid_s = self._grid_distance(support_grid_h, support_grid_w, S_all.device, S_all.dtype)
            pair_out = self._solve_pair_batch(Q_b, S_b, grid_q, grid_s)
            flat_shape = (num_query, way_num, shot_num)
            pair_axis_shape = (num_query, way_num, shot_num)
            token_s_for_aux = Ts

        shot_distance = pair_out["shot_distance"].reshape(flat_shape)
        class_distance, shot_weights = self._aggregate_shots(shot_distance)
        transport_radius = self._radius_from_shots(shot_distance)
        logits = self._score_from_distance(class_distance, transport_radius).reshape(num_query, way_num)

        rho_shot = pair_out["rho"].reshape(flat_shape)
        rho_class = (shot_weights * rho_shot).sum(dim=-1)
        transport_cost_class = class_distance * rho_class.clamp_min(self.eps)
        rho_reg = (pair_out["rho"] - self.rho_target).pow(2).mean()
        aux_loss = self.lambda_rho * rho_reg

        result: Dict[str, Any] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "rho": rho_class.detach().reshape(num_query, way_num),
            "transported_mass": rho_class.detach().reshape(num_query, way_num),
            "rho_regularization": rho_reg.detach(),
            "alpha": self.alpha.detach(),
            "score_scale": self.score_scale.detach(),
            "transport_cost": transport_cost_class.detach().reshape(num_query, way_num),
            "query_class_distance": class_distance.detach().reshape(num_query, way_num),
            "transport_radius": transport_radius.detach().reshape(num_query, way_num),
            "epsilon": transport_radius.detach().reshape(num_query, way_num),
            "shot_distance": shot_distance.detach(),
            "shot_aggregation_weights": shot_weights.detach(),
            "mean_shot_distance": shot_distance.detach().mean(),
            "mean_query_mass_entropy": (
                -(pair_out["query_mass"].detach().clamp_min(self.eps)
                  * pair_out["query_mass"].detach().clamp_min(self.eps).log()).sum(dim=-1).mean()
            ),
            "mean_support_mass_entropy": (
                -(pair_out["support_mass"].detach().clamp_min(self.eps)
                  * pair_out["support_mass"].detach().clamp_min(self.eps).log()).sum(dim=-1).mean()
            ),
            "reliability_mix": torch.tensor(
                self.reliability_mix if self.mass_mode == "reliability" else 0.0,
                device=logits.device,
                dtype=logits.dtype,
            ),
        }
        if return_aux:
            result["rho_shot"] = rho_shot.detach()
            result["query_token_mass"] = pair_out["query_mass"].detach().reshape(*pair_axis_shape, Tq)
            result["support_token_mass"] = pair_out["support_mass"].detach().reshape(
                *pair_axis_shape,
                token_s_for_aux,
            )
            result["transport_plan"] = pair_out["transport_plan"].detach().reshape(
                *pair_axis_shape,
                Tq,
                token_s_for_aux,
            )
            result["C_feat"] = pair_out["C_feat"].detach().reshape(
                *pair_axis_shape,
                Tq,
                token_s_for_aux,
            )
            result["C_final"] = pair_out["C_final"].detach().reshape(
                *pair_axis_shape,
                Tq,
                token_s_for_aux,
            )
            result["D_q"] = pair_out["D_q"].detach().reshape(*pair_axis_shape, Tq, Tq)
            result["D_s"] = pair_out["D_s"].detach().reshape(
                *pair_axis_shape,
                token_s_for_aux,
                token_s_for_aux,
            )
        return result

    @staticmethod
    def _slice_query_targets(
        query_targets: Optional[torch.Tensor],
        batch_idx: int,
        num_query_per_ep: int,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if query_targets is None:
            return None
        if query_targets.dim() == 2:
            if query_targets.shape[0] != batch_size:
                raise ValueError(
                    f"query_targets batch mismatch: expected {batch_size}, got {query_targets.shape[0]}"
                )
            return query_targets[batch_idx]
        if query_targets.dim() == 1:
            start = batch_idx * num_query_per_ep
            return query_targets[start : start + num_query_per_ep]
        raise ValueError(f"query_targets must be 1D or 2D, got {tuple(query_targets.shape)}")

    @staticmethod
    def _slice_support_targets(
        support_targets: Optional[torch.Tensor],
        batch_idx: int,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if support_targets is None:
            return None
        if support_targets.dim() == 3:
            if support_targets.shape[0] != batch_size:
                raise ValueError(
                    f"support_targets batch mismatch: expected {batch_size}, got {support_targets.shape[0]}"
                )
            return support_targets[batch_idx]
        if batch_size == 1 and support_targets.dim() == 2:
            return support_targets
        raise ValueError(f"support_targets must be 2D or 3D, got {tuple(support_targets.shape)}")

    @staticmethod
    def _merge_episode_outputs(
        outputs: list[Dict[str, Any]],
        logits: torch.Tensor,
        aux_loss: torch.Tensor,
        return_aux: bool,
    ) -> FGWUOTFSLResult:
        merged: Dict[str, Any] = {"logits": logits, "aux_loss": aux_loss}
        if return_aux:
            for key in (
                "rho",
                "transported_mass",
                "transport_cost",
                "query_class_distance",
                "transport_radius",
                "epsilon",
                "shot_distance",
                "shot_aggregation_weights",
                "rho_shot",
                "query_token_mass",
                "support_token_mass",
                "transport_plan",
                "C_feat",
                "C_final",
                "D_q",
                "D_s",
            ):
                if key in outputs[0]:
                    merged[key] = torch.cat([item[key] for item in outputs], dim=0)
            merged["rho_regularization"] = torch.stack(
                [item["rho_regularization"] for item in outputs]
            ).mean()
            merged["alpha"] = outputs[-1]["alpha"]
            merged["score_scale"] = outputs[-1]["score_scale"]
            for key in ("mean_shot_distance", "mean_query_mass_entropy", "mean_support_mass_entropy"):
                if key in outputs[0]:
                    merged[key] = torch.stack([item[key] for item in outputs]).mean()
            merged["reliability_mix"] = outputs[-1]["reliability_mix"]
        return FGWUOTFSLResult(merged)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        query_targets: Optional[torch.Tensor] = None,
        support_targets: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        needs_payload = bool(return_aux or self.training)

        if support.dim() == 6:
            batch_size = support.shape[0]
            way_num = support.shape[1]

            if query.dim() == 5:
                if query.shape[0] != batch_size:
                    raise ValueError(
                        f"query/support batch mismatch: query={tuple(query.shape)} support={tuple(support.shape)}"
                    )
                num_query_per_ep = query.shape[1]

                def get_query(batch_idx: int) -> torch.Tensor:
                    return query[batch_idx]

            elif query.dim() == 4:
                if query.shape[0] % batch_size != 0:
                    raise ValueError(
                        f"flattened query count {query.shape[0]} must be divisible by batch size {batch_size}"
                    )
                num_query_per_ep = query.shape[0] // batch_size

                def get_query(batch_idx: int) -> torch.Tensor:
                    start = batch_idx * num_query_per_ep
                    return query[start : start + num_query_per_ep]

            else:
                raise ValueError(f"query must be 4D or 5D for batched support, got {tuple(query.shape)}")

            outputs: list[Dict[str, Any]] = []
            logits_per_episode: list[torch.Tensor] = []
            aux_losses: list[torch.Tensor] = []
            for batch_idx in range(batch_size):
                out_b = self._forward_episode(
                    get_query(batch_idx),
                    support[batch_idx],
                    query_targets=self._slice_query_targets(
                        query_targets,
                        batch_idx,
                        num_query_per_ep,
                        batch_size,
                    ),
                    support_targets=self._slice_support_targets(support_targets, batch_idx, batch_size),
                    return_aux=return_aux,
                )
                outputs.append(out_b)
                logits_per_episode.append(out_b["logits"])
                aux_losses.append(out_b["aux_loss"])

            logits = torch.cat(logits_per_episode, dim=0).reshape(-1, way_num)
            aux_loss = torch.stack(aux_losses).mean()
            if not needs_payload:
                return logits
            return self._merge_episode_outputs(outputs, logits, aux_loss, return_aux=return_aux)

        if support.dim() != 5:
            raise ValueError(f"support must be 5D or 6D, got {tuple(support.shape)}")
        if query.dim() != 4:
            raise ValueError(f"query must be 4D for single-episode support, got {tuple(query.shape)}")

        out = self._forward_episode(
            query,
            support,
            query_targets=query_targets,
            support_targets=support_targets,
            return_aux=return_aux,
        )
        logits = out["logits"].reshape(-1, support.shape[0])
        if not needs_payload:
            return logits
        if return_aux:
            out["logits"] = logits
            return FGWUOTFSLResult(out)
        return FGWUOTFSLResult({"logits": logits, "aux_loss": out["aux_loss"]})
