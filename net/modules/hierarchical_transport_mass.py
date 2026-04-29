"""Transport-aware hierarchical mass allocation for J-style HROT."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


HLM_BUDGET_MODES = {"geodesic", "cost", "hybrid"}
HLM_TOKEN_MODES = {"uniform", "cost", "hybrid"}
HLM_COST_FEATURE_DIM = 9


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class HierarchicalTransportMass(nn.Module):
    """Predict shot budgets and budget-conserving token marginals.

    The module returns normalized token weights that each sum to one, then
    scales them by a learned query-class-shot budget. This keeps the learned
    budget separate from the transported mass produced by the UOT solver.
    """

    def __init__(
        self,
        *,
        min_mass: float = 0.1,
        init_mass: float = 0.8,
        budget_mode: str = "cost",
        token_mode: str = "cost",
        token_tau: float = 0.25,
        budget_hidden_dim: int = 64,
        token_hidden_dim: int = 64,
        geodesic_feature_dim: int = 4,
        detach_cost_features: bool = True,
        gate_init_bias: float = -2.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        budget_mode = str(budget_mode).strip().lower().replace("-", "_")
        token_mode = str(token_mode).strip().lower().replace("-", "_")
        if budget_mode not in HLM_BUDGET_MODES:
            raise ValueError(f"Unsupported HLM budget mode: {budget_mode}")
        if token_mode not in HLM_TOKEN_MODES:
            raise ValueError(f"Unsupported HLM token mode: {token_mode}")
        if not 0.0 < min_mass < 1.0:
            raise ValueError("min_mass must be in (0, 1)")
        if not min_mass <= init_mass <= 1.0:
            raise ValueError("init_mass must be in [min_mass, 1]")
        if token_tau <= 0.0:
            raise ValueError("token_tau must be positive")
        if budget_hidden_dim <= 0 or token_hidden_dim <= 0:
            raise ValueError("HLM hidden dimensions must be positive")
        if geodesic_feature_dim <= 0:
            raise ValueError("geodesic_feature_dim must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.min_mass = float(min_mass)
        self.init_mass = float(init_mass)
        self.budget_mode = budget_mode
        self.token_mode = token_mode
        self.token_tau = float(token_tau)
        self.geodesic_feature_dim = int(geodesic_feature_dim)
        self.detach_cost_features = bool(detach_cost_features)
        self.eps = float(eps)

        budget_input_dim = 0
        if self.budget_mode in {"cost", "hybrid"}:
            budget_input_dim += HLM_COST_FEATURE_DIM
        if self.budget_mode in {"geodesic", "hybrid"}:
            budget_input_dim += self.geodesic_feature_dim

        self.budget_mlp = nn.Sequential(
            nn.Linear(budget_input_dim, int(budget_hidden_dim)),
            nn.LayerNorm(int(budget_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(budget_hidden_dim), 1),
        )
        nn.init.zeros_(self.budget_mlp[-1].weight)
        nn.init.zeros_(self.budget_mlp[-1].bias)

        if self.token_mode == "hybrid":
            self.token_gate_mlp = nn.Sequential(
                nn.Linear(budget_input_dim, int(token_hidden_dim)),
                nn.LayerNorm(int(token_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(token_hidden_dim), 1),
            )
            nn.init.zeros_(self.token_gate_mlp[-1].weight)
            nn.init.constant_(self.token_gate_mlp[-1].bias, float(gate_init_bias))
        else:
            self.token_gate_mlp = None

        init_unit = (self.init_mass - self.min_mass) / max(self.eps, 1.0 - self.min_mass)
        self.register_buffer("rho_init_logit", torch.tensor(_inverse_sigmoid(init_unit), dtype=torch.float32))

    def _cost_stats(self, cost_qcst: torch.Tensor) -> dict[str, torch.Tensor]:
        tau = max(float(self.token_tau), self.eps)
        row_softmin = -tau * torch.logsumexp(-cost_qcst / tau, dim=-1)
        col_softmin = -tau * torch.logsumexp(-cost_qcst / tau, dim=-2)
        row_min = cost_qcst.amin(dim=-1)
        col_min = cost_qcst.amin(dim=-2)
        flat = cost_qcst.flatten(-2)
        mean_cost = flat.mean(dim=-1)
        min_cost = flat.amin(dim=-1)
        std_cost = flat.std(dim=-1, unbiased=False)
        row_softmin_mean = row_softmin.mean(dim=-1)
        col_softmin_mean = col_softmin.mean(dim=-1)
        row_min_mean = row_min.mean(dim=-1)
        col_min_mean = col_min.mean(dim=-1)
        class_mean = mean_cost.mean(dim=-1, keepdim=True)
        class_std = mean_cost.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        cost_gap_to_class = (mean_cost - class_mean) / class_std
        softmin_gap = row_softmin_mean - col_softmin_mean
        return {
            "mean_cost": mean_cost,
            "min_cost": min_cost,
            "std_cost": std_cost,
            "row_softmin_mean": row_softmin_mean,
            "col_softmin_mean": col_softmin_mean,
            "row_min_mean": row_min_mean,
            "col_min_mean": col_min_mean,
            "cost_gap_to_class": cost_gap_to_class,
            "softmin_gap": softmin_gap,
            "row_softmin": row_softmin,
            "col_softmin": col_softmin,
        }

    def _budget_features(
        self,
        stats: dict[str, torch.Tensor],
        geodesic_features: torch.Tensor | None,
    ) -> torch.Tensor:
        features = []
        if self.budget_mode in {"cost", "hybrid"}:
            cost_features = [
                stats["mean_cost"],
                stats["min_cost"],
                stats["std_cost"],
                stats["row_softmin_mean"],
                stats["col_softmin_mean"],
                stats["row_min_mean"],
                stats["col_min_mean"],
                stats["cost_gap_to_class"],
                stats["softmin_gap"],
            ]
            features.extend(f.unsqueeze(-1) for f in cost_features)
        if self.budget_mode in {"geodesic", "hybrid"}:
            if geodesic_features is None:
                raise ValueError(f"HLM budget_mode={self.budget_mode} requires geodesic_features")
            if geodesic_features.shape[-1] != self.geodesic_feature_dim:
                raise ValueError(
                    "geodesic_features last dim must be "
                    f"{self.geodesic_feature_dim}, got {geodesic_features.shape[-1]}"
                )
            features.append(geodesic_features)
        return torch.cat(features, dim=-1)

    def forward(
        self,
        cost_qcst: torch.Tensor,
        *,
        geodesic_features: torch.Tensor | None = None,
        threshold: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del threshold
        if cost_qcst.dim() != 5:
            raise ValueError(f"cost_qcst must have shape (Nq, Way, Shot, Lq, Ls), got {tuple(cost_qcst.shape)}")
        num_query, way_num, shot_num, query_len, support_len = cost_qcst.shape
        if geodesic_features is not None and tuple(geodesic_features.shape[:3]) != (num_query, way_num, shot_num):
            raise ValueError(
                "geodesic_features must align with cost_qcst in (Nq, Way, Shot), "
                f"got {tuple(geodesic_features.shape[:3])} vs {(num_query, way_num, shot_num)}"
            )

        stats = self._cost_stats(cost_qcst)
        features = self._budget_features(stats, geodesic_features)
        features_for_heads = features.detach() if self.detach_cost_features else features
        head_dtype = self.budget_mlp[0].weight.dtype
        raw = self.budget_mlp(features_for_heads.to(dtype=head_dtype)).squeeze(-1)
        raw = raw.to(device=cost_qcst.device, dtype=cost_qcst.dtype)
        rho_init = self.rho_init_logit.to(device=cost_qcst.device, dtype=cost_qcst.dtype)
        rho_unit = torch.sigmoid(raw + rho_init)
        shot_rho = self.min_mass + (1.0 - self.min_mass) * rho_unit
        shot_rho = shot_rho.clamp(min=self.min_mass + self.eps, max=1.0)

        query_weight_cost = torch.softmax((-stats["row_softmin"]) / self.token_tau, dim=-1)
        support_weight_cost = torch.softmax((-stats["col_softmin"]) / self.token_tau, dim=-1)
        uniform_q = torch.full_like(query_weight_cost, 1.0 / float(query_len))
        uniform_s = torch.full_like(support_weight_cost, 1.0 / float(support_len))

        if self.token_mode == "uniform":
            query_weight = uniform_q
            support_weight = uniform_s
            token_gate = cost_qcst.new_zeros(num_query, way_num, shot_num, 1)
        elif self.token_mode == "cost":
            query_weight = query_weight_cost
            support_weight = support_weight_cost
            token_gate = cost_qcst.new_ones(num_query, way_num, shot_num, 1)
        elif self.token_mode == "hybrid":
            if self.token_gate_mlp is None:
                raise RuntimeError("HLM hybrid token mode requires token_gate_mlp")
            gate_dtype = self.token_gate_mlp[0].weight.dtype
            token_gate = torch.sigmoid(self.token_gate_mlp(features_for_heads.to(dtype=gate_dtype)))
            token_gate = token_gate.to(device=cost_qcst.device, dtype=cost_qcst.dtype)
            query_weight = (1.0 - token_gate) * uniform_q + token_gate * query_weight_cost
            support_weight = (1.0 - token_gate) * uniform_s + token_gate * support_weight_cost
            query_weight = query_weight / query_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            support_weight = support_weight / support_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        else:
            raise ValueError(f"Unsupported HLM token mode: {self.token_mode}")

        query_mass = query_weight * shot_rho.unsqueeze(-1)
        support_mass = support_weight * shot_rho.unsqueeze(-1)

        if __debug__:
            torch._assert(torch.isfinite(shot_rho).all(), "HLM shot_rho contains non-finite values")
            torch._assert(torch.isfinite(query_mass).all(), "HLM query_mass contains non-finite values")
            torch._assert(torch.isfinite(support_mass).all(), "HLM support_mass contains non-finite values")
            torch._assert(
                torch.isclose(query_weight.sum(dim=-1), torch.ones_like(shot_rho), atol=1e-5, rtol=1e-4).all(),
                "HLM query weights must sum to one",
            )
            torch._assert(
                torch.isclose(support_weight.sum(dim=-1), torch.ones_like(shot_rho), atol=1e-5, rtol=1e-4).all(),
                "HLM support weights must sum to one",
            )

        return {
            "shot_rho": shot_rho,
            "query_weight": query_weight,
            "support_weight": support_weight,
            "query_mass": query_mass,
            "support_mass": support_mass,
            "budget_features": features,
            "token_gate": token_gate,
            "mean_cost": stats["mean_cost"],
            "min_cost": stats["min_cost"],
            "std_cost": stats["std_cost"],
            "row_softmin": stats["row_softmin"],
            "col_softmin": stats["col_softmin"],
        }
