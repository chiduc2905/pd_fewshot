"""RADA few-shot head.

Reliability-Aware Dispersion Attention (RADA) is implemented as a compact
replacement few-shot head over support/query feature tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RADAFewShotHead(nn.Module):
    """Reliability-aware prototype and dispersion-aware metric head."""

    def __init__(
        self,
        feat_dim: int,
        tau_r: float = 0.5,
        lambda_proto: float = 0.7,
        gamma_disp: float = 0.5,
        eps: float = 1e-6,
        reliability_hidden_dim: int | None = None,
        reliability_head: str = "linear",
        l2_normalize: bool = True,
        use_reliability: bool = True,
        use_dispersion_metric: bool = True,
        query_conditioned: bool = True,
        use_residual_anchor: bool = True,
        use_shrinkage: bool = True,
        disp_clamp_max: float | None = None,
    ) -> None:
        super().__init__()
        if int(feat_dim) <= 0:
            raise ValueError("feat_dim must be positive")
        if float(tau_r) <= 0.0:
            raise ValueError("tau_r must be positive")
        if not (0.0 <= float(lambda_proto) <= 1.0):
            raise ValueError("lambda_proto must be in [0, 1]")
        if not (0.0 <= float(gamma_disp) <= 1.0):
            raise ValueError("gamma_disp must be in [0, 1]")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        if disp_clamp_max is not None and float(disp_clamp_max) <= float(eps):
            raise ValueError("disp_clamp_max must be greater than eps when provided")

        self.feat_dim = int(feat_dim)
        self.tau_r = float(tau_r)
        self.lambda_proto = float(lambda_proto)
        self.gamma_disp = float(gamma_disp)
        self.eps = float(eps)
        self.l2_normalize = bool(l2_normalize)
        self.use_reliability = bool(use_reliability)
        self.use_dispersion_metric = bool(use_dispersion_metric)
        self.query_conditioned = bool(query_conditioned)
        self.use_residual_anchor = bool(use_residual_anchor)
        self.use_shrinkage = bool(use_shrinkage)
        self.disp_clamp_max = None if disp_clamp_max is None else float(disp_clamp_max)

        reliability_head = str(reliability_head).lower()
        if reliability_head not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported reliability_head: {reliability_head}")
        if reliability_head == "mlp":
            hidden_dim = int(reliability_hidden_dim or 16)
            if hidden_dim <= 0:
                raise ValueError("reliability_hidden_dim must be positive")
            self.reliability_head = nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.reliability_head = nn.Linear(5, 1)

        self.global_dispersion_param = nn.Parameter(torch.zeros(self.feat_dim))

    def cosine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return (a * b).sum(dim=-1)

    def _normalize_inputs(
        self,
        support_feat: torch.Tensor,
        query_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.l2_normalize:
            return support_feat, query_feat
        return F.normalize(support_feat, dim=-1), F.normalize(query_feat, dim=-1)

    def _uniform_alpha(
        self,
        support_feat: torch.Tensor,
        batch_size: int,
        num_query: int,
        num_way: int,
        num_shot: int,
    ) -> torch.Tensor:
        return support_feat.new_full((batch_size, num_query, num_way, num_shot), 1.0 / float(num_shot))

    def forward(
        self,
        support_feat: torch.Tensor,
        query_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if support_feat.dim() != 4:
            raise ValueError(
                f"support_feat must have shape [B, N, K, D], got {tuple(support_feat.shape)}"
            )
        if query_feat.dim() != 3:
            raise ValueError(f"query_feat must have shape [B, M, D], got {tuple(query_feat.shape)}")
        if support_feat.shape[0] != query_feat.shape[0]:
            raise ValueError("support_feat and query_feat must share the same batch size")
        if support_feat.shape[-1] != self.feat_dim or query_feat.shape[-1] != self.feat_dim:
            raise ValueError(
                f"Expected feat_dim={self.feat_dim}, got support={support_feat.shape[-1]} "
                f"query={query_feat.shape[-1]}"
            )

        batch_size, num_way, num_shot, feat_dim = support_feat.shape
        _, num_query, _ = query_feat.shape

        support_feat, query_feat = self._normalize_inputs(support_feat, query_feat)

        mu = support_feat.mean(dim=2)
        diff_seed = support_feat - mu.unsqueeze(2)
        raw_scatter = diff_seed.pow(2).sum(dim=-1).mean(dim=2)

        support_e = support_feat.unsqueeze(1).expand(-1, num_query, -1, -1, -1)
        mu_e = mu.unsqueeze(1).unsqueeze(3)
        mu_support_e = mu_e.expand(-1, num_query, -1, num_shot, -1)
        query_e = query_feat.unsqueeze(2).unsqueeze(3)
        scatter_e = raw_scatter.unsqueeze(1).unsqueeze(-1).expand(-1, num_query, -1, num_shot)

        if self.use_reliability:
            cos_z_mu = self.cosine(support_e, mu_support_e)
            d_z_mu = (support_e - mu_e).pow(2).sum(dim=-1)

            if self.query_conditioned:
                query_support_e = query_e.expand(-1, -1, num_way, num_shot, -1)
                cos_z_q = self.cosine(support_e, query_support_e)
                d_z_q = (support_e - query_e).pow(2).sum(dim=-1)
            else:
                cos_z_q = support_feat.new_zeros((batch_size, num_query, num_way, num_shot))
                d_z_q = support_feat.new_zeros((batch_size, num_query, num_way, num_shot))

            reliability_feat = torch.stack(
                [cos_z_mu, cos_z_q, d_z_mu, d_z_q, scatter_e],
                dim=-1,
            )
            reliability_logits = self.reliability_head(reliability_feat).squeeze(-1)
            alpha = F.softmax(reliability_logits / self.tau_r, dim=-1)
        else:
            reliability_logits = support_feat.new_zeros((batch_size, num_query, num_way, num_shot))
            alpha = self._uniform_alpha(support_feat, batch_size, num_query, num_way, num_shot)

        weighted_proto = (alpha.unsqueeze(-1) * support_e).sum(dim=3)
        mu_q = mu.unsqueeze(1).expand(-1, num_query, -1, -1)
        if self.use_residual_anchor:
            proto = self.lambda_proto * weighted_proto + (1.0 - self.lambda_proto) * mu_q
        else:
            proto = weighted_proto

        centered = support_e - proto.unsqueeze(3)
        delta = (alpha.unsqueeze(-1) * centered.pow(2)).sum(dim=3)

        global_disp = F.softplus(self.global_dispersion_param).view(1, 1, 1, feat_dim)
        if self.use_shrinkage:
            disp = self.gamma_disp * delta + (1.0 - self.gamma_disp) * global_disp
        else:
            disp = delta
        if self.disp_clamp_max is None:
            disp = disp.clamp(min=self.eps)
        else:
            disp = disp.clamp(min=self.eps, max=self.disp_clamp_max)

        query_class = query_feat.unsqueeze(2)
        sq_diff = (query_class - proto).pow(2)
        if self.use_dispersion_metric:
            logits = -(sq_diff / disp).sum(dim=-1)
        else:
            logits = -sq_diff.sum(dim=-1)

        alpha_safe = alpha.clamp_min(self.eps)
        aux = {
            "alpha": alpha,
            "proto": proto,
            "disp": disp,
            "mu": mu,
            "raw_scatter": raw_scatter,
            "reliability_logits": reliability_logits,
            "delta": delta,
            "global_disp": global_disp.expand(batch_size, num_query, num_way, feat_dim),
            "alpha_entropy": -(alpha_safe * alpha_safe.log()).sum(dim=-1),
            "alpha_max": alpha.max(dim=-1).values,
            "prototype_shift_norm": (proto - mu_q).norm(dim=-1),
        }
        return logits, aux
