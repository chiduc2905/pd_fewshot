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
)


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

        self.token_dim = int(token_dim)
        self.tau = float(tau)
        self.eps_sinkhorn = float(eps_sinkhorn)
        self.fgw_iters = int(fgw_iters)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.lambda_rho = float(lambda_rho)
        self.rho_target = float(rho_target)
        self.normalize_tokens = bool(normalize_tokens)
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

    def _encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.encode(images)
        if feat.dim() != 4:
            raise ValueError(f"encoded feature map must be 4D, got {tuple(feat.shape)}")
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_projector(tokens)
        if self.normalize_tokens:
            tokens = F.normalize(tokens, dim=-1)
        return tokens

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

        Q = self._encode_tokens(query)
        S_all = self._encode_tokens(support.reshape(way_num * shot_num, *support.shape[2:]))

        Tq = Q.shape[1]
        Ts = S_all.shape[1]
        token_dim = Q.shape[2]
        joint_support_tokens = shot_num * Ts
        S_class = S_all.reshape(way_num, joint_support_tokens, token_dim)

        pair_count = num_query * way_num
        Q_b = Q.unsqueeze(1).expand(-1, way_num, -1, -1).reshape(pair_count, Tq, token_dim)
        S_b = S_class.unsqueeze(0).expand(num_query, -1, -1, -1).reshape(
            pair_count,
            joint_support_tokens,
            token_dim,
        )

        C_feat = pairwise_sq_l2(Q_b, S_b)
        D_q = normalize_intra_dist(pairwise_sq_l2(Q_b, Q_b), self.eps)
        D_s = normalize_intra_dist(pairwise_sq_l2(S_b, S_b), self.eps)

        rho = self.rho_head(C_feat, D_q, D_s, self.eps).clamp(self.eps, 1.0 - self.eps)
        log_rho = rho.log()
        log_a = (log_rho - math.log(Tq)).unsqueeze(-1).expand(-1, Tq)
        log_b = (log_rho - math.log(joint_support_tokens)).unsqueeze(-1).expand(
            -1,
            joint_support_tokens,
        )

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
        logits = (self.score_scale * (-transport_cost / rho)).reshape(num_query, way_num)
        rho_reg = (rho - self.rho_target).pow(2).mean()
        aux_loss = self.lambda_rho * rho_reg

        result: Dict[str, Any] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "rho": rho.detach().reshape(num_query, way_num),
            "rho_regularization": rho_reg.detach(),
            "alpha": self.alpha.detach(),
            "score_scale": self.score_scale.detach(),
            "transport_cost": transport_cost.detach().reshape(num_query, way_num),
        }
        if return_aux:
            result["transport_plan"] = P.detach().reshape(num_query, way_num, Tq, joint_support_tokens)
            result["C_feat"] = C_feat.detach().reshape(num_query, way_num, Tq, joint_support_tokens)
            result["C_final"] = C_final.detach().reshape(num_query, way_num, Tq, joint_support_tokens)
            result["D_q"] = D_q.detach().reshape(num_query, way_num, Tq, Tq)
            result["D_s"] = D_s.detach().reshape(num_query, way_num, joint_support_tokens, joint_support_tokens)
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
            for key in ("rho", "transport_cost", "transport_plan", "C_feat", "C_final", "D_q", "D_s"):
                if key in outputs[0]:
                    merged[key] = torch.cat([item[key] for item in outputs], dim=0)
            merged["rho_regularization"] = torch.stack(
                [item["rho_regularization"] for item in outputs]
            ).mean()
            merged["alpha"] = outputs[-1]["alpha"]
            merged["score_scale"] = outputs[-1]["score_scale"]
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
