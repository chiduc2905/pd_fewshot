"""Transport-reconstruction EMD for dense ResNet12 few-shot matching."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.encoders.smnet_conv64f_encoder import build_resnet12_family_encoder
from net.modules.unbalanced_ot import sinkhorn_balanced_log as _repo_sinkhorn_balanced_log


KSHOT_MODES = {"query_condense", "mean_condense"}


def flatten_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """Convert `[B, C, H, W]` dense maps to `[B, H * W, C]` tokens."""
    if feature_map.ndim != 4:
        raise ValueError(f"feature_map must have shape [B, C, H, W], got {tuple(feature_map.shape)}")
    return feature_map.flatten(2).transpose(1, 2).contiguous()


def reshape_support_tokens(support_tokens: torch.Tensor, way: int, shot: int) -> torch.Tensor:
    """Convert flat support tokens to `[way, shot, N, C]` when needed."""
    way = int(way)
    shot = int(shot)
    if support_tokens.ndim == 4:
        if support_tokens.shape[0] != way or support_tokens.shape[1] != shot:
            raise ValueError(
                f"support_tokens leading dims must be [way, shot]=[{way}, {shot}], "
                f"got {tuple(support_tokens.shape[:2])}"
            )
        return support_tokens
    if support_tokens.ndim != 3:
        raise ValueError(
            "support_tokens must have shape [way, shot, N, C] or [way * shot, N, C], "
            f"got {tuple(support_tokens.shape)}"
        )
    if support_tokens.shape[0] != way * shot:
        raise ValueError(
            f"flat support token count must equal way * shot={way * shot}, got {support_tokens.shape[0]}"
        )
    return support_tokens.reshape(way, shot, support_tokens.shape[-2], support_tokens.shape[-1])


def condense_support_tokens(
    query_tokens: torch.Tensor,
    support_tokens: torch.Tensor,
    mode: str = "query_condense",
    tau_shot: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build query-conditioned support tokens `[num_query, way, N, C]`."""
    if query_tokens.ndim != 3:
        raise ValueError(f"query_tokens must have shape [num_query, N, C], got {tuple(query_tokens.shape)}")
    if support_tokens.ndim != 4:
        raise ValueError(f"support_tokens must have shape [way, shot, N, C], got {tuple(support_tokens.shape)}")
    if query_tokens.shape[-2:] != support_tokens.shape[-2:]:
        raise ValueError(
            f"query/support token shapes must share [N, C], got {tuple(query_tokens.shape[-2:])} "
            f"and {tuple(support_tokens.shape[-2:])}"
        )
    if tau_shot <= 0.0:
        raise ValueError("tau_shot must be positive")

    mode = str(mode).lower()
    if mode not in KSHOT_MODES:
        raise ValueError(f"kshot condensation mode must be one of {sorted(KSHOT_MODES)}, got {mode}")

    num_query = query_tokens.shape[0]
    way, shot = support_tokens.shape[:2]
    if shot == 1:
        return support_tokens[:, 0].unsqueeze(0).expand(num_query, way, -1, -1)

    if mode == "mean_condense":
        return support_tokens.mean(dim=1).unsqueeze(0).expand(num_query, way, -1, -1)

    q_global = query_tokens.mean(dim=1)
    s_global = support_tokens.mean(dim=2)
    q_global = F.normalize(q_global, dim=-1, eps=eps)
    s_global = F.normalize(s_global, dim=-1, eps=eps)
    relevance = torch.einsum("qd,wkd->qwk", q_global, s_global)
    beta = F.softmax(relevance / float(tau_shot), dim=-1)
    return torch.einsum("qwk,wknd->qwnd", beta, support_tokens)


def compute_cosine_cost(
    query_tokens: torch.Tensor,
    condensed_support_tokens: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute `1 - cosine` cost with shape `[num_query, way, N, N]`."""
    if query_tokens.ndim != 3:
        raise ValueError(f"query_tokens must have shape [num_query, N, C], got {tuple(query_tokens.shape)}")
    if condensed_support_tokens.ndim != 4:
        raise ValueError(
            "condensed_support_tokens must have shape [num_query, way, N, C], "
            f"got {tuple(condensed_support_tokens.shape)}"
        )
    if query_tokens.shape[0] != condensed_support_tokens.shape[0]:
        raise ValueError("query and condensed support must have the same num_query dimension")
    if query_tokens.shape[-2:] != condensed_support_tokens.shape[-2:]:
        raise ValueError(
            f"query/support token shapes must share [N, C], got {tuple(query_tokens.shape[-2:])} "
            f"and {tuple(condensed_support_tokens.shape[-2:])}"
        )

    query_norm = F.normalize(query_tokens, dim=-1, eps=eps)
    support_norm = F.normalize(condensed_support_tokens, dim=-1, eps=eps)
    similarity = torch.einsum("qnd,qwmd->qwnm", query_norm, support_norm)
    return 1.0 - similarity


def sinkhorn_balanced_log(
    cost: torch.Tensor,
    reg: float,
    max_iter: int,
    eps: float = 1e-8,
    tol: float = 0.0,
) -> torch.Tensor:
    """Balanced log-domain Sinkhorn with uniform marginals on the last two axes."""
    if cost.ndim < 2:
        raise ValueError(f"cost must have at least 2 dims, got {tuple(cost.shape)}")
    if reg <= 0.0:
        raise ValueError("reg must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    num_src, num_dst = cost.shape[-2], cost.shape[-1]
    a = cost.new_full(cost.shape[:-1], 1.0 / float(num_src))
    b = cost.new_full(cost.shape[:-2] + (num_dst,), 1.0 / float(num_dst))

    if cost.dtype in (torch.float16, torch.bfloat16):
        plan = _repo_sinkhorn_balanced_log(
            cost.float(),
            a.float(),
            b.float(),
            eps=float(reg),
            max_iter=int(max_iter),
            tol=float(tol),
        )
        return plan.to(dtype=cost.dtype)

    return _repo_sinkhorn_balanced_log(
        cost,
        a,
        b,
        eps=float(reg),
        max_iter=int(max_iter),
        tol=float(tol),
    )


def compute_transport_reconstruction_scores(
    Q: torch.Tensor,
    S_bar: torch.Tensor,
    C: torch.Tensor,
    P: torch.Tensor,
    eps: float = 1e-8,
    normalize_reconstruction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score EMD plus bidirectional transport-guided reconstruction."""
    if Q.ndim != 3:
        raise ValueError(f"Q must have shape [num_query, N, C], got {tuple(Q.shape)}")
    if S_bar.ndim != 4:
        raise ValueError(f"S_bar must have shape [num_query, way, N, C], got {tuple(S_bar.shape)}")
    if C.shape != P.shape:
        raise ValueError(f"C and P must have matching shapes, got {tuple(C.shape)} and {tuple(P.shape)}")

    score_emd = -(P * C).sum(dim=(-1, -2))
    Q_rec = F.normalize(Q, dim=-1, eps=eps) if normalize_reconstruction else Q
    S_rec = F.normalize(S_bar, dim=-1, eps=eps) if normalize_reconstruction else S_bar

    P_row = P / (P.sum(dim=-1, keepdim=True) + float(eps))
    Q_hat = torch.einsum("qwij,qwjd->qwid", P_row, S_rec)
    rec_q = (Q_rec.unsqueeze(1) - Q_hat).pow(2).sum(dim=-1).mean(dim=-1)

    P_col = P / (P.sum(dim=-2, keepdim=True) + float(eps))
    S_hat = torch.einsum("qwij,qid->qwjd", P_col, Q_rec)
    rec_s = (S_rec - S_hat).pow(2).sum(dim=-1).mean(dim=-1)

    score_rec = -0.5 * (rec_q + rec_s)
    return score_emd, score_rec, Q_hat, rec_q, rec_s


def compute_gram_structure_score(Q: torch.Tensor, Q_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compare query token Gram structure before and after reconstruction."""
    if Q.ndim != 3:
        raise ValueError(f"Q must have shape [num_query, N, C], got {tuple(Q.shape)}")
    if Q_hat.ndim != 4:
        raise ValueError(f"Q_hat must have shape [num_query, way, N, C], got {tuple(Q_hat.shape)}")

    Q_norm = F.normalize(Q, dim=-1, eps=eps)
    Q_hat_norm = F.normalize(Q_hat, dim=-1, eps=eps)
    G_Q = torch.matmul(Q_norm, Q_norm.transpose(-1, -2))
    G_Q_hat = torch.matmul(Q_hat_norm, Q_hat_norm.transpose(-1, -2))
    gram_err = (G_Q.unsqueeze(1) - G_Q_hat).pow(2).mean(dim=(-1, -2))
    return -gram_err


class TransportReconEMD(nn.Module):
    """DeepEMD-derived transport alignment model with reconstruction scoring."""

    def __init__(
        self,
        image_size: int = 64,
        fewshot_backbone: str = "resnet12",
        tau_shot: float = 0.2,
        kshot_mode: str = "query_condense",
        sinkhorn_reg: float = 0.05,
        sinkhorn_iter: int = 20,
        sinkhorn_tolerance: float = 1e-6,
        lambda_emd: float = 0.3,
        lambda_rec: float = 1.0,
        lambda_struct: float = 0.1,
        score_scale: float = 8.0,
        use_sfc: bool = False,
        normalize_reconstruction: bool = True,
        debug_shapes: bool = False,
        eps: float = 1e-8,
        device: str = "cuda",
        **_: Any,
    ) -> None:
        super().__init__()
        if str(fewshot_backbone).lower() != "resnet12":
            raise ValueError("TransportReconEMD requires the ResNet12 640-dim dense backbone")
        if tau_shot <= 0.0:
            raise ValueError("tau_shot must be positive")
        if str(kshot_mode).lower() not in KSHOT_MODES:
            raise ValueError(f"kshot_mode must be one of {sorted(KSHOT_MODES)}")
        if sinkhorn_reg <= 0.0:
            raise ValueError("sinkhorn_reg must be positive")
        if sinkhorn_iter <= 0:
            raise ValueError("sinkhorn_iter must be positive")
        if sinkhorn_tolerance < 0.0:
            raise ValueError("sinkhorn_tolerance must be non-negative")
        if score_scale <= 0.0:
            raise ValueError("score_scale must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if use_sfc:
            raise ValueError("TransportReconEMD does not support SFC refinement")

        self.encoder = build_resnet12_family_encoder(
            image_size=image_size,
            backbone_name="resnet12",
            pool_output=False,
            variant="fewshot",
            drop_rate=0.0,
        )
        self.feat_dim = int(self.encoder.out_channels)
        if self.feat_dim != 640:
            raise ValueError(f"TransportReconEMD expects 640 encoder channels, got {self.feat_dim}")

        self.tau_shot = float(tau_shot)
        self.kshot_mode = str(kshot_mode).lower()
        self.sinkhorn_reg = float(sinkhorn_reg)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.sinkhorn_iterations = int(sinkhorn_iter)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.lambda_emd = float(lambda_emd)
        self.lambda_rec = float(lambda_rec)
        self.lambda_struct = float(lambda_struct)
        self.score_scale = float(score_scale)
        self.use_sfc = False
        self.normalize_reconstruction = bool(normalize_reconstruction)
        self.debug_shapes = bool(debug_shapes)
        self.eps = float(eps)
        self.last_transport_debug: dict[str, torch.Tensor] | None = None
        self.to(device)

    def _debug_assert_finite(self, name: str, tensor: torch.Tensor) -> None:
        if self.debug_shapes:
            assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"

    def _debug_validate(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        condensed_support: torch.Tensor,
        cost: torch.Tensor,
        plan: torch.Tensor,
        logits: torch.Tensor,
        way: int,
        shot: int,
    ) -> None:
        if not self.debug_shapes:
            return
        num_query, num_tokens, feat_dim = query_tokens.shape
        assert query_tokens.shape == (num_query, num_tokens, 640)
        assert support_tokens.shape == (way, shot, num_tokens, 640)
        assert condensed_support.shape == (num_query, way, num_tokens, 640)
        assert cost.shape == (num_query, way, num_tokens, num_tokens)
        assert plan.shape == (num_query, way, num_tokens, num_tokens)
        assert logits.shape == (num_query, way)

        for name, tensor in {
            "query_tokens": query_tokens,
            "support_tokens": support_tokens,
            "condensed_support": condensed_support,
            "cost": cost,
            "plan": plan,
            "logits": logits,
        }.items():
            self._debug_assert_finite(name, tensor)

        uniform = plan.new_full((num_query, way, num_tokens), 1.0 / float(num_tokens))
        atol = max(5e-3, 10.0 * self.eps)
        assert torch.allclose(plan.sum(dim=-1), uniform, atol=atol, rtol=5e-2), (
            "Sinkhorn row marginals do not match uniform marginals"
        )
        assert torch.allclose(plan.sum(dim=-2), uniform, atol=atol, rtol=5e-2), (
            "Sinkhorn column marginals do not match uniform marginals"
        )

    def forward_from_tokens(
        self,
        query_tokens: torch.Tensor,
        support_tokens: torch.Tensor,
        way: int,
        shot: int,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        support_tokens = reshape_support_tokens(support_tokens, way=way, shot=shot)
        if query_tokens.shape[-1] != self.feat_dim or support_tokens.shape[-1] != self.feat_dim:
            raise ValueError(
                f"TransportReconEMD expects {self.feat_dim}-dim tokens, got "
                f"{query_tokens.shape[-1]} and {support_tokens.shape[-1]}"
            )

        S_bar = condense_support_tokens(
            query_tokens,
            support_tokens,
            mode=self.kshot_mode,
            tau_shot=self.tau_shot,
            eps=self.eps,
        )
        cost = compute_cosine_cost(query_tokens, S_bar, eps=self.eps)
        plan = sinkhorn_balanced_log(
            cost,
            reg=self.sinkhorn_reg,
            max_iter=self.sinkhorn_iter,
            eps=self.eps,
            tol=self.sinkhorn_tolerance,
        )
        score_emd = -(plan * cost).sum(dim=(-1, -2))

        zero_scores = score_emd.new_zeros(score_emd.shape)
        score_rec = zero_scores
        score_struct = zero_scores
        rec_q = zero_scores
        rec_s = zero_scores
        Q_hat = None

        needs_reconstruction = self.lambda_rec != 0.0 or self.lambda_struct != 0.0 or return_aux
        if needs_reconstruction:
            score_emd, score_rec, Q_hat, rec_q, rec_s = compute_transport_reconstruction_scores(
                query_tokens,
                S_bar,
                cost,
                plan,
                eps=self.eps,
                normalize_reconstruction=getattr(self, "normalize_reconstruction", True),
            )
            if self.lambda_struct != 0.0 or return_aux:
                score_struct = compute_gram_structure_score(query_tokens, Q_hat, eps=self.eps)

        raw_logits = (
            self.lambda_emd * score_emd
            + self.lambda_rec * score_rec
            + self.lambda_struct * score_struct
        )
        logits = self.score_scale * raw_logits

        self._debug_validate(
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            condensed_support=S_bar,
            cost=cost,
            plan=plan,
            logits=logits,
            way=int(way),
            shot=int(shot),
        )

        if not return_aux:
            self.last_transport_debug = None
            return logits

        payload: dict[str, torch.Tensor] = {
            "logits": logits,
            "raw_logits": raw_logits,
            "score_emd": score_emd,
            "score_rec": score_rec,
            "score_struct": score_struct,
            "score_scale": logits.new_tensor(self.score_scale),
            "rec_q": rec_q,
            "rec_s": rec_s,
            "cost": cost,
            "plan": plan,
            "condensed_support": S_bar,
        }
        if Q_hat is not None:
            payload["query_reconstruction"] = Q_hat
        self.last_transport_debug = payload if self.debug_shapes else None
        return payload

    @staticmethod
    def _merge_aux(outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not outputs:
            return {}
        merged: dict[str, torch.Tensor] = {}
        for key in outputs[0]:
            values = [item[key] for item in outputs if key in item]
            if values and torch.is_tensor(values[0]):
                merged[key] = torch.cat(values, dim=0)
        return merged

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        return_aux: bool = False,
        query_targets: torch.Tensor | None = None,
        support_targets: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del query_targets, support_targets
        if query.ndim != 5:
            raise ValueError(f"query must have shape [B, num_query, C, H, W], got {tuple(query.shape)}")
        if support.ndim != 6:
            raise ValueError(f"support must have shape [B, way, shot, C, H, W], got {tuple(support.shape)}")

        batch_size, num_query, channels, height, width = query.shape
        support_batch, way, shot, support_channels, support_height, support_width = support.shape
        if support_batch != batch_size:
            raise ValueError("query and support batch dimensions must match")
        if (support_channels, support_height, support_width) != (channels, height, width):
            raise ValueError("query and support image dimensions must match")

        query_feat = self.encoder.forward_features(query.reshape(-1, channels, height, width))
        support_feat = self.encoder.forward_features(support.reshape(-1, channels, height, width))
        query_tokens = flatten_tokens(query_feat).reshape(batch_size, num_query, -1, self.feat_dim)
        support_tokens = flatten_tokens(support_feat).reshape(batch_size, way, shot, -1, self.feat_dim)

        logits_per_batch: list[torch.Tensor] = []
        aux_per_batch: list[dict[str, torch.Tensor]] = []
        for batch_idx in range(batch_size):
            output = self.forward_from_tokens(
                query_tokens[batch_idx],
                support_tokens[batch_idx],
                way=way,
                shot=shot,
                return_aux=return_aux,
            )
            if return_aux:
                assert isinstance(output, dict)
                logits_per_batch.append(output["logits"])
                aux_per_batch.append(output)
            else:
                assert torch.is_tensor(output)
                logits_per_batch.append(output)

        logits = torch.cat(logits_per_batch, dim=0)
        if not return_aux:
            return logits

        payload = self._merge_aux(aux_per_batch)
        payload["logits"] = logits
        payload["aux_loss"] = logits.new_zeros(())
        return payload


TARDISEMD = TransportReconEMD
TransportReconEMDSimple = TransportReconEMD
TARDISEMDSimple = TransportReconEMD
