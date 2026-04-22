"""
FGWUOTFewShot — Fused Gromov-Wasserstein Unbalanced OT Few-Shot Classifier.

Novelty vs DeepEMD:
  1. Structural consistency via linearized GW term (end-to-end differentiable)
  2. Unbalanced transport (background tokens get low mass, not forced to match)
  3. Joint k-shot support matching (one OT solve per class, not per shot)
  4. α-weighted FGW cost as unified ground metric

Novelty vs HROT-R:
  - Single forward path (no H-anchor blend, no reliability probe)
  - Structure term IS differentiable (not detached probe)
  - 5 learnable scalars vs 20+ in HROT-R
  - Reduces cleanly to DeepEMD (α=0, τ→∞)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

from net.fewshot_common import BaseConv64FewShotModel
from net.modules.fgw_uot_solver import (
    fgw_uot_solve,
    normalize_intra_dist,
    pairwise_sq_l2,
)


# ---------------------------------------------------------------------------
# Rho head
# ---------------------------------------------------------------------------

class RhoHead(nn.Module):
    """
    Predicts per-(query, class) transport budget ρ ∈ (0, 1).

    Input: 4 episode statistics derived from C_feat, D_q, D_s:
      [0] mean(C_feat)        — average appearance cost
      [1] std(C_feat)         — cost spread / episode difficulty
      [2] mean(D_q)           — query token spread
      [3] mean(D_s)           — support token spread

    Output: ρ ∈ (0, 1)  per (query, class) pair  — (B,)
    """

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
        C_feat: torch.Tensor,   # (B, Tq, Ts)
        D_q:    torch.Tensor,   # (B, Tq, Tq)
        D_s:    torch.Tensor,   # (B, Ts, Ts)
        eps:    float = 1e-8,
    ) -> torch.Tensor:          # (B,)
        stats = torch.stack([
            C_feat.mean(dim=(-1, -2)),
            C_feat.std(dim=(-1, -2)).clamp_min(eps),
            D_q.mean(dim=(-1, -2)),
            D_s.mean(dim=(-1, -2)),
        ], dim=-1)                           # (B, 4)
        return self.net(stats).squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FGWUOTFewShot(BaseConv64FewShotModel):
    """
    Few-shot classifier using Fused Gromov-Wasserstein Unbalanced OT.

    Constructor args (all exposed to model_factory via --fgwuot_* flags):
      in_channels        : input image channels (always 3)
      hidden_dim         : backbone output channels (set by fewshot_backbone_output_dim)
      token_dim          : projected token dimension
      backbone_name      : backbone identifier (e.g. "conv64", "resnet12")
      image_size         : input image size (e.g. 84)
      tau                : KL relaxation strength for UOT
      eps_sinkhorn       : entropic regularization ε
      fgw_iters          : FGW fixed-point outer iterations
      sinkhorn_iters     : inner Sinkhorn iterations
      sinkhorn_tol       : Sinkhorn convergence tolerance
      alpha_init         : initial FGW mixing weight (0=DeepEMD, 1=pure structure)
      score_scale_init   : initial logit scale
      rho_head_hidden    : hidden dim of rho head MLP
      lambda_rho         : rho regularization coefficient
      rho_target         : rho regularization target
      normalize_tokens   : L2-normalize tokens before cost computation
      eps                : numerical epsilon
    """

    def __init__(
        self,
        in_channels:      int   = 3,
        hidden_dim:       int   = 64,
        token_dim:        int   = 128,
        backbone_name:    str   = "conv64",
        image_size:       int   = 84,
        tau:              float = 0.5,
        eps_sinkhorn:     float = 0.1,
        fgw_iters:        int   = 8,
        sinkhorn_iters:   int   = 60,
        sinkhorn_tol:     float = 1e-5,
        alpha_init:       float = 0.5,
        score_scale_init: float = 16.0,
        rho_head_hidden:  int   = 32,
        lambda_rho:       float = 0.01,
        rho_target:       float = 0.8,
        normalize_tokens: bool  = True,
        eps:              float = 1e-8,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
        )

        self.token_dim        = token_dim
        self.tau              = tau
        self.eps_sinkhorn     = eps_sinkhorn
        self.fgw_iters        = fgw_iters
        self.sinkhorn_iters   = sinkhorn_iters
        self.sinkhorn_tol     = sinkhorn_tol
        self.lambda_rho       = lambda_rho
        self.rho_target       = rho_target
        self.normalize_tokens = normalize_tokens
        self.eps              = eps

        # Token projector: hidden_dim → token_dim
        if token_dim != hidden_dim:
            self.token_projector: nn.Module = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, token_dim, bias=False),
            )
        else:
            self.token_projector = nn.Identity()

        # ---- Learnable scalars (minimal) -----------------------------------
        # alpha: FGW mix weight — raw_alpha → sigmoid → (0,1)
        _alpha_logit = math.log(max(alpha_init, eps) / max(1.0 - alpha_init, eps))
        self.raw_alpha = nn.Parameter(torch.tensor(_alpha_logit))

        # score_scale: logit scale — raw → softplus → >0
        self.raw_score_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(score_scale_init, eps))))
        )

        # Rho prediction head
        self.rho_head = RhoHead(hidden_dim=rho_head_hidden)

    # ---- Properties --------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        """FGW structure mixing weight ∈ (0, 1)."""
        return torch.sigmoid(self.raw_alpha)

    @property
    def score_scale(self) -> torch.Tensor:
        """Logit scale factor > 0."""
        return F.softplus(self.raw_score_scale)

    # ---- Encoding ----------------------------------------------------------

    def _encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        images  : (N, C, H, W)
        returns : (N, T, token_dim)  where T = H' * W'
        """
        feat = self.encode(images)                         # (N, hidden_dim, H', W')
        N, C, H, W = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(N, H * W, C)  # (N, T, C)
        tokens = self.token_projector(tokens)              # (N, T, token_dim)
        if self.normalize_tokens:
            tokens = F.normalize(tokens, dim=-1)
        return tokens

    # ---- Episode forward ---------------------------------------------------

    def _forward_episode(
        self,
        query:           torch.Tensor,              # (NumQuery, C, H, W)
        support:         torch.Tensor,              # (Way, Shot, C, H, W)
        query_targets:   Optional[torch.Tensor] = None,
        support_targets: Optional[torch.Tensor] = None,
        return_aux:      bool = False,
    ) -> Dict[str, Any]:
        """
        Core episode forward.

        Tensor shapes through the computation:
          Q         : (NQ, Tq, D)          — query tokens
          S_class   : (Way, Shot*Ts, D)    — joint support tokens per class
          Q_b       : (B, Tq, D)           — B = NQ * Way
          S_b       : (B, jTs, D)          — jTs = Shot * Ts
          C_feat    : (B, Tq, jTs)
          D_q       : (B, Tq, Tq)
          D_s       : (B, jTs, jTs)
          rho       : (B,)
          P         : (B, Tq, jTs)
          score     : (B,)
          logits    : (NQ, Way)
        """
        way_num   = support.shape[0]
        shot_num  = support.shape[1]
        num_query = query.shape[0]

        # ── 1. Encode ──────────────────────────────────────────────────────
        Q = self._encode_tokens(query)                                    # (NQ, Tq, D)
        S_flat_imgs = support.reshape(way_num * shot_num, *support.shape[2:])
        S_all = self._encode_tokens(S_flat_imgs)                         # (Way*Shot, Ts, D)

        Tq  = Q.shape[1]
        Ts  = S_all.shape[1]
        D   = Q.shape[2]
        jTs = shot_num * Ts   # joint support tokens per class

        # Joint support per class: (Way, jTs, D)
        S_class = S_all.reshape(way_num, jTs, D)

        # ── 2. Batch all (query, class) pairs ─────────────────────────────
        B = num_query * way_num

        Q_rep = Q.unsqueeze(1).expand(-1, way_num, -1, -1)          # (NQ, Way, Tq, D)
        S_rep = S_class.unsqueeze(0).expand(num_query, -1, -1, -1)  # (NQ, Way, jTs, D)

        Q_b = Q_rep.reshape(B, Tq, D)    # (B, Tq, D)
        S_b = S_rep.reshape(B, jTs, D)   # (B, jTs, D)

        # ── 3. Appearance cost and structure matrices ──────────────────────
        C_feat = pairwise_sq_l2(Q_b, S_b)      # (B, Tq, jTs)
        D_q    = pairwise_sq_l2(Q_b, Q_b)      # (B, Tq, Tq)
        D_s    = pairwise_sq_l2(S_b, S_b)      # (B, jTs, jTs)
        D_q    = normalize_intra_dist(D_q, self.eps)
        D_s    = normalize_intra_dist(D_s, self.eps)

        # ── 4. Rho prediction ─────────────────────────────────────────────
        rho = self.rho_head(C_feat, D_q, D_s, self.eps)   # (B,)
        rho = rho.clamp(self.eps, 1.0 - self.eps)

        # ── 5. Log marginals (uniform, budget-scaled) ─────────────────────
        log_rho = rho.log()                                            # (B,)
        log_a   = (log_rho - math.log(Tq)).unsqueeze(-1).expand(-1, Tq)    # (B, Tq)
        log_b   = (log_rho - math.log(jTs)).unsqueeze(-1).expand(-1, jTs)  # (B, jTs)

        # ── 6. FGW-UOT solve ──────────────────────────────────────────────
        P, C_final = fgw_uot_solve(
            C_feat, D_q, D_s,
            log_a, log_b,
            alpha=self.alpha,
            tau=self.tau,
            eps=self.eps_sinkhorn,
            fgw_iters=self.fgw_iters,
            sinkhorn_iters=self.sinkhorn_iters,
            tol=self.sinkhorn_tol,
        )   # P: (B, Tq, jTs),  C_final: (B, Tq, jTs)

        # ── 7. Score ──────────────────────────────────────────────────────
        transport_cost = (P * C_final).sum(dim=(-1, -2))              # (B,)
        score = self.score_scale * (-transport_cost / rho)            # (B,)
        logits = score.reshape(num_query, way_num)                    # (NQ, Way)

        # ── 8. Auxiliary loss ─────────────────────────────────────────────
        rho_reg  = (rho - self.rho_target).pow(2).mean()
        aux_loss = self.lambda_rho * rho_reg

        result: Dict[str, Any] = {
            "logits":             logits,
            "aux_loss":           aux_loss,
            "rho":                rho.detach().reshape(num_query, way_num),
            "rho_regularization": rho_reg.detach(),
            "alpha":              self.alpha.detach(),
            "score_scale":        self.score_scale.detach(),
            "transport_cost":     transport_cost.detach().reshape(num_query, way_num),
        }

        if return_aux:
            result["transport_plan"] = P.detach().reshape(num_query, way_num, Tq, jTs)
            result["C_feat"]         = C_feat.detach().reshape(num_query, way_num, Tq, jTs)
            result["C_final"]        = C_final.detach().reshape(num_query, way_num, Tq, jTs)
            result["D_q"]            = D_q.detach().reshape(num_query, way_num, Tq, Tq)
            result["D_s"]            = D_s.detach().reshape(num_query, way_num, jTs, jTs)

        return result

    # ---- Main forward (handles batched episodes) ---------------------------

    def forward(
        self,
        query:           torch.Tensor,
        support:         torch.Tensor,
        query_targets:   Optional[torch.Tensor] = None,
        support_targets: Optional[torch.Tensor] = None,
        return_aux:      bool = False,
    ):
        """
        Interface compatible with main.py forward_scores.

        query   : (B*NQ, C, H, W)        — B episodes × NQ queries (flattened)
        support : (B, Way, Shot, C, H, W) — B episodes

        Returns:
          (logits, aux_loss)           if return_aux=False
          (logits, aux_loss, aux_dict) if return_aux=True

          logits shape: (B*NQ, Way)
        """
        # ── Batched episodes ──────────────────────────────────────────────
        if support.dim() == 6:
            batch_size       = support.shape[0]
            way_num          = support.shape[1]
            num_query_per_ep = query.shape[0] // batch_size

            all_logits:   list = []
            all_aux_loss: list = []
            last_aux:     Dict[str, Any] = {}

            for b in range(batch_size):
                q_b  = query[b * num_query_per_ep : (b + 1) * num_query_per_ep]
                s_b  = support[b]
                qt_b = (query_targets[b * num_query_per_ep : (b + 1) * num_query_per_ep]
                        if query_targets is not None else None)
                st_b = (support_targets[b] if support_targets is not None else None)

                out_b = self._forward_episode(q_b, s_b, qt_b, st_b, return_aux)
                all_logits.append(out_b["logits"])
                all_aux_loss.append(out_b["aux_loss"])
                last_aux = out_b

            logits   = torch.cat(all_logits, dim=0).reshape(-1, way_num)
            aux_loss = torch.stack(all_aux_loss).mean()

        # ── Single episode ─────────────────────────────────────────────────
        else:
            out      = self._forward_episode(
                           query, support,
                           query_targets, support_targets, return_aux)
            logits   = out["logits"].reshape(-1, support.shape[0])
            aux_loss = out["aux_loss"]
            last_aux = out

        if return_aux:
            return logits, aux_loss, last_aux
        return logits, aux_loss
