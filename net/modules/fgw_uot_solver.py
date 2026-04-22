"""
FGW-UOT Solver: Fused Gromov-Wasserstein Unbalanced Optimal Transport.

References:
  Vayer et al. "Optimal Transport for structured data" (2019)  — linearized FGW
  Chizat et al. "Scaling algorithms for unbalanced OT" (2018) — UOT Sinkhorn
"""

import math
import torch
from typing import Tuple


def pairwise_sq_l2(
    A: torch.Tensor,   # (..., N, D)
    B: torch.Tensor,   # (..., M, D)
) -> torch.Tensor:     # (..., N, M)
    """Batched pairwise squared Euclidean distance. Safe for autograd."""
    A_sq = (A * A).sum(dim=-1, keepdim=True)          # (..., N, 1)
    B_sq = (B * B).sum(dim=-1, keepdim=True)          # (..., M, 1)
    AB   = torch.matmul(A, B.transpose(-2, -1))       # (..., N, M)
    return (A_sq + B_sq.transpose(-2, -1) - 2.0 * AB).clamp_min(0.0)


def normalize_intra_dist(
    D:   torch.Tensor,   # (..., N, N)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize intra-distribution distance matrix by its mean per batch element."""
    return D / D.mean(dim=(-1, -2), keepdim=True).clamp_min(eps)


from net.modules.unbalanced_ot import sinkhorn_unbalanced_log

def _sinkhorn_uot_log_wrapper(
    cost:    torch.Tensor,
    log_a:   torch.Tensor,
    log_b:   torch.Tensor,
    tau:     float,
    eps:     float,
    max_iter: int   = 60,
    tol:     float = 1e-5,
) -> torch.Tensor:
    a = log_a.exp()
    b = log_b.exp()
    return sinkhorn_unbalanced_log(
        cost, a, b, tau_q=tau, tau_c=tau, eps=eps, max_iter=max_iter, tol=tol
    )


def fgw_uot_solve(
    C_feat:         torch.Tensor,   # (B, Tq, Ts)  appearance cost
    D_q:            torch.Tensor,   # (B, Tq, Tq)  intra-query structure
    D_s:            torch.Tensor,   # (B, Ts, Ts)  intra-support structure
    log_a:          torch.Tensor,   # (B, Tq)      log row marginal (includes rho)
    log_b:          torch.Tensor,   # (B, Ts)      log col marginal (includes rho)
    alpha:          torch.Tensor,   # ()           FGW mixing in [0, 1]
    tau:            float,
    eps:            float,
    fgw_iters:      int   = 8,
    sinkhorn_iters: int   = 60,
    tol:            float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linearized FGW-UOT fixed-point solver.

    Algorithm (Vayer 2019, linearized around P_prev):
      P^0 = exp(log_a ⊗ log_b) / Z
      For t = 0..T-1:
          P_prev = P^t.detach()
          C_t = (1-α)*C_feat - 2α*(D_q @ P_prev @ D_s)
          P^{t+1} = UOT_Sinkhorn(C_t, log_a, log_b, τ, ε)
      return P^T, C_T

    Gradient flow:
      - Sinkhorn of final step → P (ε-smoothed, differentiable)
      - C_feat → backbone features (appearance)
      - D_q, D_s → backbone features (structure; only P is detached, not D)
      - alpha → learnable FGW mix

    Returns:
      P      : final transport plan  (B, Tq, Ts)
      C_final: last linearized cost  (B, Tq, Ts)  — used for transport_cost computation
    """
    alpha_c = alpha.clamp(0.0, 1.0)

    # Initialization: outer product (proportional to marginal product)
    P = (log_a.unsqueeze(-1) + log_b.unsqueeze(-2)).exp()   # (B, Tq, Ts)
    P = P / P.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)

    C_t = C_feat  # fallback if fgw_iters == 0

    for _ in range(fgw_iters):
        # Detach P only — D_q, D_s RETAIN gradients for backbone learning
        P_prev = P.detach()
        a_prev = P_prev.sum(dim=-1, keepdim=True)  # (B, Tq, 1)
        b_prev = P_prev.sum(dim=-2, keepdim=True)  # (B, 1, Ts)

        # L(i,j) = sum_k D_q(i,k)^2 a(k) + sum_l D_s(j,l)^2 b(l) - 2 D_q P D_s
        D_q_sq = D_q * D_q
        D_s_sq = D_s * D_s
        term1 = torch.bmm(D_q_sq, a_prev)          # (B, Tq, 1)
        term2 = torch.bmm(b_prev, D_s_sq)          # (B, 1, Ts)
        cross_term = torch.bmm(torch.bmm(D_q, P_prev), D_s)    # (B, Tq, Ts)
        
        gw_term = term1 + term2 - 2.0 * cross_term

        C_t = (1.0 - alpha_c) * C_feat + alpha_c * gw_term

        P = _sinkhorn_uot_log_wrapper(
            C_t, log_a, log_b,
            tau=tau, eps=eps,
            max_iter=sinkhorn_iters,
            tol=tol,
        )

    return P, C_t
