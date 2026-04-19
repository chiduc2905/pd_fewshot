"""Balanced and unbalanced Sinkhorn solvers with native and POT backends."""

from __future__ import annotations

import warnings

import torch

try:
    import ot
except ImportError:  # pragma: no cover - exercised only when POT is unavailable
    ot = None


EPS = 1e-8


def _validate_transport_inputs(cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float) -> None:
    if cost.dim() < 2:
        raise ValueError(f"cost must have at least 2 dims, got {tuple(cost.shape)}")
    if a.shape != cost.shape[:-1]:
        raise ValueError(f"a must have shape {tuple(cost.shape[:-1])}, got {tuple(a.shape)}")
    if b.shape != cost.shape[:-2] + (cost.shape[-1],):
        raise ValueError(f"b must have shape {tuple(cost.shape[:-2] + (cost.shape[-1],))}, got {tuple(b.shape)}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")


def resolve_ot_backend(backend: str = "native") -> str:
    backend = str(backend).lower()
    if backend == "auto":
        return "native"
    if backend in {"native", "pot"}:
        if backend == "pot" and ot is None:
            raise ImportError("POT backend requested but POT is not installed")
        return backend
    raise ValueError(f"Unsupported OT backend: {backend}")


def require_pot() -> None:
    if ot is None:  # pragma: no cover - defensive
        raise ImportError("POT is required for this solver path but is not installed")


def sinkhorn_balanced_log(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> torch.Tensor:
    _validate_transport_inputs(cost, a, b, eps)
    log_a = torch.log(a.clamp_min(EPS))
    log_b = torch.log(b.clamp_min(EPS))
    log_kernel = -cost / float(eps)

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max_iter):
        prev_log_u = log_u
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        if tol > 0.0 and (log_u - prev_log_u).abs().amax() < tol:
            break

    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return torch.exp(log_plan)


def sinkhorn_unbalanced_log(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_q: float,
    tau_c: float,
    eps: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> torch.Tensor:
    _validate_transport_inputs(cost, a, b, eps)
    if tau_q <= 0.0 or tau_c <= 0.0:
        raise ValueError("tau_q and tau_c must be positive")

    log_a = torch.log(a.clamp_min(EPS))
    log_b = torch.log(b.clamp_min(EPS))
    log_kernel = -cost / float(eps)
    rho_q = float(tau_q) / (float(tau_q) + float(eps))
    rho_c = float(tau_c) / (float(tau_c) + float(eps))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max_iter):
        prev_log_u = log_u
        log_u = rho_q * (log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1))
        log_v = rho_c * (log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2))
        if tol > 0.0 and (log_u - prev_log_u).abs().amax() < tol:
            break

    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return torch.exp(log_plan)


def _pot_apply_pairwise(
    solver,
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    if ot is None:  # pragma: no cover - defensive
        raise ImportError("POT is not installed")
    if cost.dim() == 2:
        return solver(a, b, cost)

    leading_shape = cost.shape[:-2]
    cost_flat = cost.reshape(-1, cost.shape[-2], cost.shape[-1])
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    plans = [solver(a_flat[idx], b_flat[idx], cost_flat[idx]) for idx in range(cost_flat.shape[0])]
    return torch.stack(plans, dim=0).reshape(*leading_shape, cost.shape[-2], cost.shape[-1])


def sinkhorn_balanced_pot(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> torch.Tensor:
    _validate_transport_inputs(cost, a, b, eps)
    require_pot()

    def solver(pair_a: torch.Tensor, pair_b: torch.Tensor, pair_cost: torch.Tensor) -> torch.Tensor:
        return ot.sinkhorn(
            pair_a,
            pair_b,
            pair_cost,
            reg=float(eps),
            method="sinkhorn",
            numItermax=int(max_iter),
            stopThr=float(tol),
            verbose=False,
            warn=False,
        )

    return _pot_apply_pairwise(solver, cost, a, b)


def sinkhorn_unbalanced_pot(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_q: float,
    tau_c: float,
    eps: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> torch.Tensor:
    _validate_transport_inputs(cost, a, b, eps)
    require_pot()
    if tau_q <= 0.0 or tau_c <= 0.0:
        raise ValueError("tau_q and tau_c must be positive")

    def solver(pair_a: torch.Tensor, pair_b: torch.Tensor, pair_cost: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="If reg_type = entropy, then the matrix c is overwritten by the one matrix.",
            )
            return ot.unbalanced.sinkhorn_unbalanced(
                pair_a,
                pair_b,
                pair_cost,
                reg=float(eps),
                reg_m=(float(tau_q), float(tau_c)),
                method="sinkhorn",
                reg_type="entropy",
                numItermax=int(max_iter),
                stopThr=float(tol),
                verbose=False,
                log=False,
            )

    return _pot_apply_pairwise(solver, cost, a, b)


def barycenter_balanced_pot(
    histograms: torch.Tensor,
    cost: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    method: str = "sinkhorn",
) -> torch.Tensor:
    """POT fixed-support entropic Wasserstein barycenter.

    Args:
        histograms: `(NumMeasures, SupportSize)` distributions.
        cost: `(SupportSize, SupportSize)` ground cost.
        weights: `(NumMeasures,)` barycentric weights.
    """
    if histograms.dim() != 2:
        raise ValueError(f"histograms must have shape (NumMeasures, SupportSize), got {tuple(histograms.shape)}")
    if cost.shape != (histograms.shape[1], histograms.shape[1]):
        raise ValueError(
            "cost must have shape (SupportSize, SupportSize), "
            f"got {tuple(cost.shape)} for support size {histograms.shape[1]}"
        )
    if weights.shape != (histograms.shape[0],):
        raise ValueError(f"weights must have shape ({histograms.shape[0]},), got {tuple(weights.shape)}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    require_pot()
    kwargs = dict(
        reg=float(eps),
        weights=weights,
        method=str(method),
        numItermax=int(max_iter),
        stopThr=float(tol),
        verbose=False,
        log=False,
    )
    try:
        return ot.bregman.barycenter(histograms.transpose(0, 1), cost, warn=False, **kwargs)
    except TypeError:
        return ot.bregman.barycenter(histograms.transpose(0, 1), cost, **kwargs)


def barycenter_unbalanced_pot(
    histograms: torch.Tensor,
    cost: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
    reg_m: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    method: str = "sinkhorn",
) -> torch.Tensor:
    """POT fixed-support entropic unbalanced Wasserstein barycenter."""
    if histograms.dim() != 2:
        raise ValueError(f"histograms must have shape (NumMeasures, SupportSize), got {tuple(histograms.shape)}")
    if cost.shape != (histograms.shape[1], histograms.shape[1]):
        raise ValueError(
            "cost must have shape (SupportSize, SupportSize), "
            f"got {tuple(cost.shape)} for support size {histograms.shape[1]}"
        )
    if weights.shape != (histograms.shape[0],):
        raise ValueError(f"weights must have shape ({histograms.shape[0]},), got {tuple(weights.shape)}")
    if eps <= 0.0 or reg_m <= 0.0:
        raise ValueError("eps and reg_m must be positive")

    require_pot()
    solver = getattr(ot, "barycenter_unbalanced", None)
    if solver is None:
        solver = ot.unbalanced.barycenter_unbalanced
    return solver(
        histograms.transpose(0, 1),
        cost,
        float(eps),
        float(reg_m),
        method=str(method),
        weights=weights,
        numItermax=int(max_iter),
        stopThr=float(tol),
        verbose=False,
        log=False,
    )


def compute_transport_cost(plan: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    return (plan * cost).sum(dim=(-1, -2))


def compute_transported_mass(plan: torch.Tensor) -> torch.Tensor:
    return plan.sum(dim=(-1, -2))
