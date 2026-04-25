"""Partial optimal transport solvers with native PyTorch and POT backends.

The core problem is the standard partial Wasserstein transport:

    min_P <P, C>
    s.t. P 1 <= a, P^T 1 <= b, P >= 0, sum(P) = m.

This is useful when only a fraction of two token measures should be matched,
for example foreground/evidence regions in cluttered feature maps.
"""

from __future__ import annotations

import torch

try:
    import ot
except ImportError:  # pragma: no cover - exercised only when POT is unavailable
    ot = None


EPS = 1e-8


def _validate_partial_inputs(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float,
) -> None:
    if cost.dim() < 2:
        raise ValueError(f"cost must have at least 2 dims, got {tuple(cost.shape)}")
    if a.shape != cost.shape[:-1]:
        raise ValueError(f"a must have shape {tuple(cost.shape[:-1])}, got {tuple(a.shape)}")
    if b.shape != cost.shape[:-2] + (cost.shape[-1],):
        raise ValueError(f"b must have shape {tuple(cost.shape[:-2] + (cost.shape[-1],))}, got {tuple(b.shape)}")
    if reg <= 0.0:
        raise ValueError("reg must be positive")
    if torch.any(a < 0.0) or torch.any(b < 0.0):
        raise ValueError("a and b must be non-negative")


def _prepare_transport_mass(
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor | None,
    eps: float = EPS,
) -> torch.Tensor:
    leading_shape = a.shape[:-1]
    max_mass = torch.minimum(a.sum(dim=-1), b.sum(dim=-1))

    if transport_mass is None:
        mass = max_mass
    elif isinstance(transport_mass, torch.Tensor):
        mass = transport_mass.to(device=a.device, dtype=a.dtype)
        if mass.dim() == 0:
            mass = mass.expand(leading_shape)
        elif tuple(mass.shape) != tuple(leading_shape):
            raise ValueError(
                "transport_mass tensor must be scalar or have leading shape "
                f"{tuple(leading_shape)}, got {tuple(mass.shape)}"
            )
    else:
        mass = a.new_full(leading_shape, float(transport_mass))

    if torch.any(mass < 0.0):
        raise ValueError("transport_mass must be non-negative")
    if torch.any(mass > max_mass + eps):
        raise ValueError("transport_mass must be <= min(sum(a), sum(b)) for every problem")
    return mass


def require_pot() -> None:
    if ot is None:  # pragma: no cover - defensive
        raise ImportError("POT is required for this partial OT backend but is not installed")


def entropic_partial_wasserstein(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor | None = None,
    reg: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
    eps: float = EPS,
) -> torch.Tensor:
    """Differentiable entropic partial Wasserstein plan.

    This follows the Bregman projection formulation used by POT's
    ``ot.partial.entropic_partial_wasserstein`` and supports arbitrary leading
    batch dimensions.
    """
    _validate_partial_inputs(cost, a, b, reg)
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    mass = _prepare_transport_mass(a, b, transport_mass, eps=eps)
    mass_view = mass.reshape(*mass.shape, 1, 1)

    shifted_cost = cost - cost.amin(dim=(-1, -2), keepdim=True)
    kernel = torch.exp(-shifted_cost / float(reg))
    kernel = kernel * mass_view / kernel.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)

    q1 = torch.ones_like(kernel)
    q2 = torch.ones_like(kernel)
    q3 = torch.ones_like(kernel)
    one_row = torch.ones_like(a).unsqueeze(-1)
    one_col = torch.ones_like(b).unsqueeze(-2)

    for iter_idx in range(int(max_iter)):
        previous = kernel

        kernel = kernel * q1
        row_scale = torch.minimum(
            a.unsqueeze(-1) / kernel.sum(dim=-1, keepdim=True).clamp_min(eps),
            one_row,
        )
        row_projected = kernel * row_scale
        q1 = q1 * previous / row_projected.clamp_min(eps)

        previous_row = row_projected
        row_projected = row_projected * q2
        col_scale = torch.minimum(
            b.unsqueeze(-2) / row_projected.sum(dim=-2, keepdim=True).clamp_min(eps),
            one_col,
        )
        col_projected = row_projected * col_scale
        q2 = q2 * previous_row / col_projected.clamp_min(eps)

        previous_col = col_projected
        col_projected = col_projected * q3
        kernel = col_projected * mass_view / col_projected.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)
        q3 = q3 * previous_col / kernel.clamp_min(eps)

        if not torch.isfinite(kernel).all():
            raise FloatingPointError(f"partial OT numerical error at iteration {iter_idx}")
        if tol > 0.0 and iter_idx % 10 == 0:
            delta = (kernel - previous).abs().amax()
            if delta < tol:
                break

    return kernel.clamp_min(0.0)


def _pot_apply_pairwise(
    solver,
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    mass: torch.Tensor,
) -> torch.Tensor:
    require_pot()
    if cost.dim() == 2:
        return solver(a, b, cost, mass)

    leading_shape = cost.shape[:-2]
    cost_flat = cost.reshape(-1, cost.shape[-2], cost.shape[-1])
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    mass_flat = mass.reshape(-1)
    plans = [
        solver(a_flat[idx], b_flat[idx], cost_flat[idx], mass_flat[idx])
        for idx in range(cost_flat.shape[0])
    ]
    return torch.stack(plans, dim=0).reshape(*leading_shape, cost.shape[-2], cost.shape[-1])


def partial_wasserstein_pot(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor | None = None,
    nb_dummies: int = 1,
    reg: float = 1.0,
    entropic: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-9,
    eps: float = EPS,
) -> torch.Tensor:
    """POT partial Wasserstein wrapper.

    ``entropic=False`` calls POT's exact ``partial_wasserstein``. ``entropic=True``
    calls ``entropic_partial_wasserstein`` for a regularized reference solver.
    """
    _validate_partial_inputs(cost, a, b, reg=reg if entropic else 1.0)
    if nb_dummies <= 0:
        raise ValueError("nb_dummies must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    mass = _prepare_transport_mass(a, b, transport_mass, eps=eps)

    def solver(
        pair_a: torch.Tensor,
        pair_b: torch.Tensor,
        pair_cost: torch.Tensor,
        pair_mass: torch.Tensor,
    ) -> torch.Tensor:
        if entropic:
            plan = ot.partial.entropic_partial_wasserstein(
                pair_a,
                pair_b,
                pair_cost,
                reg=float(reg),
                m=float(pair_mass.detach().cpu()),
                numItermax=int(max_iter),
                stopThr=float(tol),
                verbose=False,
                log=False,
            )
        else:
            plan = ot.partial.partial_wasserstein(
                pair_a,
                pair_b,
                pair_cost,
                m=float(pair_mass.detach().cpu()),
                nb_dummies=int(nb_dummies),
                log=False,
            )
        return plan.to(device=pair_cost.device, dtype=pair_cost.dtype)

    return _pot_apply_pairwise(solver, cost, a, b, mass)


def solve_partial_transport(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    transport_mass: float | torch.Tensor | None = None,
    *,
    backend: str = "native",
    reg: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
    nb_dummies: int = 1,
    exact: bool = False,
    eps: float = EPS,
) -> torch.Tensor:
    """Dispatch helper for reusable partial OT matching blocks."""
    backend = str(backend).lower()
    if backend == "native":
        if exact:
            raise ValueError("exact=True is only supported by backend='pot'")
        return entropic_partial_wasserstein(
            cost,
            a,
            b,
            transport_mass=transport_mass,
            reg=reg,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
        )
    if backend == "pot":
        return partial_wasserstein_pot(
            cost,
            a,
            b,
            transport_mass=transport_mass,
            nb_dummies=nb_dummies,
            reg=reg,
            entropic=not exact,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
        )
    raise ValueError(f"Unsupported partial OT backend: {backend}")


def compute_partial_transport_cost(plan: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    if plan.shape != cost.shape:
        raise ValueError(f"plan must have shape {tuple(cost.shape)}, got {tuple(plan.shape)}")
    return (plan * cost).sum(dim=(-1, -2))


def compute_partial_transported_mass(plan: torch.Tensor) -> torch.Tensor:
    return plan.sum(dim=(-1, -2))
