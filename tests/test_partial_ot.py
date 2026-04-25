from __future__ import annotations

import pytest
import torch

from net.modules.partial_ot import (
    compute_partial_transport_cost,
    compute_partial_transported_mass,
    entropic_partial_wasserstein,
    partial_wasserstein_pot,
    partial_transport_plan_residuals,
    resolve_partial_transport_mass,
    solve_partial_transport,
)


def _positive_histogram(shape):
    return torch.rand(*shape) + 0.05


def test_native_entropic_partial_ot_satisfies_partial_constraints_batched():
    torch.manual_seed(301)
    cost = torch.rand(2, 3, 4, 5)
    a = _positive_histogram((2, 3, 4))
    b = _positive_histogram((2, 3, 5))
    mass = 0.55 * torch.minimum(a.sum(dim=-1), b.sum(dim=-1))

    plan = entropic_partial_wasserstein(cost, a, b, transport_mass=mass, reg=0.2, max_iter=250, tol=1e-8)

    assert plan.shape == cost.shape
    assert torch.isfinite(plan).all()
    assert torch.all(plan >= 0.0)
    assert torch.all(plan.sum(dim=-1) <= a + 3e-4)
    assert torch.all(plan.sum(dim=-2) <= b + 3e-4)
    assert torch.allclose(compute_partial_transported_mass(plan), mass, atol=3e-4, rtol=3e-4)
    residuals = partial_transport_plan_residuals(plan, a, b, mass)
    assert torch.all(residuals["row_violation"] <= 3e-4)
    assert torch.all(residuals["column_violation"] <= 3e-4)
    assert torch.all(residuals["mass_residual"] <= 3e-4)


def test_partial_ot_mass_ratio_resolves_against_available_mass():
    torch.manual_seed(304)
    a = _positive_histogram((2, 4))
    b = _positive_histogram((2, 5))
    max_mass = torch.minimum(a.sum(dim=-1), b.sum(dim=-1))

    mass = resolve_partial_transport_mass(a, b, mass_ratio=0.4)

    assert torch.allclose(mass, 0.4 * max_mass)
    with pytest.raises(ValueError, match="cannot be set together"):
        resolve_partial_transport_mass(a, b, transport_mass=0.2, transport_mass_ratio=0.4)
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        resolve_partial_transport_mass(a, b, transport_mass_ratio=1.1)


def test_native_partial_ot_matches_pot_entropic_reference():
    pytest.importorskip("ot")
    torch.manual_seed(302)
    cost = torch.rand(4, 5)
    a = _positive_histogram((4,))
    b = _positive_histogram((5,))
    mass = 0.6 * min(float(a.sum()), float(b.sum()))

    native = entropic_partial_wasserstein(cost, a, b, transport_mass=mass, reg=0.3, max_iter=400, tol=1e-10)
    pot = partial_wasserstein_pot(
        cost,
        a,
        b,
        transport_mass=mass,
        reg=0.3,
        entropic=True,
        max_iter=400,
        tol=1e-10,
    )

    assert torch.allclose(native, pot, atol=2e-3, rtol=2e-2)
    assert torch.allclose(
        compute_partial_transport_cost(native, cost),
        compute_partial_transport_cost(pot, cost),
        atol=2e-3,
        rtol=2e-2,
    )


def test_native_entropic_partial_ot_ratio_matches_pot_entropic_cost():
    pytest.importorskip("ot")
    torch.manual_seed(305)
    cost = torch.rand(2, 4, 5)
    a = _positive_histogram((2, 4))
    b = _positive_histogram((2, 5))

    native = entropic_partial_wasserstein(
        cost,
        a,
        b,
        transport_mass_ratio=0.65,
        reg=0.25,
        max_iter=500,
        tol=1e-10,
    )
    pot = partial_wasserstein_pot(
        cost,
        a,
        b,
        mass_ratio=0.65,
        reg=0.25,
        entropic=True,
        max_iter=500,
        tol=1e-10,
    )

    expected_mass = 0.65 * torch.minimum(a.sum(dim=-1), b.sum(dim=-1))
    residuals = partial_transport_plan_residuals(native, a, b, expected_mass)
    assert torch.all(residuals["row_violation"] <= 3e-4)
    assert torch.all(residuals["column_violation"] <= 3e-4)
    assert torch.all(residuals["mass_residual"] <= 3e-4)
    assert torch.allclose(
        compute_partial_transport_cost(native, cost),
        compute_partial_transport_cost(pot, cost),
        atol=3e-3,
        rtol=3e-2,
    )


def test_pot_exact_partial_ot_uses_requested_mass_and_low_cost_pairs():
    pytest.importorskip("ot")
    a = torch.tensor([0.5, 0.5])
    b = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[0.0, 3.0], [3.0, 0.1]])

    plan = partial_wasserstein_pot(cost, a, b, transport_mass=0.5, entropic=False)

    assert plan.shape == cost.shape
    assert torch.all(plan.sum(dim=-1) <= a + 1e-6)
    assert torch.all(plan.sum(dim=-2) <= b + 1e-6)
    assert torch.allclose(compute_partial_transported_mass(plan), torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(compute_partial_transport_cost(plan, cost), torch.tensor(0.0), atol=1e-6)


def test_solve_partial_transport_dispatch_and_gradient():
    torch.manual_seed(303)
    features_q = torch.randn(3, 4, requires_grad=True)
    features_s = torch.randn(5, 4, requires_grad=True)
    cost = torch.cdist(features_q, features_s).pow(2.0)
    a = torch.full((3,), 1.0 / 3.0)
    b = torch.full((5,), 1.0 / 5.0)

    plan = solve_partial_transport(
        cost,
        a,
        b,
        transport_mass=0.45,
        backend="native",
        reg=0.25,
        max_iter=200,
        tol=1e-8,
    )
    loss = compute_partial_transport_cost(plan, cost)
    loss.backward()

    assert torch.isfinite(plan).all()
    assert features_q.grad is not None
    assert features_s.grad is not None
    assert torch.isfinite(features_q.grad).all()
    assert torch.isfinite(features_s.grad).all()
