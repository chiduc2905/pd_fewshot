from __future__ import annotations

import pytest
import torch

from net.modules.unbalanced_ot import (
    compute_transported_mass,
    compute_unbalanced_transport_objective,
    sinkhorn_balanced_log,
    sinkhorn_balanced_pot,
    sinkhorn_unbalanced_log,
    sinkhorn_unbalanced_pot,
)


def _positive_simplex(shape):
    values = torch.rand(*shape) + 0.05
    return values / values.sum(dim=-1, keepdim=True)


def test_native_balanced_sinkhorn_satisfies_marginals_batched():
    torch.manual_seed(201)
    cost = torch.rand(2, 3, 4, 5)
    a = _positive_simplex((2, 3, 4))
    b = _positive_simplex((2, 3, 5))

    plan = sinkhorn_balanced_log(cost, a, b, eps=0.15, max_iter=300, tol=1e-8)

    assert plan.shape == cost.shape
    assert torch.isfinite(plan).all()
    assert torch.all(plan >= 0.0)
    assert torch.allclose(plan.sum(dim=-1), a, atol=2e-4, rtol=2e-4)
    assert torch.allclose(plan.sum(dim=-2), b, atol=2e-4, rtol=2e-4)
    assert torch.allclose(compute_transported_mass(plan), torch.ones(2, 3), atol=2e-4, rtol=2e-4)


def test_native_balanced_sinkhorn_matches_pot_entropy_solver_batched():
    pytest.importorskip("ot")
    torch.manual_seed(202)
    cost = torch.rand(2, 4, 6)
    a = _positive_simplex((2, 4))
    b = _positive_simplex((2, 6))

    native = sinkhorn_balanced_log(cost, a, b, eps=0.2, max_iter=400, tol=1e-9)
    pot = sinkhorn_balanced_pot(cost, a, b, eps=0.2, max_iter=400, tol=1e-9)

    assert torch.allclose(native, pot, atol=2e-3, rtol=2e-2)


def test_native_unbalanced_sinkhorn_matches_pot_entropy_reg_type_batched():
    pytest.importorskip("ot")
    torch.manual_seed(203)
    cost = torch.rand(2, 3, 4, 5)
    a = 0.7 * _positive_simplex((2, 3, 4))
    b = 0.5 * _positive_simplex((2, 3, 5))

    native = sinkhorn_unbalanced_log(
        cost,
        a,
        b,
        tau_q=0.6,
        tau_c=0.8,
        eps=0.12,
        max_iter=400,
        tol=1e-9,
    )
    pot = sinkhorn_unbalanced_pot(
        cost,
        a,
        b,
        tau_q=0.6,
        tau_c=0.8,
        eps=0.12,
        max_iter=400,
        tol=1e-9,
    )

    assert torch.allclose(native, pot, atol=3e-3, rtol=3e-2)


def test_native_unbalanced_sinkhorn_recovers_balanced_limit_for_large_tau():
    torch.manual_seed(204)
    cost = torch.rand(3, 5)
    a = _positive_simplex((3,))
    b = _positive_simplex((5,))

    plan = sinkhorn_unbalanced_log(
        cost,
        a,
        b,
        tau_q=1e6,
        tau_c=1e6,
        eps=0.2,
        max_iter=300,
        tol=1e-8,
    )

    assert torch.allclose(plan.sum(dim=-1), a, atol=2e-4, rtol=2e-4)
    assert torch.allclose(plan.sum(dim=-2), b, atol=2e-4, rtol=2e-4)


def test_unbalanced_objective_penalizes_dropped_mass():
    cost = torch.zeros(1, 2, 2)
    a = torch.tensor([[0.5, 0.5]])
    b = torch.tensor([[0.5, 0.5]])
    full_plan = torch.tensor([[[0.5, 0.0], [0.0, 0.5]]])
    dropped_plan = torch.tensor([[[0.05, 0.0], [0.0, 0.05]]])

    full_objective = compute_unbalanced_transport_objective(
        full_plan,
        cost,
        a,
        b,
        tau_q=0.5,
        tau_c=0.5,
    )
    dropped_objective = compute_unbalanced_transport_objective(
        dropped_plan,
        cost,
        a,
        b,
        tau_q=0.5,
        tau_c=0.5,
    )

    assert torch.allclose(full_objective, torch.zeros_like(full_objective), atol=1e-6)
    assert torch.all(dropped_objective > full_objective)
