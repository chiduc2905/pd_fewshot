"""Tests for the fast augmented Sinkhorn partial OT solver."""

from __future__ import annotations

import time

import pytest
import torch

from net.modules.fast_partial_ot import (
    fast_partial_wasserstein,
    fast_partial_wasserstein_nesterov,
)
from net.modules.partial_ot import (
    compute_partial_transport_cost,
    compute_partial_transported_mass,
    entropic_partial_wasserstein,
    partial_transport_plan_residuals,
    resolve_partial_transport_mass,
    solve_partial_transport,
)


def _positive_histogram(shape):
    return torch.rand(*shape) + 0.05


class TestFastPOTFeasibility:
    def test_row_column_mass_constraints(self):
        torch.manual_seed(42)
        cost = torch.rand(2, 3, 8, 10)
        a = _positive_histogram((2, 3, 8))
        b = _positive_histogram((2, 3, 10))
        mass = 0.5 * torch.minimum(a.sum(dim=-1), b.sum(dim=-1))

        plan = fast_partial_wasserstein(cost, a, b, mass, reg=0.1, max_iter=40)

        assert plan.shape == cost.shape
        assert torch.isfinite(plan).all()
        assert torch.all(plan >= 0.0)
        assert torch.all(plan.sum(dim=-1) <= a + 5e-4)
        assert torch.all(plan.sum(dim=-2) <= b + 5e-4)
        assert torch.allclose(
            compute_partial_transported_mass(plan), mass, atol=5e-4, rtol=5e-4
        )

    def test_nesterov_variant_feasibility(self):
        torch.manual_seed(43)
        cost = torch.rand(4, 6, 8)
        a = _positive_histogram((4, 6))
        b = _positive_histogram((4, 8))
        mass = 0.6 * torch.minimum(a.sum(dim=-1), b.sum(dim=-1))

        plan = fast_partial_wasserstein_nesterov(
            cost, a, b, mass, reg=0.1, max_iter=30
        )

        assert torch.all(plan >= 0.0)
        assert torch.all(plan.sum(dim=-1) <= a + 5e-4)
        assert torch.all(plan.sum(dim=-2) <= b + 5e-4)
        assert torch.allclose(
            compute_partial_transported_mass(plan), mass, atol=5e-4, rtol=5e-4
        )


class TestFastPOTQuality:
    def test_cost_close_to_native(self):
        torch.manual_seed(44)
        cost = torch.rand(6, 8)
        a = _positive_histogram((6,))
        b = _positive_histogram((8,))
        mass = resolve_partial_transport_mass(a, b, mass_ratio=0.5)

        native_plan = entropic_partial_wasserstein(
            cost, a, b, transport_mass=mass, reg=0.1, max_iter=200, tol=1e-8
        )
        fast_plan = fast_partial_wasserstein(
            cost, a, b, mass, reg=0.1, max_iter=50
        )

        native_cost = compute_partial_transport_cost(native_plan, cost)
        fast_cost = compute_partial_transport_cost(fast_plan, cost)

        # Fast should produce similar cost (within 10%)
        relative_diff = (fast_cost - native_cost).abs() / native_cost.clamp_min(1e-6)
        assert relative_diff < 0.10, f"Cost diff too large: {relative_diff:.4f}"

    def test_cost_close_to_native_batched(self):
        torch.manual_seed(45)
        cost = torch.rand(3, 6, 8)
        a = _positive_histogram((3, 6))
        b = _positive_histogram((3, 8))
        mass = resolve_partial_transport_mass(a, b, mass_ratio=0.6)

        native_plan = entropic_partial_wasserstein(
            cost, a, b, transport_mass=mass, reg=0.1, max_iter=200, tol=1e-8
        )
        fast_plan = fast_partial_wasserstein(
            cost, a, b, mass, reg=0.1, max_iter=50
        )

        native_cost = compute_partial_transport_cost(native_plan, cost)
        fast_cost = compute_partial_transport_cost(fast_plan, cost)

        relative_diff = (fast_cost - native_cost).abs() / native_cost.clamp_min(1e-6)
        assert torch.all(relative_diff < 0.15)


class TestFastPOTGradient:
    def test_backward_pass(self):
        torch.manual_seed(46)
        features_q = torch.randn(6, 4, requires_grad=True)
        features_s = torch.randn(8, 4, requires_grad=True)
        cost = torch.cdist(features_q, features_s).pow(2.0)
        a = torch.full((6,), 1.0 / 6.0)
        b = torch.full((8,), 1.0 / 8.0)
        mass = torch.tensor(0.5 * min(a.sum().item(), b.sum().item()))

        plan = fast_partial_wasserstein(cost, a, b, mass, reg=0.1, max_iter=40)
        loss = compute_partial_transport_cost(plan, cost)
        loss.backward()

        assert features_q.grad is not None
        assert features_s.grad is not None
        assert torch.isfinite(features_q.grad).all()
        assert torch.isfinite(features_s.grad).all()

    def test_solve_dispatch_fast_backend(self):
        torch.manual_seed(47)
        features_q = torch.randn(4, 4, requires_grad=True)
        features_s = torch.randn(6, 4, requires_grad=True)
        cost = torch.cdist(features_q, features_s).pow(2.0)
        a = torch.full((4,), 1.0 / 4.0)
        b = torch.full((6,), 1.0 / 6.0)

        plan = solve_partial_transport(
            cost, a, b, transport_mass=0.4, backend="fast", reg=0.1, max_iter=40
        )
        loss = compute_partial_transport_cost(plan, cost)
        loss.backward()

        assert torch.isfinite(plan).all()
        assert torch.all(plan >= 0.0)
        assert features_q.grad is not None
        assert torch.isfinite(features_q.grad).all()

    def test_solve_dispatch_fast_nesterov_backend(self):
        torch.manual_seed(48)
        cost = torch.rand(4, 6)
        a = _positive_histogram((4,))
        b = _positive_histogram((6,))

        plan = solve_partial_transport(
            cost, a, b, mass_ratio=0.5, backend="fast_nesterov", reg=0.1, max_iter=30
        )

        assert torch.isfinite(plan).all()
        assert torch.all(plan >= 0.0)
        mass = resolve_partial_transport_mass(a, b, mass_ratio=0.5)
        residuals = partial_transport_plan_residuals(plan, a, b, mass)
        assert torch.all(residuals["row_violation"] <= 5e-4)
        assert torch.all(residuals["column_violation"] <= 5e-4)
        assert torch.all(residuals["mass_residual"] <= 5e-4)


class TestFastPOTSpeed:
    @pytest.mark.parametrize("batch,n,k", [(75, 25, 25), (75, 49, 49), (25, 49, 49)])
    def test_faster_than_native(self, batch, n, k):
        torch.manual_seed(50)
        cost = torch.rand(batch, n, k)
        a = _positive_histogram((batch, n))
        a = a / a.sum(-1, keepdim=True)
        b = _positive_histogram((batch, k))
        b = b / b.sum(-1, keepdim=True)
        mass = resolve_partial_transport_mass(a, b, mass_ratio=0.5)

        # Warmup
        fast_partial_wasserstein(cost, a, b, mass, reg=0.05, max_iter=50)
        entropic_partial_wasserstein(cost, a, b, transport_mass=mass, reg=0.05, max_iter=100)

        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            fast_partial_wasserstein(cost, a, b, mass, reg=0.05, max_iter=50)
        t_fast = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(reps):
            entropic_partial_wasserstein(
                cost, a, b, transport_mass=mass, reg=0.05, max_iter=100
            )
        t_native = time.perf_counter() - t0

        speedup = t_native / max(t_fast, 1e-9)
        print(f"  [{batch}x{n}x{k}] fast={t_fast:.4f}s native={t_native:.4f}s speedup={speedup:.2f}x")
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"
