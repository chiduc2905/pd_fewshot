from __future__ import annotations

import torch

from net.modules.elastic_ot import elastic_capacity_residuals, sinkhorn_elastic_log
from net.modules.unbalanced_ot import balanced_transport_plan_residuals


def _uniform_capacities(cost: torch.Tensor, mass: float = 0.8):
    a = cost.new_full(cost.shape[:-1], mass / float(cost.shape[-2]))
    b = cost.new_full(
        cost.shape[:-2] + (cost.shape[-1],),
        mass / float(cost.shape[-1]),
    )
    return a, b


def test_elastic_ot_respects_capacities_and_augmented_marginals():
    torch.manual_seed(731)
    cost = torch.randn(3, 5, 4)
    a, b = _uniform_capacities(cost)

    plan, augmented_plan, augmented_a, augmented_b = sinkhorn_elastic_log(
        cost,
        a,
        b,
        eps=0.08,
        max_iter=300,
        tol=1e-8,
        return_augmented=True,
    )

    capacity = elastic_capacity_residuals(plan, a, b)
    balanced = balanced_transport_plan_residuals(
        augmented_plan,
        augmented_a,
        augmented_b,
    )
    assert capacity["row_violation"].max().item() < 1e-5
    assert capacity["column_violation"].max().item() < 1e-5
    assert balanced["row_residual"].max().item() < 1e-5
    assert balanced["column_residual"].max().item() < 1e-5


def test_elastic_ot_matches_more_mass_when_edges_have_positive_utility():
    low_cost = torch.full((1, 4, 4), -0.4)
    high_cost = torch.full((1, 4, 4), 0.4)
    a, b = _uniform_capacities(low_cost)

    low_plan = sinkhorn_elastic_log(
        low_cost,
        a,
        b,
        eps=0.03,
        max_iter=300,
        tol=1e-8,
    )
    high_plan = sinkhorn_elastic_log(
        high_cost,
        a,
        b,
        eps=0.03,
        max_iter=300,
        tol=1e-8,
    )

    assert low_plan.sum().item() > 0.75
    assert high_plan.sum().item() < 0.05


def test_elastic_objective_is_exact_negative_threshold_mass_score():
    torch.manual_seed(732)
    original_cost = torch.rand(2, 4, 4)
    threshold = torch.tensor(0.35, requires_grad=True)
    shifted_cost = original_cost - threshold
    a, b = _uniform_capacities(original_cost)

    plan = sinkhorn_elastic_log(
        shifted_cost,
        a,
        b,
        eps=0.05,
        max_iter=300,
        tol=1e-8,
    )
    transport_cost = (plan * original_cost).sum(dim=(-1, -2))
    transported_mass = plan.sum(dim=(-1, -2))
    score = threshold * transported_mass - transport_cost
    objective = (plan * shifted_cost).sum(dim=(-1, -2))

    assert torch.allclose(score, -objective, atol=1e-6, rtol=1e-6)
    score.sum().backward()
    assert threshold.grad is not None
    assert torch.isfinite(threshold.grad)


def test_elastic_real_plan_is_invariant_to_augmented_slack_cost():
    torch.manual_seed(733)
    cost = torch.randn(2, 3, 5)
    a, b = _uniform_capacities(cost)

    low_sigma = sinkhorn_elastic_log(
        cost,
        a,
        b,
        eps=0.07,
        sigma=0.5,
        max_iter=300,
        tol=1e-8,
    )
    high_sigma = sinkhorn_elastic_log(
        cost,
        a,
        b,
        eps=0.07,
        sigma=3.0,
        max_iter=300,
        tol=1e-8,
    )

    assert torch.allclose(low_sigma, high_sigma, atol=1e-5, rtol=1e-5)
