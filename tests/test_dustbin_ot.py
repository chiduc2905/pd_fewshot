from __future__ import annotations

import torch

from net.modules.dustbin_ot import (
    dustbin_transport_diagnostics,
    dustbin_transport_from_cost,
)


def test_dustbin_ot_shapes_residuals_and_gradients():
    torch.manual_seed(941)
    cost = torch.randn(2, 3, 5, 4)
    threshold = torch.tensor(0.25, requires_grad=True)
    alpha = torch.tensor(0.0, requires_grad=True)

    out = dustbin_transport_from_cost(
        cost,
        threshold,
        alpha,
        rho=0.8,
        temperature=0.07,
        max_iter=300,
        tol=1e-8,
        return_augmented=True,
    )
    plan = out["plan"]
    assert plan.shape == cost.shape
    assert out["augmented_plan"].shape == (2, 3, 6, 5)

    diag = dustbin_transport_diagnostics(
        plan,
        out["augmented_plan"],
        out["augmented_a"],
        out["augmented_b"],
        out["scores"],
        rho=0.8,
    )
    assert diag["row_residual"].max().item() < 1e-5
    assert diag["column_residual"].max().item() < 1e-5
    assert diag["mass_residual"].max().item() < 1e-5

    loss = out["accepted_score"].sum()
    loss.backward()
    assert threshold.grad is not None
    assert alpha.grad is not None
    assert torch.isfinite(threshold.grad)
    assert torch.isfinite(alpha.grad)


def test_dustbin_ot_accepts_positive_utility_and_rejects_negative_utility():
    low_cost = torch.full((1, 4, 4), -1.0)
    high_cost = torch.full((1, 4, 4), 1.0)

    low = dustbin_transport_from_cost(
        low_cost,
        threshold=0.0,
        alpha=0.0,
        rho=0.8,
        temperature=0.03,
        max_iter=300,
        tol=1e-8,
    )
    high = dustbin_transport_from_cost(
        high_cost,
        threshold=0.0,
        alpha=0.0,
        rho=0.8,
        temperature=0.03,
        max_iter=300,
        tol=1e-8,
    )

    assert low["real_mass"].item() > 0.75
    assert high["real_mass"].item() < 0.05


def test_dustbin_ot_score_identity_matches_threshold_mass_minus_cost():
    torch.manual_seed(942)
    cost = torch.rand(2, 4, 4)
    threshold = torch.tensor(0.35, requires_grad=True)
    out = dustbin_transport_from_cost(
        cost,
        threshold,
        alpha=0.0,
        rho=0.8,
        temperature=0.05,
        max_iter=300,
        tol=1e-8,
    )

    score = threshold * out["real_mass"] - out["transport_cost"]
    assert torch.allclose(out["accepted_score"], score, atol=1e-6, rtol=1e-6)
    score.sum().backward()
    assert threshold.grad is not None
    assert torch.isfinite(threshold.grad)


def test_dustbin_alpha_is_a_rejection_boundary():
    cost = torch.full((1, 5, 5), -0.4)
    low_alpha = dustbin_transport_from_cost(
        cost,
        threshold=0.0,
        alpha=0.0,
        rho=0.8,
        temperature=0.04,
        max_iter=300,
        tol=1e-8,
    )
    high_alpha = dustbin_transport_from_cost(
        cost,
        threshold=0.0,
        alpha=0.8,
        rho=0.8,
        temperature=0.04,
        max_iter=300,
        tol=1e-8,
    )

    assert low_alpha["real_mass"].item() > 0.75
    assert high_alpha["real_mass"].item() < 0.05
