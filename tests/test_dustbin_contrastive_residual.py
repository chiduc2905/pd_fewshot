from __future__ import annotations

import torch

from net.modules.dustbin_contrastive_residual import DustbinContrastiveResidualScore


def test_dcr_preserves_negative_evidence():
    module = DustbinContrastiveResidualScore(beta=0.7, tau=0.25)
    cost = torch.full((1, 2, 1, 2, 2), 1.0)
    plan = torch.full_like(cost, 0.25)

    score, diagnostics = module(cost=cost, plan=plan, threshold=0.0)

    expected = (plan * (0.0 - cost)).sum(dim=(-1, -2))
    assert torch.allclose(score, expected, atol=1e-6, rtol=1e-6)
    assert diagnostics["dcr_removed_positive_shot"].abs().max().item() == 0.0
    assert diagnostics["dcr/removed_positive_mean"].item() == 0.0


def test_dcr_suppresses_non_specific_positive_evidence_more():
    module = DustbinContrastiveResidualScore(
        beta=1.0,
        tau=0.20,
        margin=0.0,
        min_gate=0.0,
    )
    cost = torch.ones(1, 2, 1, 2, 2)
    cost[0, 0, 0, 0, 0] = -1.00
    cost[0, 1, 0, 0, 0] = -0.70
    plan = torch.zeros_like(cost)
    plan[0, 0, 0, 0, 0] = 1.0
    plan[0, 1, 0, 0, 0] = 1.0

    score, diagnostics = module(cost=cost, plan=plan, threshold=0.0)
    removed_share = diagnostics["dcr_removed_positive_share_shot"]
    removed = diagnostics["dcr_removed_positive_shot"]

    assert removed_share[0, 1, 0] > removed_share[0, 0, 0]
    assert removed[0, 1, 0] > removed[0, 0, 0]
    assert score[0, 1, 0] < (plan * (0.0 - cost)).sum(dim=(-1, -2))[0, 1, 0]


def test_dcr_gradients_are_finite():
    torch.manual_seed(845)
    module = DustbinContrastiveResidualScore(beta=0.5, tau=0.25, detach_gate=True)
    cost = torch.randn(2, 3, 2, 3, 4, requires_grad=True)
    raw_plan = torch.randn(2, 3, 2, 3, 4, requires_grad=True)
    plan = raw_plan.softmax(dim=-1)
    threshold = torch.tensor(0.1, requires_grad=True)

    score, diagnostics = module(cost=cost, plan=plan, threshold=threshold)
    loss = score.sum() + diagnostics["dcr/shot_score_delta"]
    loss.backward()

    assert cost.grad is not None
    assert raw_plan.grad is not None
    assert threshold.grad is not None
    assert torch.isfinite(cost.grad).all()
    assert torch.isfinite(raw_plan.grad).all()
    assert torch.isfinite(threshold.grad)
