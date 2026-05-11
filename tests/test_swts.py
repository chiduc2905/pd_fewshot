"""Unit tests for Self-Weighted Transport Score (SWTS) on J-ECOT-M2."""

from __future__ import annotations

import pytest
import torch

from net.hrot_fsl import HROTFSL

from tests.test_hrot_fsl import _build_model


def test_swts_recovers_neg_cost_when_balanced():
    """When P has uniform row sums, SWTS == -C."""
    Lq, Ls = 25, 25
    D = torch.rand(2, 5, 3, 1, Lq, Ls)
    rho = 0.8
    P = torch.full_like(D, rho / (Lq * Ls))
    C = (P * D).sum(dim=(-2, -1))
    swts = HROTFSL._ecot_swts_mass_removed_score_terms(P, D, 1.0, 1e-6)
    assert torch.allclose(swts, -C, atol=1e-5), "SWTS must equal -C for uniform plan"


def test_swts_upweights_good_tokens():
    """Token with higher query marginal q(r) gets higher softmax weight w(r)."""
    P2 = torch.zeros(1, 1, 1, 1, 2, 2)
    P2[..., 0, 0] = 0.6
    P2[..., 1, 1] = 0.2
    q2 = P2.sum(dim=-1).squeeze()
    w2 = torch.softmax(q2 / 1.0, dim=-1)
    assert w2[0] > w2[1], "Token with higher marginal should get higher weight"


def test_swts_large_temp_matches_neg_cost_random_plan():
    """As softmax temperature -> inf, w -> uniform and SWTS -> -C for any fixed P."""
    torch.manual_seed(0)
    Lq, Ls = 8, 10
    P = torch.rand(1, 2, 1, 1, Lq, Ls)
    P = P / P.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-9)
    D = torch.rand_like(P)
    C = (P * D).sum(dim=(-2, -1))
    swts = HROTFSL._ecot_swts_mass_removed_score_terms(P, D, 1e4, 1e-8)
    assert torch.allclose(swts, -C, atol=1e-4)


def test_swts_rejected_when_cost_per_mass_enabled():
    with pytest.raises(ValueError, match="ecot_m2_use_swts"):
        _build_model(
            variant="J_ECOT_M2",
            ecot_m2_use_swts=True,
            ecot_m2_ablate_threshold_mass=False,
            ecot_m2_cost_per_mass_score=True,
        )


def test_swts_rejected_when_threshold_mass_not_ablated():
    with pytest.raises(ValueError, match="ecot_m2_use_swts"):
        _build_model(
            variant="J_ECOT_M2",
            ecot_m2_use_swts=True,
            ecot_m2_ablate_threshold_mass=False,
            ecot_m2_cost_per_mass_score=False,
        )
