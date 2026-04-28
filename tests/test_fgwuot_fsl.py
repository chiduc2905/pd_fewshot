"""
Behavioral tests for FGWUOTFewShot.
Run: pytest tests/test_fgwuot_fsl.py -v
"""

import math
import pytest
import torch
import torch.nn as nn

from net.fgwuot_fsl import FGWUOTFewShot
from net.modules.fgw_uot_solver import (
    pairwise_sq_l2,
    normalize_intra_dist,
    sinkhorn_uot_log,
    fgw_uot_solve,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return dict(
        in_channels=3,
        hidden_dim=64,
        token_dim=64,
        backbone_name="conv64f",
        image_size=84,
        tau=0.5,
        eps_sinkhorn=0.1,
        fgw_iters=3,        # small for tests
        sinkhorn_iters=20,
        alpha_init=0.5,
        score_scale_init=4.0,
        rho_head_hidden=16,
        lambda_rho=0.01,
        rho_target=0.8,
        normalize_tokens=True,
        mass_mode="reliability",
        reliability_mix=0.65,
        reliability_temperature=0.25,
        support_mode="shotwise",
        shot_aggregation="softmin",
        shot_softmin_beta=8.0,
        structure_prior_weight=0.12,
        score_mode="radius_margin",
        radius_alpha=0.5,
        radius_floor=0.02,
    )


@pytest.fixture
def model(cfg):
    m = FGWUOTFewShot(**cfg)
    m.eval()
    return m


def make_episode(way=4, shot=1, nq=1, img_size=84, seed=0):
    torch.manual_seed(seed)
    query   = torch.randn(nq, 3, img_size, img_size)
    support = torch.randn(way, shot, 3, img_size, img_size)
    q_tgt   = torch.zeros(nq, dtype=torch.long)
    s_tgt   = torch.arange(way).unsqueeze(1).expand(way, shot)
    return query, support, q_tgt, s_tgt


# ── Solver unit tests ────────────────────────────────────────────────────────

class TestSolverPrimitives:

    def test_pairwise_sq_l2_non_negative(self):
        A = torch.randn(4, 10, 32)
        B = torch.randn(4, 15, 32)
        D = pairwise_sq_l2(A, B)
        assert D.shape == (4, 10, 15)
        assert (D >= 0).all(), "Squared distances must be non-negative"

    def test_pairwise_sq_l2_self_is_zero(self):
        A = torch.randn(2, 8, 16)
        D = pairwise_sq_l2(A, A)
        diag = torch.diagonal(D, dim1=-2, dim2=-1)
        assert diag.abs().max().item() < 1e-4, "Self-distance should be ~0"

    def test_normalize_intra_dist_mean_one(self):
        D = torch.rand(3, 10, 10).abs()
        D_n = normalize_intra_dist(D)
        means = D_n.mean(dim=(-1, -2))
        assert torch.allclose(means, torch.ones_like(means), atol=1e-5)

    def test_sinkhorn_uot_log_plan_positive(self):
        B, N, M = 2, 8, 10
        cost  = torch.rand(B, N, M)
        log_a = torch.full((B, N), -math.log(N))
        log_b = torch.full((B, M), -math.log(M))
        P = sinkhorn_uot_log(cost, log_a, log_b, tau=0.5, eps=0.1, max_iter=30)
        assert P.shape == (B, N, M)
        assert (P >= 0).all(), "Transport plan must be non-negative"

    def test_sinkhorn_uot_balanced_limit(self):
        """With large tau, UOT approaches balanced OT (marginals satisfied)."""
        B, N, M = 2, 6, 6
        cost  = torch.rand(B, N, M)
        log_a = torch.full((B, N), -math.log(N))
        log_b = torch.full((B, M), -math.log(M))
        P = sinkhorn_uot_log(cost, log_a, log_b, tau=1000.0, eps=0.01, max_iter=200)
        row_sums = P.sum(dim=-1)
        col_sums = P.sum(dim=-2)
        a = torch.full((B, N), 1.0 / N)
        b = torch.full((B, M), 1.0 / M)
        assert torch.allclose(row_sums, a, atol=1e-3)
        assert torch.allclose(col_sums, b, atol=1e-3)

    def test_fgw_uot_solve_shapes(self):
        B, Tq, Ts = 3, 25, 25
        C_feat = torch.rand(B, Tq, Ts)
        D_q    = normalize_intra_dist(pairwise_sq_l2(
            torch.randn(B, Tq, 32), torch.randn(B, Tq, 32)))
        D_s    = normalize_intra_dist(pairwise_sq_l2(
            torch.randn(B, Ts, 32), torch.randn(B, Ts, 32)))
        log_a  = torch.full((B, Tq), -math.log(Tq))
        log_b  = torch.full((B, Ts), -math.log(Ts))
        alpha  = torch.tensor(0.5)
        P, C_final = fgw_uot_solve(
            C_feat, D_q, D_s, log_a, log_b, alpha,
            tau=0.5, eps=0.1, fgw_iters=4, sinkhorn_iters=20)
        assert P.shape     == (B, Tq, Ts)
        assert C_final.shape == (B, Tq, Ts)
        assert (P >= 0).all()

    def test_fgw_reduces_to_uot_when_alpha_zero(self):
        """When alpha=0, FGW cost equals appearance cost exactly."""
        B, Tq, Ts = 2, 10, 10
        C_feat = torch.rand(B, Tq, Ts)
        D_q    = normalize_intra_dist(torch.rand(B, Tq, Tq))
        D_s    = normalize_intra_dist(torch.rand(B, Ts, Ts))
        log_a  = torch.full((B, Tq), -math.log(Tq))
        log_b  = torch.full((B, Ts), -math.log(Ts))
        alpha  = torch.tensor(0.0)
        _, C_final = fgw_uot_solve(
            C_feat, D_q, D_s, log_a, log_b, alpha,
            tau=0.5, eps=0.1, fgw_iters=4, sinkhorn_iters=20)
        assert torch.allclose(C_final, C_feat, atol=1e-5), \
            "alpha=0 must yield C_final == C_feat"


# ── Model forward tests ──────────────────────────────────────────────────────

class TestFGWUOTModel:

    def test_forward_shapes_1shot(self, model):
        way, shot, nq = 4, 1, 2
        q, s, qt, st = make_episode(way=way, shot=shot, nq=nq)
        logits = model(q, s)
        assert logits.shape == (nq, way), f"Expected ({nq},{way}), got {logits.shape}"
        assert torch.isfinite(logits).all()

    def test_forward_shapes_5shot(self, model):
        way, shot, nq = 4, 5, 1
        q, s, qt, st = make_episode(way=way, shot=shot, nq=nq)
        logits = model(q, s)
        assert logits.shape == (nq, way)

    def test_forward_batched_episodes(self, model):
        way, shot, nq, B = 4, 1, 2, 3
        torch.manual_seed(42)
        query   = torch.randn(B, nq, 3, 84, 84)
        support = torch.randn(B, way, shot, 3, 84, 84)
        logits = model(query, support)
        assert logits.shape == (B * nq, way)

    def test_training_output_is_dict_for_main_loss(self, model):
        way, shot, nq, B = 4, 1, 2, 2
        model.train()
        torch.manual_seed(43)
        query = torch.randn(B, nq, 3, 84, 84)
        support = torch.randn(B, way, shot, 3, 84, 84)
        query_targets = torch.zeros(B * nq, dtype=torch.long)
        support_targets = torch.arange(way).view(1, way, 1).expand(B, way, shot)
        outputs = model(query, support, query_targets=query_targets, support_targets=support_targets)
        assert isinstance(outputs, dict)
        assert outputs["logits"].shape == (B * nq, way)
        assert outputs["aux_loss"].ndim == 0

    def test_return_aux_keys(self, model):
        q, s, qt, st = make_episode()
        aux_dict = model(q, s, return_aux=True)
        required_keys = {
            "logits", "aux_loss", "rho", "alpha", "score_scale",
            "transport_cost", "transport_plan", "C_feat", "C_final",
            "query_class_distance", "transport_radius", "query_token_mass",
            "support_token_mass", "shot_aggregation_weights",
        }
        for k in required_keys:
            assert k in aux_dict, f"Missing key in aux_dict: {k}"

    def test_score_matches_configured_distance_rule(self, model):
        """Verify the configured FGWUOT distance-to-logit rule."""
        q, s, qt, st = make_episode(way=4, shot=1, nq=1)
        with torch.no_grad():
            aux = model(q, s, return_aux=True)

        distance = aux["query_class_distance"]
        radius = aux["transport_radius"]
        ss  = aux["score_scale"]            # scalar
        logits = aux["logits"]              # (NQ, Way)

        if model.score_mode == "negative_distance":
            expected = ss * (-distance)
        elif model.score_mode == "radius_margin":
            expected = ss * (-(distance - radius))
        else:
            expected = ss * (-torch.relu(distance - radius))
        assert torch.allclose(logits, expected, atol=1e-4), \
            "Logit formula must match configured FGWUOT score_mode"

    def test_alpha_zero_matches_uot_only(self, model):
        """Setting alpha=0 should match pure UOT (no structure cost)."""
        with torch.no_grad():
            model.raw_alpha.fill_(-100.0)   # sigmoid(-100) ≈ 0
            q, s, _, _ = make_episode()
            aux = model(q, s, return_aux=True)
            C_feat  = aux["C_feat"]
            C_final = aux["C_final"]
        assert torch.allclose(C_feat, C_final, atol=1e-5), \
            "alpha≈0: C_final must equal C_feat"

    def test_backprop_all_parameters(self, model):
        """All parameters must receive non-zero finite gradients."""
        model.train()
        q, s, qt, st = make_episode()
        outputs = model(q, s, qt, st)
        loss = outputs["logits"].mean() + outputs["aux_loss"]
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Inf/NaN grad for {name}"

    def test_structure_cost_changes_plan(self, model):
        """alpha > 0 must change the transport plan vs alpha = 0."""
        model.eval()
        q, s, _, _ = make_episode(seed=1)

        with torch.no_grad():
            # alpha ≈ 0
            model.raw_alpha.fill_(-100.0)
            aux0 = model(q, s, return_aux=True)
            P0 = aux0["transport_plan"].clone()

            # alpha ≈ 0.8
            model.raw_alpha.fill_(1.386)  # sigmoid(1.386) ≈ 0.8
            aux1 = model(q, s, return_aux=True)
            P1 = aux1["transport_plan"].clone()

        assert not torch.allclose(P0, P1, atol=1e-4), \
            "Structure term (alpha>0) must change the transport plan"

    def test_rho_in_range(self, model):
        """Rho must be in (0, 1) for all pairs."""
        q, s, _, _ = make_episode()
        with torch.no_grad():
            aux = model(q, s, return_aux=True)
        rho = aux["rho"]
        assert (rho > 0).all() and (rho < 1).all(), \
            f"rho out of range: min={rho.min():.4f}, max={rho.max():.4f}"

    def test_rho_regularization_in_aux_loss(self, model):
        """aux_loss must equal lambda_rho * rho_reg."""
        q, s, _, _ = make_episode()
        model.train()
        aux = model(q, s, return_aux=True)
        expected_aux = model.lambda_rho * aux["rho_regularization"]
        assert torch.allclose(aux["aux_loss"], expected_aux, atol=1e-6)

    def test_shotwise_aux_shapes_5shot(self, model):
        q, s, _, _ = make_episode(way=4, shot=5, nq=2)
        with torch.no_grad():
            aux = model(q, s, return_aux=True)
        assert aux["shot_distance"].shape == (2, 4, 5)
        assert aux["shot_aggregation_weights"].shape == (2, 4, 5)
        assert aux["transport_plan"].dim() == 5
        assert torch.allclose(
            aux["shot_aggregation_weights"].sum(dim=-1),
            torch.ones(2, 4),
            atol=1e-5,
        )
