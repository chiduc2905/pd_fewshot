from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from net.hrot_fsl import HROTFSL
from net.hyperbolic.poincare_ops import frechet_mean_poincare, get_ball, hyperbolic_distance_matrix, safe_project_to_ball
from net.model_factory import build_model_from_args
from net.modules.episode_adaptive_mass import EpisodeAdaptiveMass
from net.modules.unbalanced_ot import (
    sinkhorn_balanced_log,
    sinkhorn_balanced_pot,
    sinkhorn_unbalanced_log,
    sinkhorn_unbalanced_pot,
)


def _build_model(**overrides) -> HROTFSL:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=24,
        variant="E",
        eam_hidden_dim=32,
        curvature_init=1.0,
        projection_scale=0.1,
        token_temperature=0.1,
        score_scale=8.0,
        tau_q=0.5,
        tau_c=0.5,
        sinkhorn_epsilon=0.1,
        sinkhorn_iterations=12,
        sinkhorn_tolerance=1e-5,
        fixed_mass=0.8,
        min_mass=0.1,
        mass_bonus_init=1.0,
        lambda_rho=0.05,
        rho_target=0.8,
        lambda_rho_rank=0.05,
        rho_rank_margin=0.05,
        rho_rank_temperature=0.05,
        lambda_curvature=0.01,
        min_curvature=0.05,
        normalize_euclidean_tokens=True,
        eval_use_float64=True,
        hyperbolic_backend="auto",
        ot_backend="native",
        eps=1e-6,
    )
    kwargs.update(overrides)
    return HROTFSL(**kwargs)


def test_hyperbolic_distance_and_frechet_mean_basic_properties():
    torch.manual_seed(0)
    ball = get_ball(1.0)
    points = safe_project_to_ball(torch.randn(6, 12) * 0.1, ball)
    pairwise = hyperbolic_distance_matrix(points, points, ball)
    mean = frechet_mean_poincare(points, ball)
    singleton = frechet_mean_poincare(points[:1], ball)

    assert pairwise.shape == (6, 6)
    assert torch.allclose(torch.diagonal(pairwise), torch.zeros(6), atol=1e-5, rtol=0.0)
    assert torch.isfinite(pairwise).all()
    assert mean.shape == (12,)
    assert singleton.shape == (12,)
    assert torch.allclose(singleton, points[0], atol=1e-5, rtol=0.0)


def test_geoopt_and_native_hyperbolic_distance_are_consistent():
    torch.manual_seed(11)
    native_ball = get_ball(1.0, backend="native")
    geoopt_ball = get_ball(1.0, backend="geoopt")
    tangent = torch.randn(5, 8) * 0.1

    native_points = safe_project_to_ball(tangent, native_ball)
    geoopt_points = safe_project_to_ball(tangent, geoopt_ball)
    native_dist = hyperbolic_distance_matrix(native_points, native_points, native_ball)
    geoopt_dist = hyperbolic_distance_matrix(geoopt_points, geoopt_points, geoopt_ball)

    assert torch.allclose(native_points, geoopt_points, atol=1e-5, rtol=1e-4)
    assert torch.allclose(native_dist, geoopt_dist, atol=1e-5, rtol=1e-4)


def test_episode_adaptive_mass_range_and_pairwise_shape():
    torch.manual_seed(1)
    ball = get_ball(1.0)
    query_tokens = safe_project_to_ball(torch.randn(4, 9, 16) * 0.1, ball)
    class_tokens = safe_project_to_ball(torch.randn(3, 15, 16) * 0.1, ball)
    eam = EpisodeAdaptiveMass(embed_dim=16, hidden_dim=32, min_mass=0.1, default_mass=0.8)

    rho = eam(query_tokens, class_tokens, ball)
    single = eam(query_tokens[0], class_tokens[0], ball)

    assert rho.shape == (4, 3)
    assert torch.all(rho >= 0.1)
    assert torch.all(rho <= 1.0)
    assert single.ndim == 0
    assert 0.1 <= float(single.detach()) <= 1.0


def test_sinkhorn_solvers_return_finite_nonnegative_plans():
    torch.manual_seed(2)
    cost = torch.rand(5, 7)
    a = torch.full((5,), 1.0 / 5.0)
    b = torch.full((7,), 1.0 / 7.0)

    balanced = sinkhorn_balanced_log(cost, a, b, eps=0.2, max_iter=30)
    unbalanced = sinkhorn_unbalanced_log(cost, 0.8 * a, 0.8 * b, tau_q=0.5, tau_c=0.5, eps=0.2, max_iter=30)

    assert balanced.shape == (5, 7)
    assert unbalanced.shape == (5, 7)
    assert torch.isfinite(balanced).all()
    assert torch.isfinite(unbalanced).all()
    assert torch.all(balanced >= 0.0)
    assert torch.all(unbalanced >= 0.0)
    assert torch.allclose(balanced.sum(dim=-1), a, atol=1e-3, rtol=0.0)
    assert torch.allclose(balanced.sum(dim=-2), b, atol=1e-3, rtol=0.0)


def test_native_and_pot_sinkhorn_backends_agree_on_small_problems():
    torch.manual_seed(12)
    cost = torch.rand(4, 6)
    a = torch.full((4,), 1.0 / 4.0)
    b = torch.full((6,), 1.0 / 6.0)

    balanced_native = sinkhorn_balanced_log(cost, a, b, eps=0.2, max_iter=80, tol=1e-7)
    balanced_pot = sinkhorn_balanced_pot(cost, a, b, eps=0.2, max_iter=80, tol=1e-7)
    unbalanced_native = sinkhorn_unbalanced_log(cost, 0.8 * a, 0.8 * b, tau_q=0.5, tau_c=0.7, eps=0.2, max_iter=80, tol=1e-7)
    unbalanced_pot = sinkhorn_unbalanced_pot(cost, 0.8 * a, 0.8 * b, tau_q=0.5, tau_c=0.7, eps=0.2, max_iter=80, tol=1e-7)

    assert torch.allclose(balanced_native, balanced_pot, atol=2e-3, rtol=2e-2)
    assert torch.allclose(unbalanced_native, unbalanced_pot, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("variant", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"])
def test_hrot_fsl_forward_shapes_and_variants(variant: str):
    torch.manual_seed(3)
    model = _build_model(variant=variant)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["class_scores"].shape == (2, 3)
    assert outputs["total_distance"].shape == (2, 3)
    assert outputs["transport_cost"].shape == (2, 3)
    assert outputs["transported_mass"].shape == (2, 3)
    if variant in {"G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"}:
        assert outputs["rho"].shape == (2, 3, 2)
    else:
        assert outputs["rho"].shape == (2, 3)
    assert outputs["transport_plan"].shape[:2] == (2, 3)
    assert outputs["cost_matrix"].shape == outputs["transport_plan"].shape
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["transport_plan"]).all()
    if variant in {"A", "C"}:
        assert torch.allclose(outputs["rho"], torch.ones_like(outputs["rho"]), atol=1e-6, rtol=0.0)
    else:
        assert torch.all(outputs["rho"] >= 0.1)
        assert torch.all(outputs["rho"] <= 1.0)


def test_hrot_fsl_variant_f_uses_euclidean_cost_with_learned_hyperbolic_mass():
    torch.manual_seed(13)
    model = _build_model(variant="F")
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    expected_cost = model._euclidean_cost(
        outputs["query_euclidean_tokens"],
        outputs["support_euclidean_tokens"].squeeze(0),
    )

    assert torch.allclose(outputs["cost_matrix"], expected_cost, atol=1e-5, rtol=1e-5)
    assert not model.uses_hyperbolic_geometry
    assert model.uses_unbalanced_transport
    assert model.uses_learned_mass
    assert outputs["rho_regularization"].item() >= 0.0


def test_hrot_fsl_variant_g_uses_shot_decomposed_euclidean_ot():
    torch.manual_seed(14)
    model = _build_model(variant="G")
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    flat_support = outputs["support_euclidean_tokens"].squeeze(0).reshape(6, -1, 24)
    expected_cost = model._euclidean_cost(outputs["query_euclidean_tokens"], flat_support)
    expected_cost = expected_cost.reshape(2, 3, 2, expected_cost.shape[-2], expected_cost.shape[-1])

    expected_logits = (
        -model.score_scale * outputs["shot_transport_cost"]
        + model.mass_bonus.detach().to(dtype=outputs["shot_transported_mass"].dtype) * outputs["shot_transported_mass"]
    ).mean(dim=-1)

    assert torch.allclose(outputs["cost_matrix"], expected_cost, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["logits"], expected_logits, atol=1e-5, rtol=1e-5)
    assert not model.uses_hyperbolic_geometry
    assert model.uses_unbalanced_transport
    assert model.uses_learned_mass
    assert model.uses_shot_decomposed_transport
    assert outputs["rho"].shape == (2, 3, 2)
    assert outputs["transport_plan"].shape[:3] == (2, 3, 2)
    assert outputs["cost_matrix"].shape == outputs["transport_plan"].shape


def test_hrot_fsl_variant_h_uses_geodesic_eam_with_shot_decomposed_score():
    torch.manual_seed(15)
    model = _build_model(variant="H")
    model.eval()
    captured_features = []

    original_forward_features = model.eam.forward_features

    def capture_forward_features(features):
        captured_features.append(features.detach().clone())
        return original_forward_features(features)

    model.eam.forward_features = capture_forward_features

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    flat_support = outputs["support_euclidean_tokens"].squeeze(0).reshape(6, -1, 24)
    expected_cost = model._euclidean_cost(outputs["query_euclidean_tokens"], flat_support)
    expected_cost = expected_cost.reshape(2, 3, 2, expected_cost.shape[-2], expected_cost.shape[-1])
    mass_reward = (model.score_scale * model.transport_cost_threshold.detach()).to(
        dtype=outputs["shot_transported_mass"].dtype
    )
    expected_logits = (
        -model.score_scale * outputs["shot_transport_cost"]
        + mass_reward * outputs["shot_transported_mass"]
    ).mean(dim=-1)

    assert torch.allclose(outputs["cost_matrix"], expected_cost, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        outputs["transport_cost"],
        outputs["shot_transport_cost"].mean(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(outputs["logits"], expected_logits, atol=1e-5, rtol=1e-5)
    assert not model.uses_hyperbolic_geometry
    assert model.uses_unbalanced_transport
    assert model.uses_learned_mass
    assert model.uses_shot_decomposed_transport
    assert model.uses_geodesic_eam
    assert model.uses_cost_threshold_score
    assert model.mass_bonus is None
    assert model.raw_transport_cost_threshold is not None
    assert torch.allclose(outputs["mass_bonus"], mass_reward, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        outputs["transport_cost_threshold"],
        model.transport_cost_threshold.detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    assert model.eam.network[0].in_features == 4
    assert len(captured_features) == 1
    assert captured_features[0].shape == (2, 3, 2, 4)
    assert torch.isfinite(captured_features[0]).all()
    assert torch.any(captured_features[0][..., 1] > 0.0)
    assert outputs["rho"].shape == (2, 3, 2)
    assert outputs["transport_plan"].shape[:3] == (2, 3, 2)
    assert outputs["cost_matrix"].shape == outputs["transport_plan"].shape


def test_hrot_fsl_variant_h_backpropagates_cost_threshold_and_geodesic_eam():
    torch.manual_seed(16)
    model = _build_model(variant="H", lambda_rho=0.1)
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)

    outputs = model(query, support, query_targets=targets)
    loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    model.zero_grad(set_to_none=True)
    loss.backward()

    assert model.mass_bonus is None
    assert model.raw_transport_cost_threshold is not None
    assert model.raw_transport_cost_threshold.grad is not None
    assert model.eam.network[-2].weight.grad is not None
    assert torch.isfinite(model.raw_transport_cost_threshold.grad).all()
    assert torch.isfinite(model.eam.network[-2].weight.grad).all()
    assert model.eam.network[-2].weight.grad.norm().item() > 0.0


@pytest.mark.parametrize(
    ("variant", "feature_dim", "uses_euclidean_eam", "uses_geodesic_eam", "uses_threshold", "uses_learned_mass"),
    [
        ("I", 4, True, False, True, True),
        ("J", None, False, False, True, False),
        ("K", 4, False, True, False, True),
        ("L", 3, False, True, True, True),
    ],
)
def test_hrot_fsl_post_h_ablation_variants(
    variant: str,
    feature_dim: int | None,
    uses_euclidean_eam: bool,
    uses_geodesic_eam: bool,
    uses_threshold: bool,
    uses_learned_mass: bool,
):
    torch.manual_seed(17)
    model = _build_model(variant=variant)
    model.eval()
    captured_features = []

    original_forward_features = model.eam.forward_features

    def capture_forward_features(features):
        captured_features.append(features.detach().clone())
        return original_forward_features(features)

    model.eam.forward_features = capture_forward_features

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    flat_support = outputs["support_euclidean_tokens"].squeeze(0).reshape(6, -1, 24)
    expected_cost = model._euclidean_cost(outputs["query_euclidean_tokens"], flat_support)
    expected_cost = expected_cost.reshape(2, 3, 2, expected_cost.shape[-2], expected_cost.shape[-1])
    reward_weight = (
        (model.score_scale * model.transport_cost_threshold.detach())
        if uses_threshold
        else model.mass_bonus.detach()
    ).to(dtype=outputs["shot_transported_mass"].dtype)
    expected_logits = (
        -model.score_scale * outputs["shot_transport_cost"]
        + reward_weight * outputs["shot_transported_mass"]
    ).mean(dim=-1)

    assert torch.allclose(outputs["cost_matrix"], expected_cost, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["logits"], expected_logits, atol=1e-5, rtol=1e-5)
    assert not model.uses_hyperbolic_geometry
    assert model.uses_unbalanced_transport
    assert model.uses_learned_mass is uses_learned_mass
    assert model.uses_shot_decomposed_transport
    assert model.uses_euclidean_geometric_eam is uses_euclidean_eam
    assert model.uses_geodesic_eam is uses_geodesic_eam
    assert model.uses_cost_threshold_score is uses_threshold

    if uses_threshold:
        assert model.mass_bonus is None
        assert model.raw_transport_cost_threshold is not None
        assert torch.allclose(outputs["mass_bonus"], reward_weight, atol=1e-6, rtol=1e-6)
    else:
        assert model.mass_bonus is not None
        assert model.raw_transport_cost_threshold is None

    if feature_dim is None:
        assert captured_features == []
        assert torch.allclose(
            outputs["rho"],
            torch.full_like(outputs["rho"], model.fixed_mass),
            atol=1e-6,
            rtol=0.0,
        )
    else:
        assert model.eam.network[0].in_features == feature_dim
        assert len(captured_features) == 1
        assert captured_features[0].shape == (2, 3, 2, feature_dim)
        assert torch.isfinite(captured_features[0]).all()
        assert torch.all(outputs["rho"] >= 0.1)
        assert torch.all(outputs["rho"] <= 1.0)


def test_hrot_fsl_variant_i_euclidean_eam_features_match_l2_summary():
    torch.manual_seed(18)
    model = _build_model(variant="I")
    model.eval()

    query = torch.randn(2, 5, 7)
    support = torch.randn(3, 4, 5, 7)

    features = model._build_euclidean_eam_features(query, support)

    query_mean = query.mean(dim=1)
    flat_support = support.reshape(12, 5, 7)
    shot_mean = flat_support.mean(dim=1).reshape(3, 4, 7)
    class_mean = support.reshape(3, 20, 7).mean(dim=1)
    query_var = (query - query_mean[:, None, :]).pow(2).sum(dim=-1).mean(dim=-1)
    flat_shot_mean = shot_mean.reshape(12, 7)
    shot_var = (flat_support - flat_shot_mean[:, None, :]).pow(2).sum(dim=-1).mean(dim=-1).reshape(3, 4)
    expected = torch.stack(
        [
            torch.linalg.vector_norm(query_mean[:, None, None, :] - shot_mean[None, :, :, :], dim=-1),
            torch.linalg.vector_norm(shot_mean - class_mean[:, None, :], dim=-1)[None, :, :].expand(2, 3, 4),
            query_var[:, None, None].expand(2, 3, 4),
            shot_var[None, :, :].expand(2, 3, 4),
        ],
        dim=-1,
    )

    assert torch.allclose(features, expected, atol=1e-6, rtol=0.0)


def test_hrot_fsl_variant_m_blends_post_h_ablation_paths():
    torch.manual_seed(19)
    model = _build_model(variant="M")
    model.eval()
    captured_geodesic = []
    captured_euclidean = []
    captured_reduced = []

    original_geodesic = model.eam.forward_features
    original_euclidean = model.euclidean_eam.forward_features
    original_reduced = model.reduced_geodesic_eam.forward_features

    def capture_geodesic(features):
        captured_geodesic.append(features.detach().clone())
        return original_geodesic(features)

    def capture_euclidean(features):
        captured_euclidean.append(features.detach().clone())
        return original_euclidean(features)

    def capture_reduced(features):
        captured_reduced.append(features.detach().clone())
        return original_reduced(features)

    model.eam.forward_features = capture_geodesic
    model.euclidean_eam.forward_features = capture_euclidean
    model.reduced_geodesic_eam.forward_features = capture_reduced

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    flat_support = outputs["support_euclidean_tokens"].squeeze(0).reshape(6, -1, 24)
    expected_cost = model._euclidean_cost(outputs["query_euclidean_tokens"], flat_support)
    expected_cost = expected_cost.reshape(2, 3, 2, expected_cost.shape[-2], expected_cost.shape[-1])
    reward_weight = (
        model.score_scale * model.transport_cost_threshold.detach() + model.mass_bonus.detach()
    ).to(dtype=outputs["shot_transported_mass"].dtype)
    expected_logits = (
        -model.score_scale * outputs["shot_transport_cost"]
        + reward_weight * outputs["shot_transported_mass"]
    ).mean(dim=-1)

    assert torch.allclose(outputs["cost_matrix"], expected_cost, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["logits"], expected_logits, atol=1e-5, rtol=1e-5)
    assert model.uses_hybrid_ablation_eam
    assert model.uses_hybrid_mass_reward
    assert model.uses_cost_threshold_score
    assert model.uses_geodesic_eam
    assert model.mass_bonus is not None
    assert model.raw_transport_cost_threshold is not None
    assert model.euclidean_eam is not None
    assert model.reduced_geodesic_eam is not None
    assert torch.allclose(outputs["mass_bonus"], reward_weight, atol=1e-6, rtol=1e-6)

    assert model.eam.network[0].in_features == 4
    assert model.euclidean_eam.network[0].in_features == 4
    assert model.reduced_geodesic_eam.network[0].in_features == 3
    assert len(captured_geodesic) == 1
    assert len(captured_euclidean) == 1
    assert len(captured_reduced) == 1
    assert captured_geodesic[0].shape == (2, 3, 2, 4)
    assert captured_euclidean[0].shape == (2, 3, 2, 4)
    assert captured_reduced[0].shape == (2, 3, 2, 3)
    assert torch.allclose(captured_reduced[0], captured_geodesic[0][..., [0, 2, 3]], atol=1e-6, rtol=0.0)
    assert torch.all(outputs["rho"] >= 0.1)
    assert torch.all(outputs["rho"] <= 1.0)


def test_hrot_fsl_variant_n_adds_geodesic_order_rho_rank_loss():
    torch.manual_seed(20)
    model = _build_model(
        variant="N",
        lambda_rho=0.03,
        lambda_rho_rank=0.07,
        rho_rank_margin=0.02,
        rho_rank_temperature=0.05,
    )
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    outputs = model(query, support, return_aux=True)
    geodesic_features = model._build_geodesic_eam_features(
        outputs["query_hyperbolic_tokens"],
        outputs["support_hyperbolic_tokens"].squeeze(0),
    )
    expected_rank_loss = model._rho_rank_loss(outputs["rho"], geodesic_features[..., 0])
    expected_aux_loss = (
        model.lambda_rho * outputs["rho_regularization"]
        + model.lambda_rho_rank * expected_rank_loss
        + model.lambda_curvature * outputs["curvature_regularization"]
    )

    assert model.uses_rho_rank_loss
    assert model.uses_geodesic_eam
    assert model.uses_cost_threshold_score
    assert model.uses_shot_decomposed_transport
    assert expected_rank_loss.item() > 0.0
    assert torch.allclose(outputs["rho_rank_loss"], expected_rank_loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs["aux_loss"], expected_aux_loss, atol=1e-6, rtol=1e-6)

    model.zero_grad(set_to_none=True)
    outputs["aux_loss"].backward()

    assert model.eam.network[-2].weight.grad is not None
    assert torch.isfinite(model.eam.network[-2].weight.grad).all()
    assert model.eam.network[-2].weight.grad.norm().item() > 0.0


def test_hrot_fsl_variant_o_uses_transport_aware_geodesic_eam():
    torch.manual_seed(21)
    model = _build_model(variant="O")
    model.eval()
    captured_features = []

    original_forward_features = model.eam.forward_features

    def capture_forward_features(features):
        captured_features.append(features.detach().clone())
        return original_forward_features(features)

    model.eam.forward_features = capture_forward_features

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    flat_support = outputs["support_euclidean_tokens"].squeeze(0).reshape(6, -1, 24)
    expected_cost = model._euclidean_cost(outputs["query_euclidean_tokens"], flat_support)
    expected_cost = expected_cost.reshape(2, 3, 2, expected_cost.shape[-2], expected_cost.shape[-1])
    mass_reward = (model.score_scale * model.transport_cost_threshold.detach()).to(
        dtype=outputs["shot_transported_mass"].dtype
    )
    expected_logits = (
        -model.score_scale * outputs["shot_transport_cost"]
        + mass_reward * outputs["shot_transported_mass"]
    ).mean(dim=-1)
    expected_transport_features = torch.stack(
        [
            outputs["transport_probe_cost"],
            outputs["transport_probe_mass"],
            outputs["transport_probe_entropy"],
            outputs["transport_probe_min_cost"],
        ],
        dim=-1,
    )

    assert torch.allclose(outputs["cost_matrix"], expected_cost, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["logits"], expected_logits, atol=1e-5, rtol=1e-5)
    assert model.uses_transport_aware_eam
    assert model.uses_geodesic_eam
    assert model.uses_cost_threshold_score
    assert model.uses_shot_decomposed_transport
    assert model.eam.network[0].in_features == 8
    assert len(captured_features) == 1
    assert captured_features[0].shape == (2, 3, 2, 8)
    assert torch.isfinite(captured_features[0]).all()
    expected_transport_features = expected_transport_features.to(dtype=captured_features[0].dtype)
    assert torch.allclose(captured_features[0][..., 4:], expected_transport_features, atol=1e-6, rtol=1e-6)
    assert outputs["transport_probe_cost"].shape == (2, 3, 2)
    assert outputs["transport_probe_mass"].shape == (2, 3, 2)
    assert outputs["transport_probe_entropy"].shape == (2, 3, 2)
    assert outputs["transport_probe_min_cost"].shape == (2, 3, 2)
    assert torch.isfinite(outputs["transport_probe_cost"]).all()
    assert torch.isfinite(outputs["transport_probe_mass"]).all()
    assert torch.isfinite(outputs["transport_probe_entropy"]).all()
    assert torch.isfinite(outputs["transport_probe_min_cost"]).all()


def test_hrot_fsl_variant_p_uses_hyperbolic_attentive_uot_marginals():
    torch.manual_seed(24)
    model = _build_model(variant="P", token_temperature=0.2)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    flat_support = outputs["support_euclidean_tokens"].squeeze(0).reshape(6, -1, 24)
    expected_cost = model._euclidean_cost(outputs["query_euclidean_tokens"], flat_support)
    expected_cost = expected_cost.reshape(2, 3, 2, expected_cost.shape[-2], expected_cost.shape[-1])
    flat_support_hyp = outputs["support_hyperbolic_tokens"].squeeze(0).reshape(6, -1, 24)
    flat_rho = outputs["shot_rho"].reshape(2, 6)
    expected_query_mass, expected_support_mass = model._compute_hyperbolic_token_marginals(
        outputs["query_hyperbolic_tokens"],
        flat_support_hyp,
        flat_rho,
    )
    expected_query_mass = expected_query_mass.reshape_as(outputs["query_token_mass"])
    expected_support_mass = expected_support_mass.reshape_as(outputs["support_token_mass"])
    mass_reward = (model.score_scale * model.transport_cost_threshold.detach()).to(
        dtype=outputs["shot_transported_mass"].dtype
    )
    expected_shot_logits = -model.score_scale * outputs["shot_transport_cost"] + (
        mass_reward * outputs["shot_transported_mass"]
    )
    expected_logits = expected_shot_logits.mean(dim=-1)

    assert torch.allclose(outputs["cost_matrix"], expected_cost, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["query_token_mass"], expected_query_mass, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["support_token_mass"], expected_support_mass, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["query_token_mass"].sum(dim=-1), outputs["shot_rho"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["support_token_mass"].sum(dim=-1), outputs["shot_rho"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["shot_logits"], expected_shot_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["logits"], expected_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["transport_cost"], outputs["shot_transport_cost"].mean(dim=-1), atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        outputs["transported_mass"],
        outputs["shot_transported_mass"].mean(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )
    uniform_query_mass = outputs["shot_rho"].unsqueeze(-1) / outputs["query_token_mass"].shape[-1]
    uniform_support_mass = outputs["shot_rho"].unsqueeze(-1) / outputs["support_token_mass"].shape[-1]
    assert not torch.allclose(outputs["query_token_mass"], uniform_query_mass, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(outputs["support_token_mass"], uniform_support_mass, atol=1e-4, rtol=1e-4)
    assert model.uses_hyperbolic_token_attention
    assert not model.uses_hyperbolic_geometry
    assert model.uses_geodesic_eam
    assert model.uses_cost_threshold_score
    assert model.uses_unbalanced_transport
    assert model.uses_shot_decomposed_transport
    assert model.mass_bonus is None
    assert model.raw_transport_cost_threshold is not None
    assert model.raw_token_temperature is not None
    assert outputs["token_temperature"].item() > 0.0
    assert outputs["query_token_mass"].shape[:3] == (2, 3, 2)
    assert outputs["support_token_mass"].shape[:3] == (2, 3, 2)
    assert outputs["query_token_mass"].shape[-1] == outputs["cost_matrix"].shape[-2]
    assert outputs["support_token_mass"].shape[-1] == outputs["cost_matrix"].shape[-1]
    assert torch.all(outputs["query_token_mass"] > 0.0)
    assert torch.all(outputs["support_token_mass"] > 0.0)


def test_hrot_fsl_variant_p_backpropagates_token_attention_and_geodesic_eam():
    torch.manual_seed(25)
    model = _build_model(variant="P", lambda_rho=0.1)
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)

    outputs = model(query, support, query_targets=targets)
    loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    model.zero_grad(set_to_none=True)
    loss.backward()

    assert model.eam.network[-2].weight.grad is not None
    assert model.raw_transport_cost_threshold.grad is not None
    assert model.raw_token_temperature.grad is not None
    assert torch.isfinite(model.raw_token_temperature.grad).all()
    assert torch.isfinite(model.raw_transport_cost_threshold.grad).all()
    assert torch.isfinite(model.eam.network[-2].weight.grad).all()
    assert model.raw_token_temperature.grad.abs().sum().item() > 0.0
    assert model.eam.network[-2].weight.grad.abs().sum().item() > 0.0


def test_hrot_fsl_variant_q_uses_noise_calibrated_hierarchical_uot():
    torch.manual_seed(26)
    model = _build_model(variant="Q", token_temperature=0.2)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    flat_support = outputs["support_euclidean_tokens"].squeeze(0).reshape(6, -1, 24)
    expected_cost = model._euclidean_cost(outputs["query_euclidean_tokens"], flat_support)
    expected_cost = expected_cost.reshape(2, 3, 2, expected_cost.shape[-2], expected_cost.shape[-1])
    global_threshold = model.transport_cost_threshold.detach().to(
        dtype=outputs["shot_transported_mass"].dtype
    )
    expected_shot_logits = model.score_scale * (
        outputs["adaptive_transport_cost_threshold"] * outputs["shot_transported_mass"]
        - outputs["shot_transport_cost"]
    )
    expected_q_logits = (outputs["shot_pool_weights"] * expected_shot_logits).sum(dim=-1)
    expected_q_transport_cost = (outputs["shot_pool_weights"] * outputs["shot_transport_cost"]).sum(dim=-1)
    expected_q_transport_mass = (outputs["shot_pool_weights"] * outputs["shot_transported_mass"]).sum(dim=-1)
    expected_h_anchor_shot_logits = model.score_scale * (
        global_threshold * outputs["h_anchor_shot_transported_mass"]
        - outputs["h_anchor_shot_transport_cost"]
    )
    expected_h_anchor_logits = expected_h_anchor_shot_logits.mean(dim=-1)
    expected_h_anchor_transport_cost = outputs["h_anchor_shot_transport_cost"].mean(dim=-1)
    expected_h_anchor_transport_mass = outputs["h_anchor_shot_transported_mass"].mean(dim=-1)
    q_mix = outputs["q_enhancement_mix"]
    expected_logits = (1.0 - q_mix) * expected_h_anchor_logits + q_mix * expected_q_logits
    expected_transport_cost = (
        (1.0 - q_mix) * expected_h_anchor_transport_cost
        + q_mix * expected_q_transport_cost
    )
    expected_transport_mass = (
        (1.0 - q_mix) * expected_h_anchor_transport_mass
        + q_mix * expected_q_transport_mass
    )

    assert torch.allclose(outputs["cost_matrix"], expected_cost, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["shot_logits"], expected_shot_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["q_enhanced_logits"], expected_q_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["h_anchor_shot_logits"], expected_h_anchor_shot_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["h_anchor_logits"], expected_h_anchor_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["logits"], expected_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["transport_cost"], expected_transport_cost, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["transported_mass"], expected_transport_mass, atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["query_token_mass"].sum(dim=-1), outputs["shot_rho"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(outputs["support_token_mass"].sum(dim=-1), outputs["shot_rho"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        outputs["probe_query_reliability"].sum(dim=-1),
        torch.ones_like(outputs["shot_rho"]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        outputs["probe_support_reliability"].sum(dim=-1),
        torch.ones_like(outputs["shot_rho"]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        outputs["query_hyperbolic_token_prior"].sum(dim=-1),
        torch.ones_like(outputs["shot_rho"]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        outputs["support_hyperbolic_token_prior"].sum(dim=-1),
        torch.ones_like(outputs["shot_rho"]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        outputs["shot_pool_weights"].sum(dim=-1),
        torch.ones_like(outputs["logits"]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert model.uses_noise_calibrated_transport
    assert model.uses_geodesic_eam
    assert model.uses_cost_threshold_score
    assert model.uses_unbalanced_transport
    assert model.uses_shot_decomposed_transport
    assert model.raw_token_temperature is not None
    assert model.q_eam is not None
    assert model.q_eam.network[0].in_features == 5
    assert model.query_reliability_weights is not None
    assert model.support_reliability_weights is not None
    assert model.query_token_attention_vector is not None
    assert model.support_token_attention_vector is not None
    assert model.q_shot_pool_scorer is not None
    assert model.q_threshold_scorer is not None
    assert model.raw_noise_sink_cost is not None
    assert model.raw_shot_pool_temperature is not None
    assert outputs["probe_query_reliability"].shape == outputs["query_token_mass"].shape
    assert outputs["probe_support_reliability"].shape == outputs["support_token_mass"].shape
    assert outputs["support_consensus"].shape == outputs["support_token_mass"].shape
    assert outputs["query_hyperbolic_token_prior"].shape == outputs["query_token_mass"].shape
    assert outputs["support_hyperbolic_token_prior"].shape == outputs["support_token_mass"].shape
    assert outputs["adaptive_transport_cost_threshold"].shape == (2, 3, 2)
    assert torch.all(outputs["adaptive_transport_cost_threshold"] > 0.0)
    assert outputs["h_anchor_rho"].shape == (2, 3, 2)
    assert outputs["h_anchor_shot_transport_cost"].shape == (2, 3, 2)
    assert outputs["h_anchor_shot_transported_mass"].shape == (2, 3, 2)
    assert outputs["q_eam_cross_attention"].shape == (2, 3, 2)
    assert torch.allclose(
        outputs["q_eam_cross_attention"].sum(dim=-1),
        torch.full_like(outputs["logits"], 2.0),
        atol=1e-5,
        rtol=1e-5,
    )
    assert outputs["transport_probe_cost"].shape == (2, 3, 2)
    assert outputs["noise_sink_query_mass"].shape == (2, 3, 2)
    assert outputs["noise_sink_support_mass"].shape == (2, 3, 2)
    assert outputs["noise_sink_self_mass"].shape == (2, 3, 2)
    assert torch.all(outputs["noise_sink_query_mass"] >= 0.0)
    assert torch.all(outputs["noise_sink_support_mass"] >= 0.0)
    assert outputs["noise_sink_cost"].item() > 0.0
    assert 0.0 < outputs["token_reliability_mix"].item() < 1.0
    assert 0.0 < outputs["support_consensus_mix"].item() < 1.0
    assert 0.0 < outputs["hyperbolic_token_prior_mix"].item() < 1.0
    assert outputs["eam_cross_attention_temperature"].item() > 0.0
    assert 0.0 < outputs["shot_pool_mix"].item() < 1.0
    assert 0.0 < outputs["q_enhancement_mix"].item() < 1.0


def test_hrot_fsl_variant_q_backpropagates_noise_calibration_parameters():
    torch.manual_seed(27)
    model = _build_model(variant="Q", lambda_rho=0.1)
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)

    outputs = model(query, support, query_targets=targets)
    loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    model.zero_grad(set_to_none=True)
    loss.backward()

    checked_params = [
        model.eam.network[-2].weight,
        model.q_eam.network[-2].weight,
        model.raw_transport_cost_threshold,
        model.raw_token_temperature,
        model.query_reliability_weights,
        model.support_reliability_weights,
        model.query_token_attention_vector,
        model.support_token_attention_vector,
        model.raw_token_reliability_mix,
        model.raw_support_consensus_mix,
        model.raw_hyperbolic_token_prior_mix,
        model.raw_eam_cross_attention_temperature,
        model.raw_consensus_temperature,
        model.raw_noise_sink_cost,
        model.raw_shot_pool_temperature,
        model.raw_shot_pool_mix,
        model.raw_q_enhancement_mix,
        model.q_shot_pool_scorer.weight,
        model.q_threshold_scorer.weight,
    ]
    for param in checked_params:
        assert param is not None
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()

    assert model.query_reliability_weights.grad.abs().sum().item() > 0.0
    assert model.support_reliability_weights.grad.abs().sum().item() > 0.0
    assert model.q_eam.network[-2].weight.grad.abs().sum().item() > 0.0
    assert model.raw_q_enhancement_mix.grad.abs().sum().item() > 0.0
    assert model.raw_noise_sink_cost.grad.abs().sum().item() > 0.0


def test_hrot_fsl_normalize_rho_keeps_episode_budget_mean():
    torch.manual_seed(22)
    model = _build_model(variant="O", normalize_rho=True)
    model.eval()

    def varying_rho(features):
        raw = torch.linspace(
            0.6,
            0.8,
            steps=features[..., 0].numel(),
            device=features.device,
            dtype=features.dtype,
        )
        return raw.reshape(features.shape[:-1])

    model.eam.forward_features = varying_rho

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    expected_mean = torch.full(
        (outputs["rho"].shape[0],),
        model.fixed_mass,
        device=outputs["rho"].device,
        dtype=outputs["rho"].dtype,
    )
    assert torch.all(outputs["rho"] >= model.min_mass)
    assert torch.all(outputs["rho"] <= 1.0)
    assert torch.allclose(outputs["rho"].mean(dim=(1, 2)), expected_mean, atol=1e-6, rtol=0.0)


def test_hrot_fsl_is_support_shot_permutation_invariant():
    torch.manual_seed(4)
    model = _build_model(variant="E")
    model.eval()

    query = torch.randn(1, 3, 3, 64, 64)
    support = torch.randn(1, 4, 2, 3, 64, 64)
    support_permuted = support[:, :, torch.tensor([1, 0])]

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)
        permuted = model(query, support_permuted, return_aux=True)

    assert torch.allclose(outputs["logits"], permuted["logits"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["transport_cost"], permuted["transport_cost"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["transported_mass"], permuted["transported_mass"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["rho"], permuted["rho"], atol=1e-5, rtol=0.0)


def test_hrot_fsl_backpropagates_through_core_modules():
    torch.manual_seed(5)
    model = _build_model(variant="E", lambda_rho=0.1, lambda_curvature=0.1, hyperbolic_backend="geoopt")
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)

    outputs = model(query, support, query_targets=targets)
    loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    model.zero_grad(set_to_none=True)
    loss.backward()

    assert model.token_projector[1].weight.grad is not None
    assert model.eam.network[0].weight.grad is not None
    assert model.curvature_parameter.grad is not None
    assert model.mass_bonus.grad is not None
    assert torch.isfinite(model.token_projector[1].weight.grad).all()
    assert torch.isfinite(model.eam.network[0].weight.grad).all()
    assert torch.isfinite(model.curvature_parameter.grad).all()
    assert torch.isfinite(model.mass_bonus.grad).all()


def test_hrot_fsl_pot_backend_runs_for_debug_path():
    torch.manual_seed(6)
    model = _build_model(variant="E", hyperbolic_backend="geoopt", ot_backend="pot", sinkhorn_iterations=10)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["transport_plan"]).all()


@pytest.mark.parametrize("factory_variant", ["E", "Q"])
def test_hrot_fsl_model_factory_builds_and_runs(factory_variant: str):
    args = SimpleNamespace(
        model="hrot_fsl",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        token_dim=24,
        hrot_token_dim=24,
        hrot_variant=factory_variant,
        hrot_eam_hidden_dim=32,
        hrot_curvature_init=1.0,
        hrot_projection_scale=0.1,
        hrot_token_temperature=0.1,
        hrot_score_scale=8.0,
        hrot_tau_q=0.5,
        hrot_tau_c=0.5,
        hrot_sinkhorn_epsilon=0.1,
        hrot_sinkhorn_iterations=10,
        hrot_sinkhorn_tolerance=1e-5,
        hrot_fixed_mass=0.8,
        hrot_min_mass=0.1,
        hrot_mass_bonus_init=1.0,
        hrot_lambda_rho=0.05,
        hrot_rho_target=0.8,
        hrot_lambda_rho_rank=0.05,
        hrot_rho_rank_margin=0.05,
        hrot_rho_rank_temperature=0.05,
        hrot_lambda_curvature=0.01,
        hrot_min_curvature=0.05,
        hrot_normalize_euclidean_tokens="true",
        hrot_normalize_rho="false",
        hrot_eval_use_float64="true",
        hrot_hyperbolic_backend="geoopt",
        hrot_ot_backend="native",
        hrot_eps=1e-6,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["support_hyperbolic_tokens"].shape[-1] == 24
    assert torch.isfinite(outputs["logits"]).all()
    if factory_variant == "Q":
        assert outputs["shot_pool_weights"].shape == (2, 3, 2)


def test_forward_scores_routes_query_targets_and_aux_to_hrot():
    source = Path("main.py").read_text(encoding="utf-8")
    module = ast.parse(source)

    forward_scores = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "forward_scores"
    )
    hrot_branch = None
    for node in ast.walk(forward_scores):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Attribute)
            and test.left.attr == "model"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "hrot_fsl"
        ):
            hrot_branch = node
            break

    assert hrot_branch is not None, "forward_scores must keep a dedicated hrot_fsl branch"

    call = hrot_branch.body[0].value
    assert isinstance(call, ast.Call)
    keyword_names = {kw.arg for kw in call.keywords}
    assert "query_targets" in keyword_names
    assert "support_targets" in keyword_names
    assert "return_aux" in keyword_names
