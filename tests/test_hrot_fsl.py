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


@pytest.mark.parametrize("variant", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"])
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
    if variant in {"G", "H", "I", "J", "K", "L", "M", "N"}:
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


def test_hrot_fsl_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="hrot_fsl",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        token_dim=24,
        hrot_token_dim=24,
        hrot_variant="E",
        hrot_eam_hidden_dim=32,
        hrot_curvature_init=1.0,
        hrot_projection_scale=0.1,
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
