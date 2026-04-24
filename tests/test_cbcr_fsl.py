from __future__ import annotations

from types import SimpleNamespace

import torch

from net.cbcr_fsl import CBCRFSL
from net.ubt_fsl import UBTFSL, UncertainBarycentricClass
from net.model_factory import build_model_from_args


def _build_model(**overrides) -> UBTFSL:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=12,
        sinkhorn_epsilon=0.05,
        sinkhorn_iterations=12,
        sinkhorn_tolerance=1e-5,
        barycenter_iterations=8,
        barycenter_tolerance=1e-5,
        barycenter_method="mixture",
        alpha=0.3,
        beta=0.1,
        tau=0.5,
        rho=1.0,
        score_scale=1.0,
        normalize_tokens=False,
        cost_power=2.0,
        profile=False,
        transport_backend="unbalanced_ot",
        scoring="robust_transport_envelope",
        use_query_competition=True,
        query_competition_temperature=0.5,
        ablation_mode="ubt_full",
        ot_backend="native",
        eps=1e-8,
    )
    kwargs.update(overrides)
    return UBTFSL(**kwargs)


def test_ubt_fsl_feature_episode_shapes_and_score_formula():
    torch.manual_seed(201)
    model = _build_model()
    model.eval()

    supports = torch.randn(2, 2, 2, 3)
    query = torch.randn(3, 2, 3)

    with torch.no_grad():
        outputs = model.forward_episode_features(supports, query, return_aux=True)

    expected_logits = -(outputs["query_class_distance"] - outputs["epsilon"][None, :]).clamp_min(0.0)

    assert outputs["logits"].shape == (3, 2)
    assert outputs["class_radius"].shape == (2,)
    assert outputs["transport_radius"].shape == (2,)
    assert outputs["epsilon"].shape == (2,)
    assert outputs["support_dispersion"].shape == (2,)
    assert outputs["barycenter_tokens"].shape == (2, 4, 3)
    assert outputs["barycenter_weights"].shape == (2, 4)
    assert outputs["support_token_weights"].shape == (2, 2, 2)
    assert outputs["transport_plan"].shape[:2] == (3, 2)
    assert outputs["cost_matrix"].shape == outputs["transport_plan"].shape
    assert outputs["competitive_assignment"].shape == (3, 2, 2)
    assert outputs["query_ambiguity_assignment"].shape == (3, 2, 2)
    assert outputs["competitive_query_weights"].shape == (3, 2, 2)
    assert outputs["query_ambiguity_corrected_weights"].shape == (3, 2, 2)
    assert outputs["query_token_weights"].shape == (3, 2)
    assert torch.allclose(outputs["barycenter_weights"].sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(outputs["competitive_assignment"].sum(dim=-1), torch.ones(3, 2), atol=1e-5)
    assert torch.allclose(
        outputs["competitive_query_weights"].sum(dim=-1),
        outputs["query_token_weights"],
        atol=1e-5,
    )
    assert torch.allclose(outputs["logits"], expected_logits, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(outputs["logits"]).all()


def test_ubt_fsl_class_object_ties_radius_to_barycenter():
    model = _build_model(sinkhorn_epsilon=0.01, alpha=0.3, beta=0.2)

    support = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    weights = model.compute_support_token_weights(support)
    bary_tokens, bary_weights = model.compute_barycentric_measure(support, weights)
    radius, dispersion, shot_distances = model.compute_transport_radius(support, bary_tokens, bary_weights, weights)
    class_object = model.build_uncertain_class_measure(support, class_id=7)

    assert torch.allclose(weights, torch.full_like(weights, 0.5), atol=1e-6)
    assert torch.allclose(bary_weights, torch.full_like(bary_weights, 0.5), atol=1e-6)
    assert torch.allclose(shot_distances, torch.zeros_like(shot_distances), atol=1e-6)
    assert torch.allclose(dispersion, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(radius, torch.tensor(0.2), atol=1e-6)
    assert isinstance(class_object, UncertainBarycentricClass)
    assert class_object.class_id == 7
    assert torch.allclose(class_object.radius, torch.tensor(0.2), atol=1e-6)


def test_ubt_fsl_optional_query_ambiguity_correction_can_be_disabled():
    torch.manual_seed(202)
    model = _build_model(use_query_competition=False)
    model.eval()

    supports = torch.randn(2, 2, 2, 3)
    query = torch.randn(3, 2, 3)

    with torch.no_grad():
        outputs = model.forward_episode_features(supports, query, return_aux=True)

    assert torch.allclose(outputs["query_ambiguity_assignment"], torch.ones(3, 2, 2))
    assert torch.allclose(outputs["query_ambiguity_corrected_weights"][..., 0], outputs["query_token_weights"])
    assert torch.allclose(outputs["query_ambiguity_corrected_weights"][..., 1], outputs["query_token_weights"])


def test_ubt_fsl_model_factory_builds_and_runs_new_and_legacy_names():
    args = SimpleNamespace(
        model="ubt_fsl",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        token_dim=12,
        ubt_fsl_token_dim=12,
        ubt_fsl_sinkhorn_epsilon=0.05,
        ubt_fsl_sinkhorn_iterations=5,
        ubt_fsl_sinkhorn_tolerance=1e-5,
        ubt_fsl_barycenter_iterations=5,
        ubt_fsl_barycenter_tolerance=1e-5,
        ubt_fsl_barycenter_method="mixture",
        ubt_fsl_alpha=0.3,
        ubt_fsl_beta=0.1,
        ubt_fsl_tau=0.5,
        ubt_fsl_query_competition_temperature=0.5,
        ubt_fsl_rho=1.0,
        ubt_fsl_score_scale=1.0,
        ubt_fsl_normalize_tokens="true",
        ubt_fsl_cost_power=2.0,
        ubt_fsl_profile="false",
        ubt_fsl_transport_backend="unbalanced_ot",
        ubt_fsl_scoring="robust_transport_envelope",
        ubt_fsl_use_query_competition="true",
        ubt_fsl_ablation_mode="ubt_full",
        ubt_fsl_solver_backend=None,
        ubt_fsl_ot_backend="native",
        ubt_fsl_eps=1e-8,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 1, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (1, 2)
    assert outputs["epsilon"].shape == (1, 2)
    assert outputs["transport_radius"].shape == (1, 2)
    assert outputs["barycenter_weights"].shape[:2] == (1, 2)
    assert torch.isfinite(outputs["logits"]).all()

    args.model = "cbcr_fsl"
    args.cbcr_fsl_ot_backend = "native"
    legacy_model = build_model_from_args(args)
    assert isinstance(legacy_model, CBCRFSL)
