from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from net.jsc_wdro import JSCWDRO, expected_calibration_error
from net.model_factory import build_model_from_args


def _build_model(**overrides) -> JSCWDRO:
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
        barycenter_transport="unbalanced",
        barycenter_tau=0.5,
        barycenter_method="sinkhorn",
        score_scale=4.0,
        epsilon_alpha_init=0.02,
        epsilon_beta_init=0.05,
        epsilon_floor_init=1e-4,
        learn_epsilon=True,
        normalize_tokens=False,
        cost_power=2.0,
        unbalanced_score_mode="uot_objective",
        use_competitive_diagnostics=True,
        ot_backend="pot",
        eps=1e-8,
    )
    kwargs.update(overrides)
    return JSCWDRO(**kwargs)


def test_jsc_wdro_feature_episode_shapes_and_wdro_score_formula():
    torch.manual_seed(101)
    model = _build_model()
    model.eval()

    supports = torch.randn(2, 2, 3)
    query = torch.randn(3, 3)

    with torch.no_grad():
        outputs = model.forward_episode_features(supports, query, return_aux=True)

    expected_logits = -model.score_scale * (
        outputs["query_class_distance"] - outputs["epsilon"][None, :]
    ).clamp_min(0.0)

    assert outputs["logits"].shape == (3, 2)
    assert outputs["epsilon"].shape == (2,)
    assert outputs["support_dispersion"].shape == (2,)
    assert outputs["barycenter_tokens"].shape == (2, 2, 3)
    assert outputs["barycenter_weights"].shape == (2, 2)
    assert outputs["transport_plan"].shape[:2] == (3, 2)
    assert outputs["cost_matrix"].shape == outputs["transport_plan"].shape
    assert outputs["competitive_assignment"].shape == (3, 1, 2)
    assert torch.allclose(outputs["barycenter_weights"].sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(outputs["logits"], expected_logits, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(outputs["logits"]).all()
    assert model.ot_backend == "pot"
    assert model.barycenter_transport == "unbalanced"


def test_jsc_wdro_barycenter_and_wot_degenerate_single_token_distance_zero():
    model = _build_model(sinkhorn_epsilon=0.01, epsilon_alpha_init=1e-4, epsilon_beta_init=1e-4)
    support = torch.tensor([[1.0, 0.0]])

    bary_tokens, bary_weights = model.compute_barycenter(support)
    distance = model.compute_wot(support, None, bary_tokens, bary_weights)

    assert bary_tokens.shape == (1, 2)
    assert bary_weights.shape == (1,)
    assert torch.allclose(bary_weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(distance, torch.tensor(0.0), atol=1e-6)


def test_jsc_wdro_epsilon_parameters_are_learnable():
    torch.manual_seed(102)
    model = _build_model(
        epsilon_alpha_init=1e-4,
        epsilon_beta_init=1e-4,
        epsilon_floor_init=1e-4,
        epsilon_reg_weight=0.1,
    )
    model.train()

    supports = torch.randn(2, 2, 4)
    query = torch.randn(3, 4)
    targets = torch.tensor([0, 1, 0], dtype=torch.long)

    outputs = model.forward_episode_features(supports, query, return_aux=True)
    loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    model.zero_grad(set_to_none=True)
    loss.backward()

    assert model.raw_epsilon_alpha.grad is not None
    assert model.raw_epsilon_beta.grad is not None
    assert model.raw_epsilon_floor.grad is not None
    assert torch.isfinite(model.raw_epsilon_alpha.grad).all()
    assert torch.isfinite(model.raw_epsilon_beta.grad).all()
    assert torch.isfinite(model.raw_epsilon_floor.grad).all()


def test_jsc_wdro_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="jsc_wdro",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        token_dim=12,
        jsc_wdro_token_dim=12,
        jsc_wdro_score_scale=4.0,
        jsc_wdro_sinkhorn_epsilon=0.05,
        jsc_wdro_sinkhorn_iterations=5,
        jsc_wdro_sinkhorn_tolerance=1e-5,
        jsc_wdro_barycenter_iterations=5,
        jsc_wdro_barycenter_tolerance=1e-5,
        jsc_wdro_barycenter_transport="unbalanced",
        jsc_wdro_barycenter_tau=0.5,
        jsc_wdro_barycenter_method="sinkhorn",
        jsc_wdro_tau_q=0.5,
        jsc_wdro_tau_c=0.5,
        jsc_wdro_query_transport="balanced",
        jsc_wdro_epsilon_alpha_init=0.02,
        jsc_wdro_epsilon_beta_init=0.05,
        jsc_wdro_epsilon_floor_init=1e-4,
        jsc_wdro_learn_epsilon="true",
        jsc_wdro_epsilon_dimension=None,
        jsc_wdro_epsilon_reg_weight=0.0,
        jsc_wdro_normalize_tokens="true",
        jsc_wdro_cost_power=2.0,
        jsc_wdro_normalize_unbalanced_cost="true",
        jsc_wdro_unbalanced_score_mode="uot_objective",
        jsc_wdro_use_competitive_diagnostics="true",
        jsc_wdro_competitive_temperature=0.1,
        jsc_wdro_profile="false",
        jsc_wdro_ot_backend="pot",
        jsc_wdro_eps=1e-8,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 1, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (1, 2)
    assert outputs["epsilon"].shape == (1, 2)
    assert outputs["barycenter_weights"].shape[:2] == (1, 2)
    assert torch.isfinite(outputs["logits"]).all()


def test_expected_calibration_error_is_bounded():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [2.0, 1.0]])
    targets = torch.tensor([0, 1, 1])

    ece = expected_calibration_error(logits, targets, n_bins=5)

    assert 0.0 <= float(ece) <= 1.0
