from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from net.model_factory import build_model_from_args
from net.ours import OursM2


def _tiny_ours_final(**overrides) -> OursM2:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=8,
        eam_hidden_dim=16,
        sinkhorn_iterations=4,
        sinkhorn_tolerance=1e-5,
        eval_use_float64=False,
        ecot_enable_egsm=False,
        ecot_m2_ablate_threshold_mass=False,
        ecot_rho_bank="0.8",
        ecot_base_rho=0.8,
        ecot_transport_mode="unbalanced",
    )
    kwargs.update(overrides)
    return OursM2(**kwargs)


def _episode():
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)
    return query, support


def test_token_g_none_matches_ours_final_baseline_logits():
    torch.manual_seed(510)
    baseline = _tiny_ours_final()
    token_g_none = _tiny_ours_final(token_g_kind="none")
    token_g_none.load_state_dict(baseline.state_dict())
    baseline.eval()
    token_g_none.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support)
        actual = token_g_none(query, support)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_lambda_cost_zero_keeps_logits_and_logs_token_g():
    torch.manual_seed(511)
    baseline = _tiny_ours_final()
    dmuot = _tiny_ours_final(token_g_kind="episode_mean_dist", lambda_cost=0.0)
    dmuot.load_state_dict(baseline.state_dict())
    baseline.eval()
    dmuot.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support, return_aux=True)
        actual = dmuot(query, support, return_aux=True)

    assert torch.allclose(actual["logits"], expected["logits"], atol=1e-6, rtol=1e-6)
    assert actual["token_g_query"].shape[:2] == (1, 2)
    assert actual["token_g_support"].shape[:3] == (1, 2, 1)
    assert actual["cost_modulator"].shape[:4] == (1, 2, 2, 1)
    assert torch.allclose(actual["cost_modulator"], torch.ones_like(actual["cost_modulator"]))
    assert torch.allclose(
        actual["transport_plan_modulated"].squeeze(0),
        actual["transport_plan"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_uniform_marginal_kind_keeps_logits_unchanged():
    torch.manual_seed(512)
    baseline = _tiny_ours_final()
    dmuot = _tiny_ours_final(token_g_kind="episode_mean_dist", marginal_kind="uniform")
    dmuot.load_state_dict(baseline.state_dict())
    baseline.eval()
    dmuot.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support)
        actual = dmuot(query, support)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_dmuot_active_flags_require_token_g():
    with pytest.raises(ValueError, match="lambda_cost > 0 requires token_g_kind"):
        _tiny_ours_final(lambda_cost=0.5)
    with pytest.raises(ValueError, match="marginal_kind='discriminative' requires token_g_kind"):
        _tiny_ours_final(marginal_kind="discriminative")


def test_dmuot_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        ours_final_dmuot_ablation="exp1_lambda_cost_0p5",
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )

    assert ours_final.token_g_kind == "episode_mean_dist"
    assert ours_final.lambda_cost == 0.5
    assert ours_final.marginal_kind == "uniform"
    with pytest.raises(ValueError, match="--ours_final_dmuot_ablation is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


def test_discriminative_marginals_log_shapes_and_mass_budget():
    torch.manual_seed(513)
    model = _tiny_ours_final(
        token_g_kind="episode_mean_dist",
        marginal_kind="discriminative",
        tau_marg=0.5,
    )
    model.eval()
    query, support = _episode()

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["marginal_query"].shape[:2] == (1, 2)
    assert outputs["marginal_support"].shape[:3] == (1, 2, 1)
    assert torch.allclose(
        outputs["marginal_query"].sum(dim=-1),
        torch.full(outputs["marginal_query"].shape[:-1], 0.8),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        outputs["marginal_support"].sum(dim=-1),
        torch.full(outputs["marginal_support"].shape[:-1], 0.8),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isfinite(outputs["marginal_l1_drift"])
