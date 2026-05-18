from __future__ import annotations

import math
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


def _episode(shot=1):
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, shot, 3, 64, 64)
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
    assert ours_final.dmuot_shot_strength == "none"
    shot_neutral = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            ours_final_dmuot_ablation="exp5_tau_marg_2_shot_sqrt",
            **{key: value for key, value in common.items() if key != "ours_final_dmuot_ablation"},
        )
    )
    assert shot_neutral.marginal_kind == "discriminative"
    assert shot_neutral.tau_marg == 2.0
    assert shot_neutral.dmuot_shot_strength == "inverse_sqrt"
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


def test_shot_neutralized_marginals_blend_toward_uniform_for_5shot():
    torch.manual_seed(514)
    baseline = _tiny_ours_final(
        token_g_kind="episode_mean_dist",
        marginal_kind="discriminative",
        tau_marg=2.0,
    )
    neutralized = _tiny_ours_final(
        token_g_kind="episode_mean_dist",
        marginal_kind="discriminative",
        tau_marg=2.0,
        dmuot_shot_strength="inverse_sqrt",
    )
    neutralized.load_state_dict(baseline.state_dict())
    baseline.eval()
    neutralized.eval()
    query, support = _episode(shot=5)

    with torch.no_grad():
        baseline_outputs = baseline(query, support, return_aux=True)
        neutral_outputs = neutralized(query, support, return_aux=True)

    assert neutral_outputs["dmuot_shot_strength"].item() == pytest.approx(1.0 / math.sqrt(5.0))
    uniform_query = torch.full_like(
        baseline_outputs["marginal_query"],
        0.8 / float(baseline_outputs["marginal_query"].shape[-1]),
    )
    uniform_support = torch.full_like(
        baseline_outputs["marginal_support"],
        0.8 / float(baseline_outputs["marginal_support"].shape[-1]),
    )
    baseline_query_l1 = (baseline_outputs["marginal_query"] - uniform_query).abs().sum(dim=-1).mean()
    neutral_query_l1 = (neutral_outputs["marginal_query"] - uniform_query).abs().sum(dim=-1).mean()
    baseline_support_l1 = (baseline_outputs["marginal_support"] - uniform_support).abs().sum(dim=-1).mean()
    neutral_support_l1 = (neutral_outputs["marginal_support"] - uniform_support).abs().sum(dim=-1).mean()

    assert neutral_query_l1 <= baseline_query_l1 + 1e-6
    assert neutral_support_l1 <= baseline_support_l1 + 1e-6


def test_shot_consensus_mass_score_matches_standard_for_1shot():
    torch.manual_seed(515)
    baseline = _tiny_ours_final()
    consensus = _tiny_ours_final(ecot_m2_mass_score_mode="shot_consensus")
    consensus.load_state_dict(baseline.state_dict())
    baseline.eval()
    consensus.eval()
    query, support = _episode(shot=1)

    with torch.no_grad():
        expected = baseline(query, support, return_aux=True)
        actual = consensus(query, support, return_aux=True)

    assert torch.allclose(actual["logits"], expected["logits"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        actual["shot_mass_for_score"],
        actual["shot_transported_mass"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_shot_consensus_mass_score_uses_class_mean_mass_for_5shot():
    torch.manual_seed(516)
    model = _tiny_ours_final(
        ecot_m2_mass_score_mode="shot_consensus",
        ecot_m2_consensus_mass_alpha=1.0,
    )
    model.eval()
    query, support = _episode(shot=5)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    expected_mass = outputs["shot_transported_mass"].mean(dim=-1, keepdim=True)
    expected_mass = expected_mass.expand_as(outputs["shot_mass_for_score"])
    assert torch.allclose(
        outputs["shot_mass_for_score"],
        expected_mass,
        atol=1e-6,
        rtol=1e-6,
    )
    assert outputs["ecot_m2_consensus_mass_alpha"].item() == pytest.approx(1.0)


def test_multi_shot_mass_reward_scaling_keeps_1shot_logits_unchanged():
    torch.manual_seed(517)
    baseline = _tiny_ours_final()
    scaled = _tiny_ours_final(
        ecot_m2_mass_reward_shot_scaling="multi_shot_beta",
        ecot_m2_mass_reward_beta=0.5,
    )
    scaled.load_state_dict(baseline.state_dict())
    baseline.eval()
    scaled.eval()
    query, support = _episode(shot=1)

    with torch.no_grad():
        expected = baseline(query, support, return_aux=True)
        actual = scaled(query, support, return_aux=True)

    assert torch.allclose(actual["logits"], expected["logits"], atol=1e-6, rtol=1e-6)
    assert actual["ecot_m2_mass_reward_beta_effective"].item() == pytest.approx(1.0)


def test_multi_shot_mass_reward_scaling_applies_beta_for_5shot():
    torch.manual_seed(518)
    model = _tiny_ours_final(
        ecot_m2_mass_reward_shot_scaling="multi_shot_beta",
        ecot_m2_mass_reward_beta=0.5,
    )
    model.eval()
    query, support = _episode(shot=5)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["ecot_m2_mass_reward_beta_effective"].item() == pytest.approx(0.5)


def test_per_shot_threshold_produces_varying_thresholds():
    torch.manual_seed(520)
    model = _tiny_ours_final(ecot_m2_per_shot_threshold=True, ecot_m2_pst_hidden=16)
    model.eval()
    query, support = _episode(shot=5)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()


def test_per_shot_threshold_gradient_flows():
    torch.manual_seed(521)
    model = _tiny_ours_final(ecot_m2_per_shot_threshold=True, ecot_m2_pst_hidden=16)
    model.train()
    query, support = _episode(shot=5)

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()

    scorer = model.ecot_m2_pst_scorer
    assert scorer is not None
    for param in scorer.parameters():
        assert param.grad is not None, "PST scorer param has no gradient"
        assert torch.isfinite(param.grad).all(), "PST scorer gradient is not finite"


def test_per_shot_threshold_disabled_matches_baseline():
    torch.manual_seed(522)
    baseline = _tiny_ours_final()
    pst_off = _tiny_ours_final(ecot_m2_per_shot_threshold=False)
    pst_off.load_state_dict(baseline.state_dict())
    baseline.eval()
    pst_off.eval()
    query, support = _episode(shot=5)

    with torch.no_grad():
        expected = baseline(query, support)
        actual = pst_off(query, support)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_learned_attention_token_g_produces_nonuniform_marginals():
    torch.manual_seed(530)
    model = _tiny_ours_final(
        token_g_kind="learned_attention",
        marginal_kind="discriminative",
        tau_marg=1.0,
    )
    model.eval()
    query, support = _episode(shot=1)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert "marginal_query" in outputs
    assert "marginal_support" in outputs
    mq = outputs["marginal_query"]
    ms = outputs["marginal_support"]
    num_tokens = mq.shape[-1]
    uniform_val = 0.8 / float(num_tokens)
    assert not torch.allclose(mq, torch.full_like(mq, uniform_val), atol=1e-4)
    assert torch.allclose(mq.sum(dim=-1), torch.full(mq.shape[:-1], 0.8), atol=1e-5)


def test_learned_attention_gradient_flows_to_query_vector():
    torch.manual_seed(531)
    model = _tiny_ours_final(
        token_g_kind="learned_attention",
        marginal_kind="discriminative",
        tau_marg=1.0,
    )
    model.train()
    query, support = _episode(shot=1)

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()

    assert model.token_attention_query is not None
    assert model.token_attention_query.grad is not None
    assert torch.isfinite(model.token_attention_query.grad).all()


def test_pst_and_dm_combined_runs():
    torch.manual_seed(540)
    model = _tiny_ours_final(
        ecot_m2_per_shot_threshold=True,
        ecot_m2_pst_hidden=16,
        token_g_kind="learned_attention",
        marginal_kind="discriminative",
        tau_marg=1.0,
        dmuot_shot_strength="inverse_sqrt",
    )
    model.eval()
    query, support = _episode(shot=5)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()
    assert "marginal_query" in outputs
    assert "dmuot_shot_strength" in outputs
