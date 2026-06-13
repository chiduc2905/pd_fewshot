from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from net.model_factory import build_model_from_args
from net.modules.ours_final_failure_probe import OursFinalFailureProbe
from net.modules.rival_conditional_ground_cost import RivalConditionalGroundCost
from net.ours import OursM2
from run_all_experiments import (
    build_ours_final_failure_probe_variants,
    build_ours_final_rival_cost_variants,
)


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


def test_pot_guide_flag_off_creates_no_module_and_matches_logits():
    torch.manual_seed(509)
    baseline = _tiny_ours_final()
    flag_off = _tiny_ours_final(enable_pot_guide=False)
    flag_off.load_state_dict(baseline.state_dict())
    baseline.eval()
    flag_off.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support)
        actual = flag_off(query, support)

    assert not baseline.enable_pot_guide
    assert not hasattr(baseline, "pot_guide_module")
    assert not flag_off.enable_pot_guide
    assert not hasattr(flag_off, "pot_guide_module")
    assert torch.equal(actual, expected)


def test_multiscale_flag_off_creates_no_module_and_matches_logits():
    torch.manual_seed(508)
    baseline = _tiny_ours_final()
    flag_off = _tiny_ours_final(enable_multiscale_ot=False)
    flag_off.load_state_dict(baseline.state_dict())
    baseline.eval()
    flag_off.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support)
        actual = flag_off(query, support)

    assert not hasattr(baseline, "multi_scale_tokenizer")
    assert not hasattr(baseline, "scale_weights")
    assert not hasattr(flag_off, "multi_scale_tokenizer")
    assert not hasattr(flag_off, "scale_weights")
    assert torch.equal(actual, expected)


def test_multiscale_enabled_runs_and_exposes_diagnostics():
    torch.manual_seed(507)
    model = _tiny_ours_final(enable_multiscale_ot=True, multiscale_pool_sizes="original,2x2,1x1")
    model.train()
    query, support = _episode()

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()

    assert hasattr(model, "multi_scale_tokenizer")
    assert model.scale_weights.shape == (3,)
    assert outputs["logits"].shape == (2, 2)
    assert outputs["multiscale/scale_weights"].shape == (3,)
    assert torch.allclose(
        outputs["multiscale/scale_weights"],
        torch.full_like(outputs["multiscale/scale_weights"], 1.0 / 3.0),
        atol=1e-6,
        rtol=1e-6,
    )
    for key in (
        "multiscale/score_fine_mean",
        "multiscale/score_medium_mean",
        "multiscale/score_coarse_mean",
        "multiscale/mass_fine_mean",
        "multiscale/cost_medium_mean",
    ):
        assert key in outputs
        assert torch.isfinite(outputs[key])
    assert model.scale_weights.grad is not None
    assert torch.isfinite(model.scale_weights.grad).all()


def test_region_structural_uot_enabled_runs_and_exposes_diagnostics():
    torch.manual_seed(506)
    model = _tiny_ours_final(
        enable_region_structural_uot=True,
        region_uot_grid_size="2x2",
        region_uot_strength=0.15,
        region_uot_fgw_alpha=0.30,
        region_uot_fgw_iters=1,
        region_uot_sinkhorn_iters=4,
    )
    model.train()
    query, support = _episode()

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()

    assert model.enable_region_structural_uot
    assert hasattr(model, "region_structural_uot")
    assert outputs["logits"].shape == (2, 2)
    assert outputs["region_uot_coarse_plan"].shape[:4] == (2, 2, 1, 4)
    assert outputs["region_uot_coarse_plan"].shape[-1] == 4
    assert outputs["region_uot_sparse_coarse_plan"].shape == outputs["region_uot_coarse_plan"].shape
    assert outputs["region_uot_guided_cost_matrix"].shape == outputs["cost_matrix"].shape
    assert "base_cost_matrix" in outputs
    for key in (
        "region_uot/strength",
        "region_uot/fgw_alpha",
        "region_uot/topk",
        "region_uot/coarse_mass_mean",
        "region_uot/coarse_cost_mean",
        "region_uot/affinity_peak",
        "region_uot/sparse_mass_ratio",
        "region_uot/importance_confidence",
        "region_uot/effective_strength_mean",
        "region_uot/fine_gate_mean",
        "region_uot/cost_delta_ratio",
    ):
        assert key in outputs, f"Missing diagnostic key: {key}"
        assert torch.isfinite(outputs[key]), f"Non-finite diagnostic: {key}"


def test_adaptive_region_uot_enabled_runs_and_exposes_diagnostics():
    torch.manual_seed(505)
    model = _tiny_ours_final(
        enable_adaptive_region_uot=True,
        adaptive_region_num_slots=3,
        adaptive_region_context_kernels="1,3",
        adaptive_region_cost_discount=0.10,
        adaptive_region_mass_mix=0.7,
        adaptive_region_sinkhorn_iters=4,
        adaptive_region_init_gate=0.03,
    )
    model.train()
    query, support = _episode()

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()

    assert model.enable_adaptive_region_uot
    assert hasattr(model, "adaptive_region_uot")
    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()
    assert outputs["adaptive_region_plan"].shape[:4] == (2, 2, 1, 3)
    assert outputs["adaptive_region_plan"].shape[-1] == 3
    assert outputs["adaptive_region_guided_cost_matrix"].shape == outputs["cost_matrix"].shape
    assert outputs["adaptive_region_query_masks"].shape[:2] == (2, 3)
    assert outputs["adaptive_region_support_masks"].shape[1:4] == (2, 1, 3)
    assert "base_cost_matrix" in outputs
    for key in (
        "adaptive_region/num_slots",
        "adaptive_region/cost_discount",
        "adaptive_region/mass_mix",
        "adaptive_region/cost_gate",
        "adaptive_region/mass_gate",
        "adaptive_region/effective_cost_discount",
        "adaptive_region/effective_mass_mix",
        "adaptive_region/region_cost_mean",
        "adaptive_region/region_mass_mean",
        "adaptive_region/fine_affinity_peak",
        "adaptive_region/fine_gate_mean",
        "adaptive_region/cost_delta_ratio",
        "adaptive_region/query_weight_peak",
        "adaptive_region/support_weight_peak",
        "adaptive_region/query_effective_area",
        "adaptive_region/support_effective_area",
        "adaptive_region/context_kernel_entropy",
    ):
        assert key in outputs, f"Missing diagnostic key: {key}"
        assert torch.isfinite(outputs[key]), f"Non-finite diagnostic: {key}"
    assert model.adaptive_region_uot.slot_queries.grad is not None
    assert torch.isfinite(model.adaptive_region_uot.slot_queries.grad).all()
    assert outputs["adaptive_region/cost_gate"].item() == pytest.approx(0.03, abs=1e-5)
    assert outputs["adaptive_region/mass_gate"].item() == pytest.approx(0.03, abs=1e-5)


def test_reciprocal_verified_uot_runs_with_global_residual_and_replaces_plan():
    torch.manual_seed(504)
    model = _tiny_ours_final(
        enable_reciprocal_verified_uot=True,
        rvuot_beta=0.85,
        rvuot_tau=0.10,
        rvuot_ratio_threshold=0.20,
        rvuot_kernel_size=3,
        enable_global_residual_score=True,
        global_residual_weight=0.10,
    )
    model.train()
    query, support = _episode()

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()

    assert model.enable_reciprocal_verified_uot
    assert hasattr(model, "reciprocal_verified_transport")
    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()
    assert outputs["transport_plan"].shape == outputs["rvuot_unverified_transport_plan"].shape
    assert outputs["rvuot_evidence_transport_plan"].shape == outputs["transport_plan"].shape
    assert outputs["transport_plan"].sum() <= outputs["rvuot_unverified_transport_plan"].sum() + 1e-6
    assert outputs["rvuot_evidence_transport_plan"].sum() <= outputs["transport_plan"].sum() + 1e-6
    assert outputs["rvuot/enabled"].item() == pytest.approx(1.0)
    assert outputs["rvuot/rival_gate_enabled"].item() == pytest.approx(1.0)
    assert outputs["rvuot/rival_score_mix"].item() == pytest.approx(0.0)
    assert 0.0 <= outputs["rvuot/retained_mass_ratio"].item() <= 1.0
    assert 0.0 <= outputs["rvuot/evidence_retained_mass_ratio"].item() <= 1.0
    assert "local_scores" in outputs
    assert "global_scores" in outputs
    assert outputs["global_residual_weight"].item() == pytest.approx(0.10)


def test_reciprocal_verified_uot_rejects_legacy_verified_score_combo():
    with pytest.raises(ValueError, match="enable_reciprocal_verified_uot cannot be combined"):
        _tiny_ours_final(enable_reciprocal_verified_uot=True, enable_verified_uot_score=True)


def test_pot_guide_enabled_exposes_diagnostics_and_nonuniform_marginals():
    torch.manual_seed(513)
    model = _tiny_ours_final(enable_pot_guide=True, pot_guide_max_iter=4)
    model.eval()
    query, support = _episode()

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert model.enable_pot_guide
    assert hasattr(model, "pot_guide_module")
    for key in (
        "pot_guide/alpha",
        "pot_guide/temperature",
        "pot_guide/s_mean",
        "pot_guide/pot_sparsity",
        "pot_guide/marginal_q_max",
        "pot_guide/marginal_q_min",
    ):
        assert key in outputs
        assert torch.isfinite(outputs[key])
    uniform = model.ecot_base_rho / float(outputs["transport_plan"].shape[-2])
    assert not torch.isclose(outputs["pot_guide/marginal_q_max"], outputs["pot_guide/marginal_q_min"])
    assert outputs["pot_guide/marginal_q_max"] > uniform


def test_pot_guide_adaptive_s_receives_gradient():
    torch.manual_seed(514)
    model = _tiny_ours_final(enable_pot_guide=True, pot_guide_adaptive_s=True, pot_guide_max_iter=4)
    model.train()
    query, support = _episode()

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].pow(2).sum()
    loss.backward()

    assert model.pot_guide_module.raw_alpha.grad is not None
    assert model.pot_guide_module.log_tau.grad is not None
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.pot_guide_module.s_predictor.parameters()
    )


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


def test_label_ot_disabled_matches_ours_final_baseline_logits():
    torch.manual_seed(515)
    baseline = _tiny_ours_final()
    label_ot_off = _tiny_ours_final(enable_label_ot=False)
    label_ot_off.load_state_dict(baseline.state_dict())
    baseline.eval()
    label_ot_off.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support)
        actual = label_ot_off(query, support)

    assert torch.equal(actual, expected)


def test_label_ot_is_eval_only_and_rebalances_class_bias():
    model = _tiny_ours_final(
        enable_label_ot=True,
        label_ot_epsilon=0.5,
        label_ot_iterations=40,
        label_ot_mix=1.0,
        label_ot_min_queries_per_class=1,
    )
    logits = torch.tensor(
        [
            [3.0, 0.0],
            [2.8, 0.1],
            [2.6, 0.2],
            [2.4, 0.3],
        ]
    )

    model.eval()
    adjusted, payload = model._apply_label_ot_to_logits(logits)

    assert payload["label_ot/active"].item() == pytest.approx(1.0)
    assert payload["label_ot/column_imbalance_after"] < payload["label_ot/column_imbalance_before"]
    assert adjusted[:, 0].mean() < logits[:, 0].mean()
    assert adjusted[:, 1].mean() > logits[:, 1].mean()

    model.train()
    train_adjusted, train_payload = model._apply_label_ot_to_logits(logits)
    assert torch.equal(train_adjusted, logits)
    assert train_payload["label_ot/active"].item() == pytest.approx(0.0)


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


def test_label_ot_factory_flag_is_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_label_ot=True,
        label_ot_epsilon=0.5,
        label_ot_iterations=12,
        label_ot_mix=0.75,
        label_ot_min_queries_per_class=1,
        label_ot_min_column_imbalance=0.05,
        label_ot_max_bias=1.5,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )
    assert ours_final.enable_label_ot
    assert ours_final.label_ot_epsilon == pytest.approx(0.5)
    assert ours_final.label_ot_iterations == 12
    assert ours_final.label_ot_mix == pytest.approx(0.75)

    with pytest.raises(ValueError, match="--enable_label_ot is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


def test_multiscale_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_multiscale_ot=True,
        multiscale_pool_sizes="original,2x2",
        multiscale_per_scale_T=False,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )
    assert hasattr(ours_final, "multi_scale_tokenizer")
    assert ours_final.scale_weights.shape == (2,)

    with pytest.raises(ValueError, match="--enable_multiscale_ot is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


def test_region_structural_uot_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_region_structural_uot=True,
        region_uot_grid_size="2x2",
        region_uot_strength=0.15,
        region_uot_fgw_alpha=0.30,
        region_uot_fgw_iters=1,
        region_uot_sinkhorn_iters=4,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )
    assert hasattr(ours_final, "region_structural_uot")
    assert ours_final.enable_region_structural_uot

    with pytest.raises(ValueError, match="--enable_region_structural_uot is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


def test_adaptive_region_uot_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_adaptive_region_uot=True,
        adaptive_region_num_slots=3,
        adaptive_region_context_kernels="1,3",
        adaptive_region_cost_discount=0.10,
        adaptive_region_mass_mix=0.7,
        adaptive_region_sinkhorn_iters=4,
        adaptive_region_init_gate=0.03,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )
    assert hasattr(ours_final, "adaptive_region_uot")
    assert ours_final.enable_adaptive_region_uot
    assert ours_final.adaptive_region_uot.num_slots == 3
    assert torch.sigmoid(ours_final.adaptive_region_uot.raw_cost_gate).item() == pytest.approx(0.03, abs=1e-5)

    with pytest.raises(ValueError, match="--enable_adaptive_region_uot is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


def test_reciprocal_verified_uot_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_reciprocal_verified_uot=True,
        rvuot_beta=0.8,
        rvuot_tau=0.12,
        rvuot_ratio_threshold=0.22,
        rvuot_kernel_size=3,
        rvuot_cost_quantile=0.30,
        rvuot_min_gate=0.04,
        rvuot_enable_rival_gate="true",
        rvuot_rival_score_mix=0.15,
        rvuot_rival_tau=0.11,
        rvuot_rival_margin=0.02,
        rvuot_detach_gate="true",
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )
    assert ours_final.enable_reciprocal_verified_uot
    assert hasattr(ours_final, "reciprocal_verified_transport")
    assert ours_final.rvuot_beta == pytest.approx(0.8)
    assert ours_final.rvuot_tau == pytest.approx(0.12)
    assert ours_final.rvuot_ratio_threshold == pytest.approx(0.22)
    assert ours_final.rvuot_cost_quantile == pytest.approx(0.30)
    assert ours_final.rvuot_enable_rival_gate
    assert ours_final.rvuot_rival_score_mix == pytest.approx(0.15)
    assert ours_final.rvuot_rival_tau == pytest.approx(0.11)
    assert ours_final.rvuot_rival_margin == pytest.approx(0.02)

    with pytest.raises(ValueError, match="--enable_reciprocal_verified_uot is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


def test_episode_contrastive_uot_penalizes_common_mode_before_transport():
    model = _tiny_ours_final(
        enable_episode_contrastive_uot=True,
        ect_uot_tau=0.25,
        ect_uot_margin=0.0,
        ect_uot_cost_weight=0.50,
        ect_uot_query_mass_mix=1.0,
        ect_uot_query_tau=0.25,
    )
    cost = torch.ones(1, 2, 4, 4)

    # Query token 0 is common-mode: both classes provide an equally cheap match.
    cost[:, 0, 0, 0] = 0.01
    cost[:, 1, 0, 0] = 0.01

    # Query token 1 is class-specific evidence for class 0.
    cost[:, 0, 1, 1] = 0.01
    cost[:, 1, 1, 1] = 0.80

    guided_cost, query_weight, payload = model._apply_episode_contrastive_uot(
        cost,
        query_weight=None,
        way_num=2,
        shot_num=1,
    )

    assert guided_cost.shape == cost.shape
    assert query_weight is not None
    assert query_weight.shape == (1, 2, 4)
    assert guided_cost[0, 0, 0, 0] > guided_cost[0, 0, 1, 1]
    assert query_weight[0, 0, 1] > query_weight[0, 0, 0]
    assert payload["episode_contrast/gate_mean"].item() < 1.0
    assert payload["episode_contrast/cost_delta_ratio"].item() > 0.0


def test_transport_specificity_audit_reports_common_mass():
    model = _tiny_ours_final()
    cost = torch.ones(1, 2, 4, 4)
    plan = torch.zeros(1, 2, 1, 4, 4)
    cost[:, 0, 0, 0] = 0.01
    cost[:, 1, 0, 0] = 0.01
    cost[:, 0, 1, 1] = 0.01
    cost[:, 1, 1, 1] = 0.80
    plan[:, 0, 0, 0, 0] = 0.30
    plan[:, 0, 0, 1, 1] = 0.10

    audit = model._transport_specificity_audit(
        flat_cost=cost,
        plan=plan,
        way_num=2,
        shot_num=1,
        prefix="audit",
    )

    assert audit["audit/common_mass_ratio"].item() > 0.0
    assert audit["audit/specific_mass_ratio"].item() > 0.0
    assert audit["audit/common_score_term"].item() > 0.0
    assert 0.0 <= audit["audit/common_score_positive_rate"].item() <= 1.0
    assert 0.0 <= audit["audit/common_score_abs_share"].item() <= 1.0
    assert audit["audit/cost_per_mass"].item() > 0.0


def test_residual_aligned_uot_uses_global_residual_anchor_before_transport():
    model = _tiny_ours_final(
        enable_global_residual_score=True,
        enable_residual_aligned_uot=True,
        rauot_tau=0.35,
        rauot_cost_weight=0.50,
        rauot_mass_mix=0.75,
    )
    query_tokens = torch.tensor(
        [
            [
                [2.0, 0.0],
                [-2.0, 0.0],
                [0.0, 1.0],
            ]
        ]
    )
    support_tokens = torch.tensor(
        [
            [
                [
                    [2.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ],
            [
                [
                    [-2.0, 0.0],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                ]
            ],
        ]
    )
    cost = torch.ones(1, 2, 3, 3)
    cost[0, 0, 0, 2] = 2.0
    cost[0, 1, 1, 2] = 2.0

    guided_cost, query_weight, support_weight, payload = model._apply_residual_aligned_uot(
        cost,
        query_weight=None,
        support_weight=None,
        query_tokens=query_tokens,
        support_tokens=support_tokens,
        way_num=2,
        shot_num=1,
    )

    assert guided_cost.shape == cost.shape
    assert query_weight is not None
    assert support_weight is not None
    assert query_weight.shape == (1, 2, 3)
    assert support_weight.shape == (2, 1, 3)
    assert guided_cost[0, 0, 0, 0] < guided_cost[0, 0, 2, 2]
    assert query_weight[0, 0, 0] > query_weight[0, 0, 1]
    assert query_weight[0, 1, 1] > query_weight[0, 1, 0]
    assert support_weight[0, 0, 0] > support_weight[0, 0, 2]
    assert support_weight[1, 0, 0] > support_weight[1, 0, 2]
    assert payload["residual_aligned/pair_evidence_std"].item() > 0.0
    assert payload["residual_aligned/cost_delta_ratio"].item() > 0.0


def test_residual_aligned_uot_requires_global_residual_score():
    with pytest.raises(ValueError, match="requires enable_global_residual_score"):
        _tiny_ours_final(enable_residual_aligned_uot=True)


def test_rival_aware_evidence_uot_flag_off_matches_baseline_logits():
    torch.manual_seed(518)
    baseline = _tiny_ours_final()
    flag_off = _tiny_ours_final(enable_rival_aware_evidence_uot=False)
    flag_off.load_state_dict(baseline.state_dict())
    baseline.eval()
    flag_off.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support)
        actual = flag_off(query, support)

    assert torch.equal(actual, expected)


def test_rival_gate_is_invariant_to_duplicate_rival_shots():
    model = _tiny_ours_final()
    one_shot_cost = torch.tensor(
        [
            [
                [[0.1, 0.8], [0.7, 0.2]],
                [[0.4, 0.6], [0.3, 0.9]],
            ]
        ],
        dtype=torch.float32,
    )
    five_shot_cost = (
        one_shot_cost.reshape(1, 2, 1, 2, 2)
        .expand(-1, -1, 5, -1, -1)
        .reshape(1, 10, 2, 2)
        .clone()
    )

    gate_one, _ = model._episode_contrastive_pair_gate(
        one_shot_cost,
        way_num=2,
        shot_num=1,
        tau=0.25,
        margin=0.0,
    )
    gate_five, _ = model._episode_contrastive_pair_gate(
        five_shot_cost,
        way_num=2,
        shot_num=5,
        tau=0.25,
        margin=0.0,
    )

    assert torch.allclose(gate_five[:, :, 0], gate_one[:, :, 0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(gate_five, gate_five[:, :, :1].expand_as(gate_five), atol=1e-6, rtol=1e-6)


def test_rival_aware_evidence_uot_suppresses_common_match_and_shapes_both_marginals():
    model = _tiny_ours_final(
        enable_rival_aware_evidence_uot=True,
        rae_uot_tau=0.25,
        rae_uot_marginal_mix=1.0,
    )
    query_tokens = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ]
    )
    support_tokens = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ],
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
        ]
    )
    cost = torch.ones(1, 2, 3, 3)
    cost[0, 0, 0, 0] = 0.01
    cost[0, 1, 0, 0] = 0.01
    cost[0, 0, 1, 1] = 0.01
    cost[0, 1, 1, 1] = 0.90

    unchanged_cost, query_weight, support_weight, pair_evidence, payload = (
        model._apply_rival_aware_evidence_uot(
            cost,
            query_weight=None,
            support_weight=None,
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            way_num=2,
            shot_num=1,
        )
    )

    assert query_weight is not None
    assert support_weight is not None
    assert pair_evidence is not None
    assert query_weight.shape == (1, 2, 3)
    assert support_weight.shape == (1, 2, 3)
    assert pair_evidence.shape == (1, 2, 1, 3, 3)
    assert torch.allclose(query_weight.sum(dim=-1), torch.ones(1, 2), atol=1e-6)
    assert torch.allclose(support_weight.sum(dim=-1), torch.ones(1, 2), atol=1e-6)
    assert pair_evidence[0, 0, 0, 1, 1] > pair_evidence[0, 0, 0, 0, 0]
    assert torch.equal(unchanged_cost, cost)
    assert query_weight[0, 0, 1] > query_weight[0, 0, 0]
    assert support_weight[0, 0, 1] > support_weight[0, 0, 0]
    assert payload["rae_uot/evidence_std"].item() > 0.0


def test_rival_aware_evidence_uot_common_only_case_falls_back_to_uniform_marginals():
    model = _tiny_ours_final(
        enable_rival_aware_evidence_uot=True,
        rae_uot_marginal_mix=1.0,
    )
    query_tokens = torch.tensor([[[4.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    support_tokens = torch.tensor(
        [
            [[[4.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
            [[[4.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        ]
    )
    cost = torch.ones(1, 2, 3, 3)

    _, query_weight, support_weight, pair_evidence, payload = (
        model._apply_rival_aware_evidence_uot(
            cost,
            query_weight=None,
            support_weight=None,
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            way_num=2,
            shot_num=1,
        )
    )

    assert torch.allclose(query_weight, torch.full_like(query_weight, 1.0 / 3.0), atol=1e-6)
    assert torch.allclose(support_weight, torch.full_like(support_weight, 1.0 / 3.0), atol=1e-6)
    assert torch.count_nonzero(pair_evidence) == 0
    assert payload["rae_uot/uniform_fallback_query_share"].item() == pytest.approx(1.0)
    assert payload["rae_uot/uniform_fallback_support_share"].item() == pytest.approx(1.0)


def test_rival_aware_evidence_uot_common_only_case_keeps_original_local_score():
    model = _tiny_ours_final(enable_rival_aware_evidence_uot=True)
    model.eval()
    query_tokens = torch.tensor([[[4.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    support_tokens = torch.tensor(
        [
            [[[4.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
            [[[4.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        ]
    )
    cost = torch.ones(1, 2, 3, 3)

    with torch.no_grad():
        outputs = model._forward_ecot_budget_bank(
            cost,
            way_num=2,
            shot_num=1,
            query_tokens=query_tokens,
            support_tokens=support_tokens,
        )

    assert "shot_evidence_score_mass" not in outputs
    assert torch.isfinite(outputs["shot_logits"]).all()
    assert torch.all(outputs["shot_logits"] < 0.0)
    assert torch.allclose(
        outputs["shot_logits"][:, 0],
        outputs["shot_logits"][:, 1],
        atol=1e-6,
        rtol=1e-6,
    )


def test_rival_aware_evidence_uot_is_equivariant_to_token_position_permutations():
    torch.manual_seed(519)
    model = _tiny_ours_final(enable_rival_aware_evidence_uot=True)
    way_num, shot_num = 3, 2
    query_tokens = torch.randn(2, 4, 6)
    support_tokens = torch.randn(way_num, shot_num, 5, 6)
    cost = torch.rand(2, way_num * shot_num, 4, 5)
    query_perm = torch.tensor([2, 0, 3, 1])
    support_perm = torch.tensor([4, 1, 3, 0, 2])

    original = model._apply_rival_aware_evidence_uot(
        cost,
        query_weight=None,
        support_weight=None,
        query_tokens=query_tokens,
        support_tokens=support_tokens,
        way_num=way_num,
        shot_num=shot_num,
    )
    permuted = model._apply_rival_aware_evidence_uot(
        cost[:, :, query_perm][:, :, :, support_perm],
        query_weight=None,
        support_weight=None,
        query_tokens=query_tokens[:, query_perm],
        support_tokens=support_tokens[:, :, support_perm],
        way_num=way_num,
        shot_num=shot_num,
    )

    original_cost, original_query, original_support, original_evidence, _ = original
    permuted_cost, permuted_query, permuted_support, permuted_evidence, _ = permuted
    assert torch.allclose(
        permuted_cost,
        original_cost[:, :, query_perm][:, :, :, support_perm],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(permuted_query, original_query[:, :, query_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(permuted_support, original_support[:, :, support_perm], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        permuted_evidence,
        original_evidence[:, :, :, query_perm][:, :, :, :, support_perm],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("shot", [1, 5])
def test_rival_aware_evidence_uot_forward_is_finite_for_few_shot(shot):
    torch.manual_seed(520 + shot)
    model = _tiny_ours_final(
        enable_rival_aware_evidence_uot=True,
        enable_global_residual_score=True,
        global_residual_weight=0.10,
    )
    model.eval()
    query, support = _episode(shot=shot)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert not model.ecot_m2_cost_per_mass_score
    assert model.egsm_marginal is None
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["transport_plan"]).all()
    assert "shot_evidence_score_mass" not in outputs
    assert outputs["rae_uot_pair_evidence"].shape[2] == shot
    assert torch.allclose(
        outputs["rae_uot_query_weight"].sum(dim=-1),
        torch.ones_like(outputs["rae_uot_query_weight"].sum(dim=-1)),
        atol=1e-5,
    )
    assert torch.allclose(
        outputs["rae_uot_support_weight"].sum(dim=-1),
        torch.ones_like(outputs["rae_uot_support_weight"].sum(dim=-1)),
        atol=1e-5,
    )


def test_ours_final_uniform_marginal_mode_matches_original_baseline():
    torch.manual_seed(522)
    baseline = _tiny_ours_final()
    explicit_uniform = _tiny_ours_final(ours_final_marginal_mode="uniform")
    explicit_uniform.load_state_dict(baseline.state_dict())
    baseline.eval()
    explicit_uniform.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support)
        actual = explicit_uniform(query, support)

    assert torch.equal(actual, expected)


def test_ours_final_failure_probe_detects_negative_utility_and_common_mass():
    probe = OursFinalFailureProbe(common_margin=0.10)
    cost = torch.tensor(
        [
            [
                [[0.10, 0.90], [0.90, 0.90]],
                [[0.10, 0.90], [0.20, 0.90]],
            ]
        ]
    )
    plan = torch.zeros(1, 2, 1, 2, 2)
    plan[0, 0, 0, 0, 0] = 0.4
    plan[0, 0, 0, 1, 1] = 0.4
    plan[0, 1, 0, 0, 0] = 0.4
    plan[0, 1, 0, 1, 0] = 0.4

    payload = probe(
        cost,
        plan,
        threshold=torch.tensor(0.5),
        way_num=2,
        shot_num=1,
    )

    assert payload["ours_probe_negative_utility_mass_ratio"][0, 0].item() == pytest.approx(0.5)
    assert payload["ours_probe_negative_utility_mass_ratio"][0, 1].item() == pytest.approx(0.0)
    assert payload["ours_probe_common_mass_ratio"][0, 0].item() == pytest.approx(0.5)
    assert payload["ours_probe_dead_query_ratio"][0, 0].item() == pytest.approx(0.5)
    assert payload["ours_probe_harm_share"][0, 0].item() == pytest.approx(0.5)
    assert payload["ours_probe_utility_score"][0, 1] > payload["ours_probe_utility_score"][0, 0]


def test_ours_final_failure_probe_flag_does_not_change_logits():
    torch.manual_seed(5221)
    baseline = _tiny_ours_final()
    probed = _tiny_ours_final(enable_ours_final_failure_probe=True)
    probed.load_state_dict(baseline.state_dict())
    baseline.eval()
    probed.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support, return_aux=True)
        actual = probed(query, support, return_aux=True)

    assert torch.equal(actual["logits"], expected["logits"])
    assert torch.equal(actual["transport_plan"], expected["transport_plan"])
    assert actual["ours_probe_negative_utility_mass_ratio"].shape == actual["logits"].shape
    assert torch.isfinite(actual["ours_probe/harm_share"])


def test_ours_final_failure_probe_runner_builds_baseline_and_novelty_ablation():
    variants = build_ours_final_failure_probe_variants()

    assert [variant["tag"] for variant in variants] == [
        "ours_final_probe_uniform",
        "ours_final_probe_utility_fixed",
        "ours_final_probe_utility_adaptive",
    ]
    assert all("--enable_ours_final_failure_probe" in variant["extra_args"] for variant in variants)
    assert "--ours_final_marginal_mode" not in variants[0]["extra_args"]
    assert variants[1]["extra_args"][-1] == "false"
    assert variants[2]["extra_args"][-1] == "true"


def test_rival_conditional_cost_zero_weight_is_exact_identity():
    module = RivalConditionalGroundCost(
        penalty_weight=0.0,
        temperature=0.25,
    )
    cost = torch.rand(2, 6, 4, 3)

    guided, payload = module(cost, way_num=3, shot_num=2)

    assert guided is cost
    assert torch.count_nonzero(payload["rc_cost_penalty"]) == 0
    assert payload["rc_cost/penalty_to_cost_ratio"].item() == pytest.approx(0.0)


def test_rival_conditional_cost_penalizes_common_and_rival_matches():
    module = RivalConditionalGroundCost(
        penalty_weight=0.5,
        temperature=0.25,
    )
    cost = torch.tensor(
        [
            [
                [[0.10, 0.10], [0.10, 0.20]],
                [[0.10, 0.10], [0.90, 1.00]],
            ]
        ],
        dtype=torch.float32,
    )

    guided, payload = module(cost, way_num=2, shot_num=1)
    probability = payload["rc_cost_class_probability"]
    penalty = payload["rc_cost_penalty"]

    assert probability[0, 0, 0].item() == pytest.approx(0.5, abs=1e-5)
    assert probability[0, 1, 0].item() == pytest.approx(0.5, abs=1e-5)
    assert probability[0, 0, 1] > probability[0, 1, 1]
    assert penalty[0, 0, 0] > penalty[0, 0, 1]
    assert penalty[0, 1, 1] > penalty[0, 0, 1]
    torch.testing.assert_close(
        guided.reshape(1, 2, 1, 2, 2) - cost.reshape(1, 2, 1, 2, 2),
        penalty[:, :, None, :, None].expand(-1, -1, 1, -1, 2),
    )


def test_rival_conditional_cost_has_finite_ground_cost_gradient():
    module = RivalConditionalGroundCost(
        penalty_weight=0.5,
        temperature=0.25,
    )
    cost = torch.rand(2, 3, 4, 5, requires_grad=True)

    guided, _ = module(cost, way_num=3, shot_num=1)
    guided.square().mean().backward()

    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()
    assert cost.grad.abs().sum() > 0


def test_edge_rival_cost_preserves_common_edges_and_separates_specific_edges():
    module = RivalConditionalGroundCost(
        penalty_weight=0.25,
        temperature=0.25,
        mode="edge_advantage",
    )
    cost = torch.tensor(
        [
            [
                [[0.10, 0.10], [0.10, 0.20]],
                [[0.10, 0.10], [0.90, 1.00]],
            ]
        ],
        dtype=torch.float32,
    )

    guided, payload = module(cost, way_num=2, shot_num=1)
    adjustment = payload["rc_cost_adjustment"].reshape_as(cost)
    episode_scale = cost.std(unbiased=False)

    torch.testing.assert_close(guided[0, :, 0], cost[0, :, 0], atol=1e-6, rtol=0.0)
    assert guided[0, 0, 1, 0] < cost[0, 0, 1, 0]
    assert guided[0, 1, 1, 0] > cost[0, 1, 1, 0]
    assert adjustment.abs().amax() <= 0.25 * episode_scale + 1e-6
    assert payload["rc_cost/mode_id"].item() == pytest.approx(1.0)
    assert payload["rc_cost/penalty_mean"].abs() < 1e-6


def test_rival_conditional_cost_zero_weight_preserves_model_logits():
    torch.manual_seed(5222)
    baseline = _tiny_ours_final()
    identity = _tiny_ours_final(
        enable_rival_conditional_cost=True,
        rc_cost_weight=0.0,
    )
    identity.load_state_dict(baseline.state_dict())
    baseline.eval()
    identity.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support, return_aux=True)
        actual = identity(query, support, return_aux=True)

    assert torch.equal(actual["logits"], expected["logits"])
    assert torch.equal(actual["transport_plan"], expected["transport_plan"])
    assert actual["rc_cost/enabled"].item() == pytest.approx(1.0)


def test_rival_conditional_cost_model_changes_only_ground_cost_and_backpropagates():
    torch.manual_seed(5223)
    model = _tiny_ours_final(
        enable_rival_conditional_cost=True,
        rc_cost_weight=0.5,
        rc_cost_temperature=0.25,
    )
    model.train()
    query, support = _episode()
    targets = torch.tensor([0, 1], dtype=torch.long)

    outputs = model(
        query,
        support,
        query_targets=targets,
        return_aux=True,
    )
    loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    model.zero_grad(set_to_none=True)
    loss.backward()

    assert outputs["cost_matrix"].shape == outputs["base_cost_matrix"].shape
    assert not torch.equal(outputs["cost_matrix"], outputs["base_cost_matrix"])
    assert torch.isfinite(outputs["logits"]).all()
    assert outputs["rc_cost/penalty_to_cost_ratio"] > 0.0
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0.0


def test_rival_conditional_cost_runner_isolates_one_architecture_change():
    variants = build_ours_final_rival_cost_variants()

    assert [variant["tag"] for variant in variants] == [
        "ours_final_rc_cost_baseline",
        "ours_final_rc_cost",
        "ours_final_rc_edge",
    ]
    assert "--enable_rival_conditional_cost" not in variants[0]["extra_args"]
    assert "--enable_rival_conditional_cost" in variants[1]["extra_args"]
    assert "--ours_final_marginal_mode" not in variants[1]["extra_args"]
    assert "--enable_global_residual_score" not in variants[1]["extra_args"]
    assert variants[2]["extra_args"][-1] == "edge_advantage"


def test_rival_conditional_cost_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_rival_conditional_cost="true",
        rc_cost_weight=0.4,
        rc_cost_temperature=0.3,
        rc_cost_mode="edge_advantage",
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )

    assert ours_final.enable_rival_conditional_cost
    assert ours_final.rc_cost_weight == pytest.approx(0.4)
    assert ours_final.rc_cost_temperature == pytest.approx(0.3)
    assert ours_final.rc_cost_mode == "edge_advantage"
    assert hasattr(ours_final, "rival_conditional_ground_cost")

    with pytest.raises(
        ValueError,
        match="--enable_rival_conditional_cost is supported only with --model ours_final",
    ):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


def test_score_aligned_ours_final_marginals_are_nonuniform_and_query_shared():
    model = _tiny_ours_final(
        ours_final_marginal_mode="score_aligned",
        score_marginal_tau=0.25,
        score_marginal_mix=1.0,
    )
    cost = torch.ones(1, 2, 3, 3)
    cost[0, 0, 0, 0] = 0.01
    cost[0, 1, 0, 0] = 0.01
    cost[0, 0, 1, 1] = 0.01
    cost[0, 1, 1, 1] = 0.90

    query_weight, support_weight, payload = model._apply_score_aligned_marginals(
        cost,
        query_weight=None,
        support_weight=None,
        way_num=2,
        shot_num=1,
    )

    assert query_weight.shape == (1, 3)
    assert support_weight.shape == (1, 2, 3)
    assert torch.allclose(query_weight.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(support_weight.sum(dim=-1), torch.ones(1, 2), atol=1e-6)
    assert query_weight[0, 1] > query_weight[0, 0]
    assert support_weight[0, 0, 1] > support_weight[0, 0, 0]
    assert payload["score_marginal/query_is_class_shared"].item() == pytest.approx(1.0)
    assert payload["score_marginal/query_weight_peak"].item() > 1.0 / 3.0
    assert payload["score_marginal/effective_mix"].item() <= 1.0
    assert payload["score_marginal/evidence_confidence"].item() > 0.0


def test_score_aligned_ours_final_improves_controlled_score_margin_over_uniform():
    uniform_model = _tiny_ours_final(enable_ours_final_failure_probe=True)
    score_marginal_model = _tiny_ours_final(
        ours_final_marginal_mode="score_aligned",
        score_marginal_tau=0.25,
        score_marginal_mix=1.0,
        enable_ours_final_failure_probe=True,
    )
    uniform_model.eval()
    score_marginal_model.eval()
    cost = torch.ones(1, 2, 3, 3)
    cost[0, 0, 0, 0] = 0.01
    cost[0, 1, 0, 0] = 0.01
    cost[0, 0, 1, 1] = 0.01
    cost[0, 1, 1, 1] = 0.90

    with torch.no_grad():
        uniform_outputs = uniform_model._forward_ecot_budget_bank(
            cost,
            way_num=2,
            shot_num=1,
        )
        score_marginal_outputs = score_marginal_model._forward_ecot_budget_bank(
            cost,
            way_num=2,
            shot_num=1,
        )

    uniform_margin = (
        uniform_outputs["shot_logits"][0, 0, 0]
        - uniform_outputs["shot_logits"][0, 1, 0]
    )
    score_marginal_margin = (
        score_marginal_outputs["shot_logits"][0, 0, 0]
        - score_marginal_outputs["shot_logits"][0, 1, 0]
    )
    assert score_marginal_margin > uniform_margin
    assert (
        score_marginal_outputs["ours_probe_negative_utility_mass_ratio"][0, 0]
        < uniform_outputs["ours_probe_negative_utility_mass_ratio"][0, 0]
    )
    assert (
        score_marginal_outputs["ours_probe_common_mass_ratio"][0, 0]
        < uniform_outputs["ours_probe_common_mass_ratio"][0, 0]
    )
    assert (
        score_marginal_outputs["ours_probe_harm_share"][0, 0]
        < uniform_outputs["ours_probe_harm_share"][0, 0]
    )
    assert torch.isfinite(score_marginal_outputs["score_marginal/l1_drift"])


def test_score_aligned_ours_final_common_only_case_falls_back_to_uniform():
    model = _tiny_ours_final(
        ours_final_marginal_mode="score_aligned",
        score_marginal_mix=1.0,
    )
    cost = torch.ones(1, 2, 3, 3)

    query_weight, support_weight, payload = model._apply_score_aligned_marginals(
        cost,
        query_weight=None,
        support_weight=None,
        way_num=2,
        shot_num=1,
    )

    assert torch.allclose(query_weight, torch.full_like(query_weight, 1.0 / 3.0), atol=1e-6)
    assert torch.allclose(support_weight, torch.full_like(support_weight, 1.0 / 3.0), atol=1e-6)
    assert payload["score_marginal/uniform_fallback_query_share"].item() == pytest.approx(1.0)
    assert payload["score_marginal/uniform_fallback_support_share"].item() == pytest.approx(1.0)


def test_score_aligned_ours_final_is_invariant_to_duplicate_shots():
    torch.manual_seed(523)
    model = _tiny_ours_final(ours_final_marginal_mode="score_aligned")
    way_num = 3
    cost_one = torch.rand(2, way_num, 4, 5)
    cost_five = (
        cost_one.reshape(2, way_num, 1, 4, 5)
        .expand(-1, -1, 5, -1, -1)
        .reshape(2, way_num * 5, 4, 5)
        .clone()
    )

    query_one, support_one, payload_one = model._apply_score_aligned_marginals(
        cost_one,
        query_weight=None,
        support_weight=None,
        way_num=way_num,
        shot_num=1,
    )
    query_five, support_five, payload_five = model._apply_score_aligned_marginals(
        cost_five,
        query_weight=None,
        support_weight=None,
        way_num=way_num,
        shot_num=5,
    )

    assert torch.allclose(query_five, query_one, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        support_five.reshape(2, way_num, 5, 5)[:, :, 0],
        support_one.reshape(2, way_num, 1, 5)[:, :, 0],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        payload_five["score_marginal_pair_evidence"][:, :, 0],
        payload_one["score_marginal_pair_evidence"][:, :, 0],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("shot", [1, 5])
def test_score_aligned_ours_final_forward_is_finite(shot):
    torch.manual_seed(524 + shot)
    model = _tiny_ours_final(
        ours_final_marginal_mode="score_aligned",
        enable_global_residual_score=True,
        global_residual_weight=0.10,
    )
    model.eval()
    query, support = _episode(shot=shot)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert not model.ecot_m2_cost_per_mass_score
    assert model.egsm_marginal is None
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["transport_plan"]).all()
    assert outputs["score_marginal_pair_evidence"].shape[2] == shot
    assert torch.allclose(
        outputs["score_marginal_query_weight"].sum(dim=-1),
        torch.ones_like(outputs["score_marginal_query_weight"].sum(dim=-1)),
        atol=1e-5,
    )
    assert torch.allclose(
        outputs["score_marginal_support_weight"].sum(dim=-1),
        torch.ones_like(outputs["score_marginal_support_weight"].sum(dim=-1)),
        atol=1e-5,
    )
    assert torch.isfinite(outputs["score_marginal/query_l1_drift"])
    assert torch.isfinite(outputs["score_marginal/support_l1_drift"])


def test_score_aligned_ours_final_factory_flags_are_forwarded_and_scoped():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        ours_final_marginal_mode="score_aligned",
        score_marginal_tau=0.31,
        score_marginal_mix=0.72,
        score_marginal_adaptive_mix="false",
        score_marginal_confidence_power=1.5,
        enable_ours_final_failure_probe="true",
        ours_probe_common_margin=0.12,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(model="ours_final", ours_ablation="full", **common)
    )

    assert ours_final.ours_final_marginal_mode == "score_aligned"
    assert ours_final.score_marginal_tau == pytest.approx(0.31)
    assert ours_final.score_marginal_mix == pytest.approx(0.72)
    assert not ours_final.score_marginal_adaptive_mix
    assert ours_final.score_marginal_confidence_power == pytest.approx(1.5)
    assert ours_final.enable_ours_final_failure_probe
    assert ours_final.ours_probe_common_margin == pytest.approx(0.12)
    assert not ours_final.ecot_m2_cost_per_mass_score
    assert ours_final.egsm_marginal is None

    with pytest.raises(
        ValueError,
        match="supported only with --model ours_final",
    ):
        build_model_from_args(
            SimpleNamespace(model="ours", ours_ablation="full", **common)
        )


def test_score_aligned_ours_final_rejects_competing_marginal_paths():
    with pytest.raises(ValueError, match="cannot be combined"):
        _tiny_ours_final(
            ours_final_marginal_mode="score_aligned",
            enable_rival_aware_evidence_uot=True,
        )


def test_episode_contrastive_uot_factory_flags_are_forwarded():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_episode_contrastive_uot="true",
        ect_uot_tau=0.31,
        ect_uot_margin=0.02,
        ect_uot_cost_weight=0.40,
        ect_uot_query_mass_mix=0.60,
        ect_uot_query_tau=0.70,
        ect_uot_detach="false",
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )

    assert ours_final.enable_episode_contrastive_uot
    assert ours_final.ect_uot_tau == pytest.approx(0.31)
    assert ours_final.ect_uot_margin == pytest.approx(0.02)
    assert ours_final.ect_uot_cost_weight == pytest.approx(0.40)
    assert ours_final.ect_uot_query_mass_mix == pytest.approx(0.60)
    assert ours_final.ect_uot_query_tau == pytest.approx(0.70)
    assert not ours_final.ect_uot_detach


def test_residual_aligned_uot_factory_flags_are_forwarded():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_global_residual_score="true",
        global_residual_weight=0.1,
        enable_residual_aligned_uot="true",
        rauot_tau=0.41,
        rauot_margin=0.03,
        rauot_cost_weight=0.17,
        rauot_mass_mix=0.29,
        rauot_detach="false",
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )

    assert ours_final.enable_global_residual_score
    assert ours_final.enable_residual_aligned_uot
    assert ours_final.rauot_tau == pytest.approx(0.41)
    assert ours_final.rauot_margin == pytest.approx(0.03)
    assert ours_final.rauot_cost_weight == pytest.approx(0.17)
    assert ours_final.rauot_mass_mix == pytest.approx(0.29)
    assert not ours_final.rauot_detach


def test_rival_aware_evidence_uot_factory_flags_are_forwarded_and_scoped():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_rival_aware_evidence_uot="true",
        rae_uot_tau=0.31,
        rae_uot_marginal_mix=0.72,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )

    assert ours_final.enable_rival_aware_evidence_uot
    assert ours_final.rae_uot_tau == pytest.approx(0.31)
    assert ours_final.rae_uot_marginal_mix == pytest.approx(0.72)
    assert not ours_final.ecot_m2_cost_per_mass_score
    assert ours_final.egsm_marginal is None

    with pytest.raises(
        ValueError,
        match="--enable_rival_aware_evidence_uot is supported only with --model ours_final",
    ):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


def test_rival_aware_evidence_uot_rejects_competing_local_transport_paths():
    with pytest.raises(ValueError, match="defines the complete local transport path"):
        _tiny_ours_final(
            enable_rival_aware_evidence_uot=True,
            enable_reciprocal_verified_uot=True,
        )


def test_hubness_calibrated_uot_suppresses_common_hubs_before_transport():
    model = _tiny_ours_final(
        enable_hubness_calibrated_uot=True,
        hcuot_topk=1,
        hcuot_temperature=0.25,
        hcuot_cost_weight=0.50,
        hcuot_marginal_mix=1.0,
    )
    query_tokens = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]]
    )
    support_tokens = torch.tensor(
        [
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]],
        ]
    )
    cost = torch.ones(1, 2, 3, 3)

    guided_cost, query_weight, support_weight, specificity, payload = (
        model._apply_hubness_calibrated_uot(
            cost,
            query_weight=None,
            support_weight=None,
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            way_num=2,
            shot_num=1,
        )
    )

    assert guided_cost[0, 0, 1, 1] < guided_cost[0, 0, 0, 0]
    assert specificity[0, 0, 0, 1, 1] > specificity[0, 0, 0, 0, 0]
    assert query_weight[0, 0, 1] > query_weight[0, 0, 0]
    assert support_weight[0, 0, 1] > support_weight[0, 0, 0]
    assert torch.allclose(query_weight.sum(dim=-1), torch.ones(1, 2), atol=1e-6)
    assert torch.allclose(support_weight.sum(dim=-1), torch.ones(1, 2), atol=1e-6)
    assert payload["hcuot/cost_delta_ratio"].item() > 0.0
    assert payload["hcuot/common_pair_share"].item() > 0.0


def test_hubness_calibrated_uot_common_only_case_is_neutral():
    model = _tiny_ours_final(
        enable_hubness_calibrated_uot=True,
        hcuot_topk=1,
        hcuot_marginal_mix=1.0,
    )
    shared = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    query_tokens = shared.unsqueeze(0)
    support_tokens = shared.view(1, 1, 3, 2).expand(2, -1, -1, -1).clone()
    cost = torch.ones(1, 2, 3, 3)

    guided_cost, query_weight, support_weight, specificity, payload = (
        model._apply_hubness_calibrated_uot(
            cost,
            query_weight=None,
            support_weight=None,
            query_tokens=query_tokens,
            support_tokens=support_tokens,
            way_num=2,
            shot_num=1,
        )
    )

    assert torch.allclose(guided_cost, cost, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(specificity) == 0
    assert torch.allclose(query_weight, torch.full_like(query_weight, 1.0 / 3.0), atol=1e-6)
    assert torch.allclose(support_weight, torch.full_like(support_weight, 1.0 / 3.0), atol=1e-6)
    assert payload["hcuot/uniform_fallback_query_share"].item() == pytest.approx(1.0)
    assert payload["hcuot/uniform_fallback_support_share"].item() == pytest.approx(1.0)


def test_hubness_calibrated_uot_is_invariant_to_duplicate_shots():
    torch.manual_seed(530)
    model = _tiny_ours_final(enable_hubness_calibrated_uot=True, hcuot_topk=2)
    query_tokens = torch.randn(2, 4, 6)
    support_one = torch.randn(3, 1, 5, 6)
    support_five = support_one.expand(-1, 5, -1, -1).clone()
    cost_one = torch.rand(2, 3, 4, 5)
    cost_five = (
        cost_one.reshape(2, 3, 1, 4, 5)
        .expand(-1, -1, 5, -1, -1)
        .reshape(2, 15, 4, 5)
        .clone()
    )

    one = model._apply_hubness_calibrated_uot(
        cost_one,
        query_weight=None,
        support_weight=None,
        query_tokens=query_tokens,
        support_tokens=support_one,
        way_num=3,
        shot_num=1,
    )
    five = model._apply_hubness_calibrated_uot(
        cost_five,
        query_weight=None,
        support_weight=None,
        query_tokens=query_tokens,
        support_tokens=support_five,
        way_num=3,
        shot_num=5,
    )

    one_cost, one_query, one_support, one_specificity, _ = one
    five_cost, five_query, five_support, five_specificity, _ = five
    assert torch.allclose(
        five_cost.reshape(2, 3, 5, 4, 5)[:, :, 0],
        one_cost.reshape(2, 3, 1, 4, 5)[:, :, 0],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        five_query.reshape(2, 3, 5, 4)[:, :, 0],
        one_query.reshape(2, 3, 1, 4)[:, :, 0],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        five_support.reshape(2, 3, 5, 5)[:, :, 0],
        one_support.reshape(2, 3, 1, 5)[:, :, 0],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        five_specificity[:, :, 0],
        one_specificity[:, :, 0],
        atol=1e-6,
        rtol=1e-6,
    )


def test_hubness_calibrated_uot_is_equivariant_to_token_permutations():
    torch.manual_seed(535)
    model = _tiny_ours_final(enable_hubness_calibrated_uot=True, hcuot_topk=2)
    way_num, shot_num = 3, 2
    query_tokens = torch.randn(2, 4, 6)
    support_tokens = torch.randn(way_num, shot_num, 5, 6)
    cost = torch.rand(2, way_num * shot_num, 4, 5)
    query_perm = torch.tensor([2, 0, 3, 1])
    support_perm = torch.tensor([4, 1, 3, 0, 2])

    original = model._apply_hubness_calibrated_uot(
        cost,
        query_weight=None,
        support_weight=None,
        query_tokens=query_tokens,
        support_tokens=support_tokens,
        way_num=way_num,
        shot_num=shot_num,
    )
    permuted = model._apply_hubness_calibrated_uot(
        cost[:, :, query_perm][:, :, :, support_perm],
        query_weight=None,
        support_weight=None,
        query_tokens=query_tokens[:, query_perm],
        support_tokens=support_tokens[:, :, support_perm],
        way_num=way_num,
        shot_num=shot_num,
    )

    original_cost, original_query, original_support, original_specificity, _ = original
    permuted_cost, permuted_query, permuted_support, permuted_specificity, _ = permuted
    assert torch.allclose(
        permuted_cost,
        original_cost[:, :, query_perm][:, :, :, support_perm],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        permuted_query,
        original_query[:, :, query_perm],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        permuted_support,
        original_support[:, :, support_perm],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        permuted_specificity,
        original_specificity[:, :, :, query_perm][:, :, :, :, support_perm],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("shot", [1, 5])
def test_hubness_calibrated_uot_forward_is_finite_for_few_shot(shot):
    torch.manual_seed(540 + shot)
    model = _tiny_ours_final(
        enable_hubness_calibrated_uot=True,
        enable_global_residual_score=True,
        global_residual_weight=0.10,
    )
    model.eval()
    query, support = _episode(shot=shot)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["transport_plan"]).all()
    assert outputs["hcuot_pair_specificity"].shape[2] == shot
    assert torch.allclose(
        outputs["hcuot_query_weight"].sum(dim=-1),
        torch.ones_like(outputs["hcuot_query_weight"].sum(dim=-1)),
        atol=1e-5,
    )
    assert torch.allclose(
        outputs["hcuot_support_weight"].sum(dim=-1),
        torch.ones_like(outputs["hcuot_support_weight"].sum(dim=-1)),
        atol=1e-5,
    )


def test_hubness_calibrated_uot_factory_flags_are_forwarded_and_scoped():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_hubness_calibrated_uot="true",
        hcuot_topk=2,
        hcuot_temperature=0.31,
        hcuot_cost_weight=0.42,
        hcuot_marginal_mix=0.67,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(model="ours_final", ours_ablation="full", **common)
    )

    assert ours_final.enable_hubness_calibrated_uot
    assert ours_final.hcuot_topk == 2
    assert ours_final.hcuot_temperature == pytest.approx(0.31)
    assert ours_final.hcuot_cost_weight == pytest.approx(0.42)
    assert ours_final.hcuot_marginal_mix == pytest.approx(0.67)
    assert ours_final.egsm_marginal is None

    with pytest.raises(
        ValueError,
        match="--enable_hubness_calibrated_uot is supported only with --model ours_final",
    ):
        build_model_from_args(
            SimpleNamespace(model="ours", ours_ablation="full", **common)
        )


def test_hubness_calibrated_uot_rejects_competing_local_transport_paths():
    with pytest.raises(ValueError, match="defines the complete local transport path"):
        _tiny_ours_final(
            enable_hubness_calibrated_uot=True,
            enable_rival_aware_evidence_uot=True,
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


# ──────────────────────────────────────────────────────────
# Spatial Context Enrichment tests
# ──────────────────────────────────────────────────────────


def test_context_enrichment_off_matches_baseline_logits():
    torch.manual_seed(600)
    baseline = _tiny_ours_final()
    flag_off = _tiny_ours_final(enable_context_enrichment=False)
    flag_off.load_state_dict(baseline.state_dict())
    baseline.eval()
    flag_off.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support)
        actual = flag_off(query, support)

    assert not baseline.enable_context_enrichment
    assert not hasattr(baseline, "context_enrichment")
    assert not flag_off.enable_context_enrichment
    assert not hasattr(flag_off, "context_enrichment")
    assert torch.equal(actual, expected)


def test_context_enrichment_enabled_runs_and_exposes_diagnostics():
    torch.manual_seed(601)
    model = _tiny_ours_final(enable_context_enrichment=True, context_kernel_sizes="3,5")
    model.train()
    query, support = _episode()

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()

    assert model.enable_context_enrichment
    assert hasattr(model, "context_enrichment")
    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()
    for key in (
        "context/gate_value",
        "context/branch_weight_original",
        "context/branch_weight_conv3",
        "context/branch_weight_conv5",
        "context/token_change_ratio",
    ):
        assert key in outputs, f"Missing diagnostic key: {key}"
        assert torch.isfinite(outputs[key]), f"Non-finite diagnostic: {key}"


def test_context_enrichment_gate_starts_near_zero():
    torch.manual_seed(602)
    model = _tiny_ours_final(enable_context_enrichment=True, context_kernel_sizes="3,5")
    model.eval()
    query, support = _episode()

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    gate = outputs["context/gate_value"].item()
    assert gate < 0.05, f"Initial gate should be near 0, got {gate}"
    assert outputs["context/token_change_ratio"].item() < 0.1


def test_context_enrichment_gradient_flows_to_gate_and_branches():
    torch.manual_seed(603)
    model = _tiny_ours_final(enable_context_enrichment=True, context_kernel_sizes="3,5")
    model.train()
    query, support = _episode()

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].pow(2).sum()
    loss.backward()

    ce = model.context_enrichment
    assert ce.raw_gate.grad is not None
    assert torch.isfinite(ce.raw_gate.grad).all()
    assert ce.branch_weights.grad is not None
    assert torch.isfinite(ce.branch_weights.grad).all()
    for branch in ce.branches:
        for param in branch.parameters():
            assert param.grad is not None, "Depthwise conv param has no gradient"
            assert torch.isfinite(param.grad).all()


def test_context_enrichment_bottleneck_fusion_runs():
    torch.manual_seed(604)
    model = _tiny_ours_final(
        enable_context_enrichment=True,
        context_kernel_sizes="3,5",
        context_fusion="bottleneck",
    )
    model.eval()
    query, support = _episode()

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()
    assert "context/token_change_ratio" in outputs


def test_context_enrichment_bottleneck_gradient_flows():
    torch.manual_seed(605)
    model = _tiny_ours_final(
        enable_context_enrichment=True,
        context_kernel_sizes="3",
        context_fusion="bottleneck",
    )
    model.train()
    query, support = _episode()

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].pow(2).sum()
    loss.backward()

    ce = model.context_enrichment
    for param in ce.fusion_conv.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_context_enrichment_single_kernel():
    torch.manual_seed(606)
    model = _tiny_ours_final(enable_context_enrichment=True, context_kernel_sizes="3")
    model.eval()
    query, support = _episode()

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 2)
    assert "context/branch_weight_conv3" in outputs
    assert "context/branch_weight_conv5" not in outputs


def test_context_enrichment_triple_kernel():
    torch.manual_seed(607)
    model = _tiny_ours_final(enable_context_enrichment=True, context_kernel_sizes="3,5,7")
    model.eval()
    query, support = _episode()

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 2)
    for key in (
        "context/branch_weight_conv3",
        "context/branch_weight_conv5",
        "context/branch_weight_conv7",
    ):
        assert key in outputs


def test_context_enrichment_combined_with_multiscale():
    torch.manual_seed(608)
    model = _tiny_ours_final(
        enable_context_enrichment=True,
        context_kernel_sizes="3,5",
        enable_multiscale_ot=True,
        multiscale_pool_sizes="original,2x2,1x1",
    )
    model.eval()
    query, support = _episode()

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()
    assert "context/gate_value" in outputs
    assert "multiscale/scale_weights" in outputs


def test_context_enrichment_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_context_enrichment=True,
        context_kernel_sizes="3,5",
        context_fusion="weighted_sum",
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )
    assert hasattr(ours_final, "context_enrichment")
    assert ours_final.enable_context_enrichment

    with pytest.raises(ValueError, match="--enable_context_enrichment is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


# ───────── Structural Token Augmentation tests ─────────


def test_structural_augmentation_off_matches_baseline_logits():
    torch.manual_seed(800)
    baseline = _tiny_ours_final()
    flag_off = _tiny_ours_final(enable_structural_augmentation=False)
    flag_off.load_state_dict(baseline.state_dict())
    baseline.eval()
    flag_off.eval()
    query, support = _episode()
    with torch.no_grad():
        expected = baseline(query, support)
        actual = flag_off(query, support)
    assert not baseline.enable_structural_augmentation
    assert not hasattr(baseline, "structural_augmentation")
    assert torch.equal(actual, expected)


def test_structural_augmentation_enabled_runs_and_exposes_diagnostics():
    torch.manual_seed(801)
    model = _tiny_ours_final(enable_structural_augmentation=True, struct_dim=8)
    model.train()
    query, support = _episode()
    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()
    assert model.enable_structural_augmentation
    assert hasattr(model, "structural_augmentation")
    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()
    for key in (
        "struct/token_change_ratio",
        "struct/struct_weight_norm",
        "struct/semantic_weight_norm",
        "struct/struct_vs_semantic_ratio",
    ):
        assert key in outputs, f"Missing diagnostic key: {key}"


def test_structural_augmentation_gradient_flows():
    torch.manual_seed(802)
    model = _tiny_ours_final(enable_structural_augmentation=True, struct_dim=8)
    model.train()
    query, support = _episode()
    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()
    sta = model.structural_augmentation
    assert sta.fuse.weight.grad is not None
    assert sta.fuse.weight.grad.abs().sum() > 0
    for p in sta.struct_proj.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_structural_augmentation_init_near_identity():
    torch.manual_seed(803)
    model = _tiny_ours_final(enable_structural_augmentation=True, struct_dim=8)
    sta = model.structural_augmentation
    w = sta.fuse.weight.detach()
    semantic_part = w[:, :sta.token_dim]
    struct_part = w[:, sta.token_dim:]
    identity = torch.eye(sta.token_dim)
    assert torch.allclose(semantic_part, identity, atol=1e-6)
    assert torch.allclose(struct_part, torch.zeros_like(struct_part), atol=1e-6)


def test_structural_augmentation_combined_with_multiscale():
    torch.manual_seed(804)
    model = _tiny_ours_final(
        enable_structural_augmentation=True,
        struct_dim=8,
        enable_multiscale_ot=True,
        multiscale_pool_sizes="original,2x2,1x1",
    )
    model.train()
    query, support = _episode()
    outputs = model(query, support, return_aux=True)
    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()
    assert "struct/token_change_ratio" in outputs


def test_structural_augmentation_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_structural_augmentation=True,
        struct_dim=8,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )
    assert hasattr(ours_final, "structural_augmentation")
    assert ours_final.enable_structural_augmentation

    with pytest.raises(ValueError, match="--enable_structural_augmentation is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


# Pulse-region guided UOT tests


def test_pulse_region_uot_enabled_runs_and_exposes_diagnostics():
    torch.manual_seed(900)
    model = _tiny_ours_final(
        enable_pulse_region_uot=True,
        pulse_region_kernel_size=3,
        pulse_region_cost_weight=0.25,
        pulse_saliency_mass_mix=0.5,
    )
    model.train()
    query, support = _episode(shot=1)
    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()

    assert model.enable_pulse_region_uot
    assert hasattr(model, "pulse_region_guidance")
    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()
    assert "pulse_query_saliency" in outputs
    assert "pulse_support_saliency" in outputs
    assert "pulse_query_evidence" not in outputs
    assert "pulse_support_evidence" not in outputs
    assert "pulse_guided_cost_matrix" in outputs
    assert "base_cost_matrix" in outputs
    assert outputs["cost_matrix"].shape == outputs["base_cost_matrix"].shape
    assert outputs["pulse_query_saliency"].shape[0] == outputs["logits"].shape[0]
    for key in (
        "pulse/region_cost_weight",
        "pulse/saliency_mass_mix",
        "pulse/query_saliency_peak",
        "pulse/support_saliency_peak",
        "pulse/region_cost_delta_ratio",
        "pulse/effective_strength",
    ):
        assert key in outputs, f"Missing diagnostic key: {key}"


def test_pulse_region_evidence_mask_expands_single_token_peak():
    model = _tiny_ours_final(enable_pulse_region_uot=True, pulse_region_kernel_size=3)
    saliency = torch.zeros(1, 16)
    saliency[0, 5] = 1.0

    evidence = model._pulse_region_evidence_mask(saliency, spatial_hw=(4, 4))

    assert evidence.shape == saliency.shape
    assert evidence.max().item() == pytest.approx(1.0)
    assert int((evidence > 0.0).sum().item()) > 1


def test_pulse_discriminative_evidence_gate_prefers_class_specific_matches():
    model = _tiny_ours_final(
        enable_pulse_region_uot=True,
        pulse_discriminative_tau=0.05,
        pulse_discriminative_margin=0.02,
    )
    cost = torch.full((1, 2, 2, 2), 0.6)
    cost[:, 0] = 0.05
    query_evidence = torch.ones(1, 2)
    support_evidence = torch.ones(2, 1, 2)

    pair_evidence, payload = model._pulse_discriminative_pair_evidence(
        flat_cost=cost,
        query_evidence=query_evidence,
        support_evidence=support_evidence,
        way_num=2,
        shot_num=1,
        strength=cost.new_tensor(1.0),
    )

    assert pair_evidence.shape == (1, 2, 1, 2, 2)
    assert pair_evidence[:, 0].mean() > pair_evidence[:, 1].mean()
    assert pair_evidence[:, 0].mean() > 0.95
    assert pair_evidence[:, 1].mean() < 0.05
    assert "pulse/discriminative_gate_mean" in payload


def test_origin_discriminative_uot_runs_without_pulse_region_guidance():
    torch.manual_seed(904)
    baseline = _tiny_ours_final()
    model = _tiny_ours_final(
        enable_discriminative_uot=True,
        discriminative_uot_tau=0.05,
        discriminative_uot_margin=0.02,
        discriminative_uot_background_penalty=0.25,
    )
    model.load_state_dict(baseline.state_dict())
    baseline.eval()
    model.eval()
    query, support = _episode(shot=1)

    with torch.no_grad():
        baseline_outputs = baseline(query, support, return_aux=True)
        outputs = model(query, support, return_aux=True)

    assert not model.enable_pulse_region_uot
    assert "discriminative_uot_gate" in outputs
    assert "discriminative_uot/enabled" in outputs
    assert "pulse_query_evidence" not in outputs
    assert torch.isfinite(outputs["logits"]).all()
    assert not torch.allclose(outputs["logits"], baseline_outputs["logits"])


def test_pulse_evidence_score_changes_logits_and_tracks_background_mass():
    torch.manual_seed(903)
    common = dict(
        enable_pulse_region_uot=True,
        pulse_region_kernel_size=3,
        pulse_region_cost_weight=0.25,
        pulse_saliency_mass_mix=0.5,
    )
    baseline = _tiny_ours_final(**common, pulse_evidence_score=False)
    evidence = _tiny_ours_final(**common, pulse_evidence_score=True, pulse_background_penalty=0.25)
    evidence.load_state_dict(baseline.state_dict())
    baseline.eval()
    evidence.eval()

    query = torch.zeros(1, 2, 3, 64, 64)
    support = torch.zeros(1, 2, 1, 3, 64, 64)
    query[:, 0, :, 10:22, 10:22] = 4.0
    support[:, 0, 0, :, 10:22, 10:22] = 4.0
    query[:, 1, :, 42:54, 42:54] = 4.0
    support[:, 1, 0, :, 42:54, 42:54] = 4.0

    with torch.no_grad():
        baseline_outputs = baseline(query, support, return_aux=True)
        evidence_outputs = evidence(query, support, return_aux=True)

    assert not torch.allclose(evidence_outputs["logits"], baseline_outputs["logits"])
    assert "shot_evidence_score_mass" in evidence_outputs
    assert "shot_evidence_score_background_mass" in evidence_outputs
    assert torch.all(evidence_outputs["shot_evidence_score_mass"] >= 0.0)
    assert torch.all(evidence_outputs["shot_evidence_score_background_mass"] >= 0.0)
    assert torch.all(
        evidence_outputs["shot_evidence_score_background_mass"]
        <= evidence_outputs["shot_transported_mass"] + 1e-6
    )


def test_pulse_region_train_schedule_reduces_train_guidance_but_not_eval():
    torch.manual_seed(901)
    model = _tiny_ours_final(
        enable_pulse_region_uot=True,
        pulse_region_kernel_size=3,
        pulse_region_cost_weight=0.25,
        pulse_saliency_mass_mix=0.5,
        pulse_region_train_strength=0.6,
        pulse_region_eval_strength=1.0,
        pulse_region_train_schedule="decay",
    )
    query, support = _episode(shot=1)

    model.train()
    model.set_homotopy_progress(0.5)
    train_outputs = model(query, support, return_aux=True)

    model.eval()
    with torch.no_grad():
        eval_outputs = model(query, support, return_aux=True)

    assert train_outputs["pulse/effective_strength"].item() == pytest.approx(0.3, abs=1e-6)
    assert eval_outputs["pulse/effective_strength"].item() == pytest.approx(1.0, abs=1e-6)


def test_pulse_region_multishot_strength_and_support_consensus():
    torch.manual_seed(902)
    model = _tiny_ours_final(
        enable_pulse_region_uot=True,
        pulse_region_kernel_size=3,
        pulse_region_cost_weight=0.25,
        pulse_saliency_mass_mix=0.5,
        pulse_region_train_strength=0.6,
        pulse_region_eval_strength=1.0,
        pulse_region_multishot_train_strength=0.25,
        pulse_region_multishot_eval_strength=0.45,
        pulse_support_consensus_weight=0.7,
    )
    query, support = _episode(shot=5)

    model.train()
    train_outputs = model(query, support, return_aux=True)
    model.eval()
    with torch.no_grad():
        eval_outputs = model(query, support, return_aux=True)

    assert train_outputs["pulse/effective_strength"].item() == pytest.approx(0.25, abs=1e-6)
    assert eval_outputs["pulse/effective_strength"].item() == pytest.approx(0.45, abs=1e-6)
    for key in (
        "pulse/support_consensus_weight",
        "pulse/support_consensus_peak",
        "pulse/support_consensus_entropy",
        "pulse/support_consistency_mean",
        "pulse/support_consistency_sigma_peak",
    ):
        assert key in train_outputs
        assert torch.isfinite(train_outputs[key])


def test_pulse_region_uot_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_pulse_region_uot=True,
        pulse_region_kernel_size=3,
        pulse_region_train_schedule="decay",
        pulse_region_train_strength=0.6,
        pulse_region_eval_strength=1.0,
        pulse_region_multishot_train_strength=0.25,
        pulse_region_multishot_eval_strength=0.45,
        pulse_support_consensus_weight=0.7,
        pulse_evidence_score="false",
        pulse_evidence_mass_weight=1.25,
        pulse_evidence_cost_weight=0.75,
        pulse_background_penalty=0.15,
        pulse_discriminative_evidence="false",
        pulse_discriminative_tau=0.07,
        pulse_discriminative_margin=0.03,
        pulse_discriminative_mix=0.5,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )
    assert hasattr(ours_final, "pulse_region_guidance")
    assert ours_final.enable_pulse_region_uot
    assert ours_final.pulse_region_train_schedule == "decay"
    assert ours_final.pulse_region_train_strength == pytest.approx(0.6)
    assert ours_final.pulse_region_eval_strength == pytest.approx(1.0)
    assert ours_final.pulse_region_multishot_train_strength == pytest.approx(0.25)
    assert ours_final.pulse_region_multishot_eval_strength == pytest.approx(0.45)
    assert ours_final.pulse_support_consensus_weight == pytest.approx(0.7)
    assert not ours_final.pulse_evidence_score
    assert ours_final.pulse_evidence_mass_weight == pytest.approx(1.25)
    assert ours_final.pulse_evidence_cost_weight == pytest.approx(0.75)
    assert ours_final.pulse_background_penalty == pytest.approx(0.15)
    assert not ours_final.pulse_discriminative_evidence
    assert ours_final.pulse_discriminative_tau == pytest.approx(0.07)
    assert ours_final.pulse_discriminative_margin == pytest.approx(0.03)
    assert ours_final.pulse_discriminative_mix == pytest.approx(0.5)

    with pytest.raises(ValueError, match="--enable_pulse_region_uot is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


def test_origin_discriminative_uot_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_discriminative_uot=True,
        discriminative_uot_tau=0.07,
        discriminative_uot_margin=0.03,
        discriminative_uot_mix=0.5,
        discriminative_uot_background_penalty=0.15,
        discriminative_uot_mass_weight=1.2,
        discriminative_uot_cost_weight=0.8,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )
    assert ours_final.enable_discriminative_uot
    assert ours_final.discriminative_uot_tau == pytest.approx(0.07)
    assert ours_final.discriminative_uot_margin == pytest.approx(0.03)
    assert ours_final.discriminative_uot_mix == pytest.approx(0.5)
    assert ours_final.discriminative_uot_background_penalty == pytest.approx(0.15)
    assert ours_final.discriminative_uot_mass_weight == pytest.approx(1.2)
    assert ours_final.discriminative_uot_cost_weight == pytest.approx(0.8)

    with pytest.raises(ValueError, match="--enable_discriminative_uot is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )


# ── Evidence marginals tests ──────────────────────────────────────────


def test_evidence_marginals_off_matches_baseline():
    torch.manual_seed(600)
    baseline = _tiny_ours_final()
    evidence_off = _tiny_ours_final(enable_evidence_marginals=False)
    evidence_off.load_state_dict(baseline.state_dict())
    baseline.eval()
    evidence_off.eval()
    query, support = _episode()

    with torch.no_grad():
        expected = baseline(query, support)
        actual = evidence_off(query, support)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("mode", ["query_only", "support_only", "both", "rival_aware"])
def test_evidence_marginals_forward_runs(mode):
    torch.manual_seed(601)
    model = _tiny_ours_final(
        enable_evidence_marginals=True,
        evidence_tau=0.1,
        evidence_tau_marginal=1.0,
        evidence_mode=mode,
        evidence_rival_margin=0.5,
    )
    model.train()
    query, support = _episode()

    logits = model(query, support)
    assert logits.shape == (2, 2)
    assert logits.isfinite().all()


def test_evidence_marginals_changes_logits():
    torch.manual_seed(602)
    baseline = _tiny_ours_final()
    evidence = _tiny_ours_final(
        enable_evidence_marginals=True,
        evidence_tau=0.1,
        evidence_tau_marginal=1.0,
        evidence_mode="both",
    )
    baseline.eval()
    evidence.eval()
    query, support = _episode()

    with torch.no_grad():
        logits_base = baseline(query, support)
        logits_ev = evidence(query, support)

    assert not torch.allclose(logits_base, logits_ev, atol=1e-6)


def test_evidence_marginals_multishot():
    torch.manual_seed(603)
    model = _tiny_ours_final(
        enable_evidence_marginals=True,
        evidence_tau=0.1,
        evidence_tau_marginal=1.0,
        evidence_mode="rival_aware",
        evidence_rival_margin=0.5,
    )
    model.eval()
    query, support = _episode(shot=5)

    with torch.no_grad():
        logits = model(query, support)

    assert logits.shape == (2, 2)
    assert logits.isfinite().all()


def test_evidence_marginals_gradient_flows():
    torch.manual_seed(604)
    model = _tiny_ours_final(
        enable_evidence_marginals=True,
        evidence_tau=0.1,
        evidence_tau_marginal=1.0,
        evidence_mode="both",
        evidence_detach=False,
    )
    model.train()
    query, support = _episode()

    logits = model(query, support)
    loss = logits.sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_evidence_marginals_rejects_discriminative_combo():
    with pytest.raises(ValueError, match="cannot be combined"):
        _tiny_ours_final(
            enable_evidence_marginals=True,
            marginal_kind="discriminative",
            token_g_kind="episode_mean_dist",
        )


def test_evidence_module_standalone():
    from net.modules.cost_evidence_marginals import CostEvidenceMarginals

    torch.manual_seed(605)
    module = CostEvidenceMarginals(
        tau_evidence=0.1,
        tau_marginal=1.0,
        mode="both",
        rival_margin=0.5,
    )
    cost = torch.rand(3, 4, 16, 16) * 2.0
    rho = torch.tensor(0.8)

    q_marg, s_marg, diagnostics = module(cost, rho, way_num=2, shot_num=2)

    assert q_marg is not None
    assert s_marg is not None
    assert q_marg.shape == (3, 4, 16)
    assert s_marg.shape == (2, 2, 16)
    assert q_marg.sum(dim=-1).allclose(rho.expand(3, 4), atol=1e-5)
    assert s_marg.sum(dim=-1).allclose(rho.expand(2, 2), atol=1e-5)
    assert "evidence/query_entropy" in diagnostics
    assert "evidence/support_entropy" in diagnostics


def test_evidence_module_query_only():
    from net.modules.cost_evidence_marginals import CostEvidenceMarginals

    module = CostEvidenceMarginals(mode="query_only")
    cost = torch.rand(2, 4, 8, 8)
    rho = torch.tensor(0.8)

    q_marg, s_marg, diag = module(cost, rho, way_num=2, shot_num=2)

    assert q_marg is not None
    assert s_marg is None


def test_evidence_module_support_only():
    from net.modules.cost_evidence_marginals import CostEvidenceMarginals

    module = CostEvidenceMarginals(mode="support_only")
    cost = torch.rand(2, 4, 8, 8)
    rho = torch.tensor(0.8)

    q_marg, s_marg, diag = module(cost, rho, way_num=2, shot_num=2)

    assert q_marg is None
    assert s_marg is not None


def test_evidence_module_rival_aware_differs_from_both():
    from net.modules.cost_evidence_marginals import CostEvidenceMarginals

    torch.manual_seed(606)
    cost = torch.rand(3, 6, 16, 16)
    rho = torch.tensor(0.8)

    mod_both = CostEvidenceMarginals(mode="both", tau_evidence=0.1, tau_marginal=1.0)
    mod_rival = CostEvidenceMarginals(
        mode="rival_aware", tau_evidence=0.1, tau_marginal=1.0, rival_margin=0.5,
    )

    q_both, _, _ = mod_both(cost, rho, way_num=3, shot_num=2)
    q_rival, _, _ = mod_rival(cost, rho, way_num=3, shot_num=2)

    assert not torch.allclose(q_both, q_rival, atol=1e-6)
