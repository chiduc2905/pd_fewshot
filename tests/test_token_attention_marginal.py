from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from net.model_factory import build_model_from_args
from net.modules.token_attention_marginal import TokenAttentionMarginal
from net.ours import OursFinalM2
from run_all_experiments import (
    build_tier1_diagnostic_variants,
    filter_ours_final_variants,
    ours_final_tau_profile_args,
    parse_ours_final_tau_profile,
    parse_ours_final_variant_filter,
)


def _set_temperature(module: TokenAttentionMarginal, value: float) -> None:
    with torch.no_grad():
        module.raw_temperature.copy_(
            torch.log(torch.expm1(torch.tensor(float(value))))
        )


def test_token_attention_marginal_shape_and_mass_conservation():
    torch.manual_seed(10)
    module = TokenAttentionMarginal(token_dim=7, hidden_dim=5)
    tokens = torch.randn(2, 3, 11, 7)
    rho = torch.tensor([[0.6, 0.7, 0.8], [0.5, 0.9, 0.4]])

    marginal, diagnostics = module(tokens, rho)

    assert marginal.shape == (2, 3, 11)
    assert torch.allclose(marginal.sum(dim=-1), rho, atol=1e-6, rtol=1e-6)
    assert diagnostics["token_marginal/mass_error"].item() < 1e-6


def test_token_attention_marginal_uniform_floor_one_is_uniform():
    module = TokenAttentionMarginal(
        token_dim=4,
        hidden_dim=3,
        uniform_floor=1.0,
    )
    tokens = torch.randn(2, 5, 4)
    marginal, _ = module(tokens, torch.tensor([0.5, 0.8]))
    expected = torch.tensor([[0.1] * 5, [0.16] * 5])

    assert torch.allclose(marginal, expected, atol=1e-6, rtol=1e-6)


def test_token_attention_marginal_backward_is_finite():
    torch.manual_seed(11)
    module = TokenAttentionMarginal(token_dim=6, hidden_dim=4)
    tokens = torch.randn(3, 9, 6, requires_grad=True)
    marginal, _ = module(tokens, torch.full((3,), 0.8))
    loss = (marginal.square() * torch.arange(1, 10)).sum()

    loss.backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_token_attention_marginal_temperature_controls_peakiness():
    high = TokenAttentionMarginal(
        token_dim=2,
        hidden_dim=1,
        temperature=10.0,
        uniform_floor=0.0,
    )
    low = TokenAttentionMarginal(
        token_dim=2,
        hidden_dim=1,
        temperature=0.05,
        uniform_floor=0.0,
    )
    with torch.no_grad():
        for module in (high, low):
            module.scorer[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
            module.scorer[0].bias.zero_()
            module.scorer[2].weight.fill_(1.0)
            module.scorer[2].bias.zero_()
        _set_temperature(high, 10.0)
        _set_temperature(low, 0.05)
    tokens = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]])

    high_marginal, _ = high(tokens, 1.0)
    low_marginal, _ = low(tokens, 1.0)

    assert low_marginal.max() > high_marginal.max()
    assert low_marginal.max() > 0.95


def test_tier1_suite_has_15_unique_configs_and_global_residual_baseline():
    variants = build_tier1_diagnostic_variants()

    assert len(variants) == 15
    assert len({variant["tag"] for variant in variants}) == 15
    assert len({tuple(variant["extra_args"]) for variant in variants}) == 15
    for variant in variants:
        if variant["tag"] == "t1_tam_no_gres":
            assert "--enable_global_residual_score" not in variant["extra_args"]
        else:
            assert "--enable_global_residual_score" in variant["extra_args"]
            weight_idx = variant["extra_args"].index("--global_residual_weight")
            assert variant["extra_args"][weight_idx + 1] == "0.1"

    default_alias = parse_ours_final_variant_filter("t1_T_init_default")
    selected = filter_ours_final_variants(variants, default_alias)
    assert [variant["tag"] for variant in selected] == ["t1_tau_sym_0p5"]


def test_ours_final_tau_profile_support_strict_builds_single_override_flag():
    assert parse_ours_final_tau_profile("support_strict") == (0.5, 0.8)
    assert parse_ours_final_tau_profile("0.3,0.8") == (0.3, 0.8)
    assert ours_final_tau_profile_args("t1_tau_c_strict") == [
        "--hrot_tau_q",
        "0.5",
        "--hrot_tau_c",
        "0.8",
    ]


def test_ours_final_factory_wires_token_attention_marginal():
    args = SimpleNamespace(
        model="ours_final",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        ours_ablation="full",
        enable_token_attention_marginal="true",
        tam_hidden_dim=12,
        tam_temperature_init=0.7,
        tam_uniform_floor=0.2,
        tam_detach_weights="false",
        tam_share_qk="true",
        enable_global_residual_score="true",
        global_residual_weight=0.1,
    )

    model = build_model_from_args(args)

    assert isinstance(model, OursFinalM2)
    assert model.uses_token_attention_marginal
    assert model.token_attention_marginal is not None
    assert model.token_attention_support_marginal is None
    assert model.tam_hidden_dim == 12
    assert model.tam_uniform_floor == pytest.approx(0.2)
    assert model.enable_global_residual_score
    assert model.global_residual_weight == pytest.approx(0.1)


def test_ours_final_factory_can_use_separate_query_support_tam_scorers():
    args = SimpleNamespace(
        model="ours_final",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        ours_ablation="full",
        enable_token_attention_marginal="true",
        tam_share_qk="false",
    )

    model = build_model_from_args(args)

    assert model.token_attention_marginal is not None
    assert model.token_attention_support_marginal is not None
    assert (
        model.token_attention_support_marginal
        is not model.token_attention_marginal
    )


def test_ours_final_token_attention_marginal_logs_effectiveness_diagnostics():
    torch.manual_seed(12)
    model = OursFinalM2(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=8,
        eam_hidden_dim=16,
        sinkhorn_iterations=4,
        sinkhorn_tolerance=1e-5,
        eval_use_float64=False,
        ecot_m2_ablate_threshold_mass=False,
        ecot_rho_bank="0.8",
        ecot_base_rho=0.8,
        ecot_transport_mode="unbalanced",
        enable_token_attention_marginal=True,
        enable_global_residual_score=True,
        global_residual_weight=0.1,
    )
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)
    model.eval()

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["token_marginal_query_weight"].shape[:2] == (2, 16)
    assert outputs["token_marginal_support_weight"].shape[:3] == (2, 2, 1)
    assert torch.allclose(
        outputs["token_marginal_query_weight"].sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.allclose(
        outputs["token_marginal_support_weight"].sum(dim=-1),
        torch.ones(2, 2, 1),
        atol=1e-6,
    )
    for key in (
        "token_marginal/entropy",
        "token_marginal/l1_from_uniform",
        "token_marginal/l1_drift",
        "token_marginal/transported_mass_fraction",
        "token_marginal/scorer_norm",
        "global_residual_weight",
    ):
        assert key in outputs
        assert torch.isfinite(outputs[key])


def test_token_attention_marginal_rejects_competing_marginal_path():
    with pytest.raises(
        ValueError,
        match="enable_token_attention_marginal isolates learned token marginals",
    ):
        OursFinalM2(
            backbone_name="conv64f",
            image_size=64,
            hidden_dim=64,
            token_dim=8,
            eam_hidden_dim=16,
            enable_token_attention_marginal=True,
            ours_final_marginal_mode="score_aligned",
        )
