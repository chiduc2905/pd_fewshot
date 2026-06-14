from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from net.model_factory import build_model_from_args
from net.modules.verified_region_matching import VerifiedRegionMatchingUOT
from net.ours import OursFinalM2
from run_all_experiments import (
    build_ours_final_verified_region_variants,
    filter_ours_final_variants,
    parse_ours_final_variant_filter,
)


def test_verified_region_matching_shapes_and_multiplicative_cost():
    torch.manual_seed(20)
    module = VerifiedRegionMatchingUOT(
        penalty_lambda=0.2,
        use_concentration=True,
        use_region_patch=True,
        use_rival=False,
    )
    flat_cost = torch.rand(2, 3, 4, 4)
    query_tokens = torch.randn(2, 4, 6)
    support_tokens = torch.randn(3, 1, 4, 6)

    guided, payload = module(
        flat_cost=flat_cost,
        query_tokens=query_tokens,
        support_tokens=support_tokens,
        way_num=3,
        shot_num=1,
        spatial_hw=(2, 2),
    )

    assert guided.shape == flat_cost.shape
    assert torch.all(guided >= flat_cost)
    assert payload["verified_region_pair_gate"].shape == (2, 3, 1, 4, 4)
    assert payload["verified_region_guided_cost_matrix"].shape == (2, 3, 1, 4, 4)
    for key in (
        "verified/gate_mean",
        "verified/gate_q90",
        "verified/concentration_score_mean",
        "verified/patch_consistency_mean",
        "verified/region_vs_token_consistency_gap",
        "verified/cost_delta_ratio",
    ):
        assert key in payload
        assert torch.isfinite(payload[key])


def test_verified_region_matching_full_gate_uses_soft_and_not_product():
    module = VerifiedRegionMatchingUOT(
        penalty_lambda=0.2,
        use_concentration=True,
        use_region_patch=True,
        use_rival=True,
    )
    flat_cost = torch.ones(1, 1, 2, 2)
    query_tokens = torch.randn(1, 2, 3)
    support_tokens = torch.randn(1, 1, 2, 3)
    concentration = torch.full_like(flat_cost, 0.01)
    patch = torch.full_like(flat_cost, 0.25)
    rival = torch.full_like(flat_cost, 0.64)

    module._concentration_gate = lambda cost: concentration
    module._region_patch_gate = lambda **kwargs: (patch, torch.zeros_like(flat_cost))
    module._rival_gate = lambda **kwargs: (
        rival,
        {
            "evidence/rival_margin": flat_cost.new_tensor(0.5),
            "evidence/class_evidence_std": flat_cost.new_tensor(0.0),
        },
    )

    guided, payload = module(
        flat_cost=flat_cost,
        query_tokens=query_tokens,
        support_tokens=support_tokens,
        way_num=1,
        shot_num=1,
        spatial_hw=(1, 2),
    )

    product_gate = 0.01 * 0.25 * 0.64
    soft_and_gate = product_gate ** (1.0 / 3.0)
    observed = payload["verified_region_pair_gate"].mean().item()
    assert observed == pytest.approx(soft_and_gate)
    assert observed > product_gate * 10.0
    assert guided.mean().item() == pytest.approx(1.0 + 0.2 * (1.0 - soft_and_gate))


def test_verified_region_matching_reuses_rival_gate_signal():
    torch.manual_seed(21)
    module = VerifiedRegionMatchingUOT(
        penalty_lambda=0.2,
        use_concentration=False,
        use_region_patch=False,
        use_rival=True,
    )
    flat_cost = torch.rand(2, 2, 4, 4)
    query_tokens = torch.randn(2, 4, 5)
    support_tokens = torch.randn(2, 1, 4, 5)

    guided, payload = module(
        flat_cost=flat_cost,
        query_tokens=query_tokens,
        support_tokens=support_tokens,
        way_num=2,
        shot_num=1,
        spatial_hw=(2, 2),
    )

    assert guided.shape == flat_cost.shape
    assert payload["verified/rival_enabled"].item() == pytest.approx(1.0)
    assert "verified/class_evidence_std" in payload
    assert torch.isfinite(payload["verified/rival_specificity_mean"])


def test_ours_final_factory_wires_verified_region_matching():
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
        enable_verified_region_matching_uot="true",
        vrm_lambda=0.2,
        vrm_use_concentration="true",
        vrm_use_region_patch="true",
        vrm_use_rival="false",
        enable_global_residual_score="true",
        global_residual_weight=0.1,
    )

    model = build_model_from_args(args)

    assert isinstance(model, OursFinalM2)
    assert model.enable_verified_region_matching_uot
    assert hasattr(model, "verified_region_matching_uot")
    assert model.enable_global_residual_score
    assert model.global_residual_weight == pytest.approx(0.1)


def test_ours_final_verified_region_matching_forward_logs_diagnostics():
    torch.manual_seed(22)
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
        enable_verified_region_matching_uot=True,
        enable_global_residual_score=True,
        global_residual_weight=0.1,
    )
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)
    model.eval()

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 2)
    assert outputs["verified_region_pair_gate"].shape[:3] == (2, 2, 1)
    assert outputs["verified_region_guided_cost_matrix"].shape == outputs["cost_matrix"].shape
    assert "base_cost_matrix" in outputs
    for key in (
        "verified/gate_mean",
        "verified/gate_class_gap",
        "verified/accepted_mass_ratio",
        "verified/rejected_mass_ratio",
        "verified/plan_gate_correlation",
        "verified/top10_gate_mass_ratio",
        "verified/top20_gate_mass_ratio",
        "global_residual_weight",
    ):
        assert key in outputs
        assert torch.isfinite(outputs[key])


def test_verified_region_suite_variants_and_aliases():
    variants = build_ours_final_verified_region_variants()

    assert [variant["tag"] for variant in variants] == [
        "vrm_baseline",
        "vrm_concentration",
        "vrm_patch_region",
        "vrm_conc_patch",
        "vrm_full",
    ]
    assert len({tuple(variant["extra_args"]) for variant in variants}) == 5
    selected = filter_ours_final_variants(
        variants,
        parse_ours_final_variant_filter("vrm_patch,vrm_conc_patch"),
    )
    assert [variant["tag"] for variant in selected] == [
        "vrm_patch_region",
        "vrm_conc_patch",
    ]


def test_verified_region_matching_rejects_token_attention_combination():
    with pytest.raises(
        ValueError,
        match="enable_verified_region_matching_uot isolates",
    ):
        OursFinalM2(
            backbone_name="conv64f",
            image_size=64,
            hidden_dim=64,
            token_dim=8,
            eam_hidden_dim=16,
            enable_verified_region_matching_uot=True,
            enable_token_attention_marginal=True,
        )
