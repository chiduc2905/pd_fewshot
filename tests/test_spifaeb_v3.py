from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from net.model_factory import build_model_from_args
from net.spif_aeb_v2 import SPIFAEBV2
from net.spifaeb_v3 import SPIFAEBv3, StructuredEvidenceGlobalHead


def _shared_kwargs(**overrides):
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        stable_dim=32,
        variant_dim=32,
        local_dim=24,
        gate_hidden=8,
        aeb_v2_hidden=16,
        aeb_v2_min_budget=0.1,
        aeb_v2_max_budget=1.0,
        aeb_v2_rank_temperature=0.35,
        aeb_v2_global_scale=1.0,
        aeb_v2_local_scale=1.0,
        aeb_v2_cond_global_weight=0.35,
        aeb_v2_cond_global_temperature=8.0,
        aeb_v2_cond_global_use_gate_prior=True,
        aeb_v2_query_class_attention_on=True,
        aeb_v2_query_class_attention_temperature=8.0,
        aeb_v2_query_class_attention_use_gate_prior=True,
        aeb_v2_anchor_top_r=3,
        aeb_v2_anchor_mix=0.65,
        aeb_v2_local_mix_mode="prior_adaptive",
        aeb_v2_adaptive_mix_min=0.2,
        aeb_v2_adaptive_mix_max=0.8,
        aeb_v2_adaptive_mix_scale=6.0,
        aeb_v2_local_residual_scale=0.25,
        aeb_v2_budget_prior_scale=6.0,
        aeb_v2_budget_residual_scale=0.15,
        aeb_v2_budget_rank_weight=0.1,
        aeb_v2_budget_rank_margin=0.05,
        aeb_v2_budget_residual_reg_weight=0.05,
        aeb_v2_support_coverage_temperature=8.0,
        aeb_v2_coverage_bonus_weight=0.0,
        aeb_v2_competition_weight=1.0,
        aeb_v2_competition_temperature=8.0,
        aeb_v2_local_margin_weight=0.1,
        aeb_v2_local_margin_target=0.5,
        aeb_v2_anchor_consistency_weight=0.02,
        aeb_v2_fusion_mode="margin_adaptive",
        aeb_v2_fusion_kappa=6.0,
        aeb_v2_local_score_mode="query_to_support",
        aeb_v2_share_local_head=False,
        aeb_v2_query_gate_weighting=True,
        aeb_v2_global_ce_weight=0.0,
        aeb_v2_local_ce_weight=0.2,
        aeb_v2_eps=1e-6,
    )
    kwargs.update(overrides)
    return kwargs


def _build_v2(**overrides) -> SPIFAEBV2:
    return SPIFAEBV2(**_shared_kwargs(**overrides))


def _build_v3(**overrides) -> SPIFAEBv3:
    kwargs = _shared_kwargs(
        spifaeb_v3_use_structured_global=True,
        spifaeb_v3_sigma_min=0.05,
        spifaeb_v3_use_reliability=True,
        spifaeb_v3_reliability_detach=True,
        spifaeb_v3_beta_align=1.0,
        spifaeb_v3_beta_dev=0.5,
        spifaeb_v3_beta_rel=0.2,
        spifaeb_v3_learnable_betas=False,
    )
    kwargs.update(overrides)
    return SPIFAEBv3(**kwargs)


def test_structured_evidence_global_head_matches_manual_formula():
    head = StructuredEvidenceGlobalHead(
        stable_dim=3,
        use_structured_global=True,
        sigma_min=0.05,
        use_reliability=True,
        reliability_detach=False,
        beta_align=1.0,
        beta_dev=0.5,
        beta_rel=0.2,
        learnable_betas=False,
        eps=1e-6,
    )

    query = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.2, 0.9, 0.0],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )
    support = F.normalize(
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.8, 0.6, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.8, 0.6],
                ],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )
    support_gate = torch.tensor(
        [
            [
                [[0.9], [0.5], [0.1]],
                [[0.8], [0.4], [0.2]],
            ],
            [
                [[0.7], [0.6], [0.2]],
                [[0.9], [0.5], [0.1]],
            ],
        ],
        dtype=torch.float32,
    )

    outputs = head(query, support, support_gate=support_gate)

    manual_centers = F.normalize(support.mean(dim=1), p=2, dim=-1)
    manual_dispersion = (support - manual_centers.unsqueeze(1)).square().sum(dim=-1).mean(dim=1).clamp_min(1e-4)
    gate_values = support_gate.squeeze(-1)
    manual_reliability_raw = (gate_values.std(dim=-1, unbiased=False) / (gate_values.mean(dim=-1) + 1e-6)).mean(dim=1)
    manual_reliability = torch.sigmoid(manual_reliability_raw)
    manual_alignment = torch.matmul(query, manual_centers.transpose(0, 1))
    manual_deviation = (
        (query.unsqueeze(1) - manual_centers.unsqueeze(0)).square().sum(dim=-1) / (manual_dispersion.unsqueeze(0) + 1e-6)
    )
    manual_scores = (
        outputs["beta_align"] * manual_alignment
        - outputs["beta_dev"] * manual_deviation
        + outputs["beta_rel"] * manual_reliability.unsqueeze(0)
    )

    assert torch.allclose(outputs["class_centers"], manual_centers, atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["dispersion"], manual_dispersion, atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["reliability_raw"], manual_reliability_raw, atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["reliability"], manual_reliability, atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["alignment"], manual_alignment, atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["deviation"], manual_deviation, atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["structured_scores"], manual_scores, atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["scores"], manual_scores, atol=1e-5, rtol=0.0)


def test_structured_evidence_global_head_uses_sigma_min_in_one_shot():
    head = StructuredEvidenceGlobalHead(
        stable_dim=3,
        use_structured_global=True,
        sigma_min=0.05,
        use_reliability=False,
        reliability_detach=True,
        beta_align=1.0,
        beta_dev=0.5,
        beta_rel=0.2,
        learnable_betas=False,
        eps=1e-6,
    )

    query = F.normalize(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32), p=2, dim=-1)
    support = F.normalize(
        torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )
    outputs = head(query, support, support_gate=None)

    expected_dispersion = torch.full((2,), 0.05 * 0.05, dtype=torch.float32)
    assert torch.allclose(outputs["dispersion"], expected_dispersion, atol=1e-8, rtol=0.0)
    assert torch.all(outputs["reliability"] == 0.0)


@pytest.mark.parametrize("shot_num", [1, 5])
def test_spifaeb_v3_forward_is_finite_in_low_shot_regimes(shot_num: int):
    torch.manual_seed(shot_num)
    model = _build_v3()
    model.eval()

    query = torch.randn(1, 3, 3, 64, 64)
    support = torch.randn(1, 5, shot_num, 3, 64, 64)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    assert outputs["logits"].shape == (3, 5)
    assert outputs["global_scores"].shape == (3, 5)
    assert outputs["local_scores"].shape == (3, 5)
    assert outputs["structured_alignment"].shape == (3, 5)
    assert outputs["structured_deviation"].shape == (3, 5)
    assert outputs["structured_reliability"].shape == (3, 5)
    assert outputs["structured_dispersion"].shape == (3, 5)
    assert outputs["structured_class_centers"].shape == (1, 5, 32)
    for key in (
        "logits",
        "global_scores",
        "local_scores",
        "structured_alignment",
        "structured_deviation",
        "structured_reliability",
        "structured_reliability_raw",
        "structured_dispersion",
        "rho",
        "rho_prior",
        "rho_residual",
        "row_budget_std",
        "retained_fraction",
    ):
        assert torch.isfinite(outputs[key]).all()
    for key in (
        "aux_loss",
        "global_branch_ce",
        "local_branch_ce",
        "budget_rank_loss",
        "local_margin_loss",
        "budget_residual_reg_loss",
        "anchor_consistency_loss",
        "structured_beta_align",
        "structured_beta_dev",
        "structured_beta_rel",
    ):
        assert torch.isfinite(outputs[key])

    if shot_num == 1:
        expected_dispersion = torch.full_like(outputs["structured_dispersion"], 0.05 * 0.05)
        assert torch.allclose(outputs["structured_dispersion"], expected_dispersion, atol=1e-6, rtol=0.0)


def test_spifaeb_v3_backpropagates_through_global_and_local_paths():
    torch.manual_seed(7)
    model = _build_v3(
        spifaeb_v3_learnable_betas=True,
        aeb_v2_global_ce_weight=0.2,
        aeb_v2_local_ce_weight=0.2,
    )
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 5, 1, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)
    outputs = model(query, support, query_targets=targets)

    loss = outputs["logits"].pow(2).mean() + outputs["aux_loss"]
    loss.backward()

    assert model.global_encoder.stable_head[1].weight.grad is not None
    assert model.local_token_norm.weight.grad is not None
    assert model.structured_global_head.beta_align_raw.grad is not None
    assert model.structured_global_head.beta_dev_raw.grad is not None
    assert model.structured_global_head.beta_rel_raw.grad is not None
    assert torch.isfinite(model.global_encoder.stable_head[1].weight.grad).all()
    assert torch.isfinite(model.local_token_norm.weight.grad).all()
    assert torch.isfinite(model.structured_global_head.beta_align_raw.grad).all()
    assert torch.isfinite(model.structured_global_head.beta_dev_raw.grad).all()
    assert torch.isfinite(model.structured_global_head.beta_rel_raw.grad).all()


def test_spifaeb_v3_keeps_v2_local_branch_outputs_identical():
    torch.manual_seed(11)
    v2 = _build_v2()
    v3 = _build_v3()
    load_result = v3.load_state_dict(v2.state_dict(), strict=False)

    assert set(load_result.missing_keys) == {
        "structured_global_head.beta_align_raw",
        "structured_global_head.beta_dev_raw",
        "structured_global_head.beta_rel_raw",
    }
    assert load_result.unexpected_keys == []

    v2.eval()
    v3.eval()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 2], dtype=torch.long)

    with torch.no_grad():
        outputs_v2 = v2(query, support, query_targets=targets, return_aux=True)
        outputs_v3 = v3(query, support, query_targets=targets, return_aux=True)

    for key in (
        "local_scores",
        "anchor_local_scores",
        "adaptive_local_scores",
        "raw_local_scores",
        "competitive_local_scores",
        "rho",
        "rho_prior",
        "rho_residual",
        "row_budget_std",
        "evidence_sharpness",
        "evidence_advantage",
        "evidence_dispersion",
        "evidence_quality",
        "coverage_entropy",
        "coverage_concentration",
        "coverage_bonus",
        "active_match_counts",
        "retained_fraction",
    ):
        assert torch.allclose(outputs_v2[key], outputs_v3[key], atol=1e-6, rtol=0.0)

    assert torch.allclose(
        outputs_v2["mean_query_class_attention_entropy"],
        outputs_v3["mean_query_class_attention_entropy"],
        atol=1e-6,
        rtol=0.0,
    )


def test_spifaeb_v3_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="spifaeb_v3",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        spif_stable_dim=32,
        spif_variant_dim=32,
        spif_gate_hidden=8,
        spif_alpha_init=0.7,
        spif_learnable_alpha="false",
        spif_gate_on="true",
        spif_factorization_on="true",
        spif_global_only="false",
        spif_local_only="false",
        spif_token_l2norm="true",
        aeb_v2_hidden=16,
        aeb_v2_local_dim=24,
        aeb_v2_min_budget=0.1,
        aeb_v2_max_budget=1.0,
        aeb_v2_rank_temperature=0.35,
        aeb_v2_global_scale=1.0,
        aeb_v2_local_scale=1.0,
        aeb_v2_cond_global_weight=0.35,
        aeb_v2_cond_global_temperature=8.0,
        aeb_v2_cond_global_use_gate_prior="true",
        aeb_v2_query_class_attention_on="true",
        aeb_v2_query_class_attention_temperature=8.0,
        aeb_v2_query_class_attention_use_gate_prior="true",
        aeb_v2_anchor_top_r=3,
        aeb_v2_anchor_mix=0.65,
        aeb_v2_local_mix_mode="prior_adaptive",
        aeb_v2_adaptive_mix_min=0.2,
        aeb_v2_adaptive_mix_max=0.8,
        aeb_v2_adaptive_mix_scale=6.0,
        aeb_v2_local_residual_scale=0.25,
        aeb_v2_budget_prior_scale=6.0,
        aeb_v2_budget_residual_scale=0.15,
        aeb_v2_budget_rank_weight=0.1,
        aeb_v2_budget_rank_margin=0.05,
        aeb_v2_budget_residual_reg_weight=0.05,
        aeb_v2_support_coverage_temperature=8.0,
        aeb_v2_coverage_bonus_weight=0.0,
        aeb_v2_competition_weight=1.0,
        aeb_v2_competition_temperature=8.0,
        aeb_v2_local_margin_weight=0.1,
        aeb_v2_local_margin_target=0.5,
        aeb_v2_anchor_consistency_weight=0.02,
        aeb_v2_fusion_mode="fixed",
        aeb_v2_fusion_kappa=6.0,
        aeb_v2_local_score_mode="query_to_support",
        aeb_v2_share_local_head="false",
        aeb_v2_query_gate_weighting="true",
        aeb_v2_detach_budget_context="true",
        aeb_v2_detach_local_backbone="true",
        aeb_v2_global_ce_weight=0.0,
        aeb_v2_local_ce_weight=0.2,
        aeb_v2_eps=1e-6,
        spifaeb_v3_use_structured_global="true",
        spifaeb_v3_sigma_min=0.05,
        spifaeb_v3_use_reliability="true",
        spifaeb_v3_reliability_detach="true",
        spifaeb_v3_beta_align=1.0,
        spifaeb_v3_beta_dev=0.5,
        spifaeb_v3_beta_rel=0.2,
        spifaeb_v3_learnable_betas="false",
        spifaeb_v3_global_ce_weight=0.0,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["structured_alignment"].shape == (2, 3)
    assert outputs["structured_deviation"].shape == (2, 3)
    assert outputs["structured_reliability"].shape == (2, 3)
