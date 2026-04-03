from types import SimpleNamespace

import torch

from net.model_factory import build_model_from_args
from net.spif_aeb_v2 import SPIFAEBV2


def _build_model(**overrides):
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
        aeb_v2_global_ce_weight=0.2,
        aeb_v2_local_ce_weight=0.2,
    )
    kwargs.update(overrides)
    return SPIFAEBV2(**kwargs)


def test_spifaeb_v2_budget_operator_is_monotonic():
    torch.manual_seed(0)
    model = _build_model()
    model.eval()

    query_tokens = torch.randn(2, 5, 24)
    support_token_pool = torch.randn(3, 7, 24)
    query_row_weights = torch.full((2, 5), 1.0 / 5.0)
    similarity = model.build_local_evidence_profile(query_tokens, support_token_pool)["similarity"]

    low_budget = torch.full((2, 3), 0.2)
    high_budget = torch.full((2, 3), 0.8)

    low_outputs = model.compute_local_budget_scores(
        similarity=similarity,
        rho=low_budget,
        row_budget=None,
        query_row_weights=query_row_weights,
        support_row_weights=None,
    )
    high_outputs = model.compute_local_budget_scores(
        similarity=similarity,
        rho=high_budget,
        row_budget=None,
        query_row_weights=query_row_weights,
        support_row_weights=None,
    )

    assert torch.all(high_outputs["active_match_counts"] > low_outputs["active_match_counts"])
    assert torch.all(high_outputs["retained_fraction"] > low_outputs["retained_fraction"])


def test_spifaeb_v2_forward_invariance_and_aux_outputs():
    torch.manual_seed(1)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    support_permuted = support[:, :, torch.tensor([1, 0])]
    targets = torch.tensor([0, 2], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)
        permuted_outputs = model(query, support_permuted, query_targets=targets, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["global_scores"].shape == (2, 3)
    assert outputs["base_global_scores"].shape == (2, 3)
    assert outputs["conditioned_global_scores"].shape == (2, 3)
    assert outputs["local_scores"].shape == (2, 3)
    assert outputs["anchor_local_scores"].shape == (2, 3)
    assert outputs["adaptive_local_scores"].shape == (2, 3)
    assert outputs["raw_local_scores"].shape == (2, 3)
    assert outputs["competitive_local_scores"].shape == (2, 3)
    assert outputs["rho"].shape == (2, 3)
    assert outputs["rho_prior"].shape == (2, 3)
    assert outputs["rho_residual"].shape == (2, 3)
    assert outputs["row_budget_std"].shape == (2, 3)
    assert outputs["evidence_advantage"].shape == (2, 3)
    assert outputs["evidence_dispersion"].shape == (2, 3)
    assert outputs["evidence_quality"].shape == (2, 3)
    assert outputs["coverage_entropy"].shape == (2, 3)
    assert outputs["coverage_concentration"].shape == (2, 3)
    assert outputs["coverage_bonus"].shape == (2, 3)
    assert outputs["retained_fraction"].shape == (2, 3)
    assert outputs["local_anchor_mix"].ndim == 0
    assert outputs["local_adaptive_mix"].ndim == 0
    assert torch.isfinite(outputs["mean_query_class_attention_entropy"])
    assert outputs["aux_loss"].ndim == 0
    assert outputs["global_branch_ce"].ndim == 0
    assert outputs["local_branch_ce"].ndim == 0
    assert outputs["budget_rank_loss"].ndim == 0
    assert outputs["local_margin_loss"].ndim == 0
    assert outputs["budget_residual_reg_loss"].ndim == 0
    assert outputs["anchor_consistency_loss"].ndim == 0
    assert outputs["aux_loss"].item() == 0.0
    assert torch.isfinite(outputs["global_branch_ce"])
    assert torch.isfinite(outputs["local_branch_ce"])
    assert torch.isfinite(outputs["budget_rank_loss"])
    assert torch.isfinite(outputs["local_margin_loss"])
    assert torch.allclose(outputs["logits"], permuted_outputs["logits"], atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["rho"], permuted_outputs["rho"], atol=1e-6, rtol=0.0)


def test_spifaeb_v2_returns_branch_aux_loss_during_training():
    torch.manual_seed(2)
    model = _build_model()
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)
    outputs = model(query, support, query_targets=targets)

    assert isinstance(outputs, dict)
    assert outputs["logits"].shape == (2, 3)
    assert outputs["aux_loss"].ndim == 0
    assert torch.isfinite(outputs["aux_loss"])
    assert outputs["aux_loss"].item() > 0.0
    assert torch.isfinite(outputs["budget_rank_loss"])
    assert torch.isfinite(outputs["local_margin_loss"])
    assert torch.isfinite(outputs["budget_residual_reg_loss"])
    assert torch.isfinite(outputs["anchor_consistency_loss"])


def test_spifaeb_v2_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="spifaeb_v2",
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
        aeb_v2_global_scale=1.0,
        aeb_v2_local_scale=1.0,
        aeb_v2_fusion_mode="fixed",
        aeb_v2_fusion_kappa=6.0,
        aeb_v2_local_score_mode="query_to_support",
        aeb_v2_share_local_head="false",
        aeb_v2_query_gate_weighting="true",
        aeb_v2_global_ce_weight=0.2,
        aeb_v2_local_ce_weight=0.2,
        aeb_v2_eps=1e-6,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        logits = model(query, support, query_targets=targets)

    assert logits.shape == (2, 3)
