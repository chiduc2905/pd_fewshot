from __future__ import annotations

from types import SimpleNamespace

import torch

from net.model_factory import build_model_from_args
from net.sc_lfi_v3 import SupportConditionedLatentFlowInferenceNetV3


def _build_model(**overrides) -> SupportConditionedLatentFlowInferenceNetV3:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        latent_dim=16,
        context_dim=24,
        context_hidden_dim=32,
        latent_hidden_dim=32,
        mass_hidden_dim=16,
        mass_temperature=1.0,
        memory_size=3,
        memory_num_heads=2,
        memory_ffn_multiplier=2,
        prior_num_atoms=2,
        prior_scale=0.4,
        episode_num_heads=2,
        alpha_hidden_dim=24,
        alpha_shot_scale=1.0,
        alpha_uncertainty_scale=0.5,
        flow_hidden_dim=32,
        flow_time_embedding_dim=16,
        flow_memory_num_heads=2,
        flow_conditioning_type="film",
        use_transport_flow=True,
        use_prior_measure=True,
        use_episode_adapter=True,
        use_query_reweighting=True,
        use_support_barycenter_only=False,
        use_global_proto_branch=False,
        use_align_loss=True,
        use_margin_loss=True,
        train_num_integration_steps=2,
        eval_num_integration_steps=3,
        solver_type="heun",
        fm_time_schedule="uniform",
        score_temperature=4.0,
        query_reweight_temperature=1.0,
        proto_branch_weight=0.1,
        lambda_fm=0.05,
        lambda_align=0.1,
        lambda_margin=0.1,
        lambda_reg=0.05,
        margin_value=0.1,
        support_entropy_target_ratio=0.5,
        query_entropy_target_ratio=0.4,
        relevance_entropy_target_ratio=0.35,
        score_train_num_projections=6,
        score_eval_num_projections=8,
        score_sw_p=2.0,
        score_normalize_inputs=True,
        score_train_projection_mode="fixed",
        score_eval_projection_mode="fixed",
        score_eval_num_repeats=1,
        score_projection_seed=13,
        align_distance_type="weighted_entropic_ot",
        align_train_num_projections=6,
        align_eval_num_projections=8,
        align_sw_p=2.0,
        align_normalize_inputs=True,
        align_train_projection_mode="fixed",
        align_eval_projection_mode="fixed",
        align_eval_num_repeats=1,
        align_projection_seed=17,
        sinkhorn_epsilon=0.1,
        sinkhorn_iterations=30,
        sinkhorn_cost_power=2.0,
        eps=1e-8,
    )
    kwargs.update(overrides)
    return SupportConditionedLatentFlowInferenceNetV3(**kwargs)


def test_sc_lfi_v3_forward_shapes_and_aux_outputs():
    torch.manual_seed(0)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 2], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["pairwise_distances"].shape == (2, 3)
    assert outputs["query_latents"].shape == (1, 2, outputs["query_latents"].shape[2], 16)
    assert outputs["query_masses"].shape == outputs["query_latents"].shape[:-1]
    assert outputs["support_latents"].shape[0] == 1
    assert outputs["support_latents"].shape[1] == 3
    assert outputs["support_latents"].shape[-1] == 16
    assert outputs["support_memory"].shape == (1, 3, 3, 16)
    assert outputs["class_summary"].shape == (1, 3, 24)
    assert outputs["episode_context"].shape == (1, 3, 24)
    assert outputs["weighted_mean"].shape == (1, 3, 16)
    assert outputs["uncertainty_stats"].shape == (1, 3, 4)
    assert outputs["prior_atoms"].shape == (1, 3, 2, 16)
    assert outputs["prior_masses"].shape == (1, 3, 2)
    assert outputs["alpha"].shape == (1, 3)
    assert outputs["base_atoms"].shape[0] == 1
    assert outputs["base_atoms"].shape[1] == 3
    assert outputs["posterior_atoms"].shape == outputs["base_atoms"].shape
    assert outputs["base_masses"].shape == outputs["posterior_masses"].shape
    assert outputs["query_conditioned_class_masses"].shape[:3] == (1, 2, 3)
    assert outputs["relevance_entropy"].shape == (1, 2, 3)
    assert torch.allclose(outputs["query_masses"].sum(dim=-1), torch.ones_like(outputs["query_masses"].sum(dim=-1)))
    assert torch.allclose(outputs["support_masses"].sum(dim=-1), torch.ones_like(outputs["support_masses"].sum(dim=-1)))
    assert torch.allclose(outputs["prior_masses"].sum(dim=-1), torch.ones_like(outputs["prior_masses"].sum(dim=-1)))
    assert torch.allclose(outputs["base_masses"].sum(dim=-1), torch.ones_like(outputs["base_masses"].sum(dim=-1)))
    assert torch.allclose(
        outputs["query_conditioned_class_masses"].sum(dim=-1),
        torch.ones_like(outputs["query_conditioned_class_masses"].sum(dim=-1)),
    )
    for key in ("fm_loss", "align_loss", "margin_loss", "reg_loss", "aux_loss"):
        assert outputs[key].ndim == 0
        assert torch.isfinite(outputs[key])


def test_sc_lfi_v3_is_support_permutation_invariant():
    torch.manual_seed(1)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    support_permuted = support[:, :, torch.tensor([1, 0])]
    targets = torch.tensor([1, 0], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)
        permuted = model(query, support_permuted, query_targets=targets, return_aux=True)

    assert torch.allclose(outputs["logits"], permuted["logits"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["pairwise_distances"], permuted["pairwise_distances"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["class_summary"], permuted["class_summary"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["support_memory"], permuted["support_memory"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["alpha"], permuted["alpha"], atol=1e-6, rtol=0.0)
    assert torch.allclose(
        outputs["base_masses"].sum(dim=-1),
        permuted["base_masses"].sum(dim=-1),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        outputs["query_conditioned_class_masses"].sum(dim=-1),
        permuted["query_conditioned_class_masses"].sum(dim=-1),
        atol=1e-6,
        rtol=0.0,
    )


def test_sc_lfi_v3_shot_aware_alpha_is_larger_for_more_shots():
    torch.manual_seed(2)
    model = _build_model(use_episode_adapter=False)
    support_latents = torch.randn(2, 6, 16)
    support_masses = torch.rand(2, 6)
    support_masses = support_masses / support_masses.sum(dim=-1, keepdim=True)

    with torch.no_grad():
        alpha_1 = model.posterior_context(support_latents, support_masses, shot_num=1)["alpha"]
        alpha_5 = model.posterior_context(support_latents, support_masses, shot_num=5)["alpha"]

    assert torch.all(alpha_5 > alpha_1)


def test_sc_lfi_v3_internal_shot_profile_is_more_conservative_for_one_shot():
    model = _build_model(
        lambda_fm=0.05,
        lambda_align=0.1,
        lambda_margin=0.1,
        lambda_reg=0.05,
        margin_value=0.1,
        support_entropy_target_ratio=0.5,
        query_entropy_target_ratio=0.4,
        relevance_entropy_target_ratio=0.35,
        score_temperature=8.0,
        query_reweight_temperature=1.0,
    )

    one_shot = model._shot_profile(1)
    five_shot = model._shot_profile(5)

    assert one_shot["score_temperature"] < five_shot["score_temperature"]
    assert one_shot["relevance_temperature"] > five_shot["relevance_temperature"]
    assert one_shot["reweight_strength"] > five_shot["reweight_strength"]
    assert one_shot["lambda_margin"] > five_shot["lambda_margin"]
    assert one_shot["lambda_reg"] > five_shot["lambda_reg"]
    assert one_shot["support_entropy_target_ratio"] > five_shot["support_entropy_target_ratio"]
    assert one_shot["query_entropy_target_ratio"] > five_shot["query_entropy_target_ratio"]
    assert one_shot["relevance_entropy_target_ratio"] > five_shot["relevance_entropy_target_ratio"]


def test_sc_lfi_v3_degenerate_mode_reduces_to_single_support_barycenter_measure():
    torch.manual_seed(3)
    model = _build_model(
        use_transport_flow=False,
        use_prior_measure=False,
        use_query_reweighting=False,
        use_support_barycenter_only=True,
        use_align_loss=False,
        use_margin_loss=False,
    )
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    expected_atoms = outputs["weighted_mean"].unsqueeze(2)
    assert torch.allclose(outputs["support_atoms"], expected_atoms, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["posterior_atoms"], expected_atoms, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["posterior_masses"], torch.ones_like(outputs["posterior_masses"]), atol=1e-6, rtol=0.0)


def test_sc_lfi_v3_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="sc_lfi_v3",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        latent_dim=12,
        fm_time_schedule="uniform",
        score_temperature=4.0,
        sc_lfi_v3_context_dim=20,
        sc_lfi_v3_context_hidden_dim=24,
        sc_lfi_v3_latent_hidden_dim=24,
        sc_lfi_v3_mass_hidden_dim=16,
        sc_lfi_v3_mass_temperature=1.0,
        sc_lfi_v3_memory_size=3,
        sc_lfi_v3_memory_num_heads=2,
        sc_lfi_v3_memory_ffn_multiplier=2,
        sc_lfi_v3_prior_num_atoms=2,
        sc_lfi_v3_prior_scale=0.4,
        sc_lfi_v3_episode_num_heads=2,
        sc_lfi_v3_alpha_hidden_dim=20,
        sc_lfi_v3_alpha_shot_scale=1.0,
        sc_lfi_v3_alpha_uncertainty_scale=0.5,
        sc_lfi_v3_flow_hidden_dim=24,
        sc_lfi_v3_flow_time_embedding_dim=12,
        sc_lfi_v3_flow_memory_num_heads=2,
        sc_lfi_v3_flow_conditioning_type="film",
        sc_lfi_v3_use_transport_flow="true",
        sc_lfi_v3_use_prior_measure="true",
        sc_lfi_v3_use_episode_adapter="true",
        sc_lfi_v3_use_query_reweighting="true",
        sc_lfi_v3_use_support_barycenter_only="false",
        sc_lfi_v3_use_global_proto_branch="false",
        sc_lfi_v3_use_align_loss="true",
        sc_lfi_v3_use_margin_loss="true",
        sc_lfi_v3_train_num_integration_steps=2,
        sc_lfi_v3_eval_num_integration_steps=3,
        sc_lfi_v3_solver_type="heun",
        sc_lfi_v3_query_reweight_temperature=1.0,
        sc_lfi_v3_proto_branch_weight=0.1,
        sc_lfi_v3_lambda_fm=0.05,
        sc_lfi_v3_lambda_align=0.1,
        sc_lfi_v3_lambda_margin=0.1,
        sc_lfi_v3_lambda_reg=0.05,
        sc_lfi_v3_margin_value=0.1,
        sc_lfi_v3_support_entropy_target_ratio=0.5,
        sc_lfi_v3_query_entropy_target_ratio=0.4,
        sc_lfi_v3_relevance_entropy_target_ratio=0.35,
        sc_lfi_v3_score_train_num_projections=6,
        sc_lfi_v3_score_eval_num_projections=8,
        sc_lfi_v3_score_sw_p=2.0,
        sc_lfi_v3_score_normalize_inputs="true",
        sc_lfi_v3_score_train_projection_mode="fixed",
        sc_lfi_v3_score_eval_projection_mode="fixed",
        sc_lfi_v3_score_eval_num_repeats=1,
        sc_lfi_v3_score_projection_seed=5,
        sc_lfi_v3_align_distance_type="weighted_entropic_ot",
        sc_lfi_v3_align_train_num_projections=6,
        sc_lfi_v3_align_eval_num_projections=8,
        sc_lfi_v3_align_sw_p=2.0,
        sc_lfi_v3_align_normalize_inputs="true",
        sc_lfi_v3_align_train_projection_mode="fixed",
        sc_lfi_v3_align_eval_projection_mode="fixed",
        sc_lfi_v3_align_eval_num_repeats=1,
        sc_lfi_v3_align_projection_seed=7,
        sc_lfi_v3_sinkhorn_epsilon=0.1,
        sc_lfi_v3_sinkhorn_iterations=30,
        sc_lfi_v3_sinkhorn_cost_power=2.0,
        sc_lfi_v3_eps=1e-8,
    )
    model = build_model_from_args(args)
    model.eval()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)
    assert outputs["logits"].shape == (2, 3)
