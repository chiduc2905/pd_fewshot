from __future__ import annotations

from types import SimpleNamespace

import torch

from net.model_factory import build_model_from_args
from net.sc_lfi_v4 import SupportConditionedLatentFlowInferenceNetV4


def _build_model(**overrides) -> SupportConditionedLatentFlowInferenceNetV4:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        latent_dim=16,
        context_dim=24,
        context_hidden_dim=32,
        latent_hidden_dim=32,
        mass_hidden_dim=16,
        mass_temperature=1.0,
        shot_memory_size=2,
        class_memory_size=3,
        memory_num_heads=2,
        memory_ffn_multiplier=2,
        prior_num_atoms=2,
        global_prior_size=8,
        prior_scale=0.2,
        episode_num_heads=2,
        alpha_hidden_dim=24,
        shrinkage_kappa=2.0,
        alpha_uncertainty_scale=0.5,
        use_shot_barycenter_only=False,
        flow_hidden_dim=32,
        flow_time_embedding_dim=16,
        flow_memory_num_heads=2,
        flow_conditioning_type="film",
        use_transport_flow=True,
        use_prior_measure=True,
        use_episode_adapter=True,
        use_metric_conditioning=True,
        use_align_loss=True,
        use_margin_loss=True,
        use_loo_loss=True,
        train_num_integration_steps=2,
        eval_num_integration_steps=3,
        solver_type="heun",
        fm_time_schedule="uniform",
        score_temperature=4.0,
        lambda_fm=0.05,
        lambda_align=0.1,
        lambda_margin=0.1,
        lambda_loo=0.1,
        lambda_reg=0.05,
        margin_value=0.1,
        support_token_entropy_target_ratio=0.5,
        query_entropy_target_ratio=0.4,
        shot_entropy_target_ratio=0.5,
        prior_entropy_target_ratio=0.4,
        score_distance_type="weighted_entropic_ot",
        score_train_num_projections=6,
        score_eval_num_projections=8,
        score_sw_p=2.0,
        score_normalize_inputs=True,
        score_train_projection_mode="fixed",
        score_eval_projection_mode="fixed",
        score_eval_num_repeats=1,
        score_projection_seed=13,
        score_sinkhorn_epsilon=0.1,
        score_sinkhorn_iterations=20,
        score_sinkhorn_cost_power=2.0,
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
        sinkhorn_iterations=20,
        sinkhorn_cost_power=2.0,
        eps=1e-8,
    )
    kwargs.update(overrides)
    return SupportConditionedLatentFlowInferenceNetV4(**kwargs)


def test_sc_lfi_v4_forward_shapes_and_aux_outputs():
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
    assert outputs["support_latents"].shape[2] == 2
    assert outputs["support_latents"].shape[-1] == 16
    assert outputs["shot_atoms"].shape == (1, 3, 2, 3, 16)
    assert outputs["shot_basis_masses"].shape == (1, 3, 2, 3)
    assert outputs["shot_masses"].shape == (1, 3, 2)
    assert outputs["support_atoms"].shape == (1, 3, 6, 16)
    assert outputs["support_masses"].shape == (1, 3, 6)
    assert outputs["class_memory"].shape == (1, 3, 3, 16)
    assert outputs["class_summary"].shape == (1, 3, 24)
    assert outputs["episode_context"].shape == (1, 3, 24)
    assert outputs["prior_atoms"].shape == (1, 3, 2, 16)
    assert outputs["prior_masses"].shape == (1, 3, 2)
    assert outputs["alpha"].shape == (1, 3)
    assert outputs["base_atoms"].shape == (1, 3, 8, 16)
    assert outputs["posterior_atoms"].shape == outputs["base_atoms"].shape
    assert outputs["metric_scales"].shape == (1, 2, 3, 16)
    assert torch.allclose(outputs["query_masses"].sum(dim=-1), torch.ones_like(outputs["query_masses"].sum(dim=-1)))
    assert torch.allclose(
        outputs["shot_basis_masses"].sum(dim=-1),
        torch.ones_like(outputs["shot_basis_masses"].sum(dim=-1)),
    )
    assert torch.allclose(outputs["shot_masses"].sum(dim=-1), torch.ones_like(outputs["shot_masses"].sum(dim=-1)))
    assert torch.allclose(outputs["prior_masses"].sum(dim=-1), torch.ones_like(outputs["prior_masses"].sum(dim=-1)))
    assert torch.allclose(outputs["base_masses"].sum(dim=-1), torch.ones_like(outputs["base_masses"].sum(dim=-1)))
    for key in ("fm_loss", "align_loss", "margin_loss", "loo_loss", "reg_loss", "aux_loss"):
        assert outputs[key].ndim == 0
        assert torch.isfinite(outputs[key])


def test_sc_lfi_v4_is_support_permutation_invariant():
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
    assert torch.allclose(outputs["class_memory"], permuted["class_memory"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["alpha"], permuted["alpha"], atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["metric_scales"], permuted["metric_scales"], atol=1e-5, rtol=0.0)


def test_sc_lfi_v4_alpha_increases_with_more_shots():
    torch.manual_seed(2)
    model = _build_model(use_episode_adapter=False, use_shot_barycenter_only=True)
    support_latents_1 = torch.randn(2, 1, 6, 16)
    support_latents_5 = support_latents_1.repeat(1, 5, 1, 1)
    support_masses_1 = torch.rand(2, 1, 6)
    support_masses_1 = support_masses_1 / support_masses_1.sum(dim=-1, keepdim=True)
    support_masses_5 = support_masses_1.repeat(1, 5, 1)

    with torch.no_grad():
        alpha_1 = model.posterior_context(support_latents_1, support_masses_1, shot_num=1)["alpha"]
        alpha_5 = model.posterior_context(support_latents_5, support_masses_5, shot_num=5)["alpha"]

    assert torch.all(alpha_5 > alpha_1)


def test_sc_lfi_v4_leave_one_out_loss_is_zero_for_one_shot():
    torch.manual_seed(3)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)
    targets = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    assert outputs["loo_loss"].item() == 0.0


def test_sc_lfi_v4_degenerate_barycenter_mode_matches_shot_means():
    torch.manual_seed(4)
    model = _build_model(
        use_transport_flow=False,
        use_prior_measure=False,
        use_metric_conditioning=False,
        use_align_loss=False,
        use_margin_loss=False,
        use_loo_loss=False,
        use_shot_barycenter_only=True,
    )
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    expected_shot_atoms = outputs["shot_weighted_mean"].unsqueeze(3)
    assert torch.allclose(outputs["shot_atoms"], expected_shot_atoms, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["posterior_atoms"], outputs["support_atoms"], atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["metric_scales"], torch.ones_like(outputs["metric_scales"]), atol=1e-6, rtol=0.0)


def test_sc_lfi_v4_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="sc_lfi_v4",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        latent_dim=12,
        fm_time_schedule="uniform",
        score_temperature=4.0,
        sc_lfi_v4_context_dim=20,
        sc_lfi_v4_context_hidden_dim=24,
        sc_lfi_v4_latent_hidden_dim=24,
        sc_lfi_v4_mass_hidden_dim=16,
        sc_lfi_v4_mass_temperature=1.0,
        sc_lfi_v4_shot_memory_size=2,
        sc_lfi_v4_class_memory_size=3,
        sc_lfi_v4_memory_num_heads=2,
        sc_lfi_v4_memory_ffn_multiplier=2,
        sc_lfi_v4_prior_num_atoms=2,
        sc_lfi_v4_global_prior_size=8,
        sc_lfi_v4_prior_scale=0.2,
        sc_lfi_v4_episode_num_heads=2,
        sc_lfi_v4_alpha_hidden_dim=20,
        sc_lfi_v4_shrinkage_kappa=2.0,
        sc_lfi_v4_alpha_uncertainty_scale=0.5,
        sc_lfi_v4_use_shot_barycenter_only="false",
        sc_lfi_v4_flow_hidden_dim=24,
        sc_lfi_v4_flow_time_embedding_dim=12,
        sc_lfi_v4_flow_memory_num_heads=2,
        sc_lfi_v4_flow_conditioning_type="film",
        sc_lfi_v4_use_transport_flow="true",
        sc_lfi_v4_use_prior_measure="true",
        sc_lfi_v4_use_episode_adapter="true",
        sc_lfi_v4_use_metric_conditioning="true",
        sc_lfi_v4_use_align_loss="true",
        sc_lfi_v4_use_margin_loss="true",
        sc_lfi_v4_use_loo_loss="true",
        sc_lfi_v4_train_num_integration_steps=2,
        sc_lfi_v4_eval_num_integration_steps=3,
        sc_lfi_v4_solver_type="heun",
        sc_lfi_v4_lambda_fm=0.05,
        sc_lfi_v4_lambda_align=0.1,
        sc_lfi_v4_lambda_margin=0.1,
        sc_lfi_v4_lambda_loo=0.1,
        sc_lfi_v4_lambda_reg=0.05,
        sc_lfi_v4_margin_value=0.1,
        sc_lfi_v4_support_token_entropy_target_ratio=0.5,
        sc_lfi_v4_query_entropy_target_ratio=0.4,
        sc_lfi_v4_shot_entropy_target_ratio=0.5,
        sc_lfi_v4_prior_entropy_target_ratio=0.4,
        sc_lfi_v4_score_distance_type="weighted_entropic_ot",
        sc_lfi_v4_score_train_num_projections=6,
        sc_lfi_v4_score_eval_num_projections=8,
        sc_lfi_v4_score_sw_p=2.0,
        sc_lfi_v4_score_normalize_inputs="true",
        sc_lfi_v4_score_train_projection_mode="fixed",
        sc_lfi_v4_score_eval_projection_mode="fixed",
        sc_lfi_v4_score_eval_num_repeats=1,
        sc_lfi_v4_score_projection_seed=5,
        sc_lfi_v4_score_sinkhorn_epsilon=0.1,
        sc_lfi_v4_score_sinkhorn_iterations=20,
        sc_lfi_v4_score_sinkhorn_cost_power=2.0,
        sc_lfi_v4_align_distance_type="weighted_entropic_ot",
        sc_lfi_v4_align_train_num_projections=6,
        sc_lfi_v4_align_eval_num_projections=8,
        sc_lfi_v4_align_sw_p=2.0,
        sc_lfi_v4_align_normalize_inputs="true",
        sc_lfi_v4_align_train_projection_mode="fixed",
        sc_lfi_v4_align_eval_projection_mode="fixed",
        sc_lfi_v4_align_eval_num_repeats=1,
        sc_lfi_v4_align_projection_seed=7,
        sc_lfi_v4_sinkhorn_epsilon=0.1,
        sc_lfi_v4_sinkhorn_iterations=20,
        sc_lfi_v4_sinkhorn_cost_power=2.0,
        sc_lfi_v4_eps=1e-8,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 2], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
