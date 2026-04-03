from __future__ import annotations

from types import SimpleNamespace

import torch

from net.model_factory import build_model_from_args
from net.sc_lfi_v2 import SupportConditionedLatentFlowInferenceNetV2


def _build_model(**overrides) -> SupportConditionedLatentFlowInferenceNetV2:
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
        flow_hidden_dim=32,
        flow_time_embedding_dim=16,
        flow_memory_num_heads=2,
        flow_conditioning_type="film",
        use_global_proto_branch=False,
        use_flow_branch=True,
        use_align_loss=True,
        use_margin_loss=True,
        use_smooth_loss=False,
        train_num_flow_particles=3,
        eval_num_flow_particles=4,
        train_num_integration_steps=2,
        eval_num_integration_steps=3,
        solver_type="heun",
        fm_time_schedule="uniform",
        score_temperature=4.0,
        proto_branch_weight=0.15,
        lambda_fm=0.05,
        lambda_align=0.1,
        lambda_margin=0.1,
        lambda_support_fit=0.1,
        lambda_smooth=0.0,
        margin_value=0.1,
        support_mix_min=0.45,
        support_mix_max=0.8,
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
        eval_particle_seed=19,
        eps=1e-8,
    )
    kwargs.update(overrides)
    return SupportConditionedLatentFlowInferenceNetV2(**kwargs)


def test_sc_lfi_v2_forward_shapes_and_aux_outputs():
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
    assert outputs["distribution_scores"].shape == (2, 3)
    assert outputs["global_proto_scores"].shape == (2, 3)
    assert outputs["query_latents"].shape == (1, 2, outputs["query_latents"].shape[2], 16)
    assert outputs["query_masses"].shape == outputs["query_latents"].shape[:-1]
    assert outputs["support_latents"].shape[0] == 1
    assert outputs["support_latents"].shape[1] == 3
    assert outputs["support_latents"].shape[-1] == 16
    assert outputs["support_masses"].shape == outputs["support_latents"].shape[:-1]
    assert outputs["class_summary"].shape == (1, 3, 24)
    assert outputs["support_memory"].shape == (1, 3, 3, 16)
    assert outputs["class_barycenters"].shape == (1, 3, 16)
    assert outputs["anchor_particles"].shape == (1, 3, 4, 16)
    assert outputs["anchor_masses"].shape == (1, 3, 4)
    assert outputs["generated_particles"].shape == (1, 3, 4, 16)
    assert outputs["generated_particle_masses"].shape == (1, 3, 4)
    assert outputs["class_measure_particles"].shape == (1, 3, 8, 16)
    assert outputs["class_measure_masses"].shape == (1, 3, 8)
    assert outputs["support_mix_weights"].shape == (1, 3)
    assert torch.allclose(outputs["query_masses"].sum(dim=-1), torch.ones_like(outputs["query_masses"].sum(dim=-1)))
    assert torch.allclose(outputs["support_masses"].sum(dim=-1), torch.ones_like(outputs["support_masses"].sum(dim=-1)))
    assert torch.allclose(outputs["anchor_masses"].sum(dim=-1), torch.ones_like(outputs["anchor_masses"].sum(dim=-1)))
    assert torch.allclose(
        outputs["generated_particle_masses"].sum(dim=-1),
        torch.ones_like(outputs["generated_particle_masses"].sum(dim=-1)),
    )
    assert torch.allclose(
        outputs["class_measure_masses"].sum(dim=-1),
        torch.ones_like(outputs["class_measure_masses"].sum(dim=-1)),
    )
    assert torch.all(outputs["support_mix_weights"] >= 0.45)
    assert torch.all(outputs["support_mix_weights"] <= 0.8)
    for key in ("fm_loss", "align_loss", "margin_loss", "support_fit_loss", "smooth_loss", "aux_loss"):
        assert outputs[key].ndim == 0
        assert torch.isfinite(outputs[key])


def test_sc_lfi_v2_is_support_permutation_invariant():
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
    assert torch.allclose(outputs["class_summary"], permuted["class_summary"], atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["support_memory"], permuted["support_memory"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["anchor_particles"], permuted["anchor_particles"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["anchor_masses"], permuted["anchor_masses"], atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["generated_particles"], permuted["generated_particles"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["class_measure_particles"], permuted["class_measure_particles"], atol=1e-5, rtol=0.0)
    assert torch.allclose(outputs["class_measure_masses"], permuted["class_measure_masses"], atol=1e-6, rtol=0.0)


def test_sc_lfi_v2_degenerate_mode_reduces_to_weighted_support_barycenter():
    torch.manual_seed(2)
    model = _build_model(
        use_flow_branch=False,
        use_align_loss=False,
        use_margin_loss=False,
        eval_num_flow_particles=3,
    )
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    class_barycenters = outputs["class_barycenters"].squeeze(0)
    generated_particles = outputs["generated_particles"].squeeze(0)
    expected = class_barycenters.unsqueeze(1).expand_as(generated_particles)
    assert torch.allclose(generated_particles, expected, atol=1e-6, rtol=0.0)

    query_latents = outputs["query_latents"].squeeze(0)
    query_masses = outputs["query_masses"].squeeze(0)
    particle_masses = outputs["class_measure_masses"].squeeze(0)
    manual_distances = model.score_distance.pairwise_distance(
        query_latents,
        expected,
        query_masses=query_masses,
        support_masses=particle_masses,
        reduction="none",
    )
    assert torch.allclose(outputs["pairwise_distances"], manual_distances, atol=1e-6, rtol=0.0)


def test_sc_lfi_v2_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="sc_lfi_v2",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        latent_dim=12,
        flow_conditioning_type="concat",
        sc_lfi_v2_flow_conditioning_type="film",
        fm_time_schedule="uniform",
        score_temperature=4.0,
        sc_lfi_v2_context_dim=20,
        sc_lfi_v2_context_hidden_dim=24,
        sc_lfi_v2_latent_hidden_dim=24,
        sc_lfi_v2_mass_hidden_dim=16,
        sc_lfi_v2_mass_temperature=1.0,
        sc_lfi_v2_memory_size=3,
        sc_lfi_v2_memory_num_heads=2,
        sc_lfi_v2_memory_ffn_multiplier=2,
        sc_lfi_v2_flow_hidden_dim=24,
        sc_lfi_v2_flow_time_embedding_dim=12,
        sc_lfi_v2_flow_memory_num_heads=2,
        sc_lfi_v2_use_global_proto_branch="false",
        sc_lfi_v2_use_flow_branch="true",
        sc_lfi_v2_use_align_loss="true",
        sc_lfi_v2_use_margin_loss="true",
        sc_lfi_v2_use_smooth_loss="false",
        sc_lfi_v2_train_num_flow_particles=3,
        sc_lfi_v2_eval_num_flow_particles=4,
        sc_lfi_v2_train_num_integration_steps=2,
        sc_lfi_v2_eval_num_integration_steps=3,
        sc_lfi_v2_solver_type="heun",
        sc_lfi_v2_proto_branch_weight=0.15,
        sc_lfi_v2_lambda_fm=0.05,
        sc_lfi_v2_lambda_align=0.1,
        sc_lfi_v2_lambda_margin=0.1,
        sc_lfi_v2_lambda_support_fit=0.1,
        sc_lfi_v2_lambda_smooth=0.0,
        sc_lfi_v2_margin_value=0.1,
        sc_lfi_v2_support_mix_min=0.45,
        sc_lfi_v2_support_mix_max=0.8,
        sc_lfi_v2_score_train_num_projections=6,
        sc_lfi_v2_score_eval_num_projections=8,
        sc_lfi_v2_score_sw_p=2.0,
        sc_lfi_v2_score_normalize_inputs="true",
        sc_lfi_v2_score_train_projection_mode="fixed",
        sc_lfi_v2_score_eval_projection_mode="fixed",
        sc_lfi_v2_score_eval_num_repeats=1,
        sc_lfi_v2_score_projection_seed=5,
        sc_lfi_v2_align_distance_type="weighted_entropic_ot",
        sc_lfi_v2_align_train_num_projections=6,
        sc_lfi_v2_align_eval_num_projections=8,
        sc_lfi_v2_align_sw_p=2.0,
        sc_lfi_v2_align_normalize_inputs="true",
        sc_lfi_v2_align_train_projection_mode="fixed",
        sc_lfi_v2_align_eval_projection_mode="fixed",
        sc_lfi_v2_align_eval_num_repeats=1,
        sc_lfi_v2_align_projection_seed=11,
        sc_lfi_v2_sinkhorn_epsilon=0.1,
        sc_lfi_v2_sinkhorn_iterations=20,
        sc_lfi_v2_sinkhorn_cost_power=2.0,
        sc_lfi_v2_eval_particle_seed=29,
        sc_lfi_v2_eps=1e-8,
    )
    model = build_model_from_args(args)
    model.eval()
    assert model.flow_model.velocity_field.conditioning_type == "film"

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["generated_particles"].shape == (1, 3, 4, 12)
    assert outputs["class_measure_particles"].shape == (1, 3, 8, 12)
