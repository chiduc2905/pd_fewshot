from __future__ import annotations

import torch

from net.sc_lfi import SupportConditionedLatentFlowInferenceNet


def _build_model(**overrides) -> SupportConditionedLatentFlowInferenceNet:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        latent_dim=16,
        context_dim=24,
        context_hidden_dim=32,
        latent_hidden_dim=32,
        flow_hidden_dim=32,
        flow_time_embedding_dim=16,
        class_context_type="deepsets",
        flow_conditioning_type="concat",
        distance_type="sw",
        use_global_proto_branch=False,
        use_flow_branch=True,
        use_align_loss=True,
        use_smooth_loss=False,
        num_flow_particles=4,
        num_flow_integration_steps=3,
        fm_time_schedule="uniform",
        score_temperature=4.0,
        proto_branch_weight=0.2,
        lambda_fm=0.05,
        lambda_align=0.1,
        lambda_smooth=0.0,
        distance_normalize_inputs=True,
        distance_sw_num_projections=8,
        distance_sw_p=2.0,
        distance_projection_seed=13,
        sinkhorn_epsilon=0.1,
        sinkhorn_iterations=20,
        context_num_memory_tokens=4,
        context_num_heads=4,
        context_ffn_multiplier=2,
        eval_particle_seed=17,
        eps=1e-6,
    )
    kwargs.update(overrides)
    return SupportConditionedLatentFlowInferenceNet(**kwargs)


def test_sc_lfi_forward_shapes_and_aux_outputs():
    torch.manual_seed(0)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["distribution_scores"].shape == (2, 3)
    assert outputs["global_proto_scores"].shape == (2, 3)
    assert outputs["class_contexts"].shape == (1, 3, 24)
    assert outputs["query_latents"].shape[0] == 1
    assert outputs["query_latents"].shape[1] == 2
    assert outputs["query_latents"].shape[-1] == 16
    assert outputs["support_latents"].shape[0] == 1
    assert outputs["support_latents"].shape[1] == 3
    assert outputs["support_latents"].shape[-1] == 16
    assert outputs["generated_particles"].shape == (1, 3, 4, 16)
    assert outputs["class_prototypes"].shape == (1, 3, 16)
    assert outputs["support_latents"].shape[2] > 0
    for key in ("fm_loss", "align_loss", "smooth_loss", "aux_loss"):
        assert outputs[key].ndim == 0
        assert torch.isfinite(outputs[key])


def test_sc_lfi_is_invariant_to_support_shot_permutation():
    torch.manual_seed(1)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    support_permuted = support[:, :, torch.tensor([1, 0])]

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)
        permuted_outputs = model(query, support_permuted, return_aux=True)

    assert torch.allclose(outputs["logits"], permuted_outputs["logits"], atol=1e-5, rtol=0.0)
    assert torch.allclose(
        outputs["distribution_scores"],
        permuted_outputs["distribution_scores"],
        atol=1e-5,
        rtol=0.0,
    )
    assert torch.allclose(outputs["class_contexts"], permuted_outputs["class_contexts"], atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["generated_particles"], permuted_outputs["generated_particles"], atol=1e-5, rtol=0.0)


def test_sc_lfi_degenerate_branch_reduces_to_class_mean_particles():
    torch.manual_seed(2)
    model = _build_model(
        use_flow_branch=False,
        use_align_loss=False,
        num_flow_particles=3,
    )
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    class_prototypes = outputs["class_prototypes"].squeeze(0)
    generated_particles = outputs["generated_particles"].squeeze(0)
    expected_particles = class_prototypes.unsqueeze(1).expand_as(generated_particles)
    assert torch.allclose(generated_particles, expected_particles, atol=1e-6, rtol=0.0)

    query_latents = outputs["query_latents"].squeeze(0)
    query_expanded = query_latents.unsqueeze(1).expand(-1, class_prototypes.shape[0], -1, -1)
    particle_expanded = expected_particles.unsqueeze(0).expand(query_latents.shape[0], -1, -1, -1)
    manual_scores = -model.score_temperature * model.distribution_distance(
        query_expanded,
        particle_expanded,
        reduction="none",
    )
    assert torch.allclose(outputs["distribution_scores"], manual_scores, atol=1e-6, rtol=0.0)
