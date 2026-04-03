from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from net.model_factory import build_model_from_args
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
        distance_projection_seed=19,
        sinkhorn_epsilon=0.1,
        sinkhorn_iterations=20,
        context_num_memory_tokens=4,
        context_num_heads=4,
        context_ffn_multiplier=2,
        eval_particle_seed=23,
        eps=1e-6,
    )
    kwargs.update(overrides)
    return SupportConditionedLatentFlowInferenceNet(**kwargs)


def _has_finite_nonzero_grad(module: torch.nn.Module) -> bool:
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        if torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum().item() > 0.0:
            return True
    return False


def _make_toy_episode() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support_class_0 = 0.75 + 0.05 * torch.randn(2, 3, 64, 64)
    support_class_1 = -0.75 + 0.05 * torch.randn(2, 3, 64, 64)
    query_class_0 = 0.75 + 0.05 * torch.randn(3, 64, 64)
    query_class_1 = -0.75 + 0.05 * torch.randn(3, 64, 64)

    support = torch.stack([support_class_0, support_class_1], dim=0).unsqueeze(0)
    query = torch.stack([query_class_0, query_class_1], dim=0).unsqueeze(0)
    targets = torch.tensor([0, 1], dtype=torch.long)
    return query, support, targets


def test_sc_lfi_flow_matching_loss_is_finite_and_reaches_required_modules():
    torch.manual_seed(3)
    model = _build_model()
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    outputs = model(query, support)
    fm_loss = outputs["fm_loss"]
    assert fm_loss.ndim == 0
    assert torch.isfinite(fm_loss)
    assert fm_loss.item() > 0.0

    model.zero_grad(set_to_none=True)
    fm_loss.backward()

    assert _has_finite_nonzero_grad(model.context_encoder)
    assert _has_finite_nonzero_grad(model.latent_projector)
    assert _has_finite_nonzero_grad(model.flow_model)


def test_sc_lfi_distribution_scores_improve_on_toy_episode():
    torch.manual_seed(4)
    model = _build_model(
        num_flow_particles=3,
        num_flow_integration_steps=2,
        distance_sw_num_projections=6,
        lambda_fm=0.02,
        lambda_align=0.05,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    query, support, targets = _make_toy_episode()

    model.eval()
    with torch.no_grad():
        before = model(query, support, return_aux=True)["distribution_scores"]
        before_margin = (before[torch.arange(targets.numel()), targets] - before[torch.arange(targets.numel()), 1 - targets]).mean()

    model.train()
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(query, support)
        loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        after = model(query, support, return_aux=True)["distribution_scores"]
        after_margin = (after[torch.arange(targets.numel()), targets] - after[torch.arange(targets.numel()), 1 - targets]).mean()

    assert after_margin.item() > before_margin.item()


def test_sc_lfi_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="sc_lfi",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        latent_dim=12,
        class_context_type="lightweight_set_transformer",
        flow_conditioning_type="film",
        distance_type="sw",
        use_global_proto_branch="true",
        use_flow_branch="true",
        use_align_loss="true",
        use_smooth_loss="false",
        num_flow_particles=3,
        fm_time_schedule="uniform",
        score_temperature=4.0,
        sc_lfi_context_dim=20,
        sc_lfi_context_hidden_dim=24,
        sc_lfi_latent_hidden_dim=24,
        sc_lfi_flow_hidden_dim=24,
        sc_lfi_flow_time_embedding_dim=12,
        sc_lfi_num_flow_integration_steps=2,
        sc_lfi_proto_branch_weight=0.2,
        sc_lfi_lambda_fm=0.05,
        sc_lfi_lambda_align=0.1,
        sc_lfi_lambda_smooth=0.0,
        sc_lfi_distance_normalize_inputs="true",
        sc_lfi_distance_sw_num_projections=6,
        sc_lfi_distance_sw_p=2.0,
        sc_lfi_distance_projection_seed=5,
        sc_lfi_sinkhorn_epsilon=0.1,
        sc_lfi_sinkhorn_iterations=20,
        sc_lfi_context_num_memory_tokens=3,
        sc_lfi_context_num_heads=2,
        sc_lfi_context_ffn_multiplier=2,
        sc_lfi_eval_particle_seed=29,
        sc_lfi_eps=1e-6,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["distribution_scores"].shape == (2, 3)
    assert outputs["generated_particles"].shape == (1, 3, 3, 12)
