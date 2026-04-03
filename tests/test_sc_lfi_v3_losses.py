from __future__ import annotations

import torch
import torch.nn.functional as F

from net.modules.posterior_losses_v3 import (
    compute_distribution_margin_loss_v3,
    compute_entropy_floor_loss_v3,
)
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
        lambda_align=0.05,
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


def test_sc_lfi_v3_fresh_transport_starts_close_to_identity():
    torch.manual_seed(0)
    model = _build_model()
    model.eval()

    base_atoms = torch.randn(3, 5, 16)
    class_summary = torch.randn(3, 24)
    support_memory = torch.randn(3, 3, 16)
    episode_context = torch.randn(3, 24)

    with torch.no_grad():
        transported = model.flow_model.transport(
            base_atoms,
            class_summary,
            support_memory,
            episode_context,
            num_steps=3,
            solver_type="heun",
        )

    assert torch.allclose(transported, base_atoms, atol=1e-6, rtol=0.0)


def test_sc_lfi_v3_margin_loss_behaves_as_expected():
    bad_distances = torch.tensor(
        [
            [0.5, 0.2, 0.7],
            [0.6, 0.4, 0.3],
        ],
        dtype=torch.float32,
    )
    good_distances = torch.tensor(
        [
            [0.1, 0.5, 0.7],
            [0.7, 0.1, 0.4],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1], dtype=torch.long)
    bad_loss = compute_distribution_margin_loss_v3(bad_distances, targets, margin=0.1)
    good_loss = compute_distribution_margin_loss_v3(good_distances, targets, margin=0.1)
    assert bad_loss.item() > 0.0
    assert good_loss.item() == 0.0


def test_sc_lfi_v3_entropy_floor_penalizes_collapsed_mass_more_than_uniform():
    uniform = torch.full((2, 6), 1.0 / 6.0)
    collapsed = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.97, 0.01, 0.01, 0.01, 0.0, 0.0]])
    uniform_loss = compute_entropy_floor_loss_v3(uniform, target_ratio=0.5)
    collapsed_loss = compute_entropy_floor_loss_v3(collapsed, target_ratio=0.5)
    assert collapsed_loss.item() > uniform_loss.item()


def test_sc_lfi_v3_query_conditioned_reweighting_and_fm_reach_all_main_modules():
    torch.manual_seed(1)
    model = _build_model()
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)

    outputs = model(query, support, query_targets=targets, return_aux=True)
    total = (
        F.cross_entropy(outputs["logits"], targets)
        + outputs["fm_loss"]
        + outputs["align_loss"]
        + outputs["reg_loss"]
    )
    model.zero_grad(set_to_none=True)
    total.backward()

    assert _has_finite_nonzero_grad(model.latent_projector)
    assert _has_finite_nonzero_grad(model.posterior_context)
    assert _has_finite_nonzero_grad(model.flow_model)
    assert _has_finite_nonzero_grad(model.query_transport_scorer)


def test_sc_lfi_v3_distribution_branch_improves_on_toy_episode():
    torch.manual_seed(2)
    model = _build_model(
        train_num_integration_steps=2,
        eval_num_integration_steps=2,
        lambda_fm=0.02,
        lambda_align=0.02,
        lambda_margin=0.05,
        lambda_reg=0.02,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    query, support, targets = _make_toy_episode()

    model.eval()
    with torch.no_grad():
        before = model(query, support, query_targets=targets, return_aux=True)["pairwise_distances"]
        before_margin = (
            before[torch.arange(targets.numel()), targets] - before[torch.arange(targets.numel()), 1 - targets]
        ).mean()

    model.train()
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(query, support, query_targets=targets)
        loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        after = model(query, support, query_targets=targets, return_aux=True)["pairwise_distances"]
        after_margin = (
            after[torch.arange(targets.numel()), targets] - after[torch.arange(targets.numel()), 1 - targets]
        ).mean()

    assert after_margin.item() < before_margin.item()
