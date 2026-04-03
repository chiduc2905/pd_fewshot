from __future__ import annotations

import torch
import torch.nn.functional as F

from net.modules.posterior_losses_v4 import (
    compute_distribution_margin_loss_v4,
    compute_entropy_floor_loss_v4,
)
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


def test_sc_lfi_v4_fresh_transport_starts_close_to_identity():
    torch.manual_seed(0)
    model = _build_model()
    model.eval()

    base_atoms = torch.randn(3, 5, 16)
    class_summary = torch.randn(3, 24)
    class_memory = torch.randn(3, 3, 16)
    episode_context = torch.randn(3, 24)

    with torch.no_grad():
        transported = model.flow_model.transport(
            base_atoms,
            class_summary,
            class_memory,
            episode_context,
            num_steps=3,
            solver_type="heun",
        )

    assert torch.allclose(transported, base_atoms, atol=1e-6, rtol=0.0)


def test_sc_lfi_v4_margin_loss_behaves_as_expected():
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
    bad_loss = compute_distribution_margin_loss_v4(bad_distances, targets, margin=0.1)
    good_loss = compute_distribution_margin_loss_v4(good_distances, targets, margin=0.1)
    assert bad_loss.item() > 0.0
    assert good_loss.item() == 0.0


def test_sc_lfi_v4_entropy_floor_penalizes_collapsed_mass_more_than_uniform():
    uniform = torch.full((2, 6), 1.0 / 6.0)
    collapsed = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.97, 0.01, 0.01, 0.01, 0.0, 0.0]])
    uniform_loss = compute_entropy_floor_loss_v4(uniform, target_ratio=0.5)
    collapsed_loss = compute_entropy_floor_loss_v4(collapsed, target_ratio=0.5)
    assert collapsed_loss.item() > uniform_loss.item()


def test_sc_lfi_v4_leave_one_out_loss_is_active_for_multishot():
    torch.manual_seed(1)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)

    with torch.no_grad():
        outputs = model(query, support, query_targets=targets, return_aux=True)

    assert torch.isfinite(outputs["loo_loss"])
    assert outputs["loo_loss"].item() > 0.0


def test_sc_lfi_v4_gradients_reach_all_main_modules():
    torch.manual_seed(2)
    model = _build_model()
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)

    outputs = model(query, support, query_targets=targets)
    total = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    model.zero_grad(set_to_none=True)
    total.backward()

    assert _has_finite_nonzero_grad(model.latent_projector)
    assert _has_finite_nonzero_grad(model.posterior_context)
    assert _has_finite_nonzero_grad(model.flow_model)
    assert _has_finite_nonzero_grad(model.query_transport_scorer)


def test_sc_lfi_v4_distribution_branch_improves_on_toy_episode():
    torch.manual_seed(3)
    model = _build_model(
        train_num_integration_steps=2,
        eval_num_integration_steps=2,
        lambda_fm=0.02,
        lambda_align=0.02,
        lambda_margin=0.05,
        lambda_loo=0.05,
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
