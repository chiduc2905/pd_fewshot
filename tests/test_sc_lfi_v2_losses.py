from __future__ import annotations

import torch
import torch.nn.functional as F

from net.modules.conditional_flow_v2 import FixedStepFlowSolverV2
from net.modules.flow_losses_v2 import compute_distribution_margin_loss_v2
from net.modules.transport_distance_v2 import (
    WeightedEntropicOTDistanceV2,
    WeightedTransportScoringDistanceV2,
)
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
        lambda_align=0.05,
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


def test_sc_lfi_v2_weighted_scoring_distance_matches_explicit_loop():
    torch.manual_seed(0)
    metric = WeightedTransportScoringDistanceV2(
        train_num_projections=8,
        eval_num_projections=8,
        p=2.0,
        normalize_inputs=True,
        train_projection_mode="fixed",
        eval_projection_mode="fixed",
        eval_num_repeats=1,
        projection_seed=23,
    )
    metric.eval()

    query = torch.randn(3, 5, 7)
    support = torch.randn(4, 6, 7)
    query_masses = torch.rand(3, 5)
    support_masses = torch.rand(4, 6)
    query_masses = query_masses / query_masses.sum(dim=-1, keepdim=True)
    support_masses = support_masses / support_masses.sum(dim=-1, keepdim=True)

    pairwise = metric.pairwise_distance(
        query,
        support,
        query_masses=query_masses,
        support_masses=support_masses,
        reduction="none",
    )
    rows = []
    for query_idx in range(query.shape[0]):
        cols = []
        for class_idx in range(support.shape[0]):
            cols.append(
                metric(
                    query[query_idx : query_idx + 1],
                    support[class_idx : class_idx + 1],
                    source_masses=query_masses[query_idx : query_idx + 1],
                    target_masses=support_masses[class_idx : class_idx + 1],
                    reduction="none",
                ).squeeze(0)
            )
        rows.append(torch.stack(cols))
    loop = torch.stack(rows)
    assert torch.allclose(pairwise, loop, atol=1e-6, rtol=0.0)


def test_sc_lfi_v2_weighted_entropic_ot_is_zero_on_identical_measures():
    torch.manual_seed(1)
    metric = WeightedEntropicOTDistanceV2(
        sinkhorn_epsilon=0.1,
        max_iterations=50,
        cost_power=2.0,
        normalize_inputs=False,
        reduction="none",
        eps=1e-8,
    )
    x = torch.randn(2, 5, 4)
    masses = torch.rand(2, 5)
    masses = masses / masses.sum(dim=-1, keepdim=True)
    got = metric(x, x, source_masses=masses, target_masses=masses, reduction="none")
    assert torch.all(got >= 0.0)
    assert float(got.max()) < 1e-4


def test_sc_lfi_v2_fixed_step_solver_matches_constant_velocity_solution():
    initial = torch.zeros(4, 3)

    def constant_velocity(states: torch.Tensor, time_values: torch.Tensor) -> torch.Tensor:
        del time_values
        return torch.full_like(states, 2.0)

    euler = FixedStepFlowSolverV2("euler").integrate(constant_velocity, initial, num_steps=5)
    heun = FixedStepFlowSolverV2("heun").integrate(constant_velocity, initial, num_steps=5)
    target = torch.full_like(initial, 2.0)
    assert torch.allclose(euler, target, atol=1e-6, rtol=0.0)
    assert torch.allclose(heun, target, atol=1e-6, rtol=0.0)


def test_sc_lfi_v2_margin_loss_behaves_as_expected():
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
    bad_loss = compute_distribution_margin_loss_v2(bad_distances, targets, margin=0.1)
    good_loss = compute_distribution_margin_loss_v2(good_distances, targets, margin=0.1)
    assert bad_loss.item() > 0.0
    assert good_loss.item() == 0.0


def test_sc_lfi_v2_flow_matching_gradients_reach_projector_conditioner_and_flow():
    torch.manual_seed(2)
    model = _build_model()
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)

    outputs = model(query, support, query_targets=targets)
    fm_loss = outputs["fm_loss"]
    assert fm_loss.item() > 0.0
    model.zero_grad(set_to_none=True)
    fm_loss.backward()

    assert _has_finite_nonzero_grad(model.latent_projector)
    assert _has_finite_nonzero_grad(model.support_conditioner)
    assert _has_finite_nonzero_grad(model.flow_model)


def test_sc_lfi_v2_distribution_branch_improves_on_toy_episode():
    torch.manual_seed(3)
    model = _build_model(
        train_num_flow_particles=3,
        eval_num_flow_particles=3,
        train_num_integration_steps=2,
        eval_num_integration_steps=2,
        lambda_fm=0.02,
        lambda_align=0.02,
        lambda_margin=0.05,
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
