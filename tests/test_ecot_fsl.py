from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.optimize import minimize

from net.ecot_fsl import EpisodeCompetitiveOT, semi_relaxed_sinkhorn_log
from net.model_factory import build_model_from_args, get_model_choices


def _generalized_kl_numpy(value, reference):
    value = np.clip(value, 1e-15, None)
    reference = np.clip(reference, 1e-15, None)
    return np.sum(value * (np.log(value) - np.log(reference)) - value + reference)


def test_semi_relaxed_sinkhorn_matches_scipy_constrained_optimum():
    cost_np = np.array(
        [
            [0.10, 0.70, 1.20],
            [0.90, 0.20, 0.55],
        ],
        dtype=np.float64,
    )
    source_np = np.array([0.4, 0.6], dtype=np.float64)
    target_np = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    epsilon = 0.30
    relaxation = 0.70

    cost = torch.tensor(cost_np, dtype=torch.float64).unsqueeze(0)
    source = torch.tensor(source_np, dtype=torch.float64).unsqueeze(0)
    target = torch.tensor(target_np, dtype=torch.float64).unsqueeze(0)
    plan, stats = semi_relaxed_sinkhorn_log(
        cost,
        source,
        target,
        epsilon=epsilon,
        target_relaxation=relaxation,
        max_iterations=4000,
        tolerance=1e-13,
        numerical_eps=1e-15,
    )

    reference_np = np.outer(source_np, target_np)

    def objective(flat_plan):
        candidate = flat_plan.reshape(cost_np.shape)
        target_observed = candidate.sum(axis=0)
        return (
            np.sum(candidate * cost_np)
            + epsilon * _generalized_kl_numpy(candidate, reference_np)
            + relaxation * _generalized_kl_numpy(target_observed, target_np)
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda flat_plan, row=row: (
                flat_plan.reshape(cost_np.shape)[row].sum() - source_np[row]
            ),
        }
        for row in range(cost_np.shape[0])
    ]
    scipy_result = minimize(
        objective,
        reference_np.reshape(-1),
        method="SLSQP",
        bounds=[(1e-15, None)] * reference_np.size,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 4000},
    )

    assert scipy_result.success, scipy_result.message
    scipy_plan = scipy_result.x.reshape(cost_np.shape)
    assert np.allclose(plan.squeeze(0).detach().numpy(), scipy_plan, atol=2e-5, rtol=2e-5)
    assert stats.source_marginal_l1.item() < 1e-10
    assert stats.fixed_point_residual.item() < 1e-9
    assert stats.objective.item() == pytest.approx(objective(scipy_result.x), abs=2e-8)


def test_semi_relaxed_sinkhorn_preserves_source_and_backpropagates():
    torch.manual_seed(601)
    raw_cost = torch.rand(3, 4, 7, requires_grad=True)
    source = torch.rand(3, 4)
    target = torch.rand(3, 7)

    plan, stats = semi_relaxed_sinkhorn_log(
        raw_cost,
        source,
        target,
        epsilon=0.08,
        target_relaxation=0.15,
        max_iterations=200,
        tolerance=1e-9,
    )
    normalized_source = source / source.sum(dim=-1, keepdim=True)

    assert torch.allclose(plan.sum(dim=-1), normalized_source, atol=2e-6, rtol=2e-6)
    assert torch.isfinite(plan).all()
    assert torch.isfinite(stats.objective).all()

    loss = (plan * raw_cost).sum() + stats.target_kl.mean()
    loss.backward()
    assert raw_cost.grad is not None
    assert torch.isfinite(raw_cost.grad).all()
    assert raw_cost.grad.abs().sum().item() > 0.0


class _IdentityFeatureEncoder(nn.Module):
    out_channels = 3

    def forward_features(self, images):
        return images


def _solver_only_model() -> EpisodeCompetitiveOT:
    model = EpisodeCompetitiveOT.__new__(EpisodeCompetitiveOT)
    nn.Module.__init__(model)
    model.encoder = _IdentityFeatureEncoder()
    model.feat_dim = 3
    model.epsilon = 0.05
    model.target_relaxation = 0.05
    model.sinkhorn_iterations = 300
    model.sinkhorn_tolerance = 1e-9
    model.logit_scale = 1.0
    model.claim_margin = 0.05
    model.numerical_eps = 1e-8
    return model


def test_ecot_joint_class_mass_predicts_matching_class_and_logs_diagnostics():
    model = _solver_only_model()
    query = torch.tensor(
        [[[[[1.0]], [[0.0]], [[0.0]]], [[[0.0]], [[1.0]], [[0.0]]]]]
    )
    support = torch.tensor(
        [[[[[[1.0]], [[0.0]], [[0.0]]]], [[[[0.0]], [[1.0]], [[0.0]]]]]]
    )
    targets = torch.tensor([0, 1])

    outputs = model(query, support, query_targets=targets, return_aux=True)

    assert outputs["logits"].shape == (2, 2)
    assert outputs["ecot_class_mass"].shape == (2, 2)
    assert outputs["ecot_token_class_assignment"].shape == (2, 1, 2)
    assert outputs["logits"].argmax(dim=-1).tolist() == [0, 1]
    assert torch.allclose(outputs["ecot_class_mass"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert outputs["ecot/source_marginal_l1"].item() < 1e-6
    assert outputs["ecot/mass_prediction_accuracy"].item() == pytest.approx(1.0)
    assert outputs["ecot/true_rival_mass_gap"].item() > 0.0


def test_ecot_is_equivariant_to_support_class_permutation():
    model = _solver_only_model()
    query = torch.tensor([[[[[1.0]], [[0.0]], [[0.0]]]]])
    support = torch.tensor(
        [[[[[[1.0]], [[0.0]], [[0.0]]]], [[[[0.0]], [[1.0]], [[0.0]]]]]]
    )

    logits = model(query, support)
    permuted_logits = model(query, support[:, [1, 0]])

    assert torch.allclose(permuted_logits, logits[:, [1, 0]], atol=1e-6, rtol=1e-6)


def test_ecot_model_factory_registration():
    assert "ecot_fsl" in get_model_choices()
    args = SimpleNamespace(
        model="ecot_fsl",
        device="cpu",
        image_size=84,
        fewshot_backbone="resnet12",
        ecot_epsilon=0.07,
        ecot_target_relaxation=0.12,
        ecot_sinkhorn_iterations=17,
        ecot_sinkhorn_tolerance=1e-5,
        ecot_logit_scale=1.0,
        ecot_claim_margin=0.04,
        ecot_numerical_eps=1e-8,
    )

    model = build_model_from_args(args)

    assert isinstance(model, EpisodeCompetitiveOT)
    assert model.feat_dim == 640
    assert model.epsilon == pytest.approx(0.07)
    assert model.target_relaxation == pytest.approx(0.12)
    assert model.sinkhorn_iterations == 17
