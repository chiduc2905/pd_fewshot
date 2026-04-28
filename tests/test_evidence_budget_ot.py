from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from net.evidence_budget_ot import (
    EvidenceBudgetedOTFewShot,
    EvidenceBudgetedOTMatcher,
    ScalogramPriorExtractor,
    log_sinkhorn,
)
from net.model_factory import build_model_from_args, get_model_choices


def _tiny_model(**overrides) -> EvidenceBudgetedOTFewShot:
    kwargs = dict(
        in_channels=3,
        hidden_dim=64,
        proj_dim=16,
        reliability_hidden_dim=16,
        reliability_dropout=0.0,
        backbone_name="conv64f",
        image_size=32,
        sinkhorn_epsilon=0.1,
        sinkhorn_iters=8,
        dustbin_cost=0.7,
        learnable_dustbin_cost=True,
        alpha_unmatched=0.1,
        min_budget=0.15,
        max_budget=0.95,
        gate_temperature=1.0,
        use_scalogram_priors=True,
        use_energy_prior=True,
        use_gradient_prior=True,
        use_tf_coords=True,
        prior_norm="mean",
        use_cross_reference=True,
        use_dustbin=True,
        use_evidence_budget=True,
        use_kshot_reweighting=True,
        use_aux_loss=False,
        eps=1e-6,
    )
    kwargs.update(overrides)
    return EvidenceBudgetedOTFewShot(**kwargs)


def test_prior_extractor_shapes_for_gray_and_rgb():
    torch.manual_seed(0)
    extractor = ScalogramPriorExtractor(
        use_scalogram_priors=True,
        use_energy_prior=True,
        use_gradient_prior=True,
        use_tf_coords=True,
        prior_norm="mean",
    )
    gray = torch.randn(2, 1, 32, 40)
    rgb = torch.randn(2, 3, 32, 40)

    gray_priors = extractor(gray, (4, 5))
    rgb_priors = extractor(rgb, (4, 5))

    assert gray_priors.shape == (2, 20, 6)
    assert rgb_priors.shape == (2, 20, 6)
    assert torch.isfinite(gray_priors).all()
    assert torch.isfinite(rgb_priors).all()


def test_log_sinkhorn_matches_marginals():
    torch.manual_seed(1)
    batch, tokens = 3, 6
    cost = torch.rand(batch, tokens + 1, tokens + 1)
    a = torch.rand(batch, tokens + 1)
    b = torch.rand(batch, tokens + 1)
    a = a / a.sum(dim=-1, keepdim=True)
    b = b / b.sum(dim=-1, keepdim=True)

    plan = log_sinkhorn(cost, a, b, epsilon=0.1, n_iters=120)

    assert plan.shape == cost.shape
    assert torch.isfinite(plan).all()
    assert torch.allclose(plan.sum(dim=-1), a, atol=2e-3, rtol=0.0)
    assert torch.allclose(plan.sum(dim=-2), b, atol=2e-3, rtol=0.0)


def test_matcher_scores_and_gradients_flow():
    torch.manual_seed(2)
    matcher = EvidenceBudgetedOTMatcher(
        input_dim=8,
        proj_dim=12,
        prior_dim=6,
        reliability_hidden_dim=16,
        reliability_dropout=0.0,
        sinkhorn_epsilon=0.1,
        sinkhorn_iters=12,
        dustbin_cost=0.7,
        learnable_dustbin_cost=True,
    )
    q_tokens = torch.randn(4, 5, 8, requires_grad=True)
    s_tokens = torch.randn(4, 5, 8, requires_grad=True)
    q_priors = torch.rand(4, 5, 6)
    s_priors = torch.rand(4, 5, 6)

    scores, aux = matcher(q_tokens, s_tokens, q_priors, s_priors)
    loss = -scores.mean()
    loss.backward()

    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert aux["matched_mass"].shape == (4,)
    assert matcher.projector[1].weight.grad is not None
    assert matcher.reliability.net[1].weight.grad is not None
    assert torch.isfinite(matcher.projector[1].weight.grad).all()
    assert torch.isfinite(matcher.reliability.net[1].weight.grad).all()


def test_fewshot_forward_shapes_for_kshot_and_grayscale():
    torch.manual_seed(3)
    model = _tiny_model()
    model.eval()

    query = torch.randn(2, 3, 1, 32, 32)
    support = torch.randn(2, 4, 2, 1, 32, 32)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (6, 4)
    assert outputs["shot_scores"].shape == (6, 4, 2)
    assert outputs["shot_aggregation_weights"].shape == (6, 4, 2)
    assert torch.allclose(
        outputs["shot_aggregation_weights"].sum(dim=-1),
        torch.ones(6, 4),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["matched_mass"]).all()
    maps = model.get_last_reliability_maps()
    assert maps["query"].dim() == 3
    assert maps["support"].dim() == 4


def test_fewshot_forward_shapes_for_single_episode_1shot():
    torch.manual_seed(4)
    model = _tiny_model()
    model.eval()

    query = torch.randn(3, 3, 32, 32)
    support = torch.randn(5, 1, 3, 32, 32)

    with torch.no_grad():
        logits = model(query, support)

    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()


def test_evidence_budget_ot_model_factory_builds_aliases():
    assert "evidence_budget_ot" in get_model_choices()
    assert "ebot" in get_model_choices()
    assert "ebot_scalogram" in get_model_choices()

    args = SimpleNamespace(
        model="evidence_budget_ot",
        device="cpu",
        image_size=32,
        fewshot_backbone="conv64f",
        ebot_proj_dim=16,
        ebot_reliability_hidden_dim=16,
        ebot_reliability_dropout=0.0,
        ebot_sinkhorn_epsilon=0.1,
        ebot_sinkhorn_iters=8,
        ebot_dustbin_cost=0.7,
        ebot_learnable_dustbin_cost="true",
        ebot_alpha_unmatched=0.1,
        ebot_min_budget=0.15,
        ebot_max_budget=0.95,
        ebot_gate_temperature=1.0,
        ebot_use_scalogram_priors="true",
        ebot_use_energy_prior="true",
        ebot_use_gradient_prior="true",
        ebot_use_tf_coords="true",
        ebot_log_power_alpha=10.0,
        ebot_prior_norm="mean",
        ebot_use_cross_reference="true",
        ebot_use_dustbin="true",
        ebot_use_evidence_budget="true",
        ebot_use_kshot_reweighting="true",
        ebot_lambda_score=1.0,
        ebot_lambda_mass=0.5,
        ebot_lambda_unmatched=0.5,
        ebot_symmetric_matching="false",
        ebot_use_uncertainty_prior="false",
        ebot_use_aux_loss="false",
        ebot_budget_floor=0.15,
        ebot_budget_ceiling=0.95,
        ebot_weight_budget_low=0.01,
        ebot_weight_budget_high=0.001,
        ebot_eps=1e-6,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 32, 32)
    support = torch.randn(1, 3, 1, 3, 32, 32)
    with torch.no_grad():
        logits = model(query, support)

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_tiny_episode_training_loss_decreases():
    torch.manual_seed(5)
    model = _tiny_model(sinkhorn_iters=5, use_dustbin=False, use_evidence_budget=False)
    model.train()

    query = torch.zeros(1, 2, 1, 32, 32)
    support = torch.zeros(1, 2, 1, 1, 32, 32)
    support[:, 0, 0, :, 4:16, 4:16] = 1.0
    support[:, 1, 0, :, 16:28, 16:28] = 1.0
    query[:, 0, :, 4:16, 4:16] = 1.0
    query[:, 1, :, 16:28, 16:28] = 1.0
    targets = torch.tensor([0, 1], dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(query, support)
        loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
        losses.append(float(loss.detach().item()))
        loss.backward()
        optimizer.step()

    assert torch.isfinite(torch.tensor(losses)).all()
    assert losses[-1] < losses[0]
