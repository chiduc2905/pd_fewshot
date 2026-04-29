from __future__ import annotations

import torch

from net.crj_fsl import CRJFSL


def _build_model(**overrides) -> CRJFSL:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=24,
        score_scale=8.0,
        tau_q=0.5,
        tau_c=0.5,
        sinkhorn_epsilon=0.1,
        sinkhorn_iterations=12,
        sinkhorn_tolerance=1e-5,
        fixed_mass=0.8,
        min_mass=0.1,
        mass_bonus_init=1.0,
        lambda_rho=0.0,
        lambda_rho_rank=0.0,
        lambda_curvature=0.0,
        normalize_euclidean_tokens=True,
        eval_use_float64=True,
        hyperbolic_backend="auto",
        ot_backend="native",
        eps=1e-6,
        crj_pool_gamma=0.65,
        crj_variance_penalty=0.25,
    )
    kwargs.update(overrides)
    return CRJFSL(**kwargs)


def test_crj_one_shot_reduces_to_j_pooling():
    model = _build_model(crj_pool_gamma=1.0, crj_variance_penalty=1.0)
    shot_logits = torch.tensor([[[2.0], [-1.0]], [[0.5], [3.0]]])
    shot_cost = torch.ones_like(shot_logits) * 0.25
    shot_mass = torch.ones_like(shot_logits) * 0.8

    logits, cost, mass, weights = model._pool_j_shot_scores(shot_logits, shot_cost, shot_mass)

    assert torch.allclose(logits, shot_logits.squeeze(-1))
    assert torch.allclose(cost, shot_cost.squeeze(-1))
    assert torch.allclose(mass, shot_mass.squeeze(-1))
    assert torch.allclose(weights, torch.ones_like(shot_logits))


def test_crj_penalizes_inconsistent_high_single_shot():
    model = _build_model(crj_pool_gamma=1.0, crj_variance_penalty=0.5)
    shot_logits = torch.tensor([[[0.0, 0.0, 10.0]]])
    shot_cost = torch.ones_like(shot_logits)
    shot_mass = torch.ones_like(shot_logits) * 0.8

    logits, _, _, weights = model._pool_j_shot_scores(shot_logits, shot_cost, shot_mass)
    vanilla = torch.logsumexp(shot_logits, dim=-1) - torch.log(torch.tensor(float(shot_logits.shape[-1])))

    assert logits.item() < vanilla.item()
    assert weights.shape == shot_logits.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(logits), atol=1e-6, rtol=0.0)
    assert model._last_crj_pool_diagnostics is not None
    assert "crj_shot_score_std" in model._last_crj_pool_diagnostics


def test_crj_forward_shapes_and_debug_payload():
    torch.manual_seed(7)
    model = _build_model(crj_pool_gamma=0.5, crj_variance_penalty=0.1)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["shot_logits"].shape == (2, 3, 2)
    assert outputs["shot_pool_weights"].shape == (2, 3, 2)
    assert outputs["crj_vanilla_logits"].shape == (2, 3)
    assert outputs["crj_robust_lcb"].shape == (2, 3)
    assert outputs["crj_effective_shots"].shape == (2, 3)
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["mean_crj_shot_std"])
