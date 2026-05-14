"""Tests for Ours-CPM (Cost-Per-Mass multi-budget) model."""

from types import SimpleNamespace

import torch

from net.model_factory import build_model_from_args


BASE_ARGS = dict(
    device="cpu",
    image_size=64,
    fewshot_backbone="conv64f",
    hrot_token_dim=16,
    hrot_eam_hidden_dim=16,
    hrot_sinkhorn_iterations=6,
    hrot_sinkhorn_tolerance=1e-5,
)


def _build_cpm(**overrides):
    kw = {**BASE_ARGS, **overrides}
    return build_model_from_args(SimpleNamespace(model="ours_cpm", **kw))


def test_cpm_factory_defaults():
    """Full CPM has 3 budgets, cost/mass score, and EGSM."""
    model = _build_cpm()
    assert model.variant == "J_ECOT_M2"
    assert model.ecot_rho_bank == (0.7, 0.8, 0.9)
    assert model.ecot_base_rho == 0.8
    assert model.ecot_transport_mode == "unbalanced"
    assert model.uses_unbalanced_transport
    assert model.ecot_m2_cost_per_mass_score
    assert not model.ecot_m2_ablate_threshold_mass
    assert model.ecot_m2_cost_per_mass_detach_mass
    assert model.uses_ecot_egsm_marginal
    assert not model.ecot_m2_use_aqm
    assert not model.ecot_m2_use_swts


def test_cpm_single_budget_ablation():
    """single_budget collapses to 1 budget for controlled comparison with Ours."""
    model = _build_cpm(cpm_ablation="single_budget")
    assert model.ecot_rho_bank == (0.8,)
    assert model.ecot_m2_cost_per_mass_score
    assert model.uses_ecot_egsm_marginal


def test_cpm_no_egsm_ablation():
    """no_egsm keeps multi-budget but uses uniform marginals."""
    model = _build_cpm(cpm_ablation="no_egsm")
    assert model.ecot_rho_bank == (0.7, 0.8, 0.9)
    assert not model.uses_ecot_egsm_marginal


def test_cpm_custom_alpha():
    model = _build_cpm(cpm_alpha=2.5)
    assert model.cpm_alpha == 2.5
    assert model.ecot_m2_cost_per_mass_alpha == 2.5


def test_cpm_custom_rho_bank():
    model = _build_cpm(cpm_rho_bank="0.6,0.7,0.8,0.9")
    assert model.ecot_rho_bank == (0.6, 0.7, 0.8, 0.9)
    assert model.ecot_base_rho == 0.8


def test_cpm_forward_shape():
    """Forward produces correct logit shapes for a 3-way 1-shot episode."""
    torch.manual_seed(42)
    model = _build_cpm()
    model.eval()

    way, shot, nq = 3, 1, 2
    query = torch.randn(1, nq, 3, 64, 64)
    support = torch.randn(1, way, shot, 3, 64, 64)

    with torch.no_grad():
        logits = model(query, support)
    assert logits.shape == (nq, way)


def test_cpm_forward_with_aux():
    """Aux output contains budget policy and cost/mass diagnostics."""
    torch.manual_seed(42)
    model = _build_cpm()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert "ecot_pi_budget" in outputs
    assert "ecot_budget_scores" in outputs
    assert "transported_mass" in outputs


def test_cpm_multi_budget_produces_three_experts():
    """CPM model creates 3 budget experts and produces a non-trivial policy."""
    torch.manual_seed(123)
    model = _build_cpm()
    model.eval()

    assert len(model.ecot_rho_bank) == 3

    query = torch.randn(1, 4, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    budget_scores = outputs["ecot_budget_scores"]
    assert budget_scores.shape[-1] == 3

    pi_budget = outputs["ecot_pi_budget"]
    assert pi_budget.shape[-1] == 3
    assert torch.allclose(pi_budget.sum(dim=-1), torch.ones_like(pi_budget.sum(dim=-1)), atol=1e-5)


def test_cpm_gradient_flows():
    """Gradients flow through the full CPM pipeline."""
    torch.manual_seed(7)
    model = _build_cpm()
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum() + outputs.get("aux_loss", torch.tensor(0.0))
    loss.backward()

    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    assert len(grad_norms) > 0, "no gradients computed"
    assert any(v > 0 for v in grad_norms.values()), "all gradients are zero"
