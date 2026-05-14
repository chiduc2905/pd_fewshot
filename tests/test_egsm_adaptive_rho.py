"""Tests for Direction B: EGSM episode-adaptive rho."""

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


def _build_ours_adaptive(**overrides):
    kw = {
        **BASE_ARGS,
        "hrot_ecot_egsm_adaptive_rho": "true",
        **overrides,
    }
    return build_model_from_args(SimpleNamespace(model="ours", ours_ablation="full", **kw))


def test_adaptive_rho_construction():
    """Model builds with adaptive rho enabled and EGSM has rho_head."""
    model = _build_ours_adaptive()
    assert model.ecot_egsm_adaptive_rho
    assert model.egsm_marginal is not None
    assert model.egsm_marginal.rho_head is not None
    assert model.egsm_marginal.enable_adaptive_rho


def test_adaptive_rho_disabled_by_default():
    """Standard Ours does not have adaptive rho."""
    model = build_model_from_args(SimpleNamespace(
        model="ours", ours_ablation="full", **BASE_ARGS,
    ))
    assert not model.ecot_egsm_adaptive_rho
    assert model.egsm_marginal.rho_head is None


def test_adaptive_rho_forward_shape():
    """Forward produces correct logit shapes."""
    torch.manual_seed(42)
    model = _build_ours_adaptive()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)

    with torch.no_grad():
        logits = model(query, support)
    assert logits.shape == (2, 3)


def test_adaptive_rho_returns_rho_in_aux():
    """Aux output contains egsm_rho_adaptive when enabled."""
    torch.manual_seed(42)
    model = _build_ours_adaptive()
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert "egsm_rho_adaptive" in outputs
    rho = outputs["egsm_rho_adaptive"]
    assert rho.min() >= 0.0
    assert rho.max() <= 1.0


def test_adaptive_rho_bounded():
    """Predicted rho stays within [base - delta, base + delta]."""
    torch.manual_seed(7)
    model = _build_ours_adaptive(
        hrot_ecot_egsm_rho_delta_max=0.1,
    )
    model.eval()

    query = torch.randn(1, 4, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    rho = outputs["egsm_rho_adaptive"]
    assert rho.min() >= 0.8 - 0.1 - 1e-5
    assert rho.max() <= 0.8 + 0.1 + 1e-5


def test_adaptive_rho_gradient_flows():
    """Gradients flow through rho_head parameters."""
    torch.manual_seed(42)
    model = _build_ours_adaptive()
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum() + outputs.get("aux_loss", torch.tensor(0.0))
    loss.backward()

    rho_head_grads = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if "rho_head" in name and p.grad is not None
    }
    assert len(rho_head_grads) > 0, "rho_head has no gradient"


def test_adaptive_rho_with_cpm():
    """Adaptive rho works with OursCPM (Direction B + A combined)."""
    torch.manual_seed(42)
    model = build_model_from_args(SimpleNamespace(
        model="ours_cpm",
        **BASE_ARGS,
        hrot_ecot_egsm_adaptive_rho="true",
        hrot_ecot_egsm_rho_delta_max=0.1,
    ))
    assert model.ecot_egsm_adaptive_rho
    assert len(model.ecot_rho_bank) == 3

    model.eval()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)
    with torch.no_grad():
        outputs = model(query, support, return_aux=True)
    assert outputs["logits"].shape == (2, 3)
    assert "egsm_rho_adaptive" in outputs


def test_adaptive_rho_with_mncr():
    """All three directions combined (A+B+C): CPM + adaptive rho + MNCR."""
    torch.manual_seed(42)
    model = build_model_from_args(SimpleNamespace(
        model="ours_cpm",
        **BASE_ARGS,
        hrot_ecot_egsm_adaptive_rho="true",
        hrot_use_mncr="true",
        hrot_mncr_temperature=0.5,
        hrot_mncr_lam=0.5,
    ))
    assert model.ecot_egsm_adaptive_rho
    assert model.use_mncr
    assert model.ecot_m2_cost_per_mass_score

    model.eval()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)
    with torch.no_grad():
        outputs = model(query, support, return_aux=True)
    assert outputs["logits"].shape == (2, 3)
