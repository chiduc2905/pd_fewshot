from types import SimpleNamespace

import torch


def test_mncr_with_ours_forward():
    """Ours + MNCR produces valid logits and gradients."""
    from net.model_factory import build_model_from_args

    torch.manual_seed(99)
    model = build_model_from_args(SimpleNamespace(
        model="ours",
        ours_ablation="full",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=16,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=6,
        hrot_sinkhorn_tolerance=1e-5,
        hrot_use_mncr="true",
        hrot_mncr_temperature=0.5,
        hrot_mncr_lam=0.5,
    ))
    assert model.use_mncr
    assert model.mncr_temperature == 0.5
    assert model.mncr_lam == 0.5

    model.train()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)
    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum() + outputs.get("aux_loss", torch.tensor(0.0))
    loss.backward()
    grads = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
    assert len(grads) > 0
    assert any(v > 0 for v in grads.values())


def test_mncr_with_ours_cpm_forward():
    """Ours-CPM + MNCR (Direction A+C combined) works end-to-end."""
    from net.model_factory import build_model_from_args

    torch.manual_seed(99)
    model = build_model_from_args(SimpleNamespace(
        model="ours_cpm",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=16,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=6,
        hrot_sinkhorn_tolerance=1e-5,
        hrot_use_mncr="true",
        hrot_mncr_temperature=0.5,
        hrot_mncr_lam=0.5,
    ))
    assert model.use_mncr
    assert len(model.ecot_rho_bank) == 3
    assert model.ecot_m2_cost_per_mass_score

    model.eval()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 1, 3, 64, 64)
    with torch.no_grad():
        outputs = model(query, support, return_aux=True)
    assert outputs["logits"].shape == (2, 3)
    assert "ecot_pi_budget" in outputs


def test_mncr_shape():
    """Output shape matches input shape."""
    from net.hrot_fsl import mncr_refine

    D = torch.rand(5, 25, 25)
    D_ref = mncr_refine(D, temperature=0.5, lam=0.5)
    assert D_ref.shape == D.shape


def test_mncr_identity_when_lam_zero():
    """When lam=0, D_refined == D_raw (strict reduction to baseline)."""
    from net.hrot_fsl import mncr_refine

    D = torch.rand(3, 25, 25)
    D_ref = mncr_refine(D, temperature=0.5, lam=0.0)
    assert torch.allclose(D_ref, D, atol=1e-7)


def test_mncr_reduces_cost():
    """D_refined <= D_raw element-wise (cost only decreases)."""
    from net.hrot_fsl import mncr_refine

    D = torch.rand(3, 25, 25) + 0.1
    D_ref = mncr_refine(D, temperature=0.5, lam=0.5)
    assert (D_ref <= D + 1e-7).all()


def test_mncr_non_negative():
    """D_refined >= 0 always."""
    from net.hrot_fsl import mncr_refine

    D = torch.rand(3, 25, 25) * 4
    D_ref = mncr_refine(D, temperature=0.5, lam=0.5)
    assert (D_ref >= -1e-7).all()


def test_mncr_gradient_flows():
    """Gradient flows through MNCR to D_raw."""
    from net.hrot_fsl import mncr_refine

    D = torch.rand(2, 25, 25, requires_grad=True)
    D_ref = mncr_refine(D, temperature=0.5, lam=0.5)
    D_ref.sum().backward()
    assert D.grad is not None
    assert D.grad.abs().sum() > 0
