import torch


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
