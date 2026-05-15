"""Tests for HROT Tier-1 transport projector flags."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.hrot_fsl import HROTFSL
from net.model_factory import resolve_hrot_token_dim
from net.modules.hrot_transport_projector import HROTTransportProjector, build_hrot_transport_projector


def _tiny_hrot(**kwargs) -> HROTFSL:
    defaults = dict(
        in_channels=3,
        hidden_dim=64,
        token_dim=24,
        backbone_name="conv64f",
        image_size=64,
        variant="E",
        sinkhorn_iterations=5,
        hyperbolic_backend="native",
    )
    defaults.update(kwargs)
    return HROTFSL(**defaults)


def test_build_legacy_projector_is_sequential():
    proj = build_hrot_transport_projector(640, 128, use_mlp=False, use_residual=False)
    assert isinstance(proj, nn.Sequential)
    assert len(proj) == 2


def test_build_mlp_residual_projector():
    proj = build_hrot_transport_projector(640, 128, use_mlp=True, use_residual=True, mlp_hidden_dim=256)
    assert isinstance(proj, HROTTransportProjector)
    assert proj.use_mlp and proj.use_residual
    tokens = torch.randn(2, 9, 640)
    out = proj(tokens)
    assert out.shape == (2, 9, 128)


def test_hrot_projector_mlp_forward():
    model = _tiny_hrot(projector_use_mlp=True, projector_mlp_hidden_dim=32)
    assert isinstance(model.token_projector, HROTTransportProjector)
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    with torch.no_grad():
        out = model(query, support)
    assert out["logits"].shape == (2, 3)


def test_hrot_projector_residual_gradients():
    model = _tiny_hrot(projector_use_residual=True)
    assert isinstance(model.token_projector, HROTTransportProjector)
    model.train()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    targets = torch.tensor([1, 0], dtype=torch.long)
    outputs = model(query, support, query_targets=targets)
    loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert model.token_projector.skip is not None
    assert model.token_projector.skip.weight.grad is not None
    assert torch.isfinite(model.token_projector.skip.weight.grad).all()


def test_resolve_hrot_token_dim_wide_flag():
    class Args:
        hrot_token_dim = None
        hrot_projector_wide = "true"
        hrot_projector_wide_dim = 192
        token_dim = 128

    assert resolve_hrot_token_dim(Args()) == 192

    class ArgsExplicit:
        hrot_token_dim = 160
        hrot_projector_wide = "true"
        hrot_projector_wide_dim = 192
        token_dim = 128

    assert resolve_hrot_token_dim(ArgsExplicit()) == 160


def test_raw_backbone_rejects_projector_upgrades():
    try:
        _tiny_hrot(use_raw_backbone_tokens=True, projector_use_mlp=True)
    except ValueError as exc:
        assert "raw_backbone" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for mlp + raw backbone")
