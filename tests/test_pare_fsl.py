from __future__ import annotations

import torch
from types import SimpleNamespace

from net.model_factory import build_model_from_args
from net.pare_fsl import PAREFSL


def test_pare_fsl_forward_and_gradients():
    torch.manual_seed(0)
    model = PAREFSL(
        hidden_dim=64,
        token_dim=32,
        backbone_name="conv64f",
        image_size=64,
        sinkhorn_iterations=40,
        marginal_mode="egsm",
        enable_learned_alpha=True,
    )
    model.train()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    outputs = model(query, support)
    assert outputs["logits"].shape == (2, 3)
    assert torch.isfinite(outputs["logits"]).all()
    assert outputs["partial_alpha"].shape == (2, 3, 2)
    assert torch.all(outputs["partial_alpha"] >= model.alpha_head.alpha_min)
    assert torch.all(outputs["partial_alpha"] <= model.alpha_head.alpha_max)
    loss = outputs["logits"].sum() + outputs["aux_loss"]
    loss.backward()
    assert model.token_projector[1].weight.grad is not None
    assert model.alpha_head.mlp[0].weight.grad is not None
    assert torch.isfinite(model.token_projector[1].weight.grad).all()


def test_pare_fsl_fixed_alpha_no_head():
    model = PAREFSL(
        hidden_dim=64,
        token_dim=32,
        backbone_name="conv64f",
        image_size=64,
        sinkhorn_iterations=30,
        fixed_alpha=0.5,
    )
    assert model.alpha_head is None
    query = torch.randn(1, 1, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)
    outputs = model(query, support)
    assert torch.allclose(outputs["partial_alpha"], torch.tensor(0.5), atol=1e-4)


def test_pare_fsl_factory_build():
    args = SimpleNamespace(
        model="pare_fsl",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        pare_sinkhorn_iterations=20,
    )
    model = build_model_from_args(args)
    assert model.egsm is not None
    assert model.enable_learned_alpha
