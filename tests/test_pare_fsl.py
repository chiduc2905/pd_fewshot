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
        pare_mass_ratio_bank="0.4,0.6",
        sinkhorn_iterations=40,
    )
    model.train()
    query = torch.randn(1, 2, 3, 64, 64, requires_grad=False)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    outputs = model(query, support)
    assert outputs["logits"].shape == (2, 3)
    assert torch.isfinite(outputs["logits"]).all()
    loss = outputs["logits"].sum()
    loss.backward()
    assert model.token_projector[1].weight.grad is not None
    assert torch.isfinite(model.token_projector[1].weight.grad).all()


def test_pare_fsl_factory_build():
    args = SimpleNamespace(
        model="pare_fsl",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        pare_sinkhorn_iterations=20,
    )
    model = build_model_from_args(args)
    assert model.num_mass_ratios >= 1
