from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from net.deepemd import DeepEMD
from net.dice_emd import DICEEMD
from net.model_factory import build_model_from_args, get_model_choices


def _solver_only_dice(**overrides) -> DICEEMD:
    model = DICEEMD.__new__(DICEEMD)
    nn.Module.__init__(model)
    attrs = dict(
        feat_dim=640,
        temperature=12.5,
        metric="cosine",
        norm="center",
        solver="sinkhorn",
        qpth_form="L2",
        qpth_l2_strength=1e-6,
        sinkhorn_reg=0.1,
        sinkhorn_iterations=80,
        sinkhorn_tolerance=1e-7,
        eps=1e-8,
        sfc_lr=0.1,
        sfc_update_step=1,
        sfc_bs=2,
        lambda_disc=0.2,
        tau_comp=0.1,
        use_softmin_distance=True,
        tau_softmin=0.1,
        debug_transport=True,
        lambda_flow=0.0,
        last_transport_debug=None,
    )
    attrs.update(overrides)
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def _solver_only_deepemd(**overrides) -> DeepEMD:
    model = DeepEMD.__new__(DeepEMD)
    nn.Module.__init__(model)
    attrs = dict(
        temperature=12.5,
        sinkhorn_reg=0.1,
        sinkhorn_iterations=80,
        sinkhorn_tolerance=1e-7,
        eps=1e-8,
    )
    attrs.update(overrides)
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def test_dice_emd_sinkhorn_outputs_evidence_plan_and_gradients():
    torch.manual_seed(501)
    similarity = torch.rand(2, 3, 4, 5, requires_grad=True)
    weight1 = torch.rand(2, 3, 4) + 0.1
    weight2 = torch.rand(3, 2, 5) + 0.1

    model = _solver_only_dice()
    logits = model.get_emd_distance(similarity, weight1, weight2, solver="sinkhorn")

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert model.last_transport_debug is not None
    evidence = model.last_transport_debug["evidence_g"]
    plan = model.last_transport_debug["plan"]
    assert evidence.shape == (2, 3, 4)
    assert plan.shape == (2, 3, 4, 5)
    assert torch.isfinite(evidence).all()
    assert torch.isfinite(plan).all()

    logits.sum().backward()
    assert similarity.grad is not None
    assert torch.isfinite(similarity.grad).all()


def test_dice_emd_emd_forward_keeps_640_dim_dense_tokens():
    torch.manual_seed(502)
    model = _solver_only_dice(use_softmin_distance=False)
    proto = torch.randn(3, 640, 2, 2, requires_grad=True)
    query = torch.randn(2, 640, 2, 2, requires_grad=True)

    logits = model.emd_forward(proto, query, solver="sinkhorn")

    assert logits.shape == (2, 3)
    assert model.feat_dim == 640
    loss = logits.pow(2).mean()
    loss.backward()
    assert proto.grad is not None
    assert query.grad is not None
    assert torch.isfinite(proto.grad).all()
    assert torch.isfinite(query.grad).all()


def test_dice_emd_lambda_zero_matches_deepemd_sinkhorn():
    torch.manual_seed(503)
    similarity = torch.rand(2, 4, 3, 3)
    weight1 = torch.rand(2, 4, 3) + 0.1
    weight2 = torch.rand(4, 2, 3) + 0.1

    dice = _solver_only_dice(lambda_disc=0.0, debug_transport=False)
    deepemd = _solver_only_deepemd()

    dice_logits = dice.get_emd_distance(similarity, weight1, weight2, solver="sinkhorn")
    baseline_logits = deepemd.get_emd_distance(similarity, weight1, weight2, solver="sinkhorn")

    assert torch.allclose(dice_logits, baseline_logits, atol=1e-6, rtol=1e-5)


def test_dice_emd_model_factory_registers_config_flags():
    assert "dice_emd" in get_model_choices()
    args = SimpleNamespace(
        model="dice_emd",
        device="cpu",
        image_size=64,
        fewshot_backbone="resnet12",
        deepemd_solver="sinkhorn",
        deepemd_qpth_form="L2",
        deepemd_qpth_l2_strength=1e-6,
        deepemd_sinkhorn_reg=0.12,
        deepemd_sinkhorn_iterations=7,
        deepemd_sinkhorn_tolerance=1e-5,
        deepemd_eps=1e-8,
        deepemd_sfc_lr=0.1,
        deepemd_sfc_update_step=1,
        deepemd_sfc_bs=2,
        dice_emd_lambda_disc=0.35,
        dice_emd_tau_comp=0.2,
        dice_emd_use_softmin_distance="false",
        dice_emd_tau_softmin=0.15,
        dice_emd_debug_transport="true",
        dice_emd_lambda_flow=0.0,
    )

    model = build_model_from_args(args)

    assert isinstance(model, DICEEMD)
    assert model.feat_dim == 640
    assert model.lambda_disc == pytest.approx(0.35)
    assert model.tau_comp == pytest.approx(0.2)
    assert model.use_softmin_distance is False
    assert model.tau_softmin == pytest.approx(0.15)
    assert model.debug_transport is True
