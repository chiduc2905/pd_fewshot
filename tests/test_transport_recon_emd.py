from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.model_factory import build_model_from_args, get_model_choices
from net.transport_recon_emd import (
    TransportReconEMD,
    condense_support_tokens,
)


def _token_only_model(**overrides) -> TransportReconEMD:
    model = TransportReconEMD.__new__(TransportReconEMD)
    nn.Module.__init__(model)
    attrs = dict(
        feat_dim=640,
        tau_shot=0.2,
        kshot_mode="query_condense",
        sinkhorn_reg=0.1,
        sinkhorn_iter=80,
        sinkhorn_iterations=80,
        sinkhorn_tolerance=0.0,
        lambda_emd=0.3,
        lambda_rec=1.0,
        lambda_struct=0.1,
        score_scale=8.0,
        use_sfc=False,
        normalize_reconstruction=True,
        debug_shapes=True,
        eps=1e-8,
        last_transport_debug=None,
    )
    attrs.update(overrides)
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def test_transport_recon_emd_forward_works_for_one_shot_tokens():
    torch.manual_seed(701)
    model = _token_only_model()
    query = torch.randn(3, 6, 640, requires_grad=True)
    support = torch.randn(4, 1, 6, 640, requires_grad=True)

    logits = model.forward_from_tokens(query, support, way=4, shot=1)

    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()


def test_transport_recon_emd_forward_and_backward_work_for_five_shot_tokens():
    torch.manual_seed(702)
    model = _token_only_model()
    query = torch.randn(3, 6, 640, requires_grad=True)
    support = torch.randn(4, 5, 6, 640, requires_grad=True)

    logits = model.forward_from_tokens(query, support, way=4, shot=5)
    loss = F.cross_entropy(logits, torch.tensor([0, 1, 2]))
    loss.backward()

    assert logits.shape == (3, 4)
    assert query.grad is not None
    assert support.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(support.grad).all()


def test_transport_recon_emd_one_shot_condensation_is_identity():
    torch.manual_seed(703)
    query = torch.randn(2, 5, 640)
    support = torch.randn(4, 1, 5, 640)

    condensed = condense_support_tokens(query, support, mode="query_condense", tau_shot=0.2)
    expected = support[:, 0].unsqueeze(0).expand_as(condensed)

    assert torch.allclose(condensed, expected, atol=0.0, rtol=0.0)


def test_transport_recon_emd_zero_reconstruction_lambdas_match_emd_path():
    torch.manual_seed(704)
    model = _token_only_model(lambda_emd=1.0, lambda_rec=0.0, lambda_struct=0.0, score_scale=1.0)
    query = torch.randn(2, 5, 640)
    support = torch.randn(4, 5, 5, 640)

    outputs = model.forward_from_tokens(query, support, way=4, shot=5, return_aux=True)

    assert torch.allclose(outputs["logits"], outputs["score_emd"], atol=1e-7, rtol=1e-6)
    assert outputs["cost"].shape == (2, 4, 5, 5)
    assert outputs["plan"].shape == (2, 4, 5, 5)


def test_transport_recon_emd_five_shot_path_condenses_before_transport():
    torch.manual_seed(705)
    model = _token_only_model()
    query = torch.randn(2, 5, 640)
    support = torch.randn(4, 5, 5, 640)

    outputs = model.forward_from_tokens(query, support, way=4, shot=5, return_aux=True)

    assert model.use_sfc is False
    assert not hasattr(model, "get_sfc")
    assert outputs["condensed_support"].shape == (2, 4, 5, 640)
    assert outputs["cost"].shape == (2, 4, 5, 5)


def test_transport_recon_emd_model_factory_registers_aliases_and_flags():
    assert "transport_recon_emd" in get_model_choices()
    assert "tardis_emd" in get_model_choices()
    args = SimpleNamespace(
        model="transport_recon_emd",
        device="cpu",
        image_size=64,
        fewshot_backbone="resnet12",
        deepemd_sinkhorn_reg=0.12,
        deepemd_sinkhorn_iterations=7,
        deepemd_sinkhorn_tolerance=1e-5,
        deepemd_eps=1e-8,
        transport_recon_emd_tau_shot=0.3,
        transport_recon_emd_kshot_mode="mean_condense",
        transport_recon_emd_lambda_emd=0.4,
        transport_recon_emd_lambda_rec=0.9,
        transport_recon_emd_lambda_struct=0.2,
        transport_recon_emd_score_scale=6.5,
        transport_recon_emd_normalize_reconstruction="true",
        transport_recon_emd_debug_shapes="true",
    )

    model = build_model_from_args(args)

    assert isinstance(model, TransportReconEMD)
    assert model.feat_dim == 640
    assert model.tau_shot == pytest.approx(0.3)
    assert model.kshot_mode == "mean_condense"
    assert model.sinkhorn_reg == pytest.approx(0.12)
    assert model.sinkhorn_iter == 7
    assert model.lambda_emd == pytest.approx(0.4)
    assert model.lambda_rec == pytest.approx(0.9)
    assert model.lambda_struct == pytest.approx(0.2)
    assert model.score_scale == pytest.approx(6.5)
    assert model.normalize_reconstruction is True
    assert model.debug_shapes is True


def test_transport_recon_emd_normalized_reconstruction_is_norm_stable():
    torch.manual_seed(706)
    model = _token_only_model(lambda_emd=0.2, lambda_rec=1.25, lambda_struct=0.15, score_scale=4.0)
    query = 25.0 * torch.randn(2, 5, 640)
    support = 25.0 * torch.randn(4, 5, 5, 640)

    outputs = model.forward_from_tokens(query, support, way=4, shot=5, return_aux=True)

    assert outputs["rec_q"].max() <= 4.0 + 1e-4
    assert outputs["rec_s"].max() <= 4.0 + 1e-4
    assert outputs["logits"].abs().max() < 10.0


def test_transport_recon_emd_score_scale_only_scales_raw_logits():
    torch.manual_seed(707)
    query = torch.randn(2, 5, 640)
    support = torch.randn(4, 5, 5, 640)
    base = _token_only_model(score_scale=1.0)
    scaled = _token_only_model(score_scale=8.0)

    base_outputs = base.forward_from_tokens(query, support, way=4, shot=5, return_aux=True)
    scaled_outputs = scaled.forward_from_tokens(query, support, way=4, shot=5, return_aux=True)

    assert torch.allclose(base_outputs["raw_logits"], scaled_outputs["raw_logits"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(scaled_outputs["logits"], 8.0 * base_outputs["logits"], atol=1e-6, rtol=1e-5)


def test_transport_recon_emd_merge_aux_handles_scalar_tensors():
    outputs = [
        {
            "logits": torch.randn(2, 4),
            "score_scale": torch.tensor(8.0),
        },
        {
            "logits": torch.randn(3, 4),
            "score_scale": torch.tensor(8.0),
        },
    ]

    merged = TransportReconEMD._merge_aux(outputs)

    assert merged["logits"].shape == (5, 4)
    assert merged["score_scale"].ndim == 0
    assert merged["score_scale"].item() == pytest.approx(8.0)
