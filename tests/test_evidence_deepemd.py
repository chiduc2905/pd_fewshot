from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from net.deepemd import DeepEMD
from net.evidence_deepemd import EvidenceDeepEMD, EvidenceMLP
from net.model_factory import build_model_from_args, get_model_choices


def _token_only_model(**overrides) -> EvidenceDeepEMD:
    model = EvidenceDeepEMD.__new__(EvidenceDeepEMD)
    nn.Module.__init__(model)
    feat_dim = int(overrides.pop("feat_dim", 8))
    attrs = dict(
        feat_dim=feat_dim,
        temperature=12.5,
        metric="cosine",
        norm="center",
        solver="sinkhorn",
        qpth_form="L2",
        qpth_l2_strength=1e-6,
        sinkhorn_reg=0.1,
        sinkhorn_iterations=30,
        sinkhorn_tolerance=1e-7,
        eps=1e-8,
        sfc_lr=0.1,
        sfc_update_step=1,
        sfc_bs=2,
        use_evidence_weight=True,
        use_shot_reliability=True,
        evidence_eps=1e-3,
        consensus_scale=5.0,
        shot_temperature=1.0,
        debug_evidence=True,
        last_evidence_debug=None,
    )
    attrs.update(overrides)
    for key, value in attrs.items():
        setattr(model, key, value)
    model.evidence_mlp = EvidenceMLP(feat_dim)
    return model


def _token_only_deepemd(**overrides) -> DeepEMD:
    model = DeepEMD.__new__(DeepEMD)
    nn.Module.__init__(model)
    attrs = dict(
        feat_dim=8,
        temperature=12.5,
        metric="cosine",
        norm="center",
        solver="sinkhorn",
        qpth_form="L2",
        qpth_l2_strength=1e-6,
        sinkhorn_reg=0.1,
        sinkhorn_iterations=30,
        sinkhorn_tolerance=1e-7,
        eps=1e-8,
        sfc_lr=0.1,
        sfc_update_step=1,
        sfc_bs=2,
    )
    attrs.update(overrides)
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def _factory_args(**overrides):
    args = SimpleNamespace(
        model="evidence_deepemd",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        deepemd_solver="sinkhorn",
        deepemd_qpth_form="L2",
        deepemd_qpth_l2_strength=1e-6,
        deepemd_sinkhorn_reg=0.12,
        deepemd_sinkhorn_iterations=7,
        deepemd_sinkhorn_tolerance=1e-5,
        deepemd_uot_tau_q=0.5,
        deepemd_uot_tau_c=0.5,
        deepemd_uot_score_normalize="false",
        deepemd_partial_mass_fraction=0.5,
        deepemd_partial_transport_mass=None,
        deepemd_partial_score_normalize="true",
        deepemd_partial_backend="native",
        deepemd_partial_exact="false",
        deepemd_eps=1e-8,
        deepemd_sfc_lr=0.1,
        deepemd_sfc_update_step=1,
        deepemd_sfc_bs=2,
        use_evidence_weight="true",
        use_shot_reliability="true",
        evidence_eps=1e-3,
        consensus_scale=5.0,
        shot_temperature=0.7,
        debug_evidence="false",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_evidence_deepemd_mass_sums_and_logits_are_finite():
    torch.manual_seed(601)
    model = _token_only_model()
    proto = torch.randn(3, 8, 2, 2)
    query = torch.randn(2, 8, 2, 2)
    class_proto = proto.flatten(-2).mean(dim=-1)

    logits = model.emd_forward(proto, query, solver="sinkhorn", class_proto=class_proto)
    debug = model.last_evidence_debug

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert debug is not None
    assert torch.isfinite(debug["token_reliability"]).all()
    assert torch.isfinite(debug["query_mass_sum"]).all()
    assert torch.isfinite(debug["support_mass_sum"]).all()
    assert torch.allclose(debug["query_mass_sum"], torch.ones_like(debug["query_mass_sum"]), atol=1e-6)
    assert torch.allclose(debug["support_mass_sum"], torch.ones_like(debug["support_mass_sum"]), atol=1e-6)


def test_evidence_deepemd_ablation_off_matches_deepemd_balanced_sinkhorn():
    torch.manual_seed(602)
    proto = torch.randn(3, 8, 2, 2)
    query = torch.randn(2, 8, 2, 2)
    evidence = _token_only_model(use_evidence_weight=False, use_shot_reliability=False)
    baseline = _token_only_deepemd()

    evidence_logits = evidence.emd_forward(proto, query, solver="sinkhorn")
    baseline_logits = baseline.emd_forward(proto, query, solver="sinkhorn")

    assert torch.allclose(evidence_logits, baseline_logits, atol=1e-6, rtol=1e-5)


def test_evidence_deepemd_shot_alpha_sums_for_one_and_five_shot():
    torch.manual_seed(603)
    model = _token_only_model()
    query = torch.randn(2, 8, 2, 2)

    one_shot_support = torch.randn(3, 1, 8, 2, 2)
    one_shot_logits = model.shot_reliability_forward(one_shot_support, query, solver="sinkhorn")
    one_debug = model.last_evidence_debug
    assert one_shot_logits.shape == (2, 3)
    assert one_debug is not None
    assert torch.allclose(one_debug["shot_alpha"], torch.ones_like(one_debug["shot_alpha"]))
    assert torch.allclose(one_debug["shot_alpha_sum"], torch.ones_like(one_debug["shot_alpha_sum"]))
    assert torch.isfinite(one_shot_logits).all()

    five_shot_support = torch.randn(3, 5, 8, 2, 2)
    five_shot_logits = model.shot_reliability_forward(five_shot_support, query, solver="sinkhorn")
    five_debug = model.last_evidence_debug
    assert five_shot_logits.shape == (2, 3)
    assert five_debug is not None
    assert five_debug["shot_alpha"].shape == (2, 3, 5)
    assert torch.allclose(five_debug["shot_alpha_sum"], torch.ones_like(five_debug["shot_alpha_sum"]), atol=1e-6)
    assert torch.isfinite(five_debug["shot_alpha"]).all()
    assert torch.isfinite(five_shot_logits).all()


def test_evidence_deepemd_factory_registers_flags():
    assert "evidence_deepemd" in get_model_choices()
    model = build_model_from_args(_factory_args())

    assert isinstance(model, EvidenceDeepEMD)
    assert model.use_evidence_weight is True
    assert model.use_shot_reliability is True
    assert model.evidence_eps == pytest.approx(1e-3)
    assert model.consensus_scale == pytest.approx(5.0)
    assert model.shot_temperature == pytest.approx(0.7)


def test_evidence_deepemd_full_forward_one_and_five_shot_shapes():
    torch.manual_seed(604)
    model = build_model_from_args(_factory_args(deepemd_sinkhorn_iterations=3))
    model.eval()

    one_query = torch.randn(1, 2, 3, 64, 64)
    one_support = torch.randn(1, 3, 1, 3, 64, 64)
    with torch.no_grad():
        one_logits = model(one_query, one_support, exact=False)
    assert one_logits.shape == (2, 3)
    assert torch.isfinite(one_logits).all()

    five_query = torch.randn(1, 2, 3, 64, 64)
    five_support = torch.randn(1, 3, 5, 3, 64, 64)
    with torch.no_grad():
        five_logits = model(five_query, five_support, exact=False)
    assert five_logits.shape == (2, 3)
    assert torch.isfinite(five_logits).all()
    assert model.last_evidence_debug is not None
    alpha_sum = model.last_evidence_debug["shot_alpha_sum"]
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-6)
