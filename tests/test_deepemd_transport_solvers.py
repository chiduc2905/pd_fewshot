from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from net.deepemd import DeepEMD
from net.model_factory import build_model_from_args


def _solver_only_deepemd(**overrides) -> DeepEMD:
    model = DeepEMD.__new__(DeepEMD)
    nn.Module.__init__(model)
    attrs = dict(
        temperature=12.5,
        sinkhorn_reg=0.1,
        sinkhorn_iterations=80,
        sinkhorn_tolerance=1e-7,
        uot_tau_q=0.5,
        uot_tau_c=0.5,
        uot_score_normalize=False,
        partial_mass_fraction=0.5,
        partial_transport_mass=None,
        partial_score_normalize=True,
        partial_backend="native",
        partial_exact=False,
        eps=1e-8,
    )
    attrs.update(overrides)
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def test_deepemd_uot_and_partial_ot_solvers_return_finite_logits():
    torch.manual_seed(401)
    similarity = torch.rand(2, 3, 4, 5)
    weight1 = torch.rand(2, 3, 4) + 0.1
    weight2 = torch.rand(3, 2, 5) + 0.1

    model = _solver_only_deepemd()
    for solver in ("sinkhorn", "uot", "partial_ot"):
        logits = model.get_emd_distance(similarity, weight1, weight2, solver=solver)
        assert logits.shape == (2, 3)
        assert torch.isfinite(logits).all()


def test_deepemd_partial_ot_keeps_gradients_through_similarity():
    torch.manual_seed(402)
    similarity = torch.rand(2, 3, 4, 5, requires_grad=True)
    weight1 = torch.rand(2, 3, 4) + 0.1
    weight2 = torch.rand(3, 2, 5) + 0.1
    model = _solver_only_deepemd(partial_mass_fraction=0.4)

    logits = model.get_emd_distance(similarity, weight1, weight2, solver="partial_ot")
    loss = logits.sum()
    loss.backward()

    assert similarity.grad is not None
    assert torch.isfinite(similarity.grad).all()


def test_deepemd_model_factory_passes_partial_and_uot_args():
    args = SimpleNamespace(
        model="deepemd",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        deepemd_solver="partial_ot",
        deepemd_qpth_form="L2",
        deepemd_qpth_l2_strength=1e-6,
        deepemd_sinkhorn_reg=0.12,
        deepemd_sinkhorn_iterations=7,
        deepemd_sinkhorn_tolerance=1e-5,
        deepemd_uot_tau_q=0.4,
        deepemd_uot_tau_c=0.6,
        deepemd_uot_score_normalize="false",
        deepemd_partial_mass_fraction=0.35,
        deepemd_partial_transport_mass=None,
        deepemd_partial_score_normalize="true",
        deepemd_partial_backend="native",
        deepemd_partial_exact="false",
        deepemd_eps=1e-8,
        deepemd_sfc_lr=0.1,
        deepemd_sfc_update_step=1,
        deepemd_sfc_bs=2,
    )

    model = build_model_from_args(args)

    assert model.solver == "partial_ot"
    assert model.sinkhorn_iterations == 7
    assert model.uot_tau_q == 0.4
    assert model.uot_tau_c == 0.6
    assert model.partial_mass_fraction == 0.35
