from __future__ import annotations

import torch

from net.deepemd import DeepEMD
from net.modules.deepemd_diagnostics import compute_deepemd_diagnostics


def _solver_only_deepemd() -> DeepEMD:
    model = DeepEMD.__new__(DeepEMD)
    torch.nn.Module.__init__(model)
    model.temperature = 12.5
    model.sinkhorn_reg = 0.1
    model.sinkhorn_iterations = 10
    model.sinkhorn_tolerance = 1e-6
    model.eps = 1e-8
    model.qpth_form = "L2"
    model.qpth_l2_strength = 1e-6
    model.uot_tau_q = 0.5
    model.uot_tau_c = 0.5
    model.uot_score_normalize = False
    model.partial_mass_fraction = 0.5
    model.partial_transport_mass = None
    model.partial_score_normalize = True
    model.partial_backend = "native"
    model.partial_exact = False
    return model


def test_deepemd_return_flow_preserves_sinkhorn_logits():
    model = _solver_only_deepemd()
    similarity = torch.rand(3, 2, 4, 4)
    query_weights = torch.rand(3, 2, 4)
    support_weights = torch.rand(2, 3, 4)

    logits = model.get_emd_distance(
        similarity,
        query_weights,
        support_weights,
        solver="sinkhorn",
    )
    logits_with_flow, flow = model.get_emd_distance(
        similarity,
        query_weights,
        support_weights,
        solver="sinkhorn",
        return_flow=True,
    )

    assert flow.shape == similarity.shape
    assert torch.allclose(logits, logits_with_flow, atol=1e-6, rtol=1e-6)


def test_deepemd_foreground_and_background_scores_reconstruct_logits():
    similarity = torch.tensor(
        [
            [
                [
                    [0.9, 0.2, 0.1, 0.0],
                    [0.3, 0.8, 0.1, 0.0],
                    [0.0, 0.1, 0.6, 0.2],
                    [0.0, 0.1, 0.2, 0.5],
                ],
                [
                    [0.2, 0.1, 0.4, 0.3],
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.4, 0.1, 0.0],
                    [0.4, 0.5, 0.0, 0.1],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    flow = torch.full_like(similarity, 0.25)
    query_weights = torch.ones(1, 2, 4)
    support_weights = torch.ones(2, 1, 4)
    query_images = torch.zeros(1, 1, 8, 8)
    query_images[:, :, :4, :4] = 1.0
    support_images = torch.zeros(2, 1, 1, 8, 8)
    support_images[0, :, :, :4, :4] = 1.0
    support_images[1, :, :, 4:, 4:] = 1.0
    temperature = 12.5
    logits = (flow * similarity).sum(dim=(-1, -2)) * (temperature / 4.0)

    diagnostics = compute_deepemd_diagnostics(
        similarity=similarity,
        flow=flow,
        query_weights=query_weights,
        support_weights=support_weights,
        query_images=query_images,
        support_images=support_images,
        logits=logits,
        uniform_weight_logits=logits - 0.1,
        temperature=temperature,
        normalize_score_by_mass=False,
        signal_quantile=0.7,
    )

    reconstructed = (
        diagnostics["deepemd_signal_only_score"]
        + diagnostics["deepemd_background_involved_score"]
    )
    assert torch.allclose(reconstructed, logits, atol=1e-6, rtol=1e-6)
    assert diagnostics["deepemd/score_reconstruction_error"].item() < 1e-6
    assert diagnostics["transport_plan"].shape == similarity.shape
    assert torch.allclose(diagnostics["deepemd_uniform_weight_score"], logits - 0.1)
    assert torch.all(
        (diagnostics["deepemd_fg_fg_mass_ratio"] >= 0.0)
        & (diagnostics["deepemd_fg_fg_mass_ratio"] <= 1.0)
    )
