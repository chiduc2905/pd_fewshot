from __future__ import annotations

from types import SimpleNamespace

import torch


def test_cata_forward_shape():
    from net.modules.cata import CATA

    cata = CATA(token_dim=128, num_anchors=8, num_heads=4)
    z = torch.randn(5, 25, 128)

    out = cata(z)

    assert out.shape == (5, 8, 128)


def test_cata_grad_flow():
    from net.modules.cata import CATA

    cata = CATA(token_dim=128, num_anchors=8, num_heads=4)
    z = torch.randn(3, 25, 128, requires_grad=True)

    cata(z).sum().backward()

    assert cata.anchors.grad is not None
    assert cata.anchors.grad.abs().sum().item() > 0.0


def test_m2_can_enable_cata_explicitly():
    from net.model_factory import build_model_from_args

    args = SimpleNamespace(
        model="m2",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=24,
        hrot_eam_hidden_dim=32,
        hrot_use_cata="true",
        hrot_cata_num_anchors=8,
        hrot_cata_num_heads=4,
        hrot_cata_attn_dropout=0.0,
        hrot_ecot_rho_bank="0.80",
        hrot_ecot_base_rho=0.80,
        hrot_ecot_enable_tau_shot="true",
        hrot_sinkhorn_iterations=8,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert model.use_cata
    assert model.cata is not None
    assert outputs["logits"].shape == (2, 2)
    assert outputs["query_euclidean_tokens"].shape[-2] == 8
    assert outputs["support_euclidean_tokens"].shape[-2] == 8


def test_ours_defaults_without_cata_and_learned_mass():
    from net.model_factory import build_model_from_args

    model = build_model_from_args(
        SimpleNamespace(
            model="ours",
            ours_ablation="full",
            device="cpu",
            image_size=64,
            fewshot_backbone="conv64f",
            hrot_token_dim=24,
            hrot_eam_hidden_dim=32,
            hrot_sinkhorn_iterations=8,
            hrot_sinkhorn_tolerance=1e-5,
        )
    )
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert not model.use_cata
    assert model.cata is None
    assert not model.uses_learned_mass
    assert model.ecot_m2_ablate_threshold_mass
    assert not model.ecot_m2_use_aqm
    assert not model.ecot_m2_use_swts
    assert outputs["logits"].shape == (2, 2)
    assert outputs["query_euclidean_tokens"].shape[-2] == 16
    assert outputs["support_euclidean_tokens"].shape[-2] == 16
