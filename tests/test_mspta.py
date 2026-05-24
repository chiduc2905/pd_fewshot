from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from net.model_factory import build_model_from_args
from net.modules.mspta import MSPTATokenizer
from net.ours import OursM2


def _tiny_ours_final(**overrides) -> OursM2:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=8,
        eam_hidden_dim=16,
        sinkhorn_iterations=4,
        sinkhorn_tolerance=1e-5,
        eval_use_float64=False,
        ecot_enable_egsm=False,
        ecot_m2_ablate_threshold_mass=False,
        ecot_rho_bank="0.8",
        ecot_base_rho=0.8,
        ecot_transport_mode="unbalanced",
    )
    kwargs.update(overrides)
    return OursM2(**kwargs)


def test_mspta_tokenizer_builds_area_weighted_pyramid():
    feature_map = torch.randn(2, 4, 5, 5)
    tokenizer = MSPTATokenizer(scales="1,2,3", mass_mode="area")

    records = tokenizer(feature_map)
    tokens = torch.cat([record.tokens for record in records], dim=1)
    weights = torch.cat([record.weights for record in records], dim=1)

    assert [record.spatial_hw for record in records] == [(5, 5), (3, 3), (2, 2)]
    assert tokens.shape == (2, 38, 4)
    assert weights.shape == (2, 38)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    fine_mass = weights[:, :25].mean()
    mid_mass = weights[:, 25:34].mean()
    coarse_mass = weights[:, 34:].mean()
    assert coarse_mass > mid_mass > fine_mass


def test_mspta_balanced_area_keeps_equal_scale_budget():
    feature_map = torch.randn(2, 4, 4, 4)
    tokenizer = MSPTATokenizer(scales="1,2,3", mass_mode="balanced_area")

    records = tokenizer(feature_map)
    weights = torch.cat([record.weights for record in records], dim=1)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    shares = []
    start = 0
    for record in records:
        stop = start + record.weights.shape[-1]
        shares.append(weights[:, start:stop].sum(dim=-1))
        start = stop
    for share in shares:
        assert torch.allclose(share, torch.full_like(share, 1.0 / 3.0), atol=1e-6)


def test_ours_final_mspta_forward_uses_area_marginals_and_exposes_diagnostics():
    torch.manual_seed(611)
    model = _tiny_ours_final(
        enable_mspta=True,
        mspta_scales="1,2,3",
        mspta_mass_mode="area",
        mspta_learnable_weights=True,
    )
    model.train()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum()
    loss.backward()

    assert hasattr(model, "mspta_tokenizer")
    assert model.mspta_scale_logits.shape == (3,)
    assert outputs["logits"].shape == (2, 2)
    assert outputs["query_euclidean_tokens"].shape[-2] == 24
    assert outputs["cost_matrix"].shape[-2:] == (24, 24)
    assert outputs["mspta/token_count"].item() == pytest.approx(24.0)
    assert outputs["mspta/query_weight_sum"].item() == pytest.approx(1.0, abs=1e-6)
    assert outputs["mspta/support_weight_sum"].item() == pytest.approx(1.0, abs=1e-6)
    assert outputs["mspta/query_weight_mean_s3"] > outputs["mspta/query_weight_mean_fine"]
    assert model.mspta_scale_logits.grad is not None
    assert torch.isfinite(model.mspta_scale_logits.grad).all()


def test_ours_final_mspta_default_uses_balanced_area_for_small_grids():
    torch.manual_seed(610)
    model = _tiny_ours_final(enable_mspta=True)
    model.eval()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert model.mspta_tokenizer.mass_mode == "balanced_area"
    assert outputs["mspta/query_weight_share_fine"].item() == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert outputs["mspta/query_weight_share_s2"].item() == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert outputs["mspta/query_weight_share_s3"].item() == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_mspta_factory_flags_are_ours_final_only():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=4,
        hrot_sinkhorn_tolerance=1e-5,
        enable_mspta=True,
        mspta_scales="1,2,3",
        mspta_mass_mode="area",
        mspta_learnable_weights=False,
    )
    ours_final = build_model_from_args(
        SimpleNamespace(
            model="ours_final",
            ours_ablation="full",
            **common,
        )
    )
    assert hasattr(ours_final, "mspta_tokenizer")
    assert ours_final.mspta_tokenizer.scales == (1, 2, 3)

    with pytest.raises(ValueError, match="--enable_mspta is supported only with --model ours_final"):
        build_model_from_args(
            SimpleNamespace(
                model="ours",
                ours_ablation="full",
                **common,
            )
        )
