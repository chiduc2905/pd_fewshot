from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from net.model_factory import build_model_from_args
from net.spif_rdp import ReliabilityCalibratedDistributionalHead, SPIFRDP


def _build_model(**overrides) -> SPIFRDP:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        stable_dim=32,
        variant_dim=32,
        gate_hidden=8,
        top_r=4,
        gate_on=True,
        factorization_on=True,
        global_only=False,
        local_only=False,
        token_l2norm=True,
        consistency_weight=0.0,
        decorr_weight=0.0,
        sparse_weight=0.0,
        consistency_dropout=0.1,
        rdp_lambda_init=0.5,
        rdp_alpha_init=1.2,
        rdp_tau_init=4.0,
        rdp_variance_floor=0.05,
        rdp_compact_loss_weight=0.1,
        rdp_sep_loss_weight=0.05,
        rdp_sep_margin=0.5,
        rdp_eps=1e-6,
    )
    kwargs.update(overrides)
    return SPIFRDP(**kwargs)


def test_distributional_head_matches_manual_formula():
    head = ReliabilityCalibratedDistributionalHead(
        stable_dim=3,
        lambda_init=0.7,
        alpha_init=1.3,
        tau_init=2.5,
        variance_floor=0.05,
        eps=1e-6,
    )

    query = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.3, 0.9, 0.0],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )
    support = F.normalize(
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.8, 0.6],
                ],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )

    outputs = head(query, support)

    lambda_value = outputs["lambda_value"]
    alpha_value = outputs["alpha_value"]
    tau_value = outputs["tau_value"]

    mu_bar = support.mean(dim=1)
    sq_dist_to_center = (support - mu_bar.unsqueeze(1)).square().sum(dim=-1)
    support_relevance = torch.exp(-lambda_value * sq_dist_to_center)
    support_weights = support_relevance / support_relevance.sum(dim=1, keepdim=True).clamp_min(1e-6)
    prototype = torch.sum(support_weights.unsqueeze(-1) * support, dim=1)
    diagonal_variance = torch.sum(support_weights.unsqueeze(-1) * (support - prototype.unsqueeze(1)).square(), dim=1)
    variance = diagonal_variance.clamp_min(0.05)
    compactness = diagonal_variance.mean(dim=-1)
    reliability = torch.exp(-alpha_value * compactness).clamp(min=1e-6, max=1.0)

    diff = query.unsqueeze(1) - prototype.unsqueeze(0)
    mahalanobis_distance = (diff.square() / variance.unsqueeze(0).clamp_min(1e-6)).sum(dim=-1)
    euclidean_distance = diff.square().sum(dim=-1)
    total_distance = reliability.unsqueeze(0) * mahalanobis_distance + (1.0 - reliability.unsqueeze(0)) * euclidean_distance
    global_scores = -total_distance / tau_value

    assert torch.allclose(outputs["support_weights"], support_weights, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["prototype"], prototype, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["variance"], variance, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["compactness"], compactness, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["reliability"], reliability, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["mahalanobis_distance"], mahalanobis_distance, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["euclidean_distance"], euclidean_distance, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["total_distance"], total_distance, atol=1e-6, rtol=0.0)
    assert outputs["bw_distance"].shape == total_distance.shape
    assert torch.isfinite(outputs["bw_distance"]).all()
    assert torch.allclose(outputs["global_scores"], global_scores, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["support_weights"].sum(dim=1), torch.ones(2), atol=1e-6, rtol=0.0)


def test_distributional_head_one_shot_uses_variance_floor():
    head = ReliabilityCalibratedDistributionalHead(
        stable_dim=3,
        lambda_init=0.2,
        alpha_init=1.0,
        tau_init=3.0,
        variance_floor=0.05,
        eps=1e-6,
    )

    query = F.normalize(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32), p=2, dim=-1)
    support = F.normalize(
        torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )

    outputs = head(query, support)

    expected_variance = torch.full((2, 3), 0.05, dtype=torch.float32)
    assert torch.allclose(outputs["variance"], expected_variance, atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["support_weights"], torch.ones(2, 1), atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["compactness"], torch.zeros(2), atol=1e-6, rtol=0.0)
    assert torch.allclose(outputs["reliability"], torch.ones(2), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("shot_num", [1, 5])
def test_spif_rdp_forward_is_finite_in_low_shot_regimes(shot_num: int):
    torch.manual_seed(shot_num)
    model = _build_model()
    model.eval()

    query = torch.randn(1, 3, 3, 64, 64)
    support = torch.randn(1, 5, shot_num, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (3, 5)
    assert outputs["global_scores"].shape == (3, 5)
    assert outputs["total_distance"].shape == (3, 5)
    assert outputs["bw_distance"].shape == (3, 5)
    assert outputs["mahalanobis_distance"].shape == (3, 5)
    assert outputs["euclidean_distance"].shape == (3, 5)
    assert outputs["class_reliability"].shape == (3, 5)
    assert outputs["class_compactness"].shape == (3, 5)
    assert outputs["prototype"].shape == (1, 5, 32)
    assert outputs["variance"].shape == (1, 5, 32)
    assert outputs["support_weights"].shape == (1, 5, shot_num)

    for key in (
        "logits",
        "global_scores",
        "total_distance",
        "bw_distance",
        "mahalanobis_distance",
        "euclidean_distance",
        "class_reliability",
        "class_compactness",
        "prototype",
        "variance",
        "support_weights",
        "stable_global_embeddings",
        "variant_global_embeddings",
    ):
        assert torch.isfinite(outputs[key]).all()

    for key in (
        "aux_loss",
        "mean_reliability",
        "compact_loss",
        "separation_loss",
        "factorization_aux_loss",
        "lambda_value",
        "alpha_value",
        "tau_value",
        "mean_gate",
    ):
        assert torch.isfinite(outputs[key])

    if shot_num == 1:
        expected_variance = torch.full_like(outputs["variance"], 0.05)
        assert torch.allclose(outputs["variance"], expected_variance, atol=1e-6, rtol=0.0)


def test_spif_rdp_backpropagates_through_distributional_and_local_paths():
    torch.manual_seed(7)
    model = _build_model(
        rdp_compact_loss_weight=0.1,
        rdp_sep_loss_weight=0.05,
        consistency_weight=0.05,
        decorr_weight=0.01,
        sparse_weight=0.001,
    )
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 4, 2, 3, 64, 64)
    outputs = model(query, support, return_aux=True)

    loss = outputs["logits"].pow(2).mean() + outputs["aux_loss"]
    loss.backward()

    assert model.encoder_head.stable_head[1].weight.grad is not None
    assert model.distributional_head.lambda_raw.grad is not None
    assert model.distributional_head.alpha_raw.grad is not None
    assert model.distributional_head.tau_raw.grad is not None
    assert torch.isfinite(model.encoder_head.stable_head[1].weight.grad).all()
    assert torch.isfinite(model.distributional_head.lambda_raw.grad).all()
    assert torch.isfinite(model.distributional_head.alpha_raw.grad).all()
    assert torch.isfinite(model.distributional_head.tau_raw.grad).all()


def test_spif_rdp_model_factory_builds_and_runs():
    args = SimpleNamespace(
        model="spif_rdp",
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        spif_stable_dim=32,
        spif_variant_dim=32,
        spif_gate_hidden=8,
        spif_gate_on="true",
        spif_factorization_on="true",
        spif_global_only="false",
        spif_local_only="false",
        spif_token_l2norm="true",
        spif_consistency_dropout=0.1,
        spif_rdp_top_r=4,
        spif_rdp_lambda_init=0.5,
        spif_rdp_alpha_init=1.2,
        spif_rdp_tau_init=4.0,
        spif_rdp_variance_floor=0.05,
        spif_rdp_compact_loss_weight=0.1,
        spif_rdp_sep_loss_weight=0.05,
        spif_rdp_sep_margin=0.5,
        spif_rdp_eps=1e-6,
        spif_rdp_consistency_weight=0.0,
        spif_rdp_decorr_weight=0.0,
        spif_rdp_sparse_weight=0.0,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["global_scores"].shape == (2, 3)
    assert outputs["total_distance"].shape == (2, 3)


def test_spif_rdp_fsl_mamba_backbone_runs_and_backpropagates():
    torch.manual_seed(11)
    model = _build_model(
        backbone_name="fsl_mamba",
        image_size=32,
        hidden_dim=64,
        stable_dim=24,
        variant_dim=24,
        gate_hidden=8,
        fsl_mamba_base_dim=16,
        fsl_mamba_output_dim=64,
        fsl_mamba_drop_path=0.0,
        fsl_mamba_perturb_sigma=0.0,
        rdp_use_fsl_mamba_global_prior=True,
        rdp_fsl_mamba_global_mix_init=0.4,
    )
    model.train()

    query = torch.randn(1, 2, 3, 32, 32)
    support = torch.randn(1, 3, 2, 3, 32, 32)

    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].pow(2).mean() + outputs["aux_loss"]
    loss.backward()

    assert outputs["logits"].shape == (2, 3)
    assert outputs["local_scores"].shape == (2, 3)
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["local_scores"]).all()
    assert model.fsl_mamba_global_adapter[1].weight.grad is not None
    assert torch.isfinite(model.fsl_mamba_global_adapter[1].weight.grad).all()


def test_spif_rdp_model_factory_builds_and_runs_with_fsl_mamba():
    args = SimpleNamespace(
        model="spif_rdp",
        device="cpu",
        image_size=32,
        fewshot_backbone="fsl_mamba",
        spif_stable_dim=24,
        spif_variant_dim=24,
        spif_gate_hidden=8,
        spif_gate_on="true",
        spif_factorization_on="true",
        spif_global_only="false",
        spif_local_only="false",
        spif_token_l2norm="true",
        spif_consistency_dropout=0.1,
        spif_rdp_top_r=4,
        spif_rdp_lambda_init=0.5,
        spif_rdp_alpha_init=1.2,
        spif_rdp_tau_init=4.0,
        spif_rdp_variance_floor=0.05,
        spif_rdp_compact_loss_weight=0.1,
        spif_rdp_sep_loss_weight=0.05,
        spif_rdp_sep_margin=0.5,
        spif_rdp_eps=1e-6,
        spif_rdp_consistency_weight=0.0,
        spif_rdp_decorr_weight=0.0,
        spif_rdp_sparse_weight=0.0,
        spif_rdp_beta_init=0.5,
        spif_rdp_fsl_mamba_base_dim=16,
        spif_rdp_fsl_mamba_output_dim=64,
        spif_rdp_fsl_mamba_drop_path=0.0,
        spif_rdp_fsl_mamba_perturb_sigma=0.0,
        spif_rdp_use_fsl_mamba_global_prior="true",
        spif_rdp_fsl_mamba_global_mix_init=0.4,
    )
    model = build_model_from_args(args)
    model.eval()

    query = torch.randn(1, 2, 3, 32, 32)
    support = torch.randn(1, 3, 2, 3, 32, 32)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["global_scores"].shape == (2, 3)
    assert outputs["local_scores"].shape == (2, 3)
