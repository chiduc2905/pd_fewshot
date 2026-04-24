from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from net.heads.rada_head import RADAFewShotHead
from net.model_factory import build_model_from_args
from net.rada_fsl import RADAFSL


def _build_head(**overrides) -> RADAFewShotHead:
    kwargs = dict(
        feat_dim=8,
        tau_r=0.5,
        lambda_proto=0.7,
        gamma_disp=0.5,
        eps=1e-6,
        reliability_head="linear",
        reliability_hidden_dim=None,
        l2_normalize=True,
        use_reliability=True,
        use_dispersion_metric=True,
        query_conditioned=True,
        use_residual_anchor=True,
        use_shrinkage=True,
    )
    kwargs.update(overrides)
    return RADAFewShotHead(**kwargs)


def _random_episode(batch_size: int = 2, num_way: int = 3, num_shot: int = 4, num_query: int = 5, feat_dim: int = 8):
    torch.manual_seed(batch_size * 1000 + num_way * 100 + num_shot * 10 + num_query + feat_dim)
    support = torch.randn(batch_size, num_way, num_shot, feat_dim)
    query = torch.randn(batch_size, num_query, feat_dim)
    return support, query


def test_rada_head_output_shapes():
    support, query = _random_episode()
    head = _build_head(feat_dim=support.shape[-1])

    logits, aux = head(support, query)

    assert logits.shape == (2, 5, 3)
    assert aux["alpha"].shape == (2, 5, 3, 4)
    assert aux["proto"].shape == (2, 5, 3, 8)
    assert aux["disp"].shape == (2, 5, 3, 8)
    assert aux["mu"].shape == (2, 3, 8)
    assert aux["raw_scatter"].shape == (2, 3)


def test_rada_head_alpha_normalizes_over_support_shots():
    support, query = _random_episode(batch_size=1, num_way=4, num_shot=3, num_query=2, feat_dim=6)
    head = _build_head(feat_dim=6)

    _, aux = head(support, query)

    alpha_sum = aux["alpha"].sum(dim=-1)
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-6, rtol=0.0)


def test_rada_head_dispersion_is_positive():
    support, query = _random_episode(batch_size=1, num_way=3, num_shot=5, num_query=4, feat_dim=7)
    head = _build_head(feat_dim=7)

    _, aux = head(support, query)

    assert torch.all(aux["disp"] > 0.0)


def test_rada_head_one_shot_prototype_and_dispersion_shrinkage():
    support = F.normalize(
        torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    query = F.normalize(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32), dim=-1)
    head = _build_head(feat_dim=4, gamma_disp=0.25, lambda_proto=0.7)

    logits, aux = head(support, query)

    expected_proto = support.squeeze(2).unsqueeze(1)
    expected_prior = F.softplus(head.global_dispersion_param).view(1, 1, 1, 4)
    expected_disp = (1.0 - head.gamma_disp) * expected_prior + head.eps

    assert logits.shape == (1, 1, 2)
    assert torch.allclose(aux["alpha"], torch.ones_like(aux["alpha"]), atol=1e-6, rtol=0.0)
    assert torch.allclose(aux["proto"], expected_proto, atol=1e-6, rtol=0.0)
    assert torch.allclose(aux["disp"], expected_disp.expand_as(aux["disp"]), atol=1e-5, rtol=0.0)


def test_rada_head_random_inputs_do_not_produce_nans():
    support, query = _random_episode(batch_size=3, num_way=5, num_shot=2, num_query=7, feat_dim=9)
    head = _build_head(feat_dim=9, reliability_head="mlp", reliability_hidden_dim=12)

    logits, aux = head(support, query)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux["alpha"]).all()
    assert torch.isfinite(aux["proto"]).all()
    assert torch.isfinite(aux["disp"]).all()


def test_rada_head_no_reliability_ablation_uses_uniform_alpha():
    support, query = _random_episode(batch_size=1, num_way=2, num_shot=5, num_query=3, feat_dim=4)
    head = _build_head(feat_dim=4, use_reliability=False)

    _, aux = head(support, query)

    expected = torch.full_like(aux["alpha"], 1.0 / 5.0)
    assert torch.allclose(aux["alpha"], expected, atol=1e-6, rtol=0.0)


def test_rada_model_factory_builds_and_returns_aux():
    args = SimpleNamespace(
        model="rada_fsl",
        device="cpu",
        image_size=32,
        fewshot_backbone="conv64f",
        rada_feat_dim=32,
        rada_tau_r=0.5,
        rada_lambda_proto=0.7,
        rada_gamma_disp=0.5,
        rada_eps=1e-6,
        rada_reliability_head="linear",
        rada_reliability_hidden_dim=16,
        rada_l2_normalize="true",
        rada_entropy_reg_weight=0.0,
        rada_use_reliability="true",
        rada_use_dispersion_metric="true",
        rada_query_conditioned="true",
        rada_use_residual_anchor="true",
        rada_use_shrinkage="true",
        rada_disp_clamp_max=0.0,
        rada_fsl_mamba_base_dim=16,
        rada_fsl_mamba_output_dim=64,
        rada_fsl_mamba_drop_path=0.0,
        rada_fsl_mamba_perturb_sigma=0.0,
    )
    model = build_model_from_args(args)
    assert isinstance(model, RADAFSL)
    model.eval()

    query = torch.randn(1, 2, 3, 32, 32)
    support = torch.randn(1, 3, 2, 3, 32, 32)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["episode_logits"].shape == (1, 2, 3)
    assert outputs["alpha"].shape == (1, 2, 3, 2)
    assert outputs["proto"].shape == (1, 2, 3, 32)
    assert outputs["disp"].shape == (1, 2, 3, 32)
    assert outputs["mu"].shape == (1, 3, 32)
    assert outputs["raw_scatter"].shape == (1, 3)
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["disp"]).all()
