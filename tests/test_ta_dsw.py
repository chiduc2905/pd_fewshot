from types import SimpleNamespace

import torch
import torch.nn.functional as F

from net.model_factory import build_model_from_args, get_model_choices
from net.ta_dsw import (
    TaskAdaptiveDistributionalSlicedWassersteinNet,
    sample_vmf,
    sliced_wasserstein_distance,
)


def _factory_args(model: str = "ta_dsw", **overrides):
    args = {
        "model": model,
        "device": "cpu",
        "image_size": 64,
        "fewshot_backbone": "conv64f",
        "ta_dsw_num_slices": 4,
        "ta_dsw_sw_p": 2.0,
        "ta_dsw_temperature": 0.1,
        "ta_dsw_token_dim": 32,
        "ta_dsw_hidden_dim": 16,
        "ta_dsw_token_weight_hidden_dim": 16,
        "ta_dsw_token_weight_uniform_mix": 0.2,
        "ta_dsw_num_quantiles": 8,
        "ta_dsw_normalize_tokens": "true",
        "ta_dsw_train_projection_mode": "deterministic",
        "ta_dsw_eval_projection_mode": "fixed",
        "ta_dsw_projection_seed": 7,
        "ta_dsw_shot_aggregation": "softmin",
        "ta_dsw_shot_softmin_beta": 5.0,
        "ta_dsw_proto_scale_init": 10.0,
        "ta_dsw_sw_scale_init": None,
        "ta_dsw_learnable_scales": "true",
        "ta_dsw_eps": 1e-8,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_ta_dsw_model_choices_include_canonical_and_hyphen_alias():
    choices = get_model_choices()
    assert "ta_dsw" in choices
    assert "ta-dsw" in choices


def test_sliced_wasserstein_identical_distribution_is_zero():
    torch.manual_seed(1)
    x = torch.randn(100, 64)
    thetas = F.normalize(torch.randn(64, 64), p=2, dim=1)
    distance = sliced_wasserstein_distance(x, x, thetas, p=2.0)
    assert distance < 1e-6


def test_sample_vmf_returns_unit_vectors_and_allows_mean_direction_gradient():
    torch.manual_seed(2)
    raw_mean = torch.randn(16, requires_grad=True)
    mean_dir = F.normalize(raw_mean, p=2, dim=0)
    samples = sample_vmf(mean_dir, torch.tensor(5.0), num_samples=8)

    assert samples.shape == (8, 16)
    assert torch.allclose(samples.norm(dim=1), torch.ones(8), atol=1e-5)

    loss = -(samples @ mean_dir).mean()
    loss.backward()
    assert raw_mean.grad is not None
    assert torch.isfinite(raw_mean.grad).all()


def test_ta_dsw_factory_builds_aliases_and_runs():
    for model_name in ("ta_dsw", "ta-dsw"):
        torch.manual_seed(3)
        model = build_model_from_args(_factory_args(model=model_name))
        model.eval()

        query = torch.randn(1, 2, 3, 64, 64)
        support = torch.randn(1, 3, 1, 3, 64, 64)
        with torch.no_grad():
            logits = model(query, support)

        assert logits.shape == (2, 3)
        assert torch.isfinite(logits).all()


def test_ta_dsw_training_output_and_gradient_flow():
    torch.manual_seed(4)
    model = TaskAdaptiveDistributionalSlicedWassersteinNet(
        in_channels=3,
        hidden_dim=64,
        backbone_name="conv64f",
        image_size=64,
        token_dim=32,
        num_slices=4,
        sw_p=2.0,
        temperature=0.1,
        task_hidden_dim=16,
        token_weight_hidden_dim=16,
        num_quantiles=8,
        normalize_tokens=True,
    )
    model.train()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)
    targets = torch.tensor([0, 1])

    outputs = model(query, support)
    assert isinstance(outputs, dict)
    assert outputs["logits"].shape == (2, 2)
    assert outputs["aux_loss"].ndim == 0

    loss = F.cross_entropy(outputs["logits"], targets) + outputs["aux_loss"]
    loss.backward()

    direction_grad = model.task_adaptive.direction_head.weight.grad
    assert direction_grad is not None
    assert torch.isfinite(direction_grad).all()

    backbone_grad = next(param.grad for name, param in model.named_parameters() if name.startswith("backbone."))
    assert backbone_grad is not None
    assert torch.isfinite(backbone_grad).all()


def test_ta_dsw_eval_is_deterministic_and_uses_weighted_measures():
    torch.manual_seed(5)
    model = build_model_from_args(_factory_args())
    model.eval()

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 3, 2, 3, 64, 64)
    with torch.no_grad():
        first = model(query, support, return_aux=True)
        second = model(query, support, return_aux=True)

    assert torch.allclose(first["logits"], second["logits"])
    assert first["query_token_weights"].shape[:2] == first["query_tokens"].shape[:2]
    assert first["support_token_weights"].shape == first["support_tokens"].shape[:3]
    assert first["ta_dsw_shot_distance"].shape == (2, 3, 2)
