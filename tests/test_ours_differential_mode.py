from __future__ import annotations

import csv
from types import SimpleNamespace

import torch

from net.jecot_m2 import JECOTM2
from net.model_factory import build_model_from_args
from net.ours import OursM2


def _manual_euclidean_cost(query_tokens: torch.Tensor, support_tokens: torch.Tensor) -> torch.Tensor:
    query_norm = query_tokens.pow(2).sum(dim=-1)
    support_norm = support_tokens.pow(2).sum(dim=-1)
    dot = torch.einsum("qtd,wkd->qwtk", query_tokens, support_tokens)
    return (query_norm[:, None, :, None] + support_norm[None, :, None, :] - 2.0 * dot).clamp_min(0.0)


def _tiny_ours(**overrides) -> OursM2:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=8,
        eam_hidden_dim=16,
        sinkhorn_iterations=4,
        sinkhorn_tolerance=1e-5,
        eval_use_float64=False,
    )
    kwargs.update(overrides)
    return OursM2(**kwargs)


def test_ours_differential_mode_off_matches_base_ground_cost():
    torch.manual_seed(411)
    model = _tiny_ours(use_differential_mode=False)
    model.eval()
    query = torch.randn(2, 5, model.token_dim)
    support = torch.randn(6, 5, model.token_dim)

    cost = model._ground_cost(query, support)
    expected = JECOTM2._ground_cost(model, query, support)

    assert torch.allclose(cost, expected, atol=0.0, rtol=0.0)


def test_ours_differential_mode_ground_cost_matches_manual_transform():
    torch.manual_seed(412)
    model = _tiny_ours(use_differential_mode=True, dm_alpha=0.0, dm_debug=False)
    model.eval()
    query = torch.randn(3, 4, model.token_dim)
    support = torch.randn(10, 4, model.token_dim)

    cost = model._ground_cost(query, support)

    common = support.mean(dim=0)
    expected = _manual_euclidean_cost(query - common.unsqueeze(0), support - common.unsqueeze(0))
    assert cost.shape == (3, 10, 4, 4)
    assert torch.allclose(cost, expected, atol=1e-6, rtol=1e-6)


def test_ours_differential_mode_updates_and_blends_global_template():
    torch.manual_seed(413)
    model = _tiny_ours(use_differential_mode=True, dm_alpha=0.25, dm_debug=False)
    query = torch.randn(2, 4, model.token_dim)
    support_train = torch.randn(6, 4, model.token_dim)
    support_eval = torch.randn(6, 4, model.token_dim)

    model.train()
    _ = model._ground_cost(query, support_train)
    train_template = support_train.mean(dim=0)

    assert torch.allclose(model.dm_global_template, train_template)
    assert model.dm_template_count.item() == 1.0

    model.eval()
    cost = model._ground_cost(query, support_eval)
    eval_template = support_eval.mean(dim=0)
    blended = 0.25 * train_template + 0.75 * eval_template
    expected = _manual_euclidean_cost(query - blended.unsqueeze(0), support_eval - blended.unsqueeze(0))

    assert model.dm_template_count.item() == 1.0
    assert torch.allclose(cost, expected, atol=1e-6, rtol=1e-6)


def test_ours_differential_mode_factory_and_forward_train_eval(tmp_path):
    torch.manual_seed(414)
    model = build_model_from_args(
        SimpleNamespace(
            model="ours",
            ours_ablation="full",
            use_differential_mode="true",
            dm_alpha=0.5,
            dm_debug_dir=str(tmp_path / "factory_debug"),
            dm_debug_max_episodes=2,
            device="cpu",
            image_size=64,
            fewshot_backbone="conv64f",
            hrot_token_dim=12,
            hrot_eam_hidden_dim=16,
            hrot_sinkhorn_iterations=4,
            hrot_sinkhorn_tolerance=1e-5,
        )
    )
    assert model.use_differential_mode
    assert model.dm_debug
    assert model.dm_alpha == 0.5
    assert all(not name.startswith("dm_") for name, _param in model.named_parameters())

    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)

    model.train()
    outputs = model(query, support, return_aux=True)
    loss = outputs["logits"].sum() + outputs["aux_loss"]
    loss.backward()

    assert outputs["logits"].shape == (2, 2)
    assert torch.isfinite(outputs["logits"]).all()
    assert "dm_mu_norm" in outputs
    assert "dm_residual_ratio" in outputs
    assert outputs["dm_mu_token_norm"].shape[-1] == outputs["query_euclidean_tokens"].shape[-2]
    assert model.dm_global_template.shape == outputs["query_euclidean_tokens"].shape[-2:]
    assert model.dm_template_count.item() > 0.0

    count_after_train = model.dm_template_count.item()
    model.eval()
    with torch.no_grad():
        eval_outputs = model(query, support, return_aux=True)

    assert eval_outputs["logits"].shape == (2, 2)
    assert torch.isfinite(eval_outputs["logits"]).all()
    assert model.dm_template_count.item() == count_after_train


def test_ours_differential_mode_debug_writes_metrics_and_heatmap(tmp_path):
    torch.manual_seed(415)
    model = _tiny_ours(
        use_differential_mode=True,
        dm_debug_dir=str(tmp_path),
        dm_debug_max_episodes=1,
    )
    model.eval()
    query = torch.randn(2, 4, model.token_dim)
    support = torch.randn(6, 4, model.token_dim)

    _ = model._ground_cost(query, support)
    _ = model._ground_cost(query, support)

    csv_path = tmp_path / "dmt_debug_metrics.csv"
    heatmap_path = tmp_path / "dmt_mu_heatmap_0001.png"
    assert csv_path.exists()
    assert heatmap_path.exists()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["dm_mu_norm"]) > 0.0
    assert float(rows[0]["dm_residual_ratio"]) > 0.0


def test_ours_differential_mode_debug_is_gated_by_dmt(tmp_path):
    torch.manual_seed(416)
    model = _tiny_ours(
        use_differential_mode=False,
        dm_debug=True,
        dm_debug_dir=str(tmp_path),
    )
    model.eval()
    query = torch.randn(2, 4, model.token_dim)
    support = torch.randn(6, 4, model.token_dim)

    _ = model._ground_cost(query, support)

    assert not (tmp_path / "dmt_debug_metrics.csv").exists()
    assert not list(tmp_path.glob("*.png"))


def test_ours_differential_mode_debug_can_be_disabled(tmp_path):
    torch.manual_seed(417)
    model = _tiny_ours(
        use_differential_mode=True,
        dm_debug=False,
        dm_debug_dir=str(tmp_path),
    )
    model.eval()
    query = torch.randn(2, 4, model.token_dim)
    support = torch.randn(6, 4, model.token_dim)

    _ = model._ground_cost(query, support)

    assert not (tmp_path / "dmt_debug_metrics.csv").exists()
    assert not list(tmp_path.glob("*.png"))


def test_differential_mode_factory_config_is_ours_only():
    hrot_model = build_model_from_args(
        SimpleNamespace(
            model="hrot_fsl",
            use_differential_mode="true",
            dm_alpha=0.5,
            device="cpu",
            image_size=64,
            fewshot_backbone="conv64f",
            hrot_variant="E",
            hrot_token_dim=12,
            hrot_eam_hidden_dim=16,
            hrot_sinkhorn_iterations=4,
            hrot_sinkhorn_tolerance=1e-5,
        )
    )

    assert not hasattr(hrot_model, "use_differential_mode")
