from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from net.model_factory import build_model_from_args
from net.ours import OursFinalUCOT


def _tiny_ucot(**overrides) -> OursFinalUCOT:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=8,
        eam_hidden_dim=16,
        sinkhorn_iterations=3,
        sinkhorn_tolerance=1e-5,
        eval_use_float64=False,
        ecot_enable_egsm=False,
        ecot_m2_ablate_threshold_mass=False,
        ecot_m2_cost_per_mass_score=False,
        ecot_rho_bank="0.8",
        ecot_base_rho=0.8,
        fixed_mass=0.8,
        ecot_transport_mode="unbalanced",
        transport_cost_threshold_init=0.08,
        enable_ours_final_failure_probe=True,
    )
    kwargs.update(overrides)
    return OursFinalUCOT(**kwargs)


def test_ours_final_ucot_factory_builds_standalone_defaults():
    common = dict(
        device="cpu",
        image_size=64,
        fewshot_backbone="conv64f",
        hrot_token_dim=8,
        hrot_eam_hidden_dim=16,
        hrot_sinkhorn_iterations=3,
        hrot_sinkhorn_tolerance=1e-5,
    )

    model = build_model_from_args(
        SimpleNamespace(model="ours_final_ucot", ours_ablation="full", **common)
    )

    assert isinstance(model, OursFinalUCOT)
    assert model.enable_global_residual_score
    assert model.global_residual_weight == pytest.approx(0.1)
    assert model.tau_q == pytest.approx(0.5)
    assert model.tau_c == pytest.approx(0.8)
    assert model.enable_ucot_calibration
    assert model.enable_ours_final_failure_probe
    assert model.ours_final_marginal_mode == "uniform"


def test_ucot_threshold_calibration_is_label_free_and_raises_low_threshold():
    model = _tiny_ucot(ucot_threshold_quantile=0.70, ucot_threshold_mix=1.0)
    flat_cost = torch.full((2, 4, 3, 3), 0.7)
    flat_cost[:, :, :, 0] = 0.30

    threshold, payload = model._compute_ucot_threshold(flat_cost, way_num=2, shot_num=2)

    assert threshold.item() > model.transport_cost_threshold.item()
    assert threshold.item() == pytest.approx(0.30, abs=1e-5)
    assert payload["ucot/calibrated_threshold"].item() > payload["ucot/base_threshold"].item()


def test_ucot_quotas_conserve_probability():
    torch.manual_seed(10)
    model = _tiny_ucot()
    flat_cost = torch.rand(3, 4, 5, 6)
    threshold, _ = model._compute_ucot_threshold(flat_cost, way_num=2, shot_num=2)

    query_prob, support_prob, payload = model._compute_ucot_marginals(
        flat_cost,
        threshold,
        way_num=2,
        shot_num=2,
    )

    assert query_prob.shape == (3, 5)
    assert support_prob.shape == (3, 4, 6)
    assert torch.allclose(query_prob.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.allclose(support_prob.sum(dim=-1), torch.ones(3, 4), atol=1e-6)
    assert torch.all(query_prob >= 0.0)
    assert torch.all(support_prob >= 0.0)
    assert torch.isfinite(payload["ucot/query_entropy"])


def test_ucot_uniform_fallback_equal_costs_has_no_nan_and_stays_uniform():
    model = _tiny_ucot(ucot_uniform_floor=0.05)
    flat_cost = torch.full((2, 2, 4, 4), 0.35)
    threshold, _ = model._compute_ucot_threshold(flat_cost, way_num=2, shot_num=1)

    query_prob, support_prob, payload = model._compute_ucot_marginals(
        flat_cost,
        threshold,
        way_num=2,
        shot_num=1,
    )

    assert torch.isfinite(query_prob).all()
    assert torch.isfinite(support_prob).all()
    assert torch.allclose(query_prob, torch.full_like(query_prob, 0.25), atol=1e-5)
    assert torch.allclose(support_prob, torch.full_like(support_prob, 0.25), atol=1e-5)
    assert payload["ucot/query_l1_from_uniform"].item() == pytest.approx(0.0, abs=1e-5)


def test_ucot_threshold_override_reaches_score_and_changes_logits():
    torch.manual_seed(11)
    full = _tiny_ucot(ucot_ablation="full")
    off = _tiny_ucot(ucot_ablation="off")
    off.load_state_dict(full.state_dict())
    flat_cost = torch.full((2, 2, 4, 4), 0.65)
    flat_cost[:, 0, :, 0] = 0.30
    flat_cost[:, 1, :, 1] = 0.35

    full.eval()
    off.eval()
    with torch.no_grad():
        full_out = full._forward_ecot_budget_bank(flat_cost, way_num=2, shot_num=1)
        off_out = off._forward_ecot_budget_bank(flat_cost, way_num=2, shot_num=1)

    assert "ecot_threshold" in full_out
    assert full_out["ecot_threshold"].item() > full.transport_cost_threshold.item()
    assert not torch.allclose(full_out["logits"], off_out["logits"], atol=1e-6)


def test_ucot_diagnostics_surface_and_are_finite():
    torch.manual_seed(12)
    model = _tiny_ucot()
    flat_cost = torch.rand(2, 2, 4, 4)

    model.eval()
    with torch.no_grad():
        outputs = model._forward_ecot_budget_bank(flat_cost, way_num=2, shot_num=1)

    required = (
        "ucot/enabled",
        "ucot/ablation_id",
        "ucot/base_threshold",
        "ucot/calibrated_threshold",
        "ucot/threshold_ratio",
        "ucot/positive_edge_rate",
        "ucot/effective_mix",
        "ucot/query_l1_from_uniform",
        "ucot/support_l1_from_uniform",
        "ucot/transported_mass_fraction",
        "ucot/query_l1_drift",
        "ucot/support_l1_drift",
        "ours_probe/negative_utility_mass_ratio",
    )
    for key in required:
        assert key in outputs, f"missing {key}"
        assert torch.isfinite(outputs[key]).all(), f"non-finite {key}"
