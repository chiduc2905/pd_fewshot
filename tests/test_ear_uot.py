from __future__ import annotations

import torch

from net.modules.evidence_adaptive_relaxation_uot import (
    EvidenceAdaptiveRelaxationUOT,
)
from net.modules.unbalanced_ot import (
    sinkhorn_unbalanced_log,
    sinkhorn_unbalanced_log_adaptive,
)
from net.ours import OursM2
from run_all_experiments import (
    build_ours_final_adaptive_relaxation_variants,
    filter_ours_final_variants,
    parse_ours_final_variant_filter,
)


def _tiny_ours_final(**overrides) -> OursM2:
    kwargs = dict(
        backbone_name="conv64f",
        image_size=64,
        hidden_dim=64,
        token_dim=8,
        eam_hidden_dim=16,
        sinkhorn_iterations=8,
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


def test_adaptive_uot_matches_scalar_uot_for_constant_tau():
    torch.manual_seed(6101)
    cost = torch.rand(3, 4, 5)
    a = torch.full((3, 4), 0.8 / 4.0)
    b = torch.full((3, 5), 0.8 / 5.0)
    tau_q = torch.full_like(a, 0.5)
    tau_c = torch.full_like(b, 0.5)

    expected = sinkhorn_unbalanced_log(
        cost,
        a,
        b,
        tau_q=0.5,
        tau_c=0.5,
        eps=0.1,
        max_iter=200,
        tol=1e-8,
    )
    actual = sinkhorn_unbalanced_log_adaptive(
        cost,
        a,
        b,
        tau_q=tau_q,
        tau_c=tau_c,
        eps=0.1,
        max_iter=200,
        tol=1e-8,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_adaptive_uot_preserves_cost_gradients():
    torch.manual_seed(6103)
    cost = torch.rand(2, 4, 5, requires_grad=True)
    a = torch.full((2, 4), 0.8 / 4.0)
    b = torch.full((2, 5), 0.8 / 5.0)
    tau_q = torch.rand_like(a).mul(0.95).add(0.05)
    tau_c = torch.rand_like(b).mul(0.95).add(0.05)

    plan = sinkhorn_unbalanced_log_adaptive(
        cost,
        a,
        b,
        tau_q=tau_q,
        tau_c=tau_c,
        eps=0.1,
        max_iter=40,
        tol=0.0,
    )
    (plan * cost).sum().backward()

    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()
    assert cost.grad.abs().sum() > 0.0


def test_ear_uot_assigns_higher_relaxation_to_specific_matchable_tokens():
    cost = torch.tensor(
        [
            [
                [
                    [0.05, 0.10, 0.90, 0.90],
                    [0.08, 0.06, 0.90, 0.90],
                    [0.80, 0.85, 0.90, 0.90],
                    [0.85, 0.80, 0.90, 0.90],
                ],
                [
                    [0.65, 0.70, 0.80, 0.85],
                    [0.70, 0.65, 0.85, 0.80],
                    [0.75, 0.80, 0.70, 0.75],
                    [0.80, 0.75, 0.75, 0.70],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    module = EvidenceAdaptiveRelaxationUOT(
        tau_min=0.05,
        tau_max=1.0,
        temperature=0.25,
        spatial_mix=0.5,
        kernel_size=3,
    )

    tau_q, tau_c, diagnostics = module(
        cost,
        threshold=0.5,
        way_num=2,
        shot_num=1,
        spatial_hw=(2, 2),
    )

    assert tau_q.shape == (1, 2, 4)
    assert tau_c.shape == (1, 2, 4)
    assert torch.isfinite(tau_q).all()
    assert torch.isfinite(tau_c).all()
    assert tau_q[0, 0, :2].mean() > tau_q[0, 1].mean()
    assert diagnostics["ear_uot/positive_rival_advantage_share"].item() > 0.0


def test_ear_uot_model_forward_exposes_plan_level_diagnostics():
    torch.manual_seed(6102)
    model = _tiny_ours_final(
        enable_evidence_adaptive_relaxation_uot=True,
        ear_uot_tau_min=0.05,
        ear_uot_tau_max=1.0,
        ear_uot_temperature=0.25,
        ear_uot_spatial_mix=0.5,
    )
    model.eval()
    query = torch.randn(1, 2, 3, 64, 64)
    support = torch.randn(1, 2, 1, 3, 64, 64)

    with torch.no_grad():
        outputs = model(query, support, return_aux=True)

    assert outputs["logits"].shape == (2, 2)
    assert outputs["transport_plan"].shape[:3] == (2, 2, 1)
    assert outputs["ear_uot/enabled"].item() == 1.0
    assert outputs["ear_uot_reliable_mass"].shape == (2, 2)
    assert outputs["ear_uot_low_reliability_destroyed_fraction"].shape == (2, 2)
    assert torch.isfinite(outputs["transport_plan"]).all()
    assert torch.isfinite(outputs["ear_uot/mass_reliability_correlation"])


def test_ear_uot_scalar_control_matches_original_uot_model():
    for score_mode in ("threshold_mass", "uot_energy"):
        torch.manual_seed(6104)
        baseline = _tiny_ours_final(ours_final_score_mode=score_mode)
        scalar_control = _tiny_ours_final(
            ours_final_score_mode=score_mode,
            enable_evidence_adaptive_relaxation_uot=True,
            ear_uot_tau_min=0.5,
            ear_uot_tau_max=0.5,
            ear_uot_spatial_mix=0.0,
        )
        scalar_control.load_state_dict(baseline.state_dict(), strict=False)
        baseline.eval()
        scalar_control.eval()
        query = torch.randn(1, 2, 3, 64, 64)
        support = torch.randn(1, 2, 1, 3, 64, 64)

        with torch.no_grad():
            baseline_outputs = baseline(query, support, return_aux=True)
            control_outputs = scalar_control(query, support, return_aux=True)

        assert torch.equal(baseline_outputs["logits"], control_outputs["logits"])
        assert torch.equal(
            baseline_outputs["transport_plan"],
            control_outputs["transport_plan"],
        )
        assert torch.equal(
            baseline_outputs["ecot_uot_classification_energy_bank"],
            control_outputs["ecot_uot_classification_energy_bank"],
        )


def test_ear_uot_ablation_suite_has_baseline_and_mechanism_controls():
    variants = build_ours_final_adaptive_relaxation_variants()
    assert [variant["tag"] for variant in variants] == [
        "ours_final_ear_baseline",
        "ours_final_ear_scalar_control",
        "ours_final_ear_no_spatial",
        "ours_final_ear_uot",
    ]
    selected = filter_ours_final_variants(
        variants,
        parse_ours_final_variant_filter("ear_uot,ear_uot_scalar_control"),
    )
    assert [variant["tag"] for variant in selected] == [
        "ours_final_ear_scalar_control",
        "ours_final_ear_uot",
    ]
