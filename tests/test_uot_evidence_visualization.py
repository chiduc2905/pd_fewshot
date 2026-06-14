import torch

from visualization.nr_ot_figure import export_nr_ot_debiasing_figure
from visualization.noise_diagnostics import (
    _contrast_transport_image,
    _transport_plan_by_shot,
    export_uot_evidence_figure,
)


def test_transport_contrast_suppresses_uniform_entropy_floor():
    uniform = torch.full((8, 8), 0.1)
    contrasted_uniform = _contrast_transport_image(uniform)

    sparse = torch.zeros(8, 8)
    sparse[2, 3] = 0.1
    sparse[5, 6] = 0.05
    contrasted_sparse = _contrast_transport_image(sparse)

    assert float(contrasted_uniform.max()) == 0.0
    assert contrasted_sparse[2, 3] > contrasted_sparse.mean()
    assert contrasted_sparse[5, 6] > 0.0


def test_transport_plan_by_shot_prefers_rvuot_evidence_plan_for_visualization():
    score_plan = torch.zeros(1, 2, 1, 4, 4)
    evidence_plan = torch.zeros_like(score_plan)
    score_plan[0, 0, 0, 0, 0] = 1.0
    evidence_plan[0, 0, 0, 1, 1] = 1.0

    selected = _transport_plan_by_shot(
        {
            "transport_plan": score_plan,
            "rvuot_evidence_transport_plan": evidence_plan,
        },
        way_num=2,
        shot_num=1,
    )

    assert selected is not None
    assert selected[0, 0, 0, 1, 1] == 1.0
    assert selected[0, 0, 0, 0, 0] == 0.0


def test_export_uot_evidence_figure_draws_top_transport_correspondences(tmp_path):
    query_images = torch.rand(2, 1, 16, 16)
    support_images = torch.rand(2, 1, 1, 16, 16)
    plan = torch.zeros(2, 2, 1, 4, 4)
    plan[0, 0, 0, 0, 0] = 0.30
    plan[0, 0, 0, 1, 2] = 0.18
    plan[0, 0, 0, 3, 3] = 0.12
    plan[1, 1, 0, 2, 1] = 0.25
    outputs = {
        "transport_plan": plan,
        "shot_transported_mass": plan.sum(dim=(-1, -2)),
        "shot_rho": torch.ones(2, 2, 1),
    }
    save_path = tmp_path / "uot_evidence.png"

    rows = export_uot_evidence_figure(
        outputs=outputs,
        query_images=query_images,
        support_images=support_images,
        logits=torch.tensor([[2.0, 0.1], [0.2, 1.5]]),
        preds=torch.tensor([0, 1]),
        targets=torch.tensor([0, 1]),
        class_names=["PD-A", "PD-B"],
        save_path=str(save_path),
        episode_index=3,
        query_indices=[0],
    )

    assert save_path.exists()
    assert (tmp_path / "uot_evidence_all_classes.png").exists()
    assert save_path.stat().st_size > 0
    assert len(rows) == 1
    assert rows[0]["top_match_count"] == 3
    assert 0.0 < rows[0]["top_match_mass_fraction"] <= 1.0
    assert rows[0]["max_match_mass_fraction"] > 0.0
    assert "pulse_to_pulse_mass_ratio" in rows[0]
    assert 0.0 <= rows[0]["pulse_to_pulse_mass_ratio"] <= 1.0


def test_export_uot_evidence_figure_prefers_positive_threshold_evidence(tmp_path):
    query_images = torch.rand(1, 1, 16, 16)
    support_images = torch.rand(1, 1, 1, 16, 16)
    plan = torch.zeros(1, 1, 1, 4, 4)
    plan[0, 0, 0, 0, 0] = 0.30
    plan[0, 0, 0, 1, 2] = 0.18
    plan[0, 0, 0, 3, 3] = 0.12
    cost = torch.ones_like(plan) * 0.20
    cost[0, 0, 0, 0, 0] = 0.02
    cost[0, 0, 0, 3, 3] = 0.05
    outputs = {
        "transport_plan": plan,
        "cost_matrix": cost,
        "transport_cost_threshold": torch.tensor(0.10),
        "shot_transported_mass": plan.sum(dim=(-1, -2)),
        "shot_rho": torch.ones(1, 1, 1),
    }
    save_path = tmp_path / "uot_positive_evidence.png"

    rows = export_uot_evidence_figure(
        outputs=outputs,
        query_images=query_images,
        support_images=support_images,
        logits=torch.tensor([[1.0]]),
        preds=torch.tensor([0]),
        targets=torch.tensor([0]),
        class_names=["PD-A"],
        save_path=str(save_path),
        episode_index=4,
        query_indices=[0],
    )

    assert save_path.exists()
    assert (tmp_path / "uot_positive_evidence_all_classes.png").exists()
    assert rows[0]["evidence_map_source"] == "positive_evidence"
    assert abs(rows[0]["positive_evidence_total"] - 0.03) < 1e-6
    assert 0.0 < rows[0]["top_match_score_fraction"] <= 1.0
    assert rows[0]["positive_pulse_to_pulse_ratio"] is not None


def test_export_uot_evidence_figure_nr_ot_uses_nr_payload(tmp_path):
    query_images = torch.rand(1, 1, 16, 16)
    support_images = torch.rand(2, 1, 1, 16, 16)
    base_plan = torch.zeros(1, 2, 1, 4, 4)
    nr_plan = torch.zeros_like(base_plan)
    nr_plan[0, 0, 0, 1, 2] = 0.30
    nr_plan[0, 0, 0, 2, 3] = 0.15
    nr_cost = torch.ones_like(nr_plan) * 0.20
    nr_cost[0, 0, 0, 1, 2] = 0.02
    nr_cost[0, 0, 0, 2, 3] = 0.04
    outputs = {
        "transport_plan": base_plan,
        "nr_ot_class_transport_plan": nr_plan,
        "nr_ot_class_cost_matrix": nr_cost,
        "nr_ot_transport_cost_threshold": torch.tensor(0.10),
        "nr_ot_class_shot_transported_mass": nr_plan.sum(dim=(-1, -2)),
        "nr_ot_class_shot_expected_mass": torch.ones(1, 2, 1),
        "nr_ot_class_evidence": torch.tensor([[0.5, 0.1]]),
        "nr_ot_ref_evidence": torch.tensor([[0.2, 0.1]]),
        "nr_ot_debias_gap": torch.tensor([[0.3, 0.0]]),
    }
    save_path = tmp_path / "nr_ot_evidence.png"

    rows = export_uot_evidence_figure(
        outputs=outputs,
        query_images=query_images,
        support_images=support_images,
        logits=torch.tensor([[1.0, 0.2]]),
        preds=torch.tensor([0]),
        targets=torch.tensor([0]),
        class_names=["PD-A", "PD-B"],
        save_path=str(save_path),
        episode_index=8,
        query_indices=[0],
        transport_kind="nr_ot",
    )

    assert save_path.exists()
    assert len(rows) == 1
    assert rows[0]["transport_kind"] == "nr_ot"
    assert rows[0]["top_match_count"] == 2
    assert rows[0]["evidence_map_source"] == "positive_evidence"
    assert abs(rows[0]["positive_evidence_total"] - 0.033) < 1e-6
    assert abs(rows[0]["nr_ot_class_evidence"] - 0.5) < 1e-6
    assert abs(rows[0]["nr_ot_ref_evidence"] - 0.2) < 1e-6
    assert abs(rows[0]["nr_ot_debias_gap"] - 0.3) < 1e-6


def test_export_nr_ot_debiasing_figure_handles_non_square_tokens(tmp_path):
    outputs = {
        "nr_ot_query_marginal": torch.rand(1, 2, 20),
        "nr_ot_class_transport_plan": torch.rand(1, 2, 1, 20, 20),
        "nr_ot_ref_transport_plan": torch.rand(1, 2, 20, 40),
        "nr_ot_class_evidence": torch.tensor([[0.7, 0.2]]),
        "nr_ot_ref_evidence": torch.tensor([[0.3, 0.1]]),
        "nr_ot_debias_gap": torch.tensor([[0.4, 0.1]]),
    }
    save_path = tmp_path / "nested" / "nr_ot_mechanism_source.png"

    rows = export_nr_ot_debiasing_figure(
        outputs=outputs,
        query_images=torch.rand(1, 1, 16, 16),
        preds=torch.tensor([0]),
        targets=torch.tensor([0]),
        save_path=str(save_path),
        query_indices=[0, 99],
        class_names=["PD-A", "PD-B"],
        file_format="png",
    )

    expected_path = tmp_path / "nested" / "nr_ot_mechanism_source_nr_ot_mechanism.png"
    assert expected_path.exists()
    assert len(rows) == 1
    assert rows[0]["pred_class_name"] == "PD-A"
    assert abs(rows[0]["nr_ot_debias_gap"] - 0.4) < 1e-6
    assert rows[0]["paths"] == [str(expected_path)]


def test_export_nr_ot_debiasing_figure_requires_nr_payload(tmp_path):
    rows = export_nr_ot_debiasing_figure(
        outputs={"transport_plan": torch.zeros(1, 1, 1, 4, 4)},
        query_images=torch.rand(1, 1, 16, 16),
        preds=torch.tensor([0]),
        targets=torch.tensor([0]),
        save_path=str(tmp_path / "missing.png"),
        query_indices=[0],
    )

    assert rows == []


def test_export_uot_evidence_figure_paper_style_is_compact(tmp_path):
    query_images = torch.rand(1, 1, 16, 16)
    support_images = torch.rand(2, 1, 1, 16, 16)
    plan = torch.zeros(1, 2, 1, 4, 4)
    plan[0, 0, 0, 0, 0] = 0.30
    plan[0, 0, 0, 1, 1] = 0.22
    plan[0, 0, 0, 2, 2] = 0.16
    plan[0, 0, 0, 3, 3] = 0.10
    plan[0, 1, 0, 0, 3] = 0.12
    plan[0, 1, 0, 2, 1] = 0.08
    outputs = {
        "transport_plan": plan,
        "shot_transported_mass": plan.sum(dim=(-1, -2)),
        "shot_rho": torch.ones(1, 2, 1),
    }
    save_path = tmp_path / "uot_paper.png"

    rows = export_uot_evidence_figure(
        outputs=outputs,
        query_images=query_images,
        support_images=support_images,
        logits=torch.tensor([[2.0, 0.5]]),
        preds=torch.tensor([0]),
        targets=torch.tensor([0]),
        class_names=["PD-A", "PD-B"],
        save_path=str(save_path),
        episode_index=5,
        query_indices=[0],
        visual_style="paper",
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0
    assert len(rows) == 1
    assert 0 < rows[0]["top_match_count"] <= 6
    assert (tmp_path / "uot_paper_mass_overlay.png").exists()
    assert (tmp_path / "uot_paper_transport_matrix.png").exists()
    assert rows[0]["mass_overlay_path"].endswith("uot_paper_mass_overlay.png")
    assert rows[0]["transport_matrix_path"].endswith("uot_paper_transport_matrix.png")
    assert rows[0]["all_class_match_path"] == ""


def test_export_uot_evidence_figure_paper_style_full_ot_uses_mass_not_threshold(tmp_path):
    query_images = torch.rand(1, 1, 16, 16)
    support_images = torch.rand(1, 1, 1, 16, 16)
    plan = torch.full((1, 1, 1, 4, 4), 1.0 / 16.0)
    cost = torch.full_like(plan, 0.05)
    outputs = {
        "transport_plan": plan,
        "cost_matrix": cost,
        "transport_cost_threshold": torch.tensor(0.10),
        "shot_transported_mass": plan.sum(dim=(-1, -2)),
        "shot_rho": torch.ones(1, 1, 1),
    }
    save_path = tmp_path / "full_ot_paper.png"

    rows = export_uot_evidence_figure(
        outputs=outputs,
        query_images=query_images,
        support_images=support_images,
        logits=torch.tensor([[1.0]]),
        preds=torch.tensor([0]),
        targets=torch.tensor([0]),
        class_names=["PD-A"],
        save_path=str(save_path),
        episode_index=6,
        query_indices=[0],
        transport_kind="balanced_ot",
        visual_style="paper",
    )

    assert save_path.exists()
    assert (tmp_path / "full_ot_paper_mass_overlay.png").exists()
    assert (tmp_path / "full_ot_paper_transport_matrix.png").exists()
    assert rows[0]["evidence_map_source"] == "transport_mass"
    assert rows[0]["mass_overlay_path"].endswith("full_ot_paper_mass_overlay.png")
    assert rows[0]["transport_matrix_path"].endswith("full_ot_paper_transport_matrix.png")
    assert rows[0]["top_match_count"] > 0


def test_export_uot_evidence_figure_paper_style_handles_region_uot_payload(tmp_path):
    query_images = torch.rand(1, 1, 16, 16)
    support_images = torch.rand(2, 1, 1, 16, 16)
    fine_plan = torch.zeros(1, 2, 1, 4, 4)
    fine_plan[0, 0, 0, 0, 0] = 0.30
    fine_plan[0, 0, 0, 1, 1] = 0.22
    fine_plan[0, 0, 0, 2, 2] = 0.16
    fine_plan[0, 1, 0, 0, 3] = 0.12
    region_plan = torch.zeros(1, 2, 1, 9, 9)
    region_plan[0, 0, 0, 0, 0] = 0.25
    region_plan[0, 0, 0, 1, 1] = 0.20
    region_plan[0, 0, 0, 4, 4] = 0.18
    region_plan[0, 1, 0, 2, 6] = 0.13
    outputs = {
        "transport_plan": fine_plan,
        "region_uot_coarse_plan": region_plan,
        "region_uot_sparse_coarse_plan": region_plan,
        "shot_transported_mass": fine_plan.sum(dim=(-1, -2)),
        "shot_rho": torch.ones(1, 2, 1),
    }
    save_path = tmp_path / "uot_region_paper.png"

    rows = export_uot_evidence_figure(
        outputs=outputs,
        query_images=query_images,
        support_images=support_images,
        logits=torch.tensor([[2.0, 0.5]]),
        preds=torch.tensor([0]),
        targets=torch.tensor([0]),
        class_names=["PD-A", "PD-B"],
        save_path=str(save_path),
        episode_index=6,
        query_indices=[0],
        visual_style="paper",
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0
    assert len(rows) == 1
    assert rows[0]["top_match_count"] == 3


def test_export_uot_evidence_figure_paper_style_handles_adaptive_region_payload(tmp_path):
    query_images = torch.rand(1, 1, 16, 16)
    support_images = torch.rand(2, 1, 1, 16, 16)
    fine_plan = torch.zeros(1, 2, 1, 4, 4)
    fine_plan[0, 0, 0, 0, 0] = 0.30
    fine_plan[0, 0, 0, 1, 1] = 0.22
    fine_plan[0, 0, 0, 2, 2] = 0.16
    fine_plan[0, 1, 0, 0, 3] = 0.12
    region_plan = torch.zeros(1, 2, 1, 3, 3)
    region_plan[0, 0, 0, 0, 0] = 0.25
    region_plan[0, 0, 0, 1, 2] = 0.20
    region_plan[0, 1, 0, 2, 1] = 0.13
    query_masks = torch.tensor(
        [
            [
                [0.70, 0.20, 0.05, 0.05],
                [0.05, 0.70, 0.20, 0.05],
                [0.05, 0.05, 0.20, 0.70],
            ]
        ],
        dtype=torch.float32,
    )
    support_masks = query_masks.repeat(2, 1, 1, 1)
    outputs = {
        "transport_plan": fine_plan,
        "adaptive_region_plan": region_plan,
        "adaptive_region_query_masks": query_masks,
        "adaptive_region_support_masks": support_masks,
        "shot_transported_mass": fine_plan.sum(dim=(-1, -2)),
        "shot_rho": torch.ones(1, 2, 1),
    }
    save_path = tmp_path / "uot_adaptive_region_paper.png"

    rows = export_uot_evidence_figure(
        outputs=outputs,
        query_images=query_images,
        support_images=support_images,
        logits=torch.tensor([[2.0, 0.5]]),
        preds=torch.tensor([0]),
        targets=torch.tensor([0]),
        class_names=["PD-A", "PD-B"],
        save_path=str(save_path),
        episode_index=7,
        query_indices=[0],
        visual_style="paper",
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0
    assert len(rows) == 1
    assert rows[0]["top_match_count"] == 3
