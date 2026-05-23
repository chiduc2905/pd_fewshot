import torch

from visualization.noise_diagnostics import export_uot_evidence_figure


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
    assert rows[0]["all_class_match_path"] == ""


def test_export_uot_evidence_figure_paper_style_draws_region_uot_prior(tmp_path):
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


def test_export_uot_evidence_figure_paper_style_draws_adaptive_region_prior(tmp_path):
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
