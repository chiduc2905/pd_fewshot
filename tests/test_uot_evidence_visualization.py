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
