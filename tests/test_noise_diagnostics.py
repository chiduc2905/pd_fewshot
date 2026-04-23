from __future__ import annotations

import torch

from visualization.noise_diagnostics import (
    compute_scalogram_statistics,
    compute_support_episode_distribution,
    extract_focus_maps,
    infer_token_hw,
)


def test_infer_token_hw_prefers_near_square_factorization():
    assert infer_token_hw(36) == (6, 6)
    assert infer_token_hw(35) == (5, 7)


def test_extract_focus_maps_uses_competitive_assignment_when_available():
    query_images = torch.rand(2, 1, 8, 8)
    support_images = torch.rand(3, 2, 1, 8, 8)
    outputs = {
        "competitive_assignment": torch.tensor(
            [
                [
                    [0.8, 0.1, 0.1],
                    [0.6, 0.2, 0.2],
                    [0.2, 0.7, 0.1],
                    [0.1, 0.2, 0.7],
                ],
                [
                    [0.2, 0.7, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.3, 0.2, 0.5],
                    [0.2, 0.1, 0.7],
                ],
            ],
            dtype=torch.float32,
        )
    }

    payload = extract_focus_maps(model=torch.nn.Identity(), outputs=outputs, query_images=query_images, support_images=support_images)

    assert payload["source"] == "competitive_assignment"
    assert payload["class_specific"] is True
    assert payload["class_maps"].shape == (2, 3, 8, 8)
    assert torch.isclose(torch.tensor(payload["class_maps"][0, 0].sum()), torch.tensor(1.0), atol=1e-4)


def test_compute_scalogram_statistics_noise_proxy_distinguishes_dispersed_images():
    focused = torch.zeros(2, 1, 8, 8)
    focused[0, 0, 3:5, 3:5] = 1.0
    focused[1, 0] = 1.0
    labels = torch.tensor([0, 0], dtype=torch.long)

    rows = compute_scalogram_statistics(focused, labels, split_name="train", class_names=["A"])

    assert rows[0]["top20_ratio"] > rows[1]["top20_ratio"]
    assert rows[1]["noise_proxy"] > rows[0]["noise_proxy"]


def test_compute_support_episode_distribution_flags_outlier_shot():
    support_images = torch.zeros(1, 3, 1, 8, 8)
    support_images[0, 0, 0, 2:5, 2:5] = 1.0
    support_images[0, 1, 0, 2:5, 2:5] = 1.0
    support_images[0, 2, 0, 0:2, 0:2] = 1.0
    outputs = {
        "shot_rho": torch.tensor([[[0.8, 0.75, 0.2]]], dtype=torch.float32),
    }

    rows, summary = compute_support_episode_distribution(
        support_images,
        class_names=["PD"],
        episode_index=5,
        outputs=outputs,
    )

    assert len(rows) == 3
    shot0 = next(row for row in rows if row["shot_index"] == 0)
    shot2 = next(row for row in rows if row["shot_index"] == 2)
    assert shot2["outlier_score"] > shot0["outlier_score"]
    assert shot2["shot_rho"] < shot0["shot_rho"]
    assert summary["support_outlier_score_max"] >= shot2["outlier_score"]
