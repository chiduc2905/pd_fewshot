from __future__ import annotations

from unittest.mock import patch

import torch

from visualization.noise_diagnostics import (
    compute_scalogram_statistics,
    compute_support_episode_distribution,
    export_dataset_noise_profile,
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


def test_export_dataset_noise_profile_handles_empty_splits():
    train_images = torch.rand(4, 1, 8, 8)
    train_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    empty_images = torch.empty(0, 1, 8, 8)
    empty_labels = torch.empty(0, dtype=torch.long)
    with patch("visualization.noise_diagnostics._write_csv") as write_csv_mock, patch(
        "visualization.noise_diagnostics._plot_dataset_profile"
    ) as plot_mock:
        payload = export_dataset_noise_profile(
            {
                "train": (train_images, train_labels, None),
                "val": (empty_images, empty_labels, None),
                "test_clean": (empty_images, empty_labels, None),
            },
            class_names=["A", "B"],
            save_dir="ignored_dir",
            run_stem="robust_profile",
        )

        assert payload["metrics"]["train_count"] == 4.0
        assert "val_count" not in payload["metrics"]
        assert write_csv_mock.call_count == 2
        assert plot_mock.call_count == 1
