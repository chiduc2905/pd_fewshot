from __future__ import annotations

import torch

from main import get_args, infer_hrot_arch_overrides_from_state_dict, infer_hrot_variant_from_state_dict


def test_hrot_sinkhorn_iters_cli_alias(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--model", "hrot_fsl", "--hrot_sinkhorn_iters", "7"],
    )

    args = get_args()

    assert args.hrot_sinkhorn_iterations == 7


def test_infer_hrot_variant_detects_variant_r_from_noise_calibrated_state():
    state_dict = {
        "raw_transport_cost_threshold": torch.tensor(0.0),
        "raw_structure_cost_weight": torch.tensor(0.0),
        "raw_token_temperature": torch.tensor(0.0),
        "raw_q_enhancement_mix": torch.tensor(0.0),
        "query_reliability_weights": torch.randn(4),
        "support_reliability_weights": torch.randn(4),
        "query_token_attention_vector": torch.randn(128),
        "support_token_attention_vector": torch.randn(128),
        "q_eam.network.0.weight": torch.randn(256, 5),
        "q_eam.network.0.bias": torch.randn(256),
        "eam.network.0.weight": torch.randn(256, 4),
        "eam.network.0.bias": torch.randn(256),
        "token_projector.1.weight": torch.randn(128, 640),
    }

    assert infer_hrot_variant_from_state_dict(state_dict) == "R"


def test_infer_hrot_arch_overrides_recovers_variant_and_token_shape():
    state_dict = {
        "raw_transport_cost_threshold": torch.tensor(0.0),
        "raw_structure_cost_weight": torch.tensor(0.0),
        "raw_token_temperature": torch.tensor(0.0),
        "raw_q_enhancement_mix": torch.tensor(0.0),
        "query_reliability_weights": torch.randn(4),
        "support_reliability_weights": torch.randn(4),
        "query_token_attention_vector": torch.randn(96),
        "support_token_attention_vector": torch.randn(96),
        "q_eam.network.0.weight": torch.randn(192, 5),
        "q_eam.network.0.bias": torch.randn(192),
        "eam.network.0.weight": torch.randn(192, 4),
        "eam.network.0.bias": torch.randn(192),
        "token_projector.1.weight": torch.randn(96, 640),
    }

    overrides = infer_hrot_arch_overrides_from_state_dict(state_dict)

    assert overrides["hrot_variant"] == "R"
    assert overrides["hrot_token_dim"] == 96
    assert overrides["hrot_eam_hidden_dim"] == 192
    assert overrides["hrot_use_raw_backbone_tokens"] == "false"


def test_infer_hrot_arch_overrides_prefers_checkpoint_args_when_present():
    state_dict = {
        "eam.network.0.weight": torch.randn(256, 259),
        "eam.network.0.bias": torch.randn(256),
    }
    checkpoint_args = {
        "hrot_variant": "Q",
        "hrot_token_dim": 128,
        "hrot_eam_hidden_dim": 320,
        "hrot_use_raw_backbone_tokens": "true",
    }

    overrides = infer_hrot_arch_overrides_from_state_dict(state_dict, checkpoint_args=checkpoint_args)

    assert overrides["hrot_variant"] == "Q"
    assert overrides["hrot_token_dim"] == 128
    assert overrides["hrot_eam_hidden_dim"] == 320
    assert overrides["hrot_use_raw_backbone_tokens"] == "true"
