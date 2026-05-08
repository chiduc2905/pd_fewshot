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


def test_hrot_ground_cost_cli_accepts_cosine(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--model", "hrot_fsl", "--hrot_ground_cost", "cosine"],
    )

    args = get_args()

    assert args.hrot_ground_cost == "cosine"


def test_hrot_eam_mode_cli_accepts_legacy(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--model", "hrot_fsl", "--hrot_eam_mode", "legacy"],
    )

    args = get_args()

    assert args.hrot_eam_mode == "legacy"


def test_hrot_variant_cli_accepts_j_egtw(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--model",
            "hrot_fsl",
            "--hrot_variant",
            "J_EGTW",
            "--hrot_egtw_tau",
            "0.7",
            "--hrot_egtw_lambda",
            "1.3",
            "--hrot_egtw_attention_temperature",
            "0.2",
            "--hrot_egtw_uniform_mix",
            "0.1",
        ],
    )

    args = get_args()

    assert args.hrot_variant == "J_EGTW"
    assert args.hrot_egtw_tau == 0.7
    assert args.hrot_egtw_lambda == 1.3
    assert args.hrot_egtw_attention_temperature == 0.2
    assert args.hrot_egtw_uniform_mix == 0.1


def test_hrot_variant_cli_accepts_j_ecot(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--model",
            "hrot_fsl",
            "--hrot_variant",
            "J_ECOT",
            "--hrot_ecot_rho_bank",
            "0.45,0.80",
            "--hrot_ecot_lambda_init",
            "-12.0",
            "--hrot_ecot_controller_hidden",
            "16",
        ],
    )

    args = get_args()

    assert args.hrot_variant == "J_ECOT"
    assert args.hrot_ecot_rho_bank == "0.45,0.80"
    assert args.hrot_ecot_lambda_init == -12.0
    assert args.hrot_ecot_controller_hidden == 16


def test_hrot_variant_cli_accepts_j_ecot_m2(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--model",
            "hrot_fsl",
            "--hrot_variant",
            "J_ECOT_M2",
        ],
    )

    args = get_args()

    assert args.hrot_variant == "J_ECOT_M2"
    assert args.hrot_ecot_rho_bank is None
    assert args.hrot_ecot_base_rho is None


def test_hrot_variant_cli_accepts_j_ecot_care(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--model",
            "hrot_fsl",
            "--hrot_variant",
            "J_ECOT_CARE",
            "--care_enable_fwec",
            "false",
            "--care_mdr_lambda",
            "0.2",
        ],
    )

    args = get_args()

    assert args.hrot_variant == "J_ECOT_CARE"
    assert args.care_enable_fwec == "false"
    assert args.care_mdr_lambda == 0.2


def test_hrot_variant_cli_accepts_cp_ecot(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--model",
            "hrot_fsl",
            "--hrot_variant",
            "CP_ECOT",
            "--hrot_ecot_consensus_tau_mode",
            "sqrt",
            "--hrot_ecot_consensus_tau",
            "1.2",
        ],
    )

    args = get_args()

    assert args.hrot_variant == "CP_ECOT"
    assert args.hrot_ecot_rho_bank is None
    assert args.hrot_ecot_consensus_tau_mode == "sqrt"
    assert args.hrot_ecot_consensus_tau == 1.2


def test_hrot_variant_cli_accepts_j_ncet(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--model",
            "hrot_fsl",
            "--hrot_variant",
            "J_NCET",
            "--hrot_ncet_mix_init",
            "0.35",
            "--hrot_ncet_real_penalty_init",
            "0.4",
            "--hrot_ncet_null_penalty_init",
            "0.08",
            "--hrot_ncet_sink_cost_init",
            "1.2",
        ],
    )

    args = get_args()

    assert args.hrot_variant == "J_NCET"
    assert args.hrot_ncet_mix_init == 0.35
    assert args.hrot_ncet_real_penalty_init == 0.4
    assert args.hrot_ncet_null_penalty_init == 0.08
    assert args.hrot_ncet_sink_cost_init == 1.2


def test_infer_hrot_variant_detects_j_ncet_state():
    state_dict = {
        "raw_transport_cost_threshold": torch.tensor(0.0),
        "raw_ncet_mix": torch.tensor(0.0),
        "raw_ncet_real_penalty": torch.tensor(0.0),
        "raw_ncet_null_penalty": torch.tensor(0.0),
        "raw_noise_sink_cost": torch.tensor(0.0),
    }

    assert infer_hrot_variant_from_state_dict(state_dict) == "J_NCET"


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


def test_infer_hrot_variant_normalizes_j_egtw_checkpoint_args():
    state_dict = {
        "raw_transport_cost_threshold": torch.tensor(0.0),
        "eam.network.0.weight": torch.randn(256, 259),
        "eam.network.0.bias": torch.randn(256),
    }
    checkpoint_args = {
        "hrot_variant": "J_EGTW",
        "hrot_egtw_tau": 0.7,
        "hrot_egtw_lambda": 1.3,
        "hrot_egtw_attention_temperature": 0.2,
        "hrot_egtw_detach_masses": "true",
    }

    overrides = infer_hrot_arch_overrides_from_state_dict(state_dict, checkpoint_args=checkpoint_args)

    assert infer_hrot_variant_from_state_dict(state_dict, checkpoint_args=checkpoint_args) == "JE"
    assert overrides["hrot_variant"] == "JE"
    assert overrides["hrot_egtw_tau"] == 0.7
    assert overrides["hrot_egtw_lambda"] == 1.3
    assert overrides["hrot_egtw_attention_temperature"] == 0.2
    assert overrides["hrot_egtw_detach_masses"] == "true"


def test_infer_hrot_variant_uses_cp_ecot_checkpoint_args():
    state_dict = {
        "raw_ecot_lambda": torch.tensor(0.0),
        "episode_controller.network.0.weight": torch.randn(32, 11),
    }
    checkpoint_args = {
        "hrot_variant": "CP_ECOT",
        "hrot_ecot_rho_bank": "0.50,0.80,0.95",
        "hrot_ecot_consensus_tau_mode": "sqrt",
        "hrot_ecot_consensus_tau": 1.2,
    }

    overrides = infer_hrot_arch_overrides_from_state_dict(state_dict, checkpoint_args=checkpoint_args)

    assert infer_hrot_variant_from_state_dict(state_dict, checkpoint_args=checkpoint_args) == "CP_ECOT"
    assert overrides["hrot_variant"] == "CP_ECOT"
    assert overrides["hrot_ecot_rho_bank"] == "0.50,0.80,0.95"
    assert overrides["hrot_ecot_consensus_tau_mode"] == "sqrt"
    assert overrides["hrot_ecot_consensus_tau"] == 1.2


def test_infer_hrot_variant_uses_j_ecot_m2_checkpoint_args():
    state_dict = {
        "raw_ecot_lambda": torch.tensor(0.0),
        "episode_controller.network.0.weight": torch.randn(32, 11),
    }
    checkpoint_args = {
        "hrot_variant": "J_ECOT_M2",
        "hrot_ecot_rho_bank": "0.80",
        "hrot_ecot_base_rho": 0.80,
    }

    overrides = infer_hrot_arch_overrides_from_state_dict(state_dict, checkpoint_args=checkpoint_args)

    assert infer_hrot_variant_from_state_dict(state_dict, checkpoint_args=checkpoint_args) == "J_ECOT_M2"
    assert overrides["hrot_variant"] == "J_ECOT_M2"
    assert overrides["hrot_ecot_rho_bank"] == "0.80"
    assert overrides["hrot_ecot_base_rho"] == 0.80


def test_infer_hrot_variant_maps_old_j_hlm_state_to_j_ecot():
    state_dict = {
        "raw_transport_cost_threshold": torch.tensor(0.0),
        "hierarchical_transport_mass.budget_mlp.0.weight": torch.randn(64, 9),
        "hierarchical_transport_mass.budget_mlp.0.bias": torch.randn(64),
        "token_projector.1.weight": torch.randn(96, 640),
    }
    checkpoint_args = {
        "hrot_variant": "J_HLM",
        "hrot_hlm_budget_mode": "cost",
        "hrot_hlm_token_mode": "cost",
        "hrot_hlm_token_tau": 0.25,
    }

    overrides = infer_hrot_arch_overrides_from_state_dict(state_dict, checkpoint_args=checkpoint_args)

    assert infer_hrot_variant_from_state_dict(state_dict) == "J_ECOT"
    assert infer_hrot_variant_from_state_dict(state_dict, checkpoint_args=checkpoint_args) == "J_ECOT"
    assert overrides["hrot_variant"] == "J_ECOT"
    assert "hrot_hlm_budget_mode" not in overrides
    assert "hrot_hlm_token_mode" not in overrides
    assert "hrot_hlm_token_tau" not in overrides


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


def test_infer_hrot_arch_overrides_marks_old_h_checkpoints_as_legacy_eam():
    state_dict = {
        "raw_transport_cost_threshold": torch.tensor(0.0),
        "eam.network.0.weight": torch.randn(192, 4),
        "eam.network.0.bias": torch.randn(192),
        "token_projector.1.weight": torch.randn(96, 640),
    }

    overrides = infer_hrot_arch_overrides_from_state_dict(state_dict)

    assert overrides["hrot_variant"] == "H"
    assert overrides["hrot_eam_mode"] == "legacy"


def test_infer_hrot_arch_overrides_preserves_checkpoint_eam_mode_when_present():
    state_dict = {
        "raw_transport_cost_threshold": torch.tensor(0.0),
        "eam.network.0.weight": torch.randn(192, 4),
        "eam.network.0.bias": torch.randn(192),
    }
    checkpoint_args = {
        "hrot_variant": "H",
        "hrot_eam_mode": "compact",
        "hrot_compact_eam_prior_mix": 0.25,
    }

    overrides = infer_hrot_arch_overrides_from_state_dict(state_dict, checkpoint_args=checkpoint_args)

    assert overrides["hrot_variant"] == "H"
    assert overrides["hrot_eam_mode"] == "compact"
    assert overrides["hrot_compact_eam_prior_mix"] == 0.25


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
