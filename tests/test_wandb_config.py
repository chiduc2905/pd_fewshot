from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from main import (
    accumulate_diagnostics,
    build_wandb_init_config,
    dcr_debug_enabled,
    is_elastic_ot_debug_metric,
    finalize_diagnostics,
    is_dcr_debug_metric,
    is_uot_energy_debug_metric,
    summarize_ours_final_audit_batch,
    summarize_score_diagnostics,
    write_dcr_debug_report,
    write_elastic_ot_debug_report,
    write_ours_final_audit_report,
    write_uot_energy_debug_report,
)


def _model_meta():
    return {"architecture": "arch", "metric": "metric"}


def test_diagnostic_metrics_use_per_key_denominators():
    sums = defaultdict(float)
    accumulate_diagnostics(sums, {"always": 2.0, "probe": 1.0}, weight=4)
    accumulate_diagnostics(sums, {"always": 4.0}, weight=4)

    result = finalize_diagnostics(sums, total_weight=8)

    assert result["always"] == 3.0
    assert result["probe"] == 1.0


def test_ours_final_audit_report_summarizes_transport_bottleneck(tmp_path):
    output_dir = tmp_path / f"ours_final_audit_{uuid4().hex}"
    output_dir.mkdir(parents=True)
    args = SimpleNamespace(
        model="ours_final",
        dataset_name="knee_aug_split",
        training_samples=60,
        shot_num=1,
        seed=42,
        final_test_seed=42,
        test_protocol="clean",
        effective_test_protocol="clean",
        current_test_name="",
        experiment_tag="audit_smoke",
        path_results=str(output_dir),
        ours_ablation="full",
        hrot_ecot_transport_mode="unbalanced",
        ours_final_score_mode="threshold_mass",
        enable_global_residual_score="true",
        global_residual_weight=0.1,
    )
    targets = torch.tensor([0, 1])
    logits = torch.tensor([[0.2, 0.9], [0.1, 0.8]])
    scores = {
        "logits": logits,
        "transport_cost": torch.tensor([[0.30, 0.10], [0.35, 0.12]]),
        "transported_mass": torch.tensor([[0.80, 0.80], [0.80, 0.80]]),
        "transport_cost_threshold": torch.tensor(0.20),
    }

    try:
        audit_diag = summarize_ours_final_audit_batch(scores, logits, targets, eps=1e-8)
        path = write_ours_final_audit_report(
            args,
            {"pred_acc": 0.5, "local_score_gap": -0.1, "global_score_gap": 0.2},
            audit_diag,
            accuracy=0.5,
        )

        assert path is not None
        text = Path(path).read_text(encoding="utf-8")
        assert "OURS-FINAL LOCAL TRANSPORT AUDIT" in text
        assert "Verdict: LOCAL-UOT-BOTTLENECK" in text
        assert "Descriptor/cost bottleneck" in text
        assert "audit_avg_cost_gap_vs_logit_runner" in text
        assert "NOVELTY CHECK" in text
    finally:
        for file_path in output_dir.iterdir():
            file_path.unlink()
        output_dir.rmdir()


def test_uot_energy_debug_report_is_separate_and_seeded(tmp_path):
    output_dir = tmp_path / f"uot_energy_debug_{uuid4().hex}"
    output_dir.mkdir(parents=True)
    args = SimpleNamespace(
        model="ours_final",
        dataset_name="knee_aug_split",
        training_samples=60,
        shot_num=1,
        seed=43,
        final_test_seed=42,
        test_protocol="clean",
        effective_test_protocol="clean",
        current_test_name="",
        experiment_tag="seed43_ours_final_score_uot_energy",
        path_results=str(output_dir),
        ours_final_score_mode="uot_energy",
    )
    diagnostics = {
        "pred_acc": 0.82,
        "ours_probe_utility_pred_acc": 0.80,
        "uot_energy_pred_acc": 0.82,
        "uot_energy/marginal_penalty_share": 0.14,
        "ours_probe/harm_share": 0.98,
        "transport_audit/common_mass_ratio": 0.75,
    }

    try:
        path = write_uot_energy_debug_report(
            args,
            diagnostics,
            accuracy=0.82,
        )

        assert path is not None
        assert "seed43" in path
        text = Path(path).read_text()
        assert "Verdict: SUPPORTED" in text
        assert "UOT-Energy Accuracy Delta: +0.020000" in text
        assert "ours_probe/harm_share: 0.980000" in text
        assert is_uot_energy_debug_metric("uot_energy/query_kl")
        assert is_uot_energy_debug_metric("ours_probe/harm_share")
        assert not is_uot_energy_debug_metric("pred_acc")
    finally:
        for file_path in output_dir.iterdir():
            file_path.unlink()
        output_dir.rmdir()


def test_elastic_ot_debug_report_compares_same_checkpoint_uniform_uot(tmp_path):
    output_dir = tmp_path / f"elastic_ot_debug_{uuid4().hex}"
    output_dir.mkdir(parents=True)
    args = SimpleNamespace(
        model="ours_final",
        dataset_name="knee_aug_split",
        training_samples=60,
        shot_num=1,
        seed=42,
        final_test_seed=42,
        test_protocol="clean",
        effective_test_protocol="clean",
        experiment_tag="seed42_ours_final_score_elastic_ot",
        path_results=str(output_dir),
        ours_final_score_mode="elastic_ot",
    )
    diagnostics = {
        "pred_acc": 0.83,
        "elastic_probe_uniform_uot_pred_acc": 0.80,
        "elastic_probe_fix_rate": 0.05,
        "elastic_probe_harm_rate": 0.02,
        "elastic_ot/score_identity_error": 1e-8,
        "elastic_ot/row_capacity_violation": 1e-8,
        "elastic_ot/column_capacity_violation": 1e-8,
        "elastic_ot/transported_fraction": 0.31,
    }

    try:
        path = write_elastic_ot_debug_report(
            args,
            diagnostics,
            accuracy=0.83,
        )

        assert path is not None
        assert "seed42" in path
        text = Path(path).read_text()
        assert "Verdict: SUPPORTED" in text
        assert "Accuracy Delta: +0.030000" in text
        assert "Fix Rate: 0.050000" in text
        assert is_elastic_ot_debug_metric("elastic_ot/transported_fraction")
        assert is_elastic_ot_debug_metric("elastic_probe_fix_rate")
        assert not is_elastic_ot_debug_metric("pred_acc")
    finally:
        for file_path in output_dir.iterdir():
            file_path.unlink()
        output_dir.rmdir()


def test_dcr_debug_report_logs_residual_evidence_gap(tmp_path):
    output_dir = tmp_path / f"dcr_debug_{uuid4().hex}"
    output_dir.mkdir(parents=True)
    args = SimpleNamespace(
        model="ours_final",
        dataset_name="knee_aug_split",
        training_samples=60,
        shot_num=1,
        seed=42,
        final_test_seed=42,
        test_protocol="clean",
        effective_test_protocol="clean",
        experiment_tag="seed42_ours_final_score_dcr",
        path_results=str(output_dir),
        ours_final_score_mode="threshold_mass",
        enable_dustbin_contrastive_score="true",
    )
    diagnostics = {
        "pred_acc": 0.81,
        "dcr_unadjusted_pred_acc": 0.78,
        "dcr_accuracy_delta": 0.03,
        "dcr_fix_rate": 0.05,
        "dcr_harm_rate": 0.02,
        "dcr/shot_logit_delta_abs": 0.03,
        "dcr_removed_positive_distance_gap": 0.12,
        "dcr/removed_positive_share": 0.22,
    }

    try:
        path = write_dcr_debug_report(
            args,
            diagnostics,
            accuracy=0.81,
        )

        assert path is not None
        assert "seed42" in path
        text = Path(path).read_text()
        assert "Verdict: SUPPORTED" in text
        assert "Accuracy Delta: +0.030000" in text
        assert "Fix Rate: 0.050000" in text
        assert "Harm Rate: 0.020000" in text
        assert "dcr_removed_positive_distance_gap: 0.120000" in text
        assert dcr_debug_enabled(args)
        assert is_dcr_debug_metric("dcr/removed_positive_share")
        assert is_dcr_debug_metric("dcr_removed_positive_distance_gap")
        assert not is_dcr_debug_metric("pred_acc")
    finally:
        for file_path in output_dir.iterdir():
            file_path.unlink()
        output_dir.rmdir()


def test_dcr_prediction_comparison_metrics_use_same_examples():
    targets = torch.tensor([0, 0, 1, 1])
    unadjusted = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    )
    active = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    )

    metrics = summarize_score_diagnostics(
        {
            "dcr_logits": active,
            "dcr_unadjusted_logits": unadjusted,
        },
        active,
        targets,
    )

    assert metrics["dcr_unadjusted_pred_acc"] == 0.25
    assert metrics["dcr_pred_acc"] == 0.50
    assert metrics["dcr_fix_rate"] == 0.50
    assert metrics["dcr_harm_rate"] == 0.25
    assert metrics["dcr_prediction_change_rate"] == 0.75
    assert metrics["dcr_accuracy_delta"] == 0.25


def test_score_diagnostics_preserve_ucot_scalar_metrics():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])

    metrics = summarize_score_diagnostics(
        {
            "ucot/common_query_share": torch.tensor(0.625),
            "ucot/discriminative_query_share": torch.tensor(0.375),
        },
        logits,
        targets,
    )

    assert metrics["ucot/common_query_share"] == pytest.approx(0.625)
    assert metrics["ucot/discriminative_query_share"] == pytest.approx(0.375)


def test_ours_final_wandb_config_hides_other_model_flags_and_keeps_global_residual():
    args = SimpleNamespace(
        model="ours_final",
        dataset_path="data",
        dataset_name="pd",
        way_num=4,
        shot_num=1,
        image_size=84,
        fewshot_backbone="resnet12",
        num_epochs=100,
        lr=5e-4,
        seed=42,
        mode="train",
        token_dim=128,
        hrot_token_dim=None,
        hrot_score_scale=16.0,
        hrot_fixed_mass=0.8,
        hrot_ecot_rho_bank=None,
        hrot_ecot_base_rho=None,
        hrot_ecot_transport_mode=None,
        hrot_ecot_enable_egsm=None,
        hrot_ecot_egsm_hidden_dim=32,
        ours_ablation="full",
        ours_final_dmuot_ablation="off",
        ours_final_evidence_ablation="off",
        enable_global_residual_score=True,
        global_residual_mode="residual",
        global_residual_weight=0.1,
        enable_mspta=False,
        mspta_mass_mode="compact_area",
        aeb_hidden=16,
        spifaeb_v3_sigma_min=0.05,
        spif_gate_on="true",
        rada_tau_r=0.5,
        care_enable_fwec="true",
        deepemd_solver="sinkhorn",
        sc_lfi_context_dim=128,
        hrot_variant="CARE",
        ec_mrot_response_mode="anchor_free_functional",
        hrot_hlm_min_mass=0.1,
    )

    cfg = build_wandb_init_config(
        args,
        _model_meta(),
        selection_split="val",
        merge_val_into_train=False,
    )

    assert cfg["model"] == "ours_final"
    assert cfg["enable_global_residual_score"] is True
    assert cfg["global_residual_mode"] == "residual"
    assert cfg["global_residual_weight"] == 0.1
    assert cfg["architecture"] == "arch"
    assert cfg["distance_metric"] == "metric"
    assert cfg["selection_split"] == "val"
    assert cfg["merge_val_into_train"] is False

    for unrelated_key in (
        "aeb_hidden",
        "spifaeb_v3_sigma_min",
        "spif_gate_on",
        "rada_tau_r",
        "care_enable_fwec",
        "deepemd_solver",
        "sc_lfi_context_dim",
        "hrot_variant",
        "ec_mrot_response_mode",
        "hrot_hlm_min_mass",
        "mspta_mass_mode",
        "hrot_ecot_enable_egsm",
        "hrot_ecot_egsm_hidden_dim",
        "hrot_ecot_egsm_adaptive_rho",
    ):
        assert unrelated_key not in cfg


def test_ours_final_wandb_config_logs_verified_uot_for_verified_alias():
    args = SimpleNamespace(
        model="ours_final_verified_uot",
        verified_uot_beta=0.75,
        verified_uot_tau=0.1,
        verified_uot_ratio_threshold=0.35,
        verified_uot_kernel_size=3,
        enable_verified_uot_score=False,
        care_enable_qesm="true",
    )

    cfg = build_wandb_init_config(
        args,
        _model_meta(),
        selection_split="test",
        merge_val_into_train=True,
    )

    assert cfg["verified_uot_beta"] == 0.75
    assert cfg["verified_uot_tau"] == 0.1
    assert "care_enable_qesm" not in cfg


def test_ours_final_wandb_config_logs_hcuot_only_when_enabled():
    args = SimpleNamespace(
        model="ours_final",
        enable_hubness_calibrated_uot=True,
        hcuot_topk=3,
        hcuot_temperature=0.25,
        hcuot_cost_weight=0.35,
        hcuot_marginal_mix=0.50,
        deepemd_solver="sinkhorn",
    )

    cfg = build_wandb_init_config(
        args,
        _model_meta(),
        selection_split="val",
        merge_val_into_train=False,
    )

    assert cfg["enable_hubness_calibrated_uot"] is True
    assert cfg["hcuot_topk"] == 3
    assert cfg["hcuot_temperature"] == 0.25
    assert cfg["hcuot_cost_weight"] == 0.35
    assert cfg["hcuot_marginal_mix"] == 0.50
    assert "deepemd_solver" not in cfg


def test_ours_final_wandb_config_logs_score_aligned_marginal_mode():
    args = SimpleNamespace(
        model="ours_final",
        ours_final_marginal_mode="score_aligned",
        score_marginal_tau=0.25,
        score_marginal_mix=0.65,
        score_marginal_adaptive_mix="true",
        score_marginal_confidence_power=1.0,
        deepemd_solver="sinkhorn",
    )

    cfg = build_wandb_init_config(
        args,
        _model_meta(),
        selection_split="val",
        merge_val_into_train=False,
    )

    assert cfg["ours_final_marginal_mode"] == "score_aligned"
    assert cfg["score_marginal_tau"] == 0.25
    assert cfg["score_marginal_mix"] == 0.65
    assert cfg["score_marginal_adaptive_mix"] == "true"
    assert cfg["score_marginal_confidence_power"] == 1.0
    assert "deepemd_solver" not in cfg


def test_ours_final_wandb_config_logs_failure_probe_only_when_enabled():
    args = SimpleNamespace(
        model="ours_final",
        enable_ours_final_failure_probe="true",
        ours_probe_common_margin=0.10,
        deepemd_solver="sinkhorn",
    )

    cfg = build_wandb_init_config(
        args,
        _model_meta(),
        selection_split="val",
        merge_val_into_train=False,
    )

    assert cfg["enable_ours_final_failure_probe"] == "true"
    assert cfg["ours_probe_common_margin"] == 0.10
    assert "deepemd_solver" not in cfg
