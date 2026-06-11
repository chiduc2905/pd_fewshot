from types import SimpleNamespace

from main import build_wandb_init_config


def _model_meta():
    return {"architecture": "arch", "metric": "metric"}


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
        "hrot_ecot_egsm_hidden_dim",
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
