"""Validate execution quality for the SPIF-HMamba dual-branch family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.dataloader import FewshotDataset
from dataset import load_dataset
from function.function import seed_func
from main import get_model, load_model_weights, resolve_runtime_device


def _bool_flag(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def get_args():
    parser = argparse.ArgumentParser(description="Validate SPIF-HMamba branch execution on episodic splits")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="knee_aug_split")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--model", type=str, default="spifharmce", choices=["spifharmce", "spifharmmax"])
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--fewshot_backbone", type=str, default="resnet12", choices=["default", "resnet12", "conv64f"])
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--way_num", type=int, default=4)
    parser.add_argument("--shot_num", type=int, default=1)
    parser.add_argument("--query_num", type=int, default=1)
    parser.add_argument("--episode_num", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--persistent_workers", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--prefetch_factor", type=int, default=2)

    parser.add_argument("--spif_stable_dim", type=int, default=64)
    parser.add_argument("--spif_variant_dim", type=int, default=64)
    parser.add_argument("--spif_gate_hidden", type=int, default=16)
    parser.add_argument("--spif_gate_on", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_factorization_on", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_global_only", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--spif_local_only", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--spif_token_l2norm", type=str, default="true", choices=["true", "false"])

    parser.add_argument("--spif_papersw_train_num_projections", type=int, default=128)
    parser.add_argument("--spif_papersw_eval_num_projections", type=int, default=512)
    parser.add_argument("--spif_papersw_p", type=float, default=2.0)
    parser.add_argument("--spif_papersw_normalize_inputs", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--spif_papersw_train_projection_mode", type=str, default="resample", choices=["resample", "fixed"])
    parser.add_argument("--spif_papersw_eval_projection_mode", type=str, default="fixed", choices=["resample", "fixed"])
    parser.add_argument("--spif_papersw_eval_num_repeats", type=int, default=1)
    parser.add_argument("--spif_papersw_projection_seed", type=int, default=7)

    parser.add_argument("--spif_harm_global_scale", type=float, default=16.0)
    parser.add_argument("--spif_harm_fusion_hidden_dim", type=int, default=32)
    parser.add_argument("--spif_harm_local_score_scale", type=float, default=8.0)
    parser.add_argument("--spif_harm_factor_consistency_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_factor_decorr_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_factor_sparse_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_factor_consistency_dropout", type=float, default=0.1)
    parser.add_argument("--spif_harm_final_sparse_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_global_ce_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_local_ce_weight", type=float, default=0.0)
    parser.add_argument("--spif_harm_beta_init", type=float, default=0.1)
    parser.add_argument("--spif_harm_learnable_beta", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--spif_harm_use_final_gate", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--ssm_state_dim", type=int, default=16)
    parser.add_argument("--spif_mamba_depth", type=int, default=1)
    parser.add_argument("--spif_mamba_ffn_multiplier", type=int, default=2)
    parser.add_argument("--spif_mamba_d_conv", type=int, default=4)
    parser.add_argument("--spif_mamba_expand", type=int, default=2)
    parser.add_argument("--spif_mamba_dropout", type=float, default=0.0)

    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()
    args.device = resolve_runtime_device(args)
    return args


def _loader_kwargs(args):
    kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": _bool_flag(args.pin_memory, default=True),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = _bool_flag(args.persistent_workers, default=True)
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def _select_split(dataset, split: str):
    if split == "val":
        return dataset.X_val, dataset.y_val
    return dataset.X_test, dataset.y_test


def main():
    args = get_args()
    seed_func(args.seed)

    dataset = load_dataset(args.dataset_path, image_size=args.image_size)
    split_X, split_y = _select_split(dataset, args.split)
    split_X = torch.from_numpy(split_X.astype(np.float32))
    split_y = torch.from_numpy(split_y).long()

    if args.way_num != len(torch.unique(split_y)):
        print(f"Adjusting way_num from {args.way_num} to {len(torch.unique(split_y))} to match split labels")
        args.way_num = int(len(torch.unique(split_y)))

    loader = DataLoader(
        FewshotDataset(
            split_X,
            split_y,
            args.episode_num,
            args.way_num,
            args.shot_num,
            args.query_num,
            seed=args.seed,
        ),
        **_loader_kwargs(args),
    )

    model = get_model(args)
    if args.weights:
        load_model_weights(model, args.weights, args.device)
        print(f"Loaded weights: {args.weights}")

    model.eval()
    device = torch.device(args.device)
    non_blocking = _bool_flag(args.pin_memory, default=True) and device.type == "cuda"

    fused_correct = 0
    global_correct = 0
    local_correct = 0
    total = 0

    fused_global_agree = []
    fused_local_agree = []
    global_local_agree = []
    fusion_gate_mean = []
    beta_values = []
    final_gate_mean = []
    pool_entropy_mean = []
    encoder_gate_mean = []
    fused_margin = []
    global_margin = []
    local_margin = []

    with torch.no_grad():
        for query, q_labels, support, support_labels in loader:
            batch_size = query.shape[0]
            channels, height, width = query.shape[2], query.shape[3], query.shape[4]
            support = support.view(batch_size, args.way_num, args.shot_num, channels, height, width).to(
                device,
                non_blocking=non_blocking,
            )
            query = query.to(device, non_blocking=non_blocking)
            targets = q_labels.view(-1).to(device)

            outputs = model(query, support, return_aux=True, query_targets=targets)
            fused_logits = outputs["logits"]
            global_logits = outputs["global_logits"]
            local_logits = outputs["local_logits"]

            fused_preds = fused_logits.argmax(dim=1)
            global_preds = global_logits.argmax(dim=1)
            local_preds = local_logits.argmax(dim=1)

            fused_correct += (fused_preds == targets).sum().item()
            global_correct += (global_preds == targets).sum().item()
            local_correct += (local_preds == targets).sum().item()
            total += targets.numel()

            fused_global_agree.append((fused_preds == global_preds).float().mean().item())
            fused_local_agree.append((fused_preds == local_preds).float().mean().item())
            global_local_agree.append((global_preds == local_preds).float().mean().item())
            fusion_gate_mean.append(outputs["fusion_gate"].mean().item())
            beta_values.append(float(outputs["beta"]))
            final_gate_mean.append(outputs["final_gate_mean"].item())
            pool_entropy_mean.append(outputs["pool_entropy_mean"].item())
            encoder_gate_mean.append(outputs["mean_gate"].item())

            fused_top2 = torch.topk(fused_logits, k=min(2, fused_logits.shape[-1]), dim=-1).values
            global_top2 = torch.topk(global_logits, k=min(2, global_logits.shape[-1]), dim=-1).values
            local_top2 = torch.topk(local_logits, k=min(2, local_logits.shape[-1]), dim=-1).values
            if fused_top2.shape[-1] == 2:
                fused_margin.append((fused_top2[:, 0] - fused_top2[:, 1]).mean().item())
                global_margin.append((global_top2[:, 0] - global_top2[:, 1]).mean().item())
                local_margin.append((local_top2[:, 0] - local_top2[:, 1]).mean().item())

    summary = {
        "model": args.model,
        "split": args.split,
        "episodes": args.episode_num,
        "fused_acc": fused_correct / max(total, 1),
        "global_acc": global_correct / max(total, 1),
        "local_acc": local_correct / max(total, 1),
        "fused_global_agree": float(np.mean(fused_global_agree)) if fused_global_agree else None,
        "fused_local_agree": float(np.mean(fused_local_agree)) if fused_local_agree else None,
        "global_local_agree": float(np.mean(global_local_agree)) if global_local_agree else None,
        "fusion_gate_mean": float(np.mean(fusion_gate_mean)) if fusion_gate_mean else None,
        "beta_mean": float(np.mean(beta_values)) if beta_values else None,
        "global_final_gate_mean": float(np.mean(final_gate_mean)) if final_gate_mean else None,
        "global_pool_entropy_mean": float(np.mean(pool_entropy_mean)) if pool_entropy_mean else None,
        "encoder_gate_mean": float(np.mean(encoder_gate_mean)) if encoder_gate_mean else None,
        "fused_margin_mean": float(np.mean(fused_margin)) if fused_margin else None,
        "global_margin_mean": float(np.mean(global_margin)) if global_margin else None,
        "local_margin_mean": float(np.mean(local_margin)) if local_margin else None,
    }

    print(json.dumps(summary, indent=2))

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved validation summary to {save_path}")


if __name__ == "__main__":
    main()
