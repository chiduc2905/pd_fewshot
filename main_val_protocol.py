"""Validation-only MLFork-style entrypoint for pulse_fewshot.

This file leaves ``main.py`` untouched and reuses its training pipeline, while
changing only the final evaluation protocol:
- best checkpoint is still selected by validation inside the original train loop
- final reported accuracy is computed on validation episodes
- no final test-set evaluation is performed here
"""

from __future__ import annotations

import os

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from dataloader.dataloader import FewshotDataset

import main as base


def build_val_loader(val_X, val_y, args):
    val_seed = args.seed + 1
    val_ds = FewshotDataset(
        val_X,
        val_y,
        args.episode_num_val,
        args.way_num,
        args.shot_num,
        args.query_num_val,
        seed=val_seed,
    )
    return DataLoader(
        val_ds,
        **base._build_loader_kwargs(
            args,
            batch_size=1,
            shuffle=False,
            worker_seed=val_seed,
        ),
    )


def final_eval_mlfork_style(net, loader, args):
    meta = base.get_model_metadata(args.model)
    acc, avg_loss = base.evaluate(net, loader, args)
    num_episodes = len(loader)
    total_queries = num_episodes * args.way_num * args.query_num_val

    print(f"\n{'=' * 60}")
    print(f"Final Eval (VAL / MLFork): {meta['display_name']} | {args.dataset_name} | {args.shot_num}-shot")
    print(f"{num_episodes} episodes x {args.way_num} classes x {args.query_num_val} query")
    print("=" * 60)
    print(f"  Accuracy      : {acc * 100:.2f}%")
    print(f"  Average Loss  : {avg_loss:.4f}")
    print(f"  Total Queries : {total_queries}")

    wandb.log(
        {
            "final_accuracy": acc,
            "final_loss": avg_loss,
            "val_accuracy_mlfork": acc,
            "val_loss_mlfork": avg_loss,
        }
    )
    wandb.run.summary["final_eval_split"] = "val"
    wandb.run.summary["final_eval_protocol"] = "mlfork"
    wandb.run.summary["final_accuracy"] = acc
    wandb.run.summary["final_loss"] = avg_loss
    wandb.run.summary["val_accuracy_mlfork"] = acc
    wandb.run.summary["val_loss_mlfork"] = avg_loss

    samples_str = f"{args.training_samples}samples" if args.training_samples else "allsamples"
    txt_path = os.path.join(
        args.path_results,
        f"results_{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot.txt",
    )
    with open(txt_path, "w") as handle:
        handle.write(f"Model: {meta['display_name']} ({args.model})\n")
        handle.write(f"Dataset: {args.dataset_name}\n")
        handle.write(f"Shot: {args.shot_num}\n")
        handle.write(f"Training Samples: {args.training_samples if args.training_samples else 'All'}\n")
        handle.write("Evaluation Split: val\n")
        handle.write("Evaluation Protocol: mlfork\n")
        handle.write("-" * 40 + "\n")
        handle.write(f"Accuracy : {acc:.4f} +/- 0.0000\n")
        handle.write(f"Loss : {avg_loss:.4f}\n")
        handle.write(f"Episodes : {num_episodes}\n")
        handle.write(f"Queries per episode : {args.query_num_val}\n")
        handle.write(f"Total Queries : {total_queries}\n")
    print(f"Results saved to {txt_path}")


def main():
    args = base.get_args()
    args.device = base.resolve_runtime_device(args)
    args.fewshot_backbone = base.resolve_fewshot_backbone(args)
    model_meta = base.get_model_metadata(args.model)

    if args.query_num is not None:
        args.query_num_train = args.query_num
        args.query_num_val = args.query_num
        args.query_num_test = args.query_num

    print(f"\n{'=' * 72}")
    print(model_meta["display_name"])
    print("=" * 72)
    print(f"Mode        : {args.mode}")
    print(f"Config      : {args.model} | {args.shot_num}-shot | {args.num_epochs} epochs | device={args.device}")
    print(f"Dataset     : {args.dataset_path} ({args.dataset_name})")
    print(f"Architecture: {model_meta['architecture']}")
    print(f"Backbone    : {args.fewshot_backbone}")
    print("Final Eval  : split=val, protocol=mlfork")

    samples_str = f"{args.training_samples}samples" if args.training_samples else "all"
    run_name = f"{args.model}_{args.dataset_name}_{samples_str}_{args.shot_num}shot"
    config = vars(args).copy()
    config["architecture"] = model_meta["architecture"]
    config["distance_metric"] = model_meta["metric"]
    config["final_eval_split"] = "val"
    config["final_eval_protocol"] = "mlfork"

    wandb.init(
        project=args.project,
        config=config,
        name=run_name,
        group=f"{args.model}_{args.dataset_name}",
        job_type=args.mode,
    )

    base.seed_func(args.seed)
    base.configure_cudnn_runtime(args)
    print(
        f"Runtime     : workers={args.num_workers}, pin_memory={base._pin_memory_enabled(args)}, "
        f"persistent_workers={base._persistent_workers_enabled(args)}, "
        f"cudnn(det={torch.backends.cudnn.deterministic}, bench={torch.backends.cudnn.benchmark})"
    )

    os.makedirs(args.path_weights, exist_ok=True)
    os.makedirs(args.path_results, exist_ok=True)

    dataset = base.load_dataset(args.dataset_path, image_size=args.image_size)

    def to_tensor(images, labels):
        return torch.from_numpy(images.astype(np.float32)), torch.from_numpy(labels).long()

    train_X, train_y = to_tensor(dataset.X_train, dataset.y_train)
    val_X, val_y = to_tensor(dataset.X_val, dataset.y_val)
    test_X, test_y = to_tensor(dataset.X_test, dataset.y_test)
    train_file_paths = [path for path, _ in getattr(dataset, "train_files", [])] if hasattr(dataset, "train_files") else None
    val_file_paths = [path for path, _ in getattr(dataset, "val_files", [])] if hasattr(dataset, "val_files") else None

    pretty_map = {
        "surface": "Surface",
        "internal": "Internal",
        "corona": "Corona",
        "notpd": "NotPD",
        "nopd": "NotPD",
    }
    dataset_classes = list(getattr(dataset, "classes", []))
    if dataset_classes:
        all_class_names = [pretty_map.get(class_name.lower(), class_name) for class_name in dataset_classes]
    else:
        num_classes = int(len(torch.unique(train_y)))
        all_class_names = [f"Class{i}" for i in range(num_classes)]

    if args.selected_classes:
        selected = [int(class_id.strip()) for class_id in args.selected_classes.split(",")]
        if any(class_id < 0 or class_id >= len(all_class_names) for class_id in selected):
            raise ValueError(f"selected_classes={selected} out of range for classes={all_class_names}")
        print(f"\nUsing only selected classes: {selected}")
        args.class_names = [all_class_names[class_id] for class_id in selected]
        args.way_num = len(selected)

        def filter_classes(images, labels, selected_classes, file_paths=None):
            mask = torch.zeros(len(labels), dtype=torch.bool)
            for class_id in selected_classes:
                mask |= labels == class_id

            filtered_images = images[mask]
            filtered_labels = labels[mask]
            label_map = {old: new for new, old in enumerate(selected_classes)}
            remapped_labels = torch.tensor([label_map[label.item()] for label in filtered_labels])
            filtered_paths = None
            if file_paths is not None and len(file_paths) == len(labels):
                filtered_paths = [path for path, keep in zip(file_paths, mask.tolist()) if keep]
            return filtered_images, remapped_labels, filtered_paths

        train_X, train_y, train_file_paths = filter_classes(train_X, train_y, selected, train_file_paths)
        val_X, val_y, val_file_paths = filter_classes(val_X, val_y, selected, val_file_paths)
        test_X, test_y, _ = filter_classes(test_X, test_y, selected, None)
        print(f"Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
    else:
        args.class_names = all_class_names
        if args.way_num != len(args.class_names):
            print(
                f"way_num={args.way_num} does not match dataset classes={len(args.class_names)}. "
                f"Using way_num={len(args.class_names)}."
            )
            args.way_num = len(args.class_names)

    wandb.config.update(
        {
            "way_num": args.way_num,
            "query_num_train": args.query_num_train,
            "query_num_val": args.query_num_val,
            "query_num_test": args.query_num_test,
            "final_eval_split": "val",
            "final_eval_protocol": "mlfork",
        },
        allow_val_change=True,
    )

    if args.training_samples:
        if args.training_samples % args.way_num != 0:
            raise ValueError(
                f"training_samples ({args.training_samples}) must be divisible by way_num ({args.way_num}) "
                "for balanced class sampling."
            )
        per_class = args.training_samples // args.way_num
        sample_images, sample_labels = [], []
        for class_id in range(args.way_num):
            indices = (train_y == class_id).nonzero(as_tuple=True)[0]
            if len(indices) < per_class:
                raise ValueError(f"Class {class_id}: need {per_class}, have {len(indices)}")
            generator = torch.Generator().manual_seed(args.seed)
            perm = torch.randperm(len(indices), generator=generator)[:per_class]
            sample_images.append(train_X[indices[perm]])
            sample_labels.append(train_y[indices[perm]])
        train_X = torch.cat(sample_images)
        train_y = torch.cat(sample_labels)
        print(f"Using {args.training_samples} training samples ({per_class}/class)")

    val_loader = build_val_loader(val_X, val_y, args)

    net = base.get_model(args)
    base.log_model_parameters(net, args.model, device=args.device, image_size=args.image_size)

    if args.mode == "train":
        if args.model == "adchot":
            base.train_adchot_loop(net, train_X, train_y, val_X, val_y, args)
        else:
            base.train_loop(net, train_X, train_y, val_X, val_y, args)
        path = base.get_best_model_path(args)
        print(f"Evaluating with best checkpoint on val: {path}")
        base.load_model_weights(net, path, args.device)
        base.prepare_dataset_conditioned_model(net, train_X, train_y, args)
        final_eval_mlfork_style(net, val_loader, args)
    else:
        if args.weights:
            base.load_model_weights(net, args.weights, args.device)
            base.prepare_dataset_conditioned_model(net, train_X, train_y, args)
            final_eval_mlfork_style(net, val_loader, args)
        else:
            print("Error: Please specify --weights for test mode")

    wandb.finish()


if __name__ == "__main__":
    main()
