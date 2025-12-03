# Training & Testing Guide

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_path` | `./prpd_images_for_cnn/` | Dataset path |
| `--path_weights` | `checkpoints/` | Checkpoint directory |
| `--path_results` | `results/` | Results directory |
| `--weights` | None | Custom weights for testing |
| `--model` | `covamnet` | Model: cosine, protonet, covamnet |
| `--way_num` | 2 | N-way (classes per episode) |
| `--shot_num` | 1 | K-shot (support samples) |
| `--query_num` | 19/15 | Query samples (auto by shot) |
| `--training_samples` | None | Limit total training samples |
| `--episode_num_train` | 100 | Training episodes per epoch |
| `--episode_num_test` | 75 | Test episodes |
| `--num_epochs` | 100/50 | Epochs (auto by shot) |
| `--batch_size` | 1 | Episodes per batch |
| `--lr` | 1e-4 | Learning rate |
| `--step_size` | 10 | LR scheduler step |
| `--gamma` | 0.1 | LR decay factor |
| `--seed` | 42 | Random seed |
| `--device` | cuda | Device (cuda/cpu) |
| `--mode` | train | Mode: train or test |
| `--loss` | contrastive | Loss: contrastive, supcon, triplet |
| `--temp` | 0.1 | Temperature for SupCon |
| `--margin` | 0.5 | Margin for Triplet |
| `--lambda_center` | 0.001 | Weight for Center Loss |

## Training

```bash
<<<<<<< HEAD
# Default (CovaMNet 3-way 1-shot, Contrastive + Center Loss)
python main.py --mode train --lambda_center 0.001

# SupCon Loss + Center Loss
python main.py --mode train --loss supcon --temp 0.1 --lambda_center 0.001

# Triplet Loss + Center Loss
python main.py --mode train --loss triplet --margin 0.5 --lambda_center 0.001

# ProtoNet 5-shot
python main.py --model protonet --shot_num 5 --mode train
=======
# Default (CovaMNet 2-way 1-shot)
python main.py --mode train

# ProtoNet 5-shot
python main.py --model protonet --shot_num 5 --mode train

# Limited samples (40 total = 20/class)
python main.py --model cosine --training_samples 40 --mode train
>>>>>>> 051989f71e953b580b9f773333b5ca5a9e8d0716
```

## Testing

```bash
# Auto-load best checkpoint
python main.py --model covamnet --shot_num 1 --mode test

# Custom weights
python main.py --model covamnet --weights checkpoints/my_model.pth --mode test
```

## Batch Experiments

```bash
# Run all combinations
python run_all.py --models cosine protonet covamnet --shots 1 5
```

