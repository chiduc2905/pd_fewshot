# Training & Testing Guide

## Quick Start

```bash
# Train 1-shot RelationNet
python main.py --model relationnet --shot_num 1 --mode train

# Train 5-shot ProtoNet with limited samples
python main.py --model protonet --shot_num 5 --training_samples 60 --mode train

# Test a trained model
python main.py --model relationnet --shot_num 1 --mode test
```

## Command Line Arguments

### Model & Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `covamnet` | Model: protonet, matchingnet, relationnet, covamnet, dn4, feat, deepemd, siamese, baseline, cosine |
| `--backbone` | `conv64f` | Encoder: conv64f, resnet12 |
| `--shot_num` | `1` | K-shot (support samples per class) |
| `--way_num` | `3` | N-way (classes per episode) |
| `--query_num` | `1` | Query samples per class |
| `--training_samples` | `None` | Limit total training samples |
| `--num_epochs` | `100/70` | Training epochs (auto: 100 for 1-shot, 70 for 5-shot) |
| `--mode` | `train` | Mode: train or test |

### Paths
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_path` | `./scalogram/` | Dataset directory |
| `--dataset_name` | `prpd` | Dataset name for logging |
| `--path_weights` | `checkpoints/` | Checkpoint save directory |
| `--path_results` | `results/` | Results output directory |
| `--weights` | `None` | Custom checkpoint for testing |

### Loss & Optimization
| Argument | Default | Description |
|----------|---------|-------------|
| `--loss` | `contrastive` | Loss: contrastive, triplet |
| `--lambda_center` | `0.0` | Center Loss weight |
| `--margin` | `0.1` | Triplet Loss margin |
| `--lr` | `1e-4` | Learning rate |
| `--step_size` | `10` | LR scheduler step |
| `--gamma` | `0.1` | LR decay factor |

### Experiment Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--image_size` | `128` | Input size (default: 128) |
| `--episode_num_train` | `100` | Training episodes per epoch |
| `--episode_num_val` | `150` | Validation episodes |
| `--episode_num_test` | `200` | Test episodes |
| `--seed` | `42` | Random seed |
| `--project` | `prpd` | WandB project name |

## Training Examples

```bash
# Default: CovaMNet 3-way 1-shot
python main.py --mode train

# RelationNet 5-shot
python main.py --model relationnet --shot_num 5 --mode train

# MatchingNet with ResNet12 backbone (128x128)
python main.py --model matchingnet --backbone resnet12 --image_size 128 --mode train

# Limited training samples (60 total = 20/class for 3-way)
python main.py --model cosine --training_samples 60 --mode train

# Triplet Loss + Center Loss
python main.py --model protonet --loss triplet --lambda_center 0.01 --mode train
```

## Testing Examples

```bash
# Auto-load best checkpoint (based on model/shot/samples config)
python main.py --model relationnet --shot_num 1 --mode test

# Test with custom weights
python main.py --model covamnet --weights checkpoints/custom_model.pth --mode test
```

## Batch Experiments

Run all model × shot × sample combinations:

```bash
python run_all_experiments.py --project my_wandb_project
```

This will train and evaluate all configured models with 1-shot, 5-shot, and 10-shot settings.

## Output Files

After training and testing, results are saved to:

```
checkpoints/
└── {dataset}_{model}_{samples}_{shot}shot_best.pth

results/
├── results_{dataset}_{model}_{samples}_{shot}shot.txt
├── confusion_matrix_{dataset}_{model}_{samples}_{shot}shot.png
└── tsne_{dataset}_{model}_{samples}_{shot}shot.png
```

All metrics are also logged to WandB for visualization.
