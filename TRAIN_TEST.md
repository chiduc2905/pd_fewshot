# Training and Testing Guide

This document provides details on how to train and test the few-shot learning models using `main.py`.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| **Dataset & Paths** | | | |
| `--dataset_path` | str | `./scalogram_images/` | Path to dataset. |
| `--image_size` | int | `224` | Input image size (e.g. 224, 84). |
| `--path_weights` | str | `checkpoints/` | Directory to save model checkpoints. |
| `--path_results` | str | `results/` | Directory to save result files and plots. |
| `--weights` | str | `None` | Path to specific weight file (`.pth`) for testing mode. |
| **Model** | | | |
| `--model` | str | `covamnet` | Model architecture: `cosine`, `protonet`, or `covamnet`. |
| `--classifier_type` | str | `learnable` | For CovaMNet: `learnable` (Conv1d) or `metric` (spatial mean). |
| **Few-shot Settings** | | | |
| `--way_num` | int | `3` | Number of classes per episode (N-way). |
| `--shot_num` | int | `1` | Number of support samples per class (K-shot). |
| `--query_num` | int | `None` | Number of query samples per class. Default: 19 for 1-shot, 15 for 5-shot. |
| **Training Settings** | | | |
| `--training_samples` | int | `None` | Total training samples across all classes. `None` = all data. |
| `--episode_num_train`| int | `None` | Number of episodes per epoch during training. Default: 1000 for 1-shot, 600 for 5-shot. |
| `--episode_num_val` | int | `600` | Number of episodes per epoch during validation. |
| `--episode_num_test` | int | `600` | Number of episodes for final testing. |
| `--num_epochs` | int | `None` | Total training epochs. Default: 100 for 1-shot, 50 for 5-shot (use early stopping). |
| `--batch_size` | int | `1` | Number of episodes per batch. |
| `--lr` | float| `5e-3` | Initial learning rate. |
| `--step_size` | int | `1000` | Step size: decay every 1000 epochs (100,000 episodes). |
| `--gamma` | float| `0.5` | Gamma factor for learning rate scheduler. |
| `--seed` | int | `42` | Random seed for reproducibility. |
| `--device` | str | `cuda` (if avail) | Device to run on: `cuda` or `cpu`. |
| **Mode** | | | |
| `--mode` | str | `train` | Execution mode: `train` or `test`. |

## Training Examples

### 1. Default Training (CovaMNet, 3-way 1-shot)
Trains CovaMNet with default settings.
```bash
python main.py --mode train --model covamnet --way_num 3 --shot_num 1
```

### 2. Train ProtoNet (3-way 5-shot)
```bash
python main.py --mode train --model protonet --way_num 3 --shot_num 5
```

### 3. Train CosineNet with Limited Training Samples
Train CosineNet using only 60 training samples in total.
```bash
python main.py --mode train --model cosine --training_samples 60
```

### 4. Train CovaMNet with Metric Classifier
Use the metric-based classifier (spatial mean) instead of the learnable one for CovaMNet.
```bash
python main.py --mode train --model covamnet --classifier_type metric
```

## Testing Examples

### 1. Test Best Saved Model
If you have already trained a model, the script saves the best checkpoint to `checkpoints/`. By default, `test` mode looks for `checkpoints/{model}_{shot}shot_best.pth`.

```bash
python main.py --mode test --model covamnet --shot_num 1
```

### 2. Test Specific Weight File
To test a specific checkpoint file:
```bash
python main.py --mode test --model covamnet --weights checkpoints/covamnet_1shot_best.pth
```

## Running Multiple Experiments
You can use `run_all.py` to run multiple combinations of models and shots sequentially.

```bash
# Edit run_all.py to configure the lists of models and shots, then run:
python run_all.py
```
Or pass arguments directly to `run_all.py` (supported args: `--models`, `--shots`):
```bash
python run_all.py --models cosine protonet --shots 1 5
```

