# PD Scalogram Few-Shot Learning Project

This project implements few-shot learning algorithms (CovaMNet, ProtoNet, CosineNet) for Partial Discharge (PD) classification using scalogram images.

## Structure

- `main.py`: Unified entry point for training and testing all models.
- `dataset.py`: Data loading logic for the raw image dataset.
- `dataloader/`: Contains the `FewshotDataset` class for episodic data generation.
- `net/`: Contains model definitions and reusable components.
  - `covamnet.py`: Covariance Metric Network.
  - `protonet.py`: Prototypical Network.
  - `cosine.py`: Cosine Similarity Network.
  - `encoder.py`: Shared convolutional backbone.
- `function/`: Utility functions (metrics, losses).
- `checkpoints/`: Directory where model weights are saved.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The `main.py` script handles both training and testing. You can specify the model architecture, shot number, and other hyperparameters via command line arguments.

### Training

**Train CovaMNet (1-shot):**
```bash
python main.py --model covamnet --shot_num 1 --mode train
```

**Train ProtoNet (5-shot):**
```bash
python main.py --model protonet --shot_num 5 --mode train
```

**Train CosineNet (1-shot):**
```bash
python main.py --model cosine --shot_num 1 --mode train
```

### Testing

To test a trained model, specify the `mode` as `test`. It will automatically look for the best checkpoint in `checkpoints/` unless `--weights` is specified.

```bash
python main.py --model covamnet --shot_num 1 --mode test
```

**Specify custom weights:**
```bash
python main.py --model covamnet --shot_num 1 --mode test --weights checkpoints/my_model.pth
```

### Common Arguments

- `--dataset_path`: Path to the image dataset (default: `./scalogram_images/`).
- `--way_num`: Number of classes per episode (default: 3).
- `--shot_num`: Number of support samples per class (1 or 5).
- `--query_num`: Number of query samples per class. Default: 19 for 1-shot, 15 for 5-shot.
- `--episode_num_train`: Number of episodes per epoch during training. Default: 1000 for 1-shot, 600 for 5-shot.
- `--num_epochs`: Total number of training epochs. Default: 100 for 1-shot, 50 for 5-shot (use early stopping).
- `--device`: `cuda` or `cpu`.

### Dataset Configuration

- Dataset split: **70% train / 15% validation / 15% test**
- N-way K-shot setup:
  - **1-shot**: N=3, K=1, Q=19 (episodes/epoch=1000, epochs=100-300)
  - **5-shot**: N=3, K=5, Q=15 (episodes/epoch=600, epochs=50-200)

## Results

Results (checkpoints, confusion matrices) are saved in the `checkpoints/` folder (or root for images).
