# PD Scalogram Few-Shot Learning

Few-shot classification of Partial Discharge patterns using scalogram images.

## Models

| Model | Description |
|-------|-------------|
| **CovaMNet** | Covariance Metric Network |
| **ProtoNet** | Prototypical Network |
| **CosineNet** | Cosine Similarity Network |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (1-shot)
python main.py --model covamnet --shot_num 1 --mode train

# Train with limited samples
python main.py --model protonet --shot_num 5 --training_samples 60 --mode train

# Test
python main.py --model covamnet --shot_num 1 --mode test
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | covamnet | Model: cosine, protonet, covamnet |
| `--way_num` | 3 | Classes per episode |
| `--shot_num` | 1 | Support samples per class |
| `--query_num` | 1 | Query samples per class |
| `--training_samples` | all | Total training samples (e.g., 30=10/class) |
| `--num_epochs` | 100/70 | Training epochs (1-shot/5-shot) |
| `--lr` | 1e-3 | Learning rate |

## Evaluation Protocol

### All Phases (Train/Val/Test)
- **Query**: 1 per class per episode
- **Support**: K-shot per class

| Phase | Episodes | Total Predictions |
|-------|----------|-------------------|
| Training | 100/epoch | 300 (100 × 3) |
| Validation | 75 | 225 (75 × 3) |
| Final Test | 150 | 450 (150 × 3) |

### Final Test Metrics
- Accuracy, Precision, Recall, F1, p-value
- Confusion matrix: each row sums to 150

## Dataset

```
scalogram_images/
├── corona/    # Class 0
├── no_pd/     # Class 1  
└── surface/   # Class 2
```

- **Input**: 64×64 RGB
- **Split**: 75/class for val/test, rest for train
- **Normalization**: Auto-computed from dataset

## Results

Results saved to `results/`:
- `summary_*.txt` - Metrics table
- `confusion_matrix_*.png` - Confusion matrix
- `tsne_*.png` - t-SNE visualization
