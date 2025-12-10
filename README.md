# PD Scalogram Few-Shot Learning

Few-shot classification of Partial Discharge patterns using scalogram images.

## Models

| Model | Description |
|-------|-------------|
| **CovaMNet** | Covariance Metric Network |
| **ProtoNet** | Prototypical Network |
| **CosineNet** | Cosine Similarity Network |
| **MatchingNet** | Matching Network |
| **RelationNet** | Relation Network |

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
| `--model` | covamnet | Model: cosine, protonet, covamnet, matchingnet, relationnet |
| `--way_num` | 3 | Classes per episode |
| `--shot_num` | 1 | Support samples per class |
| `--query_num` | 1 | Query samples per class |
| `--training_samples` | all | Total training samples |
| `--num_epochs` | 100/70 | Training epochs (1-shot/5-shot) |
| `--lr` | 1e-4 | Learning rate |

## Evaluation Protocol

### All Phases (Train/Val/Test)
- **Query**: 1 per class per episode
- **Support**: K-shot per class

| Phase | Episodes | Total Predictions |
|-------|----------|-------------------|
| Training | 100/epoch | 300 (100 × 3) |
| Validation | 100 | 300 (100 × 3) |
| Final Test | 150 | 450 (150 × 3) |

### Final Test Metrics
- Accuracy, Precision, Recall, F1, p-value
- Confusion matrix, t-SNE plots

## Dataset

```
scalogram_images/
├── surface/   # Class 0
├── corona/    # Class 1
└── no_pd/     # Class 2
```

- **Input**: 64×64 RGB
- **Split**: 40/class for val, 40/class for test, rest for train
- **Normalization**: Auto-computed from training set

## Results

Results saved to `results/`:
- `results_*.txt` - Metrics
- `confusion_matrix_*.png` - Confusion matrix
- `tsne_*.png` - t-SNE visualization
