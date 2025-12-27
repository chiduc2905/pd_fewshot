# Project Structure

## Architecture Overview

```
pd_fewshot/
├── main.py                     # Main training & evaluation script
├── main_domain_shift.py        # Domain shift experiments
├── run_all_experiments.py      # Automated batch experiment runner
├── dataset.py                  # Dataset loader with auto-normalization
├── split_dataset.py            # Dataset splitting utilities
│
├── net/                        # Model implementations
│   ├── protonet.py             # Prototypical Networks
│   ├── matchingnet.py          # Matching Networks
│   ├── relationnet.py          # Relation Networks
│   ├── covamnet.py             # Covariance Metric Networks
│   ├── dn4.py                  # Dense Nearest-Neighbor (DN4)
│   ├── feat.py                 # FEAT (Transformer-based)
│   ├── deepemd.py              # DeepEMD (Earth Mover's Distance)
│   ├── siamesenet.py           # Siamese Networks
│   ├── cosine.py               # Cosine Classifier
│   ├── cosine_classifier.py    # Baseline++ (learnable temperature)
│   └── encoders/               # CNN backbone implementations
│       ├── base_encoder.py     # Conv64F encoder
│       ├── resnet12_encoder.py # ResNet12 encoder
│       └── resnet18_encoder.py # ResNet18 encoder
│
├── dataloader/                 # Few-shot episode generation
│   └── dataloader.py           # FewshotDataset class
│
├── function/                   # Loss functions & utilities
│   └── function.py             # ContrastiveLoss, TripletLoss, CenterLoss, etc.
│
├── visualization/              # Feature visualization tools
│   └── feature_visualizer.py   # t-SNE, feature maps, heatmaps
│
└── visualize_features.py       # CLI for feature map visualization
```

## Models

| Model | Pooling | Distance Metric |
|-------|---------|-----------------|
| **ProtoNet** | AvgPool | Negative Euclidean Distance |
| **MatchingNet** | LSTM Attention | Cosine Similarity |
| **RelationNet** | Concat | Learned Relation Score (CNN) |
| **CovaMNet** | CovaBlock | Covariance Metric |
| **DN4** | None | Local Descriptor k-NN |
| **FEAT** | Transformer | Adapted Euclidean Distance |
| **DeepEMD** | Set Matching | Earth Mover's Distance |
| **SiameseNet** | Concat | Learned Distance (MLP) |
| **Baseline++** | AvgPool | Scaled Cosine Similarity |

## Loss Functions

| Loss | Description |
|------|-------------|
| **Contrastive** | Standard Softmax Cross-Entropy on similarity scores (Default) |
| **Triplet** | Triplet Loss with margin on feature embeddings |
| **Center Loss** | Auxiliary loss to minimize intra-class variance |

## Data Flow

### Training
```
Train Data → FewshotDataset → 100 episodes/epoch × K-shot × 5-query → Model → Loss
```

### Validation (Model Selection)
```
Val Data → 150 episodes × K-shot × 1-query/class → Accuracy → Save Best Model
```

### Final Test
```
Test Data → 200 episodes × K-shot × 1-query → Metrics + Visualization
```

**Outputs:**
- Accuracy (mean ± std, worst-case, best-case)
- Precision, Recall, F1-score
- Confusion Matrix, t-SNE plots
- All metrics logged to WandB