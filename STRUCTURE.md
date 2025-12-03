| ProtoNet | AvgPool | Negative Euclidean |
| CovaMNet | CovaBlock | Covariance-based |

### Loss Functions

| Loss | Description |
|------|-------------|
| **Contrastive** | Standard Softmax Cross-Entropy on similarity scores (Default) |
| **SupCon** | Supervised Contrastive Loss on features (Support + Query) |
| **Triplet** | Triplet Loss with Batch Hard Mining on features (Support + Query) |
| **Center Loss** | Auxiliary loss to minimize intra-class variance (Combined with above) |

## Files

```
├── main.py              # Training & evaluation (WandB integrated)
├── dataset.py           # Data loading (64×64, auto-norm)
├── dataloader/
│   └── dataloader.py    # Episode generator
├── net/
│   ├── encoder.py       # Conv64F backbone
│   ├── cosine.py        
│   ├── protonet.py      
│   └── covamnet.py      
├── function/
│   └── function.py      # Loss functions (Contrastive, SupCon, Triplet, Center) & visualization
├── checkpoints/         # Model weights
└── results/             # Local plots (Confusion Matrix, t-SNE)
```

## Data Flow

### Training
```
Train Data → FewshotDataset → 100 episodes (K-shot, 1-query) → Model → Loss (Main + Center)
```
*   **Logging**: Metrics (Loss, Acc) logged to **WandB**.

### Validation (Model Selection)
```
Val Data → 75 episodes × K-shot × 1-query/class → Accuracy → Save Best
```

### Final Test
```
Test Data → 150 episodes × 1-shot × 1-query → Metrics + Plots
```
*   **Metrics**: Accuracy, Precision, Recall, F1, p-value (Logged to WandB).
*   **Plots**: Confusion Matrix, t-SNE (Logged to WandB & saved locally).

## Commands

```bash
# Example: CovaMNet 1-shot with SupCon + Center Loss
python main.py --model covamnet --shot_num 1 --loss supcon --lambda_center 0.001
```