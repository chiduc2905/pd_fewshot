# Project Structure

## Architecture

```
Input (3×64×64) → Conv64F_Encoder (64×16×16) → Model Head → Scores (way_num)
```

### Models

| Model | Head | Metric |
|-------|------|--------|
| CosineNet | AvgPool → FC | Cosine similarity |
| ProtoNet | AvgPool | Negative Euclidean |
| CovaMNet | CovaBlock | Covariance-based |

## Files

```
├── main.py              # Training & evaluation
├── dataset.py           # Data loading (64×64, auto-norm)
├── dataloader/
│   └── dataloader.py    # Episode generator
├── net/
│   ├── encoder.py       # Conv64F backbone
│   ├── cosine.py        
│   ├── protonet.py      
│   └── covamnet.py      
├── function/
│   └── function.py      # Loss & visualization
├── checkpoints/         # Model weights
└── results/             # Metrics & plots
```

## Data Flow

### Training
```
Train Data → FewshotDataset → 100 episodes (K-shot, 1-query) → Model → Loss
```

### Validation (Model Selection)
```
Val Data → 75 episodes × K-shot × 1-query/class → Accuracy → Save Best
```

### Final Test
```
Test Data → 150 episodes × 1-shot × 1-query → Metrics + Plots
```

## Commands

```bash
# All models with 30 samples
for model in covamnet protonet cosine; do
    for shot in 1 5; do
        python main.py --model $model --shot_num $shot --training_samples 30
    done
done
```
