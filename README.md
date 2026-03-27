# Few-Shot Learning for Partial Discharge Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive framework for **Few-Shot Learning** applied to Partial Discharge (PD) pattern classification in high-voltage electrical systems. This project implements a unified benchmark suite spanning classical few-shot baselines, local-descriptor methods, and recent A*-style meta-learning models.

## 🎯 Highlights

- **12 benchmark models**: ProtoNet, MatchingNet, RelationNet, CovaMNet, DN4, FEAT, DeepEMD, MAML, CAN, FRN, DeepBDC, Cosine
- **98.67% accuracy** with only 1-shot learning (1 sample per class)
- **Episodic meta-learning** framework with N-way K-shot configuration
- **Paper-aligned backbones**: Conv4, Conv64F, Conv4-32, ResNet12
- **Signal-to-image pipeline**: CWT scalogram transformation
- **Comprehensive experiment automation** with WandB integration

## 📊 Results

| Model | 1-shot | 5-shot | 10-shot | Metric Type |
|-------|--------|--------|---------|-------------|
| **RelationNet** | **98.67%** | **98.11%** | - | Learned Relation Score |
| CovaMNet | 97.33% | 96.78% | - | Covariance Metric |
| Cosine Classifier | 96.89% | 97.00% | - | Cosine Similarity |
| MatchingNet | 97.56% | 96.44% | - | Attention LSTM |
| ProtoNet | 91.33% | 95.67% | - | Euclidean Distance |
| DN4 | 90.00% | 97.11% | - | Local k-NN |
| DeepEMD | 94.78% | 95.67% | - | Earth Mover's Distance |
| FEAT | 95.89% | 95.44% | - | Transformer-adapted |

## 🏗️ Project Structure

```
pd_fewshot/
├── main.py                 # Main training/evaluation script
├── dataset.py              # Dataset loader with auto-normalization
├── net/                    # Model implementations
│   ├── protonet.py         # Prototypical Networks
│   ├── matchingnet.py      # Matching Networks
│   ├── relationnet.py      # Relation Networks
│   ├── covamnet.py         # Covariance Metric Networks
│   ├── dn4.py              # Dense Nearest-Neighbor (DN4)
│   ├── feat.py             # FEAT (ResNet12)
│   ├── deepemd.py          # DeepEMD (ResNet12)
│   ├── can.py              # Cross Attention Network
│   ├── frn.py              # Feature Map Reconstruction Networks
│   ├── deepbdc.py          # DeepBDC
│   ├── maml.py             # MAML
│   └── encoders/           # CNN backbones
│       ├── base_encoder.py
│       ├── resnet12_encoder.py
│       └── resnet18_encoder.py
├── dataloader/             # Few-shot episode generator
├── function/               # Loss functions & utilities
├── visualization/          # Feature visualization tools
└── run_all_experiments.py  # Automated experiment runner
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/chiduc2905/pd_fewshot.git
cd pd_fewshot
pip install -r requirements.txt
```

### Training

```bash
# Train 1-shot with RelationNet
python main.py --model relationnet --shot_num 1 --mode train

# Train 5-shot with ProtoNet
python main.py --model protonet --shot_num 5 --mode train

# Train with limited samples (60 total)
python main.py --model covamnet --shot_num 1 --training_samples 60 --mode train
```

### Testing

```bash
# Test a trained model
python main.py --model relationnet --shot_num 1 --mode test --weights checkpoints/best_model.pth
```

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | covamnet | Model: protonet, matchingnet, relationnet, covamnet, dn4, feat, deepemd, can, frn, deepbdc, maml, cosine, mann |
| `--way_num` | 3 | Number of classes per episode |
| `--shot_num` | 1 | Support samples per class |
| `--query_num` | 1 | Query samples per class |
| `--image_size` | 128 | Input image size (default: 128) |
| `--training_samples` | all | Limit training samples |
| `--loss` | contrastive | Loss: contrastive, triplet |
| `--num_epochs` | 100/70 | Training epochs (1-shot/5-shot) |

## 📁 Dataset Preparation

Organize your dataset in the following structure:

```
scalogram/
├── train/
│   ├── corona/     # Class 0
│   ├── surface/    # Class 1
│   └── void/       # Class 2
├── val/
│   └── ...
└── test/
    └── ...
```

Input images should be RGB (default: 128×128).

## 🔬 Implemented Algorithms

### Metric-based Methods
- **ProtoNet** (Snell et al., NeurIPS 2017) - Euclidean distance to class prototypes
- **MatchingNet** (Vinyals et al., NeurIPS 2016) - Attention-based matching with LSTM
- **RelationNet** (Sung et al., CVPR 2018) - Learned relation scoring

### Distribution-based Methods
- **CovaMNet** (Li et al., AAAI 2019) - Covariance metric learning

### Local Descriptor Methods
- **DN4** (Li et al., CVPR 2019) - Dense k-NN with local descriptors

### Transformer-based Methods
- **FEAT** (Ye et al., CVPR 2020) - Set-to-set feature adaptation with ResNet12

### Cross-Attention / Reconstruction Methods
- **CAN** (Hou et al., NeurIPS 2019) - Class-specific cross-attention matching
- **FRN** (Wertheimer et al., CVPR 2021) - Feature-map reconstruction metric

### Second-Order Methods
- **DeepBDC** (Xie et al., CVPR 2022) - Brownian distance covariance pooling

### Optimal Transport Methods
- **DeepEMD** (Zhang et al., CVPR 2020) - Earth Mover's Distance with ResNet12 local descriptors

### Optimization-based Methods
- **MAML** (Finn et al., ICML 2017) - Gradient-based task adaptation
  Default is now paper-style second-order; use `--maml_first_order` only when you explicitly want the approximation.

## 📈 Experiment Tracking

This project integrates with [Weights & Biases](https://wandb.ai) for experiment tracking:

```bash
# Run with WandB logging
python main.py --model relationnet --shot_num 1 --project my_project
```

Tracked metrics:
- Training/validation accuracy and loss
- Test accuracy (mean ± std, worst-case, best-case)
- Confusion matrix and t-SNE visualizations
- Inference time per episode
- Model parameters and FLOPs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{pd_fewshot2024,
  author = {Chi Duc Nguyen},
  title = {Few-Shot Learning for Partial Discharge Classification},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/chiduc2905/pd_fewshot}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ProtoNet, MatchingNet, RelationNet implementations inspired by original papers
- Encoder architectures adapted from few-shot learning benchmarks
- WandB for experiment tracking
