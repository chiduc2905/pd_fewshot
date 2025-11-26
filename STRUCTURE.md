# Model Training & Testing Commands

Use the following commands to train and test the different models. Ensure your dataset is located at `./scalogram_images/` or specify the path with `--dataset_path`.

## Dataset Configuration
- Dataset split: **70% train / 15% validation / 15% test**
- N-way K-shot setup:
  - **1-shot**: N=3, K=1, Q=19 (episodes/epoch=1000, epochs=100-300)
  - **5-shot**: N=3, K=5, Q=15 (episodes/epoch=600, epochs=50-200)

## Common Arguments
- `--model`: `covamnet`, `protonet`, or `cosine`.
- `--shot_num`: Number of support samples (1 or 5).
- `--query_num`: Number of query samples per class. Default: 19 for 1-shot, 15 for 5-shot.
- `--training_samples`: **Total** number of training samples across all classes (e.g., 30, 60). If set to 60 with 3 classes, it uses 20 samples per class. Omit to use all available data.
- `--episode_num_train`: Number of episodes per epoch during training. Default: 1000 for 1-shot, 600 for 5-shot.
- `--num_epochs`: Number of training epochs. Default: 100 for 1-shot, 50 for 5-shot (use early stopping).
- `--mode`: `train` or `test`.

---

## 1. CovaMNet (Covariance Metric Network)

**1-Shot Training (default: Q=19, episodes/epoch=1000, epochs=100):**
```bash
python main.py --model covamnet --shot_num 1 --mode train
```

**1-Shot Training with Limited Samples (Total 30 training samples - 10/class):**
```bash
python main.py --model covamnet --shot_num 1 --training_samples 30 --mode train
```

**5-Shot Training (default: Q=15, episodes/epoch=600, epochs=50):**dd
```bash
python main.py --model covamnet --shot_num 5 --mode train
```

**Testing Only:**
```bash
python main.py --model covamnet --shot_num 1 --mode test
```

---

## 2. ProtoNet (Prototypical Network)

**1-Shot Training:**
```bash
python main.py --model protonet --shot_num 1 --mode train
```

**5-Shot Training:**
```bash
python main.py --model protonet --shot_num 5 --mode train
```

**Testing Only:**
```bash
python main.py --model protonet --shot_num 1 --mode test
```

---

## 3. CosineNet (Cosine Similarity Network)

**1-Shot Training:**
```bash
python main.py --model cosine --shot_num 1 --mode train
```

**5-Shot Training:**
```bash
python main.py --model cosine --shot_num 5 --mode train
```

**Testing Only:**
```bash
python main.py --model cosine --shot_num 1 --mode test
```
