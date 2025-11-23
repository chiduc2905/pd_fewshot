# Model Training & Testing Commands

Use the following commands to train and test the different models. Ensure your dataset is located at `./scalogram_images/` or specify the path with `--dataset_path`.

## Common Arguments
- `--model`: `covamnet`, `protonet`, or `cosine`.
- `--shot_num`: Number of support samples (1 or 5).
- `--query_num`: Number of query samples (e.g., 15).
- `--training_samples`: **Total** number of training samples across all classes (e.g., 30, 60). If set to 60 with 3 classes, it uses 20 samples per class. Omit to use all available data.
- `--num_epochs`: Number of training epochs (default 100).
- `--mode`: `train` or `test`.

---

## 1. CovaMNet (Covariance Metric Network)

**1-Shot Training (Total 30 training samples - 10/class):**
```bash
python main.py --model covamnet --shot_num 1 --query_num 15 --training_samples 30 --num_epochs 100 --mode train
```

**1-Shot Training (Total 60 training samples - 20/class):**
```bash
python main.py --model covamnet --shot_num 1 --query_num 15 --training_samples 60 --num_epochs 100 --mode train
```

**5-Shot Training (All available data):**
```bash
python main.py --model covamnet --shot_num 5 --query_num 15 --num_epochs 100 --mode train
```

**Testing Only:**
```bash
python main.py --model covamnet --shot_num 1 --query_num 15 --mode test
```

---

## 2. ProtoNet (Prototypical Network)

**1-Shot Training (Total 30 training samples):**
```bash
python main.py --model protonet --shot_num 1 --query_num 15 --training_samples 30 --num_epochs 100 --mode train
```

**5-Shot Training (All available data):**
```bash
python main.py --model protonet --shot_num 5 --query_num 15 --num_epochs 100 --mode train
```

**Testing Only:**
```bash
python main.py --model protonet --shot_num 5 --query_num 15 --mode test
```

---

## 3. CosineNet (Cosine Similarity Network)

**1-Shot Training (Total 30 training samples):**
```bash
python main.py --model cosine --shot_num 1 --query_num 15 --training_samples 30 --num_epochs 100 --mode train
```

**5-Shot Training (All available data):**
```bash
python main.py --model cosine --shot_num 5 --query_num 15 --num_epochs 100 --mode train
```

**Testing Only:**
```bash
python main.py --model cosine --shot_num 1 --query_num 15 --mode test
```
