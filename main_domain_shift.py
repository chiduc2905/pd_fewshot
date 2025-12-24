"""
Domain Shift Meta-Learning for PD Classification.

Train on OLD dataset (scalogram_augmented), test on NEW domain (scalogram_minh).
Key: Domain shift = independent class distributions between train and test.

Data redistribution:
- Keep train/ as-is (augmented)
- Fill val/ to 50/class from test/
- Remaining test/ → add to train/
"""
import os
import argparse
import random
import numpy as np
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import torchvision.transforms as transforms
import wandb

# FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from dataloader.dataloader import FewshotDataset
from function.function import (ContrastiveLoss, RelationLoss, CenterLoss, 
                               seed_func, plot_confusion_matrix, plot_tsne)
from net.cosine import CosineNet
from net.cosine_classifier import CosineClassifier
from net.protonet import ProtoNet
from net.covamnet import CovaMNet
from net.matchingnet import MatchingNet
from net.relationnet import RelationNet
from net.siamesenet import SiameseNetFast as SiameseNet
from net.dn4 import DN4Fast as DN4
from net.feat import FEAT
from net.deepemd import DeepEMDSimple as DeepEMD


# =============================================================================
# Configuration
# =============================================================================

# Class mapping for OLD dataset
OLD_CLASS_MAP = {'surface': 0, 'corona': 1, 'nopd': 2}

# Class mapping for NEW dataset (scalogram_minh)
# Independent from OLD - just 0,1,2 for the 3 classes
NEW_CLASS_MAP = {'corona': 0, 'surface': 1, 'void': 2}  # Will handle surface/PD subdirectory


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Domain Shift Meta-Learning')
    
    # Paths
    parser.add_argument('--dataset_path', type=str, default='./scalogram_augmented/',
                        help='Path to OLD dataset for training')
    parser.add_argument('--test_dataset_path', type=str, 
                        default='/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/scalogram_minh',
                        help='Path to NEW dataset for domain shift testing')
    parser.add_argument('--path_weights', type=str, default='checkpoints/')
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--dataset_name', type=str, default='domain_shift')
    
    # Model
    parser.add_argument('--model', type=str, default='covamnet',
                        choices=['cosine', 'baseline', 'protonet', 'covamnet', 'matchingnet', 
                                 'relationnet', 'siamese', 'dn4', 'feat', 'deepemd'])
    parser.add_argument('--backbone', type=str, default='conv64f',
                        choices=['conv64f', 'resnet12', 'resnet18'])
    parser.add_argument('--use_base_encoder', action='store_true')
    
    # Few-shot settings
    parser.add_argument('--way_num', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=64)
    
    # Training
    parser.add_argument('--val_per_class', type=int, default=50,
                        help='Samples per class for validation')
    parser.add_argument('--episode_num_train', type=int, default=130)
    parser.add_argument('--episode_num_val', type=int, default=150)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    # Testing
    parser.add_argument('--test_episodes', type=int, default=500)
    parser.add_argument('--test_query_num', type=int, default=5,
                        help='Query samples per class for testing')
    
    # Loss
    parser.add_argument('--loss', type=str, default='contrastive',
                        choices=['contrastive', 'triplet'])
    parser.add_argument('--lambda_center', type=float, default=0.0)
    
    # WandB
    parser.add_argument('--project', type=str, default='prpd-domain-shift')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'stats'])
    
    return parser.parse_args()


# =============================================================================
# Dataset Loading - NO DATA LEAKAGE
# =============================================================================

class DomainShiftDataset:
    """
    Load OLD dataset with redistribution:
    - Keep train/ as-is (augmented)
    - Fill val/ to val_per_class from test/
    - Remaining test/ → add to train/
    
    IMPORTANT: Ensures NO data leakage between train and val.
    """
    
    def __init__(self, data_path, val_per_class=50, image_size=64, seed=42):
        self.data_path = os.path.abspath(data_path)
        self.val_per_class = val_per_class
        self.image_size = image_size
        self.seed = seed
        self.class_map = OLD_CLASS_MAP
        
        # Storage
        self.train_files = []  # [(path, label), ...]
        self.val_files = []
        
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.mean, self.std = None, None
        
        # Base transform
        self._base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f"\n{'='*60}")
        print(f"LOADING OLD DATASET (DOMAIN SOURCE)")
        print(f"{'='*60}")
        print(f"Path: {self.data_path}")
        
        # 1. Scan and redistribute
        self._scan_and_redistribute()
        
        # 2. Compute stats on train only
        self._compute_stats()
        
        # 3. Load images
        self._load_images()
        
        # 4. Shuffle
        self._shuffle_all()
        
        self._print_statistics()
    
    def _scan_folders(self, split_name):
        """Scan a split folder and return files grouped by class."""
        files_by_class = {0: [], 1: [], 2: []}
        split_path = os.path.join(self.data_path, split_name)
        
        if not os.path.exists(split_path):
            print(f"Warning: {split_name} folder not found")
            return files_by_class
        
        for class_name, label in self.class_map.items():
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            files = sorted([f for f in os.listdir(class_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                           and 'labeled' not in f.lower()])
            
            files_by_class[label] = [os.path.join(class_path, f) for f in files]
        
        return files_by_class
    
    def _scan_and_redistribute(self):
        """
        Redistribute data with NO leakage:
        1. Load original train, val, test splits
        2. Fill val to val_per_class from test
        3. Remaining test → train
        """
        # Scan all splits
        train_by_class = self._scan_folders('train')
        val_by_class = self._scan_folders('val')
        test_by_class = self._scan_folders('test')
        
        print(f"\nOriginal split counts (per class):")
        for label in range(3):
            class_name = [k for k, v in self.class_map.items() if v == label][0]
            print(f"  {class_name}: train={len(train_by_class[label])}, "
                  f"val={len(val_by_class[label])}, test={len(test_by_class[label])}")
        
        # Redistribute with fixed seed for reproducibility
        rng = random.Random(self.seed)
        
        new_train_files = []
        new_val_files = []
        
        for label in range(3):
            # 1. Start with original val
            current_val = val_by_class[label].copy()
            
            # 2. Shuffle test before taking
            test_files = test_by_class[label].copy()
            rng.shuffle(test_files)
            
            # 3. Fill val to val_per_class
            needed = self.val_per_class - len(current_val)
            if needed > 0:
                if needed > len(test_files):
                    print(f"Warning: Class {label} has only {len(test_files)} test samples, "
                          f"need {needed} to fill val")
                    needed = len(test_files)
                
                current_val.extend(test_files[:needed])
                remaining_test = test_files[needed:]
            else:
                remaining_test = test_files
            
            # 4. Current train + remaining test
            current_train = train_by_class[label].copy() + remaining_test
            
            # Store as (path, label) tuples
            new_train_files.extend([(f, label) for f in current_train])
            new_val_files.extend([(f, label) for f in current_val])
        
        self.train_files = new_train_files
        self.val_files = new_val_files
        
        # Verify no overlap (data leakage check)
        train_paths = set(f[0] for f in self.train_files)
        val_paths = set(f[0] for f in self.val_files)
        overlap = train_paths & val_paths
        
        if overlap:
            raise ValueError(f"DATA LEAKAGE DETECTED! {len(overlap)} files in both train and val")
        
        print(f"\n✓ No data leakage detected")
    
    def _compute_stats(self):
        """Compute mean/std on TRAINING data only."""
        print("\nComputing mean/std on training set...")
        pixels = []
        
        for fpath, _ in self.train_files:
            img = Image.open(fpath).convert('RGB')
            pixels.append(self._base_transform(img).numpy())
        
        all_imgs = np.stack(pixels)
        self.mean = all_imgs.mean(axis=(0, 2, 3)).tolist()
        self.std = all_imgs.std(axis=(0, 2, 3)).tolist()
        
        print(f"  Mean: {[f'{m:.4f}' for m in self.mean]}")
        print(f"  Std:  {[f'{s:.4f}' for s in self.std]}")
        
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _load_images(self):
        """Load all images with normalization."""
        X_train, y_train = [], []
        X_val, y_val = [], []
        
        for fpath, label in self.train_files:
            img = Image.open(fpath).convert('RGB')
            X_train.append(self.transform(img).numpy())
            y_train.append(label)
        
        for fpath, label in self.val_files:
            img = Image.open(fpath).convert('RGB')
            X_val.append(self.transform(img).numpy())
            y_val.append(label)
        
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_val = np.array(X_val)
        self.y_val = np.array(y_val)
    
    def _shuffle_all(self):
        """Shuffle with fixed seeds."""
        # Shuffle train
        idx = np.arange(len(self.X_train))
        np.random.default_rng(self.seed).shuffle(idx)
        self.X_train = self.X_train[idx]
        self.y_train = self.y_train[idx]
        
        # Shuffle val
        idx = np.arange(len(self.X_val))
        np.random.default_rng(self.seed + 1).shuffle(idx)
        self.X_val = self.X_val[idx]
        self.y_val = self.y_val[idx]
    
    def _print_statistics(self):
        """Print dataset statistics."""
        print(f"\n{'='*60}")
        print("FINAL DATASET STATISTICS (AFTER REDISTRIBUTION)")
        print('='*60)
        
        print(f"\nTraining set: {len(self.X_train)} samples")
        for label in range(3):
            count = (self.y_train == label).sum()
            class_name = [k for k, v in self.class_map.items() if v == label][0]
            print(f"  {class_name} (label {label}): {count} samples")
        
        print(f"\nValidation set: {len(self.X_val)} samples")
        for label in range(3):
            count = (self.y_val == label).sum()
            class_name = [k for k, v in self.class_map.items() if v == label][0]
            print(f"  {class_name} (label {label}): {count} samples")
        
        print('='*60)


class NewDomainDataset:
    """
    Load NEW domain dataset for testing (scalogram_minh).
    Structure: corona/, surface/PD/, void/
    """
    
    def __init__(self, data_path, image_size=64, mean=None, std=None):
        self.data_path = os.path.abspath(data_path)
        self.image_size = image_size
        self.class_map = NEW_CLASS_MAP
        
        # Use provided mean/std (from training data) or default
        if mean is None or std is None:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            self.mean = mean
            self.std = std
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        print(f"\n{'='*60}")
        print("LOADING NEW DOMAIN DATASET (TARGET DOMAIN)")
        print('='*60)
        print(f"Path: {self.data_path}")
        
        self.X_test, self.y_test = self._load_all()
        self._print_statistics()
    
    def _load_all(self):
        """Load all images from new domain."""
        X, y = [], []
        
        # Handle special directory structures
        class_dirs = {
            'corona': ['corona'],
            'surface': ['surface', 'surface/PD'],  # Try both
            'void': ['void']
        }
        
        for class_name, label in self.class_map.items():
            found = False
            for subdir in class_dirs.get(class_name, [class_name]):
                class_path = os.path.join(self.data_path, subdir)
                if os.path.exists(class_path):
                    files = sorted([f for f in os.listdir(class_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    
                    for f in files:
                        img = Image.open(os.path.join(class_path, f)).convert('RGB')
                        X.append(self.transform(img).numpy())
                        y.append(label)
                    
                    found = True
                    break
            
            if not found:
                print(f"Warning: Class {class_name} not found in {self.data_path}")
        
        return np.array(X), np.array(y)
    
    def _print_statistics(self):
        """Print dataset statistics."""
        print(f"\nTest set (NEW DOMAIN): {len(self.X_test)} samples")
        for label in range(3):
            count = (self.y_test == label).sum()
            class_name = [k for k, v in self.class_map.items() if v == label][0]
            print(f"  {class_name} (label {label}): {count} samples")
        print('='*60)


# =============================================================================
# Model
# =============================================================================

def get_model(args):
    """Initialize model based on args."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model == 'protonet':
        model = ProtoNet(use_base_encoder=args.use_base_encoder, device=device)
    elif args.model == 'covamnet':
        model = CovaMNet(device=device)
    elif args.model == 'matchingnet':
        model = MatchingNet(backbone=args.backbone, device=device)
    elif args.model == 'relationnet':
        model = RelationNet(device=device)
    elif args.model == 'siamese':
        model = SiameseNet(device=device)
    elif args.model == 'dn4':
        model = DN4(k_neighbors=3, device=device)
    elif args.model == 'feat':
        model = FEAT(temperature=0.2, device=device)
    elif args.model == 'deepemd':
        model = DeepEMD(device=device)
    elif args.model == 'baseline':
        model = CosineClassifier(temperature=10.0, learnable_temp=True, device=device)
    else:
        model = CosineNet(device=device)
    
    return model.to(device)


# =============================================================================
# Training
# =============================================================================

def train_loop(net, train_loader, val_X, val_y, args):
    """Train with validation-based model selection."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss function
    if args.model == 'relationnet':
        criterion_main = RelationLoss().to(device)
    else:
        criterion_main = ContrastiveLoss().to(device)
    
    # Calculate feature dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.image_size, args.image_size).to(device)
        dummy_feat = net.encoder(dummy)
        feat_dim = dummy_feat.view(1, -1).size(1)
    
    criterion_center = CenterLoss(num_classes=args.way_num, feat_dim=feat_dim, device=device)
    
    optimizer = optim.Adam([
        {'params': net.parameters()},
        {'params': criterion_center.parameters()}
    ], lr=args.lr)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    best_acc = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        net.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}')
        for query, q_labels, support, s_labels in pbar:
            optimizer.zero_grad()
            
            B = query.shape[0]
            C, H, W = query.shape[2], query.shape[3], query.shape[4]
            
            # Use 1 query for training
            support = support.view(B, args.way_num, args.shot_num, C, H, W).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)
            
            scores = net(query, support)
            loss = criterion_main(scores, targets)
            
            if args.lambda_center > 0:
                q_flat = query.view(-1, C, H, W)
                features = net.encoder(q_flat)
                features = F.normalize(features.view(features.size(0), -1), p=2, dim=1)
                loss += args.lambda_center * criterion_center(features, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        scheduler.step()
        
        # Validate
        val_ds = FewshotDataset(val_X, val_y, args.episode_num_val,
                                args.way_num, args.shot_num, 1, args.seed + epoch)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        val_acc = evaluate(net, val_loader, args)
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}')
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_acc": val_acc,
        })
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(args.path_weights, 
                               f'{args.dataset_name}_{args.model}_{args.shot_num}shot_best.pth')
            torch.save(net.state_dict(), path)
            print(f'  → Best model saved ({val_acc:.4f})')
    
    return best_acc


def evaluate(net, loader, args):
    """Compute accuracy on loader."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for query, q_labels, support, s_labels in loader:
            B = query.shape[0]
            C, H, W = query.shape[2], query.shape[3], query.shape[4]
            
            shot_num = support.shape[1] // args.way_num
            support = support.view(B, args.way_num, shot_num, C, H, W).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)
            
            scores = net(query, support)
            preds = scores.argmax(dim=1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    return correct / total if total > 0 else 0


# =============================================================================
# Domain Shift Testing
# =============================================================================

def calculate_p_value(acc, baseline, n):
    """Z-test for proportion significance."""
    from scipy.stats import norm
    if n <= 0:
        return 1.0
    z = (acc - baseline) / np.sqrt(baseline * (1 - baseline) / n)
    return 2 * norm.sf(abs(z))


def test_domain_shift(net, test_X, test_y, shot_num, args):
    """
    Test on NEW domain with K-shot.
    
    Args:
        shot_num: K for K-shot (1, 5, or 10)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensor
    test_X_t = torch.from_numpy(test_X.astype(np.float32))
    test_y_t = torch.from_numpy(test_y).long()
    
    # Create test loader
    test_ds = FewshotDataset(test_X_t, test_y_t, args.test_episodes,
                             args.way_num, shot_num, args.test_query_num, args.seed)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"\n{'='*60}")
    print(f"DOMAIN SHIFT TEST: {shot_num}-shot")
    print(f"Episodes: {args.test_episodes}, Query/class: {args.test_query_num}")
    print('='*60)
    
    net.eval()
    all_preds, all_targets, all_features = [], [], []
    episode_accuracies = []
    
    with torch.no_grad():
        for query, q_labels, support, s_labels in tqdm(test_loader, desc=f'{shot_num}-shot test'):
            B, NQ, C, H, W = query.shape
            
            support = support.view(B, args.way_num, shot_num, C, H, W).to(device)
            query = query.to(device)
            targets = q_labels.view(-1).to(device)
            
            scores = net(query, support)
            preds = scores.argmax(dim=1)
            
            episode_acc = (preds == targets).float().mean().item()
            episode_accuracies.append(episode_acc)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Features for t-SNE
            q_flat = query.view(-1, C, H, W)
            if hasattr(net, 'encoder'):
                feat = net.encoder(q_flat)
                if feat.dim() == 4:
                    feat = nn.functional.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)
                all_features.append(feat.cpu().numpy())
    
    # Metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    episode_accuracies = np.array(episode_accuracies)
    
    acc_mean = episode_accuracies.mean()
    acc_std = episode_accuracies.std()
    acc_worst = episode_accuracies.min()
    acc_best = episode_accuracies.max()
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds,
        labels=list(range(args.way_num)),
        average='macro',
        zero_division=0
    )
    p_val = calculate_p_value(acc_mean, 1.0/args.way_num, len(all_targets))
    
    # Print results
    print(f"\nAccuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Worst: {acc_worst:.4f}, Best: {acc_best:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"p-value: {p_val:.2e}")
    
    # Log to WandB
    wandb.log({
        f"test_{shot_num}shot_accuracy": acc_mean,
        f"test_{shot_num}shot_std": acc_std,
        f"test_{shot_num}shot_f1": f1,
    })
    
    # Plots
    if shot_num == 1:  # Only plot once
        cm_base = os.path.join(args.path_results,
                               f"confusion_matrix_{args.dataset_name}_{args.model}_{shot_num}shot")
        plot_confusion_matrix(all_targets, all_preds, args.way_num, cm_base)
        wandb.log({"confusion_matrix": wandb.Image(f"{cm_base}_2col.png")})
        
        if all_features:
            features = np.vstack(all_features)
            tsne_base = os.path.join(args.path_results,
                                     f"tsne_{args.dataset_name}_{args.model}_{shot_num}shot")
            plot_tsne(features, all_targets, args.way_num, tsne_base)
            wandb.log({"tsne_plot": wandb.Image(f"{tsne_base}_2col.png")})
    
    return acc_mean, acc_std


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seed_func(args.seed)
    os.makedirs(args.path_weights, exist_ok=True)
    os.makedirs(args.path_results, exist_ok=True)
    
    # Load OLD dataset (source domain)
    old_dataset = DomainShiftDataset(
        args.dataset_path,
        val_per_class=args.val_per_class,
        image_size=args.image_size,
        seed=args.seed
    )
    
    if args.mode == 'stats':
        print("\n[Stats mode] Dataset statistics printed above.")
        return
    
    # Convert to tensors
    train_X = torch.from_numpy(old_dataset.X_train.astype(np.float32))
    train_y = torch.from_numpy(old_dataset.y_train).long()
    val_X = torch.from_numpy(old_dataset.X_val.astype(np.float32))
    val_y = torch.from_numpy(old_dataset.y_val).long()
    
    # Initialize WandB
    run_name = f"domain_shift_{args.model}"
    wandb.init(project=args.project, config=vars(args), name=run_name)
    
    # Initialize model
    net = get_model(args)
    print(f"\nModel: {args.model}")
    print(f"Device: {args.device}")
    
    if args.mode == 'train':
        # Test all shot configurations
        for shot_num in [1, 5, 10]:
            args.shot_num = shot_num
            print(f"\n{'#'*60}")
            print(f"TRAINING {shot_num}-SHOT")
            print('#'*60)
            
            # Re-initialize model for each shot
            net = get_model(args)
            
            # Create train loader
            train_ds = FewshotDataset(train_X, train_y, args.episode_num_train,
                                      args.way_num, shot_num, 1, args.seed)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            
            # Train
            best_acc = train_loop(net, train_loader, val_X, val_y, args)
            print(f"\n{shot_num}-shot training complete. Best val acc: {best_acc:.4f}")
            
            # Load best model
            path = os.path.join(args.path_weights,
                               f'{args.dataset_name}_{args.model}_{shot_num}shot_best.pth')
            net.load_state_dict(torch.load(path))
            
            # Load NEW domain dataset (target domain)
            new_dataset = NewDomainDataset(
                args.test_dataset_path,
                image_size=args.image_size,
                mean=old_dataset.mean,
                std=old_dataset.std
            )
            
            # Test on new domain
            test_domain_shift(net, new_dataset.X_test, new_dataset.y_test, shot_num, args)
    
    wandb.finish()
    print("\n✓ Domain shift experiment complete!")


if __name__ == '__main__':
    main()
