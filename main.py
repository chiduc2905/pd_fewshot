"""PD Scalogram Few-Shot Learning - Training and Evaluation."""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import wandb

# FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. Run 'pip install thop' for FLOPs calculation.")

from dataset import load_dataset
from dataloader.dataloader import FewshotDataset
from function.function import ContrastiveLoss, RelationLoss, SiameseLoss, CenterLoss, TripletLoss, seed_func, plot_confusion_matrix, plot_tsne, plot_model_comparison_bar, plot_training_curves
from net.cosine import CosineNet  # User's custom cosine similarity model
from net.cosine_classifier import CosineClassifier  # Baseline++ (Chen et al. ICLR 2019)
from net.protonet import ProtoNet
from net.covamnet import CovaMNet
from net.matchingnet import MatchingNet
from net.relationnet import RelationNet
# New models
from net.siamesenet import SiameseNetFast as SiameseNet
from net.dn4 import DN4Fast as DN4
from net.feat import FEAT
from net.deepemd import DeepEMDSimple as DeepEMD


# =============================================================================
# Configuration
# =============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PD Scalogram Few-shot Learning')
    
    # Paths
    parser.add_argument('--dataset_path', type=str, default='/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/scalogram_official')
    parser.add_argument('--path_weights', type=str, default='checkpoints/')
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--weights', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--dataset_name', type=str, default='scalogram',
                        help='Dataset name for checkpoint naming (e.g., original, augmented)')
    
    # Model
    parser.add_argument('--model', type=str, default='covamnet', 
                        choices=['cosine', 'baseline', 'protonet', 'covamnet', 'matchingnet', 'relationnet',
                                 'siamese', 'dn4', 'feat', 'deepemd'])
    parser.add_argument('--use_base_encoder', action='store_true',
                        help='Use Conv64F_Encoder (GroupNorm) for ProtoNet instead of paper-specific encoder')
    parser.add_argument('--backbone', type=str, default='conv64f',
                        choices=['conv64f', 'resnet12', 'resnet18'],
                        help='Backbone for MatchingNet: conv64f (paper, 1024 dim), resnet12 (512 dim), or resnet18 (512 dim)')

    
    # Few-shot settings
    parser.add_argument('--way_num', type=int, default=4)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--query_num', type=int, default=5, help='Queries per class (same for train/val/test)')
    parser.add_argument('--image_size', type=int, default=128, choices=[64, 84, 128],
                        help='Input image size: 64 (required for conv64f) or 84 (required for resnet12/18)')
    
    # Training
    parser.add_argument('--training_samples', type=int, default=None, 
                        help='Total training samples (e.g. 30=10/class)')
    parser.add_argument('--episode_num_train', type=int, default=100)
    parser.add_argument('--episode_num_val', type=int, default=150)
    parser.add_argument('--episode_num_test', type=int, default=150)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--eta_min', type=float, default=1e-5, help='Min LR for CosineAnnealingLR')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42)
    # Device
    # parser.add_argument('--device', type=str, 
    #                     default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss
    parser.add_argument('--loss', type=str, default='contrastive', 
                        choices=['contrastive', 'triplet'],
                        help='Loss function: contrastive (default) or triplet')
    parser.add_argument('--temp', type=float, default=0.01,
                        help='Temperature for SupCon loss (default: 0.01)')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Margin for Triplet loss (default: 0.1)')
    
    # Center Loss
    parser.add_argument('--lambda_center', type=float, default=0.0, 
                        help='Weight for Center Loss (default: 0.0, disabled)')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_last_checkpoint', action='store_true', default=False,
                        help='Use final epoch checkpoint for test (default: use best val)')
    
    # WandB
    parser.add_argument('--project', type=str, default='prpd',
                        help='WandB project name')
    
    return parser.parse_args()


def get_model(args):
    """Initialize model based on args."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = (device.type == 'cuda')
    
    if args.model == 'protonet':
        # Only ProtoNet supports encoder selection
        model = ProtoNet(use_base_encoder=args.use_base_encoder, device=device)
    elif args.model == 'covamnet':
        model = CovaMNet(device=device)
    elif args.model == 'matchingnet':
        # MatchingNet: supports conv64f (paper) or resnet12 backbone
        model = MatchingNet(backbone=args.backbone, device=device)
    elif args.model == 'relationnet':
        # RelationNet: paper-specific encoder only (RelationBlock expects 4x4 features)
        model = RelationNet(device=device)
    elif args.model == 'siamese':
        # Siamese Network with learned distance
        model = SiameseNet(device=device)
    elif args.model == 'dn4':
        # DN4: Local descriptors with k-NN
        model = DN4(k_neighbors=3, device=device)
    elif args.model == 'feat':
        # FEAT: Transformer-based embedding adaptation
        model = FEAT(temperature=0.2, device=device)
    elif args.model == 'deepemd':
        # DeepEMD: Earth Mover's Distance (simplified version)
        model = DeepEMD(device=device)
    elif args.model == 'baseline':
        # Baseline++ (Chen et al. ICLR 2019) - Cosine with learnable temperature
        model = CosineClassifier(temperature=10.0, learnable_temp=True, device=device)
    else:  # cosine (user's original)
        model = CosineNet(device=device)
    
    return model.to(device)


# =============================================================================
# Training
# =============================================================================

def train_loop(net, train_loader, val_X, val_y, args):
    """Train with validation-based model selection.
    
    Args:
        net: Model to train
        train_loader: Training data loader
        val_X: Validation images tensor
        val_y: Validation labels tensor
        args: Training arguments
    
    Returns:
        best_acc: Best validation accuracy
        history: Dict with training curves (train_acc, val_acc, train_loss, val_loss)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss functions - Auto-select based on model
    if args.model == 'relationnet':
        # RelationNet uses MSE loss (paper-specific)
        criterion_main = RelationLoss().to(device)
    elif args.model == 'siamese':
        # SiameseNet uses true contrastive loss with margin (Koch et al. 2015)
        # Applied to embeddings from support+query pairs
        criterion_main = SiameseLoss(margin=1.0).to(device)
    elif args.loss == 'triplet':
        criterion_main = TripletLoss(margin=args.margin).to(device)
    else:
        # Default: ContrastiveLoss (CrossEntropyLoss)
        criterion_main = ContrastiveLoss().to(device)
        
    # Calculate feature dimension dynamically
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 64, 64).to(device)
        dummy_feat = net.encoder(dummy_input)
        feat_dim = dummy_feat.view(1, -1).size(1)
        
    criterion_center = CenterLoss(num_classes=args.way_num, feat_dim=feat_dim, device=device)
    
    # Optimizer with weight decay (AdamW for proper decoupled weight decay)
    optimizer = optim.AdamW([
        {'params': net.parameters()},
        {'params': criterion_center.parameters()}
    ], lr=args.lr, weight_decay=args.weight_decay)
    
    # CosineAnnealingLR Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs,
        eta_min=args.eta_min
    )
    
    # Training history for plotting
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
    best_acc = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        net.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}')
        for query, q_labels, support, s_labels in pbar:
            optimizer.zero_grad()  # Reset gradients!
            
            B = query.shape[0]
            C, H, W = query.shape[2], query.shape[3], query.shape[4]
            
            support = support.view(B, args.way_num, args.shot_num, C, H, W).to(args.device)
            query = query.to(args.device)
            targets = q_labels.view(-1).to(args.device)
            
            # Forward Main
            scores = net(query, support)
            
            # 1. Main Loss
            if args.loss == 'triplet' or args.model == 'siamese':
                # For metric learning losses (Triplet, Siamese Contrastive),
                # we need to combine support and query features for pair-wise training
                
                # Extract query features
                q_flat = query.view(-1, C, H, W)
                q_feats = net.encoder(q_flat)
                if q_feats.dim() == 4:  # Feature maps
                    q_feats = F.adaptive_avg_pool2d(q_feats, 1)
                q_feats = q_feats.view(q_feats.size(0), -1)
                q_targets = targets
                
                # Extract support features
                s_flat = support.view(-1, C, H, W)
                s_feats = net.encoder(s_flat)
                if s_feats.dim() == 4:  # Feature maps
                    s_feats = F.adaptive_avg_pool2d(s_feats, 1)
                s_feats = s_feats.view(s_feats.size(0), -1)
                s_targets = s_labels.view(-1).to(args.device)
                
                # Concatenate all embeddings and labels
                all_feats = torch.cat([q_feats, s_feats], dim=0)
                all_targets = torch.cat([q_targets, s_targets], dim=0)
                
                # Apply contrastive/triplet loss on embeddings
                loss_main = criterion_main(all_feats, all_targets)
                
            else:
                # CE-based losses (ContrastiveLoss, RelationLoss) use scores
                loss_main = criterion_main(scores, targets)
            
            # 2. Center Loss
            # Extract features from query images
            q_flat = query.view(-1, C, H, W)
            features = net.encoder(q_flat)
            features = features.view(features.size(0), -1) # Flatten to (N, feat_dim)
            
            # Normalize features for stability (Center Loss works best with normalized features)
            features = F.normalize(features, p=2, dim=1)
            
            loss_center = criterion_center(features, targets)
            
            # Total Loss
            loss = loss_main + args.lambda_center * loss_center
            
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr:.2e}')
        
        scheduler.step()
        
        # Evaluate on training set (same episodes used for training)
        train_acc, _ = evaluate(net, train_loader, args)
        
        # Validate - Create new validation dataset each epoch with seed+epoch
        # This ensures different episodes per epoch while maintaining reproducibility
        val_ds = FewshotDataset(val_X, val_y, args.episode_num_val,
                                args.way_num, args.shot_num, args.query_num, args.seed + epoch)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        
        val_acc, val_loss = evaluate(net, val_loader, args, criterion_main, criterion_center)
        avg_loss = total_loss / len(train_loader)
        
        # Track history for plotting
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss if val_loss else 0.0)
        
        # Calculate train-val gap (indicator of overfitting)
        train_val_gap = train_acc - val_acc
        
        print(f'Epoch {epoch}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f} (gap={train_val_gap:+.4f})')
        
        # Log to WandB with grouped metrics for combined charts
        wandb.log({
            "epoch": epoch,
            # Grouped for combined charts
            "loss/train": avg_loss,
            "loss/val": val_loss,
            "accuracy/train": train_acc,
            "accuracy/val": val_acc,
            # Individual metrics (for backward compatibility)
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_val_gap": train_val_gap,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            # Naming with dataset: dataset_model_samples_shot
            samples_suffix = f'{args.training_samples}samples' if args.training_samples else 'all'
            model_filename = f'{args.dataset_name}_{args.model}_{samples_suffix}_{args.shot_num}shot_best.pth'
            path = os.path.join(args.path_weights, model_filename)
            torch.save(net.state_dict(), path)
            print(f'  → Best model saved ({val_acc:.4f})')
            wandb.run.summary["best_val_acc"] = best_acc
            

    
    # Plot training curves
    samples_str = f"{args.training_samples}samples" if args.training_samples else "allsamples"
    curves_path = os.path.join(args.path_results, 
                               f"training_{args.dataset_name}_{args.model}_{samples_str}_{args.shot_num}shot")
    plot_training_curves(history, curves_path)
    
    # Log to WandB
    if os.path.exists(f"{curves_path}_curves.png"):
        wandb.log({"training_curves": wandb.Image(f"{curves_path}_curves.png")})
    
    # Save final epoch model (for proper protocol)
    samples_suffix = f'{args.training_samples}samples' if args.training_samples else 'all'
    final_model_filename = f'{args.dataset_name}_{args.model}_{samples_suffix}_{args.shot_num}shot_final.pth'
    final_path = os.path.join(args.path_weights, final_model_filename)
    torch.save(net.state_dict(), final_path)
    print(f'Final model saved: {final_path}')
    
    return best_acc, history


def evaluate(net, loader, args, criterion_main=None, criterion_center=None):
    """Compute accuracy and optionally loss on loader."""
    net.eval()
    correct, total = 0, 0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for query, q_labels, support, s_labels in loader:
            B = query.shape[0]
            C, H, W = query.shape[2], query.shape[3], query.shape[4]
            
            # Infer shot_num from support shape
            shot_num = support.shape[1] // args.way_num
            
            support = support.view(B, args.way_num, shot_num, C, H, W).to(args.device)
            query = query.to(args.device)
            targets = q_labels.view(-1).to(args.device)
            
            scores = net(query, support)
            preds = scores.argmax(dim=1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            # Calculate loss if criterion provided
            if criterion_main is not None:
                if args.loss == 'triplet':
                    # Extract query features
                    q_flat = query.view(-1, C, H, W)
                    q_feats = net.encoder(q_flat)
                    q_feats = q_feats.view(q_feats.size(0), -1)
                    
                    # Extract support features
                    s_flat = support.view(-1, C, H, W)
                    s_feats = net.encoder(s_flat)
                    s_feats = s_feats.view(s_feats.size(0), -1)
                    s_targets = s_labels.view(-1).to(args.device)
                    
                    all_feats = torch.cat([q_feats, s_feats], dim=0)
                    all_targets = torch.cat([targets, s_targets], dim=0)
                    all_feats = F.normalize(all_feats, p=2, dim=1)
                    
                    loss_main = criterion_main(all_feats, all_targets)
                else:
                    loss_main = criterion_main(scores, targets)
                
                # Center loss
                if criterion_center is not None:
                    q_flat = query.view(-1, C, H, W)
                    features = net.encoder(q_flat)
                    features = features.view(features.size(0), -1)
                    features = F.normalize(features, p=2, dim=1)
                    loss_center = criterion_center(features, targets)
                    loss = loss_main + args.lambda_center * loss_center
                else:
                    loss = loss_main
                
                total_loss += loss.item()
                num_batches += 1
    
    acc = correct / total if total > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else None
    
    return acc, avg_loss


# =============================================================================
# Testing
# =============================================================================

def calculate_p_value(acc, baseline, n):
    """Z-test for proportion significance."""
    from scipy.stats import norm
    if n <= 0:
        return 1.0
    z = (acc - baseline) / np.sqrt(baseline * (1 - baseline) / n)
    return 2 * norm.sf(abs(z))


def test_final(net, loader, args):
    """
    Final evaluation with detailed metrics.
    
    Metrics:
    - Accuracy: mean ± std (worst-case, best-case)
    - Precision, Recall, F1, p-value
    - Inference time per episode
    
    Plots: Confusion Matrix, t-SNE
    """
    import time
    
    num_episodes = len(loader)
    print(f"\n{'='*60}")
    print(f"Final Test: {args.dataset_name}/{args.model} | {args.shot_num}-shot")
    print(f"{num_episodes} episodes × {args.way_num} classes × {args.query_num} query = {num_episodes * args.way_num * args.query_num} predictions")
    print('='*60)
    
    net.eval()
    all_preds, all_targets, all_features = [], [], []
    
    # Track per-episode metrics
    episode_accuracies = []
    episode_times = []
    
    with torch.no_grad():
        for query, q_labels, support, s_labels in tqdm(loader, desc='Testing'):
            # Start timing
            start_time = time.perf_counter()
            
            B, NQ, C, H, W = query.shape
            
            # Use same shot_num as training
            support = support.view(B, args.way_num, args.shot_num, C, H, W).to(args.device)
            query = query.to(args.device)
            targets = q_labels.view(-1).to(args.device)
            
            scores = net(query, support)
            preds = scores.argmax(dim=1)
            
            # End timing (inference only, exclude data prep for consistency)
            end_time = time.perf_counter()
            episode_time_ms = (end_time - start_time) * 1000
            episode_times.append(episode_time_ms)
            
            # Calculate per-episode accuracy
            episode_correct = (preds == targets).float().mean().item()
            episode_accuracies.append(episode_correct)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Extract features for t-SNE
            q_flat = query.view(-1, C, H, W)
            if hasattr(net, 'encoder'):
                feat = net.encoder(q_flat)
                # Handle both 4D feature maps and 2D flattened features
                if feat.dim() == 4:  # (B, C, H, W) - default encoder
                    feat = nn.functional.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)
                elif feat.dim() == 2:  # (B, feat_dim) - paper encoder, already flattened
                    pass  # Already flattened
                all_features.append(feat.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    episode_accuracies = np.array(episode_accuracies)
    episode_times = np.array(episode_times)
    
    # =========================================================================
    # Accuracy Metrics: mean ± std (worst, best)
    # =========================================================================
    acc_mean = episode_accuracies.mean()
    acc_std = episode_accuracies.std()
    acc_worst = episode_accuracies.min()
    acc_best = episode_accuracies.max()
    
    # =========================================================================
    # Inference Time Metrics
    # =========================================================================
    time_mean = episode_times.mean()
    time_std = episode_times.std()
    time_min = episode_times.min()
    time_max = episode_times.max()
    
    # =========================================================================
    # Classification Metrics
    # =========================================================================
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, 
        labels=list(range(args.way_num)),
        average='macro', 
        zero_division=0
    )
    p_val = calculate_p_value(acc_mean, 1.0/args.way_num, len(all_targets))
    
    # =========================================================================
    # Print Results
    # =========================================================================
    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print('='*60)
    print(f"  Mean Accuracy : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Worst-case    : {acc_worst:.4f}")
    print(f"  Best-case     : {acc_best:.4f}")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  p-value       : {p_val:.2e}")
    
    print(f"\n{'='*60}")
    print("INFERENCE TIME (per episode)")
    print('='*60)
    print(f"  Mean Time     : {time_mean:.2f} ± {time_std:.2f} ms")
    print(f"  Min Time      : {time_min:.2f} ms")
    print(f"  Max Time      : {time_max:.2f} ms")
    
    # =========================================================================
    # Log to WandB
    # =========================================================================
    wandb.log({
        # Accuracy metrics
        "test_accuracy_mean": acc_mean,
        "test_accuracy_std": acc_std,
        "test_accuracy_worst": acc_worst,
        "test_accuracy_best": acc_best,
        # Legacy: keep for compatibility
        "test_accuracy": acc_mean,
        # Classification metrics
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_p_value": p_val,
        # Inference time metrics
        "inference_time_mean_ms": time_mean,
        "inference_time_std_ms": time_std,
        "inference_time_min_ms": time_min,
        "inference_time_max_ms": time_max,
    })
    
    # Update WandB summary for easy access
    wandb.run.summary["test_accuracy_mean"] = acc_mean
    wandb.run.summary["test_accuracy_std"] = acc_std
    wandb.run.summary["test_accuracy_worst"] = acc_worst
    wandb.run.summary["inference_time_mean_ms"] = time_mean
    
    # =========================================================================
    # Plots (include dataset_name in filenames)
    # =========================================================================
    samples_str = f"_{args.training_samples}samples" if args.training_samples else "_allsamples"
    
    # Confusion Matrix
    cm_base = os.path.join(args.path_results, 
                           f"confusion_matrix_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot")
    plot_confusion_matrix(all_targets, all_preds, args.way_num, cm_base)
    wandb.log({"confusion_matrix": wandb.Image(f"{cm_base}_2col.png")})
    
    # t-SNE
    if all_features:
        features = np.vstack(all_features)
        tsne_base = os.path.join(args.path_results, 
                                 f"tsne_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot")
        plot_tsne(features, all_targets, args.way_num, tsne_base)
        wandb.log({"tsne_plot": wandb.Image(f"{tsne_base}_2col.png")})
    
    print(f"\nResults logged to WandB and plots saved to {args.path_results}")
    
    # =========================================================================
    # Save results to text file (include dataset_name)
    # =========================================================================
    txt_path = os.path.join(args.path_results, 
                            f"results_{args.dataset_name}_{args.model}_{samples_str.strip('_')}_{args.shot_num}shot.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Shot: {args.shot_num}\n")
        f.write(f"Loss: {args.loss}\n")
        f.write(f"Lambda Center: {args.lambda_center}\n")
        f.write(f"Training Samples: {args.training_samples if args.training_samples else 'All'}\n")
        f.write(f"Test Episodes: {num_episodes}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy      : {acc_mean:.4f} ± {acc_std:.4f}\n")
        f.write(f"Worst-case    : {acc_worst:.4f}\n")
        f.write(f"Best-case     : {acc_best:.4f}\n")
        f.write(f"Precision     : {prec:.4f}\n")
        f.write(f"Recall        : {rec:.4f}\n")
        f.write(f"F1-Score      : {f1:.4f}\n")
        f.write(f"p-value       : {p_val:.2e}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Inference Time: {time_mean:.2f} ± {time_std:.2f} ms/episode\n")
        f.write(f"Min Time      : {time_min:.2f} ms\n")
        f.write(f"Max Time      : {time_max:.2f} ms\n")
    
    print(f"Results saved to {txt_path}")


def log_model_comparison_bar(args):
    """
    Read all results files and generate model comparison bar chart.
    Log to wandb.
    """
    import re
    
    samples_str = f"{args.training_samples}samples" if args.training_samples else "allsamples"
    results_dir = args.path_results
    
    # Model name mapping for display
    model_display_names = {
        'cosine': 'Cosine Classifier',
        'baseline': 'Baseline++',
        'protonet': 'ProtoNet',
        'covamnet': 'CovaMNet',
        'matchingnet': 'MatchingNet',
        'relationnet': 'RelationNet',
        'siamese': 'SiameseNet',
        'dn4': 'DN4',
        'feat': 'FEAT',
        'deepemd': 'DeepEMD'
    }
    
    # Collect results
    model_results = {}
    models = ['cosine', 'baseline', 'protonet', 'covamnet', 'matchingnet', 'relationnet',
              'siamese', 'dn4', 'feat', 'deepemd']
    
    for model in models:
        display_name = model_display_names.get(model, model)
        model_results[display_name] = {'1shot': None, '5shot': None}
        
        for shot in [1, 5]:
            result_file = os.path.join(results_dir, 
                f"results_{model}_{samples_str}_{shot}shot.txt")
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    content = f.read()
                    # Parse accuracy
                    match = re.search(r'Accuracy\s*:\s*([\d.]+)', content)
                    if match:
                        acc = float(match.group(1))
                        model_results[display_name][f'{shot}shot'] = acc
    
    # Remove models with missing data
    model_results = {k: v for k, v in model_results.items() 
                     if v['1shot'] is not None and v['5shot'] is not None}
    
    if len(model_results) == 0:
        print("No complete results found for model comparison chart.")
        return
    
    # Generate chart
    training_samples = args.training_samples if args.training_samples else 'All'
    save_path = os.path.join(results_dir, f"model_comparison_{samples_str}.png")
    
    fig = plot_model_comparison_bar(model_results, training_samples, save_path)
    
    # Log to wandb
    wandb.log({"model_comparison_bar": wandb.Image(save_path)})
    
    # Also log as wandb bar chart table
    data = []
    for model_name, accs in model_results.items():
        data.append([model_name, "1 Shot", accs['1shot'] * 100])
        data.append([model_name, "5 Shot", accs['5shot'] * 100])
    
    table = wandb.Table(data=data, columns=["Model", "Shot", "Accuracy (%)"])
    wandb.log({"model_comparison_table": table})
    
    print(f"Model comparison chart saved to {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    
    # Validate: conv64f requires 64x64, resnet12/18 require 84x84
    if args.backbone == 'conv64f' and args.image_size != 64:
        print(f"Error: conv64f backbone requires 64x64 image size, got {args.image_size}")
        print("Use --backbone resnet12 or --backbone resnet18 for 84x84.")
        return
    if args.backbone in ['resnet12', 'resnet18'] and args.image_size != 84:
        print(f"Error: {args.backbone} backbone requires 84x84 image size, got {args.image_size}")
        print("Use --backbone conv64f for 64x64.")
        return
    
    # Set defaults based on shot_num
    if args.num_epochs is None:
        args.num_epochs = 100 if args.shot_num == 1 else 70

    # Auto-detect device if not specified (handled by argparse default)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Determine encoder info for logging
    if args.model == 'protonet':
        if args.use_base_encoder:
            encoder_info = "Conv64F_Encoder (GroupNorm) [--use_base_encoder]"
        else:
            encoder_info = "Conv64F_Paper_Encoder (BatchNorm) [default]"
    elif args.model == 'covamnet' or args.model == 'cosine':
        encoder_info = "Conv64F_Encoder (GroupNorm)"
    elif args.model == 'matchingnet':
        if args.backbone == 'resnet12':
            encoder_info = "ResNet12Encoder (BatchNorm, 512 dim) [--backbone resnet12]"
        elif args.backbone == 'resnet18':
            encoder_info = "ResNet18Encoder (BatchNorm, 512 dim) [--backbone resnet18]"
        else:
            encoder_info = "MatchingNetEncoder (BatchNorm, 1024 dim) [default]"
    elif args.model == 'relationnet':
        encoder_info = "RelationNetEncoder (BatchNorm, paper-only)"
    else:
        encoder_info = "unknown"
    
    print(f"Config: {args.model} | {args.shot_num}-shot | {args.num_epochs} epochs | Device: {args.device}")
    print(f"Encoder: {encoder_info}")
    
    # Initialize WandB with a descriptive run name
    samples_str = f"{args.training_samples}samples" if args.training_samples else "all"
    # Simplified run name: model_samples_shot (without contrastive/lambda)
    run_name = f"{args.model}_{samples_str}_{args.shot_num}shot"
    
    # Model metric/distance types (what similarity measure each model uses)
    MODEL_METRICS = {
        'cosine': 'Cosine Similarity',
        'baseline': 'Scaled Cosine Similarity (learnable temperature)',
        'protonet': 'Squared Euclidean Distance',
        'covamnet': 'Covariance Metric (distribution-based)',
        'matchingnet': 'Cosine Similarity + Attention LSTM',
        'relationnet': 'Learned Relation Score (CNN)',
        'siamese': 'Learned Distance (MLP)',
        'dn4': 'Local Descriptor k-NN (Cosine)',
        'feat': 'Transformer-adapted Euclidean Distance',
        'deepemd': 'Earth Mover\'s Distance (EMD)'
    }
    
    # Add metric info to config
    config = vars(args).copy()
    config['distance_metric'] = MODEL_METRICS.get(args.model, 'unknown')
    config['encoder_type'] = encoder_info
    
    wandb.init(project=args.project, config=config, name=run_name, group=run_name, job_type=args.mode)
    
    seed_func(args.seed)
    
    # Log model parameters and FLOPs after model is created
    def calculate_flops(model, input_size=(3, 64, 64), device='cuda'):
        """
        Calculate FLOPs/MACs for the encoder part of the model.
        
        Args:
            model: The few-shot model with encoder attribute
            input_size: Input tensor size (C, H, W)
            device: Device to run calculation on
            
        Returns:
            tuple: (macs, params) or (None, None) if calculation fails
        """
        if not THOP_AVAILABLE:
            return None, None
            
        try:
            # Create dummy input
            dummy_input = torch.randn(1, *input_size).to(device)
            
            # Calculate FLOPs for encoder only (feature extraction part)
            if hasattr(model, 'encoder'):
                encoder = model.encoder
                macs, params = profile(encoder, inputs=(dummy_input,), verbose=False)
                return macs, params
            else:
                # Fallback: try to profile entire model forward with dummy support
                return None, None
        except Exception as e:
            print(f"Warning: Could not calculate FLOPs: {e}")
            return None, None
    
    def log_model_parameters(model, model_name, device='cuda', image_size=64):
        """Log model parameters and FLOPs to wandb."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Calculate FLOPs
        macs, flops_params = calculate_flops(model, input_size=(3, image_size, image_size), device=device)
        
        # Convert to readable format
        if macs is not None:
            macs_readable, params_readable = clever_format([macs, flops_params], "%.2f")
            flops = macs * 2  # 1 MAC ≈ 2 FLOPs
            flops_readable = clever_format([flops], "%.2f")[0]
        else:
            macs_readable = "N/A"
            flops_readable = "N/A"
            flops = None
        
        # Log to WandB
        log_dict = {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/non_trainable_parameters": non_trainable_params,
        }
        
        if macs is not None:
            log_dict["model/macs"] = macs
            log_dict["model/flops"] = flops
            log_dict["model/macs_readable"] = macs_readable
            log_dict["model/flops_readable"] = flops_readable
        
        wandb.log(log_dict)
        
        # Also update run summary for easy access
        wandb.run.summary["model_total_params"] = total_params
        wandb.run.summary["model_trainable_params"] = trainable_params
        if macs is not None:
            wandb.run.summary["model_macs"] = macs
            wandb.run.summary["model_flops"] = flops
            wandb.run.summary["model_macs_readable"] = macs_readable
            wandb.run.summary["model_flops_readable"] = flops_readable
        
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print(f"MACs (encoder): {macs_readable}")
        print(f"FLOPs (encoder): {flops_readable}")
        print(f"{'='*50}\n")
    os.makedirs(args.path_weights, exist_ok=True)
    os.makedirs(args.path_results, exist_ok=True)
    
    # Load dataset (auto-detects pre-split or auto-split structure)
    dataset = load_dataset(args.dataset_path, image_size=args.image_size)
    
    def to_tensor(X, y):
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y).long()
        return X, y
    
    train_X, train_y = to_tensor(dataset.X_train, dataset.y_train)
    val_X, val_y = to_tensor(dataset.X_val, dataset.y_val)
    test_X, test_y = to_tensor(dataset.X_test, dataset.y_test)
    
    # Limit training samples if specified
    if args.training_samples:
        per_class = args.training_samples // args.way_num
        X_list, y_list = [], []
        
        for c in range(args.way_num):
            idx = (train_y == c).nonzero(as_tuple=True)[0]
            if len(idx) < per_class:
                raise ValueError(f"Class {c}: need {per_class}, have {len(idx)}")
            
            g = torch.Generator().manual_seed(args.seed)
            perm = torch.randperm(len(idx), generator=g)[:per_class]
            X_list.append(train_X[idx[perm]])
            y_list.append(train_y[idx[perm]])
        
        train_X = torch.cat(X_list)
        train_y = torch.cat(y_list)
        print(f"Using {args.training_samples} training samples ({per_class}/class)")
    
    # Create data loaders
    # Training: Fixed seed, shuffled loader for varied batch order
    train_ds = FewshotDataset(train_X, train_y, args.episode_num_train,
                              args.way_num, args.shot_num, 1, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
    # Validation: Will be recreated each epoch with seed+epoch in train_loop
    # This ensures different episodes per epoch but reproducibility across program runs
    
    # Test: Fixed seed ensures identical episodes across all runs
    test_ds = FewshotDataset(test_X, test_y, args.episode_num_test,
                             args.way_num, args.shot_num, args.query_num, args.seed)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Initialize Model
    net = get_model(args)
    
    # Log model parameters to wandb
    log_model_parameters(net, args.model, device=args.device, image_size=args.image_size)
    
    if args.mode == 'train':
        best_acc, history = train_loop(net, train_loader, val_X, val_y, args)
        
        # Load model for testing
        samples_suffix = f'{args.training_samples}samples' if args.training_samples else 'all'
        
        if args.use_last_checkpoint:
            # Proper protocol: use final epoch checkpoint
            path = os.path.join(args.path_weights, f'{args.dataset_name}_{args.model}_{samples_suffix}_{args.shot_num}shot_final.pth')
            print(f'Testing with FINAL checkpoint (epoch {args.num_epochs}): {path}')
        else:
            # Legacy: use best validation checkpoint
            path = os.path.join(args.path_weights, f'{args.dataset_name}_{args.model}_{samples_suffix}_{args.shot_num}shot_best.pth')
            print(f'Testing with BEST val checkpoint: {path}')
        
        net.load_state_dict(torch.load(path))
        test_final(net, test_loader, args)
        
    else:  # Test only
        if args.weights:
            net.load_state_dict(torch.load(args.weights))
            test_final(net, test_loader, args)
        else:
            print("Error: Please specify --weights for test mode")


if __name__ == '__main__':
    main()
