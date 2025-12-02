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

from dataset import PDScalogram
from dataloader.dataloader import FewshotDataset
from function.function import ContrastiveLoss, CenterLoss, SupConLoss, TripletLoss, seed_func, plot_confusion_matrix, plot_tsne
from net.cosine import CosineNet
from net.protonet import ProtoNet
from net.covamnet import CovaMNet


# =============================================================================
# Configuration
# =============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PD Scalogram Few-shot Learning')
    
    # Paths
    parser.add_argument('--dataset_path', type=str, default='./scalogram_images/')
    parser.add_argument('--path_weights', type=str, default='checkpoints/')
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--weights', type=str, default=None, help='Checkpoint for testing')
    
    # Model
    parser.add_argument('--model', type=str, default='covamnet', 
                        choices=['cosine', 'protonet', 'covamnet'])

    
    # Few-shot settings
    parser.add_argument('--way_num', type=int, default=3)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--query_num', type=int, default=1, help='Queries per class per episode')
    
    # Training
    parser.add_argument('--training_samples', type=int, default=None, 
                        help='Total training samples (e.g. 30=10/class)')
    parser.add_argument('--episode_num_train', type=int, default=100)
    parser.add_argument('--episode_num_val', type=int, default=75)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    # Device
    # parser.add_argument('--device', type=str, 
    #                     default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss
    parser.add_argument('--loss', type=str, default='contrastive', 
                        choices=['contrastive', 'supcon', 'triplet'],
                        help='Loss function: contrastive (default), supcon, or triplet')
    parser.add_argument('--temp', type=float, default=0.01,
                        help='Temperature for SupCon loss (default: 0.01)')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Margin for Triplet loss (default: 0.1)')
    
    # Center Loss
    parser.add_argument('--lambda_center', type=float, default=0.1, 
                        help='Weight for Center Loss (default: 0.1)')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    return parser.parse_args()


def get_model(args):
    """Initialize model based on args."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = (device.type == 'cuda')
    
    if args.model == 'covamnet':
        model = CovaMNet(device=device)
    elif args.model == 'protonet':
        model = ProtoNet(device=device)
    else:  # cosine
        model = CosineNet(device=device)
    
    return model.to(device)


# =============================================================================
# Training
# =============================================================================

def train_loop(net, train_loader, val_loader, args):
    """Train with validation-based model selection."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss functions
    if args.loss == 'supcon':
        criterion_main = SupConLoss(temperature=args.temp).to(device)
    elif args.loss == 'triplet':
        criterion_main = TripletLoss(margin=args.margin).to(device)
    else:
        criterion_main = ContrastiveLoss().to(device)
        
    # Calculate feature dimension dynamically
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 64, 64).to(device)
        dummy_feat = net.encoder(dummy_input)
        feat_dim = dummy_feat.view(1, -1).size(1)
        
<<<<<<< HEAD
    criterion_center = CenterLoss(num_classes=args.way_num, feat_dim=feat_dim, device=device)
=======
    criterion_center = CenterLoss(num_classes=args.way_num, feat_dim=feat_dim, device=args.device)
>>>>>>> 7bcb50ced6163ac4d72da6a057bd3db392a60616
    
    # Optimizer (optimize both model and center loss parameters)
    optimizer = optim.Adam([
        {'params': net.parameters()},
        {'params': criterion_center.parameters()}
    ], lr=args.lr)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    best_acc = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        net.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}')
        for query, q_labels, support, s_labels in pbar:
            B = query.shape[0]
            C, H, W = query.shape[2], query.shape[3], query.shape[4]
            
            support = support.view(B, args.way_num, args.shot_num, C, H, W).to(args.device)
            query = query.to(args.device)
            targets = q_labels.view(-1).to(args.device)
            
            optimizer.zero_grad()
            scores = net(query, support)
            
            # 1. Main Loss (Contrastive, SupCon, or Triplet)
            # 1. Main Loss (Contrastive, SupCon, or Triplet)
            if args.loss in ['supcon', 'triplet']:
                # For metric learning losses, we need to combine support and query features
                # to ensure we have enough positive pairs (at least 1 support + 1 query per class)
                
                # Extract query features
                q_flat = query.view(-1, C, H, W)
                q_feats = net.encoder(q_flat)
                q_feats = q_feats.view(q_feats.size(0), -1)
                q_targets = targets
                
                # Extract support features
                # support shape: (B, Way, Shot, C, H, W) -> (B*Way*Shot, C, H, W)
                s_flat = support.view(-1, C, H, W)
                s_feats = net.encoder(s_flat)
                s_feats = s_feats.view(s_feats.size(0), -1)
                
                # Create support targets
                # s_labels shape: (B, Way, Shot) -> (B*Way*Shot)
                s_targets = s_labels.view(-1).to(args.device)
                
                # Concatenate
                all_feats = torch.cat([q_feats, s_feats], dim=0)
                all_targets = torch.cat([q_targets, s_targets], dim=0)
                
                # Normalize features for stability
                all_feats = F.normalize(all_feats, p=2, dim=1)
                
                if args.loss == 'supcon':
                    # SupCon expects (batch, n_views, feat_dim)
                    # We treat each sample as a separate view
                    all_feats = all_feats.unsqueeze(1)
                    loss_main = criterion_main(all_feats, all_targets)
                else:
                    # Triplet
                    loss_main = criterion_main(all_feats, all_targets)
                
            else:
                # Contrastive needs scores
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
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        scheduler.step()
        
        # Validate
        val_acc = evaluate(net, val_loader, args)
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}')
        
        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(args.path_weights, f'{args.model}_{args.shot_num}shot_best.pth')
            torch.save(net.state_dict(), path)
            print(f'  → Best model saved ({val_acc:.4f})')
            # Log best model as artifact if needed, or just log the metric
            wandb.run.summary["best_val_acc"] = best_acc
    
    return best_acc


def evaluate(net, loader, args):
    """Compute accuracy on loader."""
    net.eval()
    correct, total = 0, 0
    
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
    
    return correct / total if total > 0 else 0


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
    Final evaluation: 150 episodes, K-shot (same as training), 1-query.
    
    Metrics: Accuracy, Precision, Recall, F1, p-value
    Plots: Confusion Matrix (rows sum to 150), t-SNE
    """
    print(f"\n{'='*50}")
    print(f"Final Test: {args.model} | {args.shot_num}-shot")
    print(f"150 episodes × {args.way_num} classes × 1 query = 450 predictions")
    print('='*50)
    
    net.eval()
    all_preds, all_targets, all_features = [], [], []
    
    with torch.no_grad():
        for query, q_labels, support, s_labels in tqdm(loader, desc='Testing'):
            B, NQ, C, H, W = query.shape
            
            # Use same shot_num as training
            support = support.view(B, args.way_num, args.shot_num, C, H, W).to(args.device)
            query = query.to(args.device)
            targets = q_labels.view(-1).to(args.device)
            
            scores = net(query, support)
            preds = scores.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Extract features for t-SNE
            q_flat = query.view(-1, C, H, W)
            if hasattr(net, 'encoder'):
                feat = net.encoder(q_flat)
                feat = nn.functional.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)
                all_features.append(feat.cpu().numpy())
    
    # Metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = (all_preds == all_targets).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    p_val = calculate_p_value(acc, 1.0/args.way_num, len(all_targets))
    
    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"p-value  : {p_val:.2e}")
    
    # Log metrics to WandB
    wandb.log({
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_p_value": p_val
    })
    
    # Plots
    samples_str = f"_{args.training_samples}samples" if args.training_samples else "_allsamples"
    
    # Confusion Matrix
    cm_path = os.path.join(args.path_results, 
                           f"confusion_matrix_{args.model}_{args.shot_num}shot{samples_str}.png")
    plot_confusion_matrix(all_targets, all_preds, args.way_num, cm_path)
    wandb.log({"confusion_matrix": wandb.Image(cm_path)})
    
    # t-SNE
    if all_features:
        features = np.vstack(all_features)
        tsne_path = os.path.join(args.path_results, 
                                 f"tsne_{args.model}_{args.shot_num}shot{samples_str}.png")
        plot_tsne(features, all_targets, args.way_num, tsne_path)
        wandb.log({"tsne_plot": wandb.Image(tsne_path)})
    
    print(f"\nResults logged to WandB and plots saved to {args.path_results}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    
    # Set defaults based on shot_num
    if args.num_epochs is None:
        args.num_epochs = 100 if args.shot_num == 1 else 70

    # Auto-detect device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Config: {args.model} | {args.shot_num}-shot | {args.num_epochs} epochs | Device: {args.device}")
    
    # Initialize WandB with a descriptive run name
    samples_str = f"{args.training_samples}samples" if args.training_samples else "all"
    run_name = f"{args.model}_{args.shot_num}shot_{args.loss}_{samples_str}"
    
    wandb.init(project="pd_fewshot", config=vars(args), name=run_name, group=run_name, job_type=args.mode)
    
    seed_func(args.seed)
    os.makedirs(args.path_weights, exist_ok=True)
    os.makedirs(args.path_results, exist_ok=True)
    
    # Load dataset
    dataset = PDScalogram(args.dataset_path)
    
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
    
    # Create data loaders (all use 1 query per class)
    train_ds = FewshotDataset(train_X, train_y, args.episode_num_train,
                              args.way_num, args.shot_num, 1, args.seed)
    
    val_ds = FewshotDataset(val_X, val_y, args.episode_num_val,
                            args.way_num, args.shot_num, 1, args.seed)
    
    test_ds = FewshotDataset(test_X, test_y, 150,  # Fixed 150 episodes for test
                             args.way_num, args.shot_num, 1, args.seed)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Initialize Model
    net = get_model(args)
    
    if args.mode == 'train':
        train_loop(net, train_loader, val_loader, args)
        
        # Load best model for testing
        path = os.path.join(args.path_weights, f'{args.model}_{args.shot_num}shot_best.pth')
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
