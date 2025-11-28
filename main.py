"""PD Scalogram Few-Shot Learning - Training and Evaluation."""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from dataset import PDScalogram
from dataloader.dataloader import FewshotDataset
from function.function import ContrastiveLoss, seed_func, plot_confusion_matrix, plot_tsne
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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    return parser.parse_args()


def get_model(name, device):
    """Initialize model by name."""
    models = {
        'cosine': CosineNet,
        'protonet': ProtoNet,
        'covamnet': CovaMNet,
    }
    model = models[name](use_gpu=(device == 'cuda'))
    return model.to(device)


# =============================================================================
# Training
# =============================================================================

def train_loop(net, train_loader, val_loader, args):
    """Train with validation-based model selection."""
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = ContrastiveLoss().to(args.device)
    
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
            loss = loss_fn(scores, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        scheduler.step()
        
        # Validate
        acc = evaluate(net, val_loader, args)
        print(f'Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Val Acc={acc:.4f}')
        
        # Save best
        if acc > best_acc:
            best_acc = acc
            path = os.path.join(args.path_weights, f'{args.model}_{args.shot_num}shot_best.pth')
            torch.save(net.state_dict(), path)
            print(f'  → Best model saved ({acc:.4f})')
    
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
    Final evaluation: 150 episodes, 1-shot 1-query.
    
    Metrics: Accuracy, Precision, Recall, F1, p-value
    Plots: Confusion Matrix (rows sum to 150), t-SNE
    """
    print(f"\n{'='*50}")
    print(f"Final Test: {args.model} | {args.shot_num}-shot training")
    print(f"150 episodes × {args.way_num} classes × 1 query = 450 predictions")
    print('='*50)
    
    net.eval()
    all_preds, all_targets, all_features = [], [], []
    
    with torch.no_grad():
        for query, q_labels, support, s_labels in tqdm(loader, desc='Testing'):
            B, NQ, C, H, W = query.shape
            
            # Final test always uses 1-shot
            support = support.view(B, args.way_num, 1, C, H, W).to(args.device)
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
    
    # Save results
    samples_str = f"_{args.training_samples}samples" if args.training_samples else "_allsamples"
    
    result_file = os.path.join(args.path_results, 
                               f"results_{args.model}_{args.shot_num}shot{samples_str}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Shot: {args.shot_num}\n")
        f.write(f"Training Samples: {args.training_samples}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"p-value: {p_val:.2e}\n")
    
    # Append to summary
    summary_file = os.path.join(args.path_results, f"summary{samples_str}.txt")
    write_header = not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0
    with open(summary_file, 'a') as f:
        if write_header:
            f.write("Model\tShot\tAccuracy\tPrecision\tRecall\tF1\tp-value\n")
        f.write(f"{args.model}\t{args.shot_num}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{p_val:.2e}\n")
    
    # Plots
    cm_path = os.path.join(args.path_results, 
                           f"confusion_matrix_{args.model}_{args.shot_num}shot{samples_str}.png")
    plot_confusion_matrix(all_targets, all_preds, args.way_num, cm_path)
    
    if all_features:
        features = np.vstack(all_features)
        tsne_path = os.path.join(args.path_results, 
                                 f"tsne_{args.model}_{args.shot_num}shot{samples_str}.png")
        plot_tsne(features, all_targets, args.way_num, tsne_path)
    
    print(f"\nResults saved to {args.path_results}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    
    # Set defaults based on shot_num
    if args.num_epochs is None:
        args.num_epochs = 100 if args.shot_num == 1 else 70
    
    print(f"Config: {args.model} | {args.shot_num}-shot | {args.num_epochs} epochs")
    
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
    
    # Final test: 150 episodes, 1-shot, 1-query
    test_ds = FewshotDataset(test_X, test_y, 150,
                             args.way_num, 1, 1, args.seed)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Model
    net = get_model(args.model, args.device)
    
    if args.mode == 'train':
        train_loop(net, train_loader, val_loader, args)
        
        # Load best and run final test
        best_path = os.path.join(args.path_weights, f'{args.model}_{args.shot_num}shot_best.pth')
        if os.path.exists(best_path):
            net.load_state_dict(torch.load(best_path))
            test_final(net, test_loader, args)
    
    elif args.mode == 'test':
        # Load weights
        if args.weights:
            path = args.weights
        else:
            path = os.path.join(args.path_weights, f'{args.model}_{args.shot_num}shot_best.pth')
        
        if os.path.exists(path):
            net.load_state_dict(torch.load(path))
            print(f"Loaded: {path}")
        else:
            print(f"Warning: {path} not found, using random init")
        
        test_final(net, test_loader, args)


if __name__ == '__main__':
    main()
