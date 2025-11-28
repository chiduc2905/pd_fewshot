"""Training and testing script for PD Scalogram few-shot classification."""
import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from dataset import PDScalogram
from dataloader.dataloader import FewshotDataset
from function.function import ContrastiveLoss, seed_func, plot_confusion_matrix, plot_tsne
from net.cosine import CosineNet
from net.protonet import ProtoNet
from net.covamnet import CovaMNet

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PD Scalogram Few-shot Learning')
    
    # Dataset / Paths
    parser.add_argument('--dataset_path', type=str, default='./scalogram_images/', help='Path to dataset')
    parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--path_results', type=str, default='results/', help='Directory to save results')
    parser.add_argument('--weights', type=str, default=None, help='Path to specific weight file for testing')
    
    # Model
    parser.add_argument('--model', type=str, default='covamnet', choices=['cosine', 'protonet', 'covamnet'], help='Model architecture')
    parser.add_argument('--covamnet_classifier', action='store_true', help='Use learnable classifier in CovaMNet (default: False, use similarity scores only)')
    
    # Fewshot settings
    parser.add_argument('--way_num', type=int, default=3, help='Number of classes per episode')
    parser.add_argument('--shot_num', type=int, default=1, help='Number of support samples per class')
    parser.add_argument('--query_num', type=int, default=None, help='Number of query samples per class. Default: 19 for 1-shot, 15 for 5-shot')
    
    # Training settings
    parser.add_argument('--training_samples', type=int, default=None, help='Total number of training samples across ALL classes (e.g. 30, 60). If None, use all.')
    parser.add_argument('--episode_num_train', type=int, default=100, help='Number of episodes per epoch (train)')
    parser.add_argument('--episode_num_test', type=int, default=75, help='Number of episodes per epoch (test)')
    parser.add_argument('--num_epochs', type=int, default=None, help='Total training epochs. Default: 100 for 1-shot, 50 for 5-shot (use early stopping)')
    parser.add_argument('--batch_size', type=int, default=1, help='Episodes per batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (init value)')
    parser.add_argument('--step_size', type=int, default=10, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='Scheduler gamma')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Execution mode')
    
    return parser.parse_args()

def get_classifier_suffix(args):
    """Get classifier suffix for filenames when using CovaMNet."""
    if args.model == 'covamnet':
        return "_classifier" if args.covamnet_classifier else "_similarity"
    return ""

def get_model(model_name, device, use_classifier=None):
    """Initialize model by name."""
    if model_name == 'cosine':
        model = CosineNet(use_gpu=(device=='cuda'))
    elif model_name == 'protonet':
        model = ProtoNet(use_gpu=(device=='cuda'))
    elif model_name == 'covamnet':
        model = CovaMNet(use_classifier=use_classifier, use_gpu=(device=='cuda'))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)

def train_loop(net, train_loader, val_loader, args):
    """Training loop with validation-based model selection."""
    device = args.device
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = ContrastiveLoss().to(device)
    
    best_acc = 0.0
    history = {'loss': [], 'acc': []}
    
    print(f"Starting training for {args.num_epochs} epochs...")
    
    for epoch in range(1, args.num_epochs + 1):
        net.train()
        running_loss = 0.0
        total_batches = 0
        
        # Training
        with tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', unit='batch') as t:
            for query_imgs, query_targets, support_imgs, support_targets in t:
                B, NQ_total, C, H, W = query_imgs.shape
                _, NS_total, _, _, _ = support_imgs.shape
                
                s = support_imgs.view(B, args.way_num, args.shot_num, C, H, W).to(device)
                q = query_imgs.to(device)
                targets = query_targets.view(-1).to(device) 
                
                optimizer.zero_grad()
                
                scores = net(q, s)
                loss = loss_fn(scores, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                total_batches += 1
                
                t.set_postfix(loss=running_loss/total_batches)
        
        scheduler.step()
        avg_loss = running_loss / total_batches
        history['loss'].append(avg_loss)
        
        # Validation (Model Selection)
        acc = evaluate(net, val_loader, args)
        history['acc'].append(acc)
        
        print(f"Val Accuracy: {acc:.4f}")
        
        # Save Best Model
        if acc > best_acc:
            best_acc = acc
            classifier_suffix = get_classifier_suffix(args)
            save_name = f"{args.model}_{args.shot_num}shot{classifier_suffix}_best.pth"
            save_path = os.path.join(args.path_weights, save_name)
            torch.save(net.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
            
    return history

def evaluate(net, loader, args):
    """Evaluate model accuracy on loader."""
    net.eval()
    total_correct = 0
    total_samples = 0
    device = args.device
    
    with torch.no_grad():
        for query_imgs, query_targets, support_imgs, support_targets in loader:
            B, NQ_total, C, H, W = query_imgs.shape
            
            s = support_imgs.view(B, args.way_num, args.shot_num, C, H, W).to(device)
            q = query_imgs.to(device)
            targets = query_targets.view(-1).to(device)
            
            scores = net(q, s)
            preds = torch.argmax(scores, dim=1)
            
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            
    return total_correct / total_samples if total_samples > 0 else 0

def calculate_p_value(acc, baseline=0.33, n=75):
    """Calculate p-value using z-test for proportions."""
    if n <= 0:
        return 1.0
    p = acc
    p0 = baseline
    z = (p - p0) / np.sqrt(p0 * (1 - p0) / n)
    
    # Two-tailed p-value from z-score
    from scipy.stats import norm
    # Use survival function (sf = 1 - cdf) for better precision with small p-values
    p_value = 2 * norm.sf(abs(z))
    return p_value

def test_full(net, loader, args, shot_num=None, query_num=None, generate_plots=True):
    """Full evaluation with metrics, confusion matrix, and t-SNE.

    Args:
        generate_plots: Whether to generate confusion matrix and t-SNE plots.
                        Only True for final test phase (150 episodes, 1-query).

    For final test phase: 150 episodes, 1-shot/1-query per class.
    Confusion matrix: each row sums to 150 (one prediction per class per episode).
    t-SNE: visualizes all 150 * way_num query features.
    """
    # Use provided shot_num/query_num or fall back to args
    shot_num = shot_num if shot_num is not None else args.shot_num
    query_num = query_num if query_num is not None else args.query_num
    
    print(f"\n{'='*20} Testing: {args.model} ({shot_num}-shot, {query_num}-query) {'='*20}")

    # Calculate expected samples for debugging
    if hasattr(args, 'episode_num_test'):
        expected_episodes = args.episode_num_test
    else:
        # For final test, use 150 episodes
        expected_episodes = 150

    expected_total = expected_episodes * args.way_num * query_num
    print(f"Expected: {expected_episodes} episodes × {args.way_num} classes × {query_num} queries = {expected_total} samples")

    all_preds = []
    all_targets = []
    all_features = []
    
    device = args.device
    net.eval()
    
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for query_imgs, query_targets, support_imgs, support_targets in tqdm(loader, desc="Testing"):
            B, NQ_total, C, H, W = query_imgs.shape
            s = support_imgs.view(B, args.way_num, shot_num, C, H, W).to(device)
            q = query_imgs.to(device)
            targets = query_targets.view(-1).to(device)
            
            scores = net(q, s)
            preds = torch.argmax(scores, dim=1)
            
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Extract Features
            q_flat = q.view(-1, C, H, W)
            if hasattr(net, 'encoder'):
                feat = net.encoder(q_flat)
            elif hasattr(net, 'features'):
                feat = net.features(q_flat)
            else:
                feat = None
            
            if feat is not None:
                 if feat.dim() > 2:
                     feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                     feat = feat.view(feat.size(0), -1)
                 all_features.append(feat.cpu().numpy())
                 
    final_acc = total_correct / total_samples
    
    # Debug: Print actual sample counts
    print(f"Actual results: {len(all_targets)} total samples, {len(set(all_targets))} unique classes")
    cm = confusion_matrix(all_targets, all_preds)
    print(f"Confusion matrix shape: {cm.shape}, sum per class: {cm.sum(axis=1)}")

    # Calculate Metrics: Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average='macro')

    # Calculate p-value (approximate, assuming random chance baseline = 1/num_classes)
    try:
        from scipy.stats import norm
        p_value = calculate_p_value(final_acc, baseline=1.0/args.way_num, n=total_samples)
    except ImportError:
        p_value = 0.0
        print("Warning: scipy not installed, p-value set to 0.0")

    # Display Results
    print("\n" + "="*50)
    print(f"RESULTS SUMMARY: {args.model} {shot_num}-shot, {query_num}-query")
    print("="*50)
    print(f"Accuracy : {final_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"p-value  : {p_value:.4e}")
    print("="*50 + "\n")
    
    # Save Results to File
    res_dir = args.path_results
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    training_samples_str = f"_{args.training_samples}samples" if args.training_samples is not None else "_allsamples"
    classifier_suffix = get_classifier_suffix(args)
    res_file = os.path.join(res_dir, f"results_{args.model}_{args.shot_num}shot{classifier_suffix}{training_samples_str}.txt")
    
    # Save individual result
    with open(res_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Shot: {args.shot_num}\n")
        f.write(f"Training Samples: {args.training_samples}\n")
        f.write(f"Accuracy: {final_acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"p-value: {p_value:.4e}\n")
    print(f"Results saved to {res_file}")

    # Append to summary file for this sample size
    summary_file = os.path.join(res_dir, f"summary{training_samples_str}.txt")
    with open(summary_file, 'a') as f:
        # Check if file is empty to write header
        if os.stat(summary_file).st_size == 0:
            f.write("Model\tShot\tAccuracy\tPrecision\tRecall\tF1\tp-value\n")
        f.write(f"{args.model}\t{args.shot_num}\t{final_acc:.4f}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t{p_value:.4e}\n")
    print(f"Summary appended to {summary_file}")


    # Plotting (only for final test phase with 150 episodes)
    if generate_plots:
        # Confusion Matrix
        classifier_suffix = get_classifier_suffix(args)
        save_path_cm = os.path.join(res_dir, f"confusion_matrix_{args.model}_{args.shot_num}shot{classifier_suffix}{training_samples_str}.png")
        try:
            plot_confusion_matrix(all_targets, all_preds, num_classes=args.way_num, save_path=save_path_cm)
        except Exception as e:
            print(f"Skipping confusion matrix: {e}")

        # t-SNE
        if all_features:
            all_features = np.vstack(all_features)
            all_targets_arr = np.array(all_targets)
            save_path_tsne = os.path.join(res_dir, f"tsne_{args.model}_{args.shot_num}shot{classifier_suffix}{training_samples_str}.png")
            try:
                # Use only a subset if points > 2000 for speed, or full set for accuracy
                plot_tsne(all_features, all_targets_arr, num_classes=args.way_num, save_path=save_path_tsne)
            except Exception as e:
                print(f"Skipping t-SNE: {e}")
        else:
            print("No features extracted for t-SNE")
    else:
        print("Skipping plots (not final test phase)")



def main():
    """Main entry point."""
    args = get_args()
    
    # Set defaults based on shot_num
    if args.query_num is None:
        args.query_num = 15  # 15 queries per class for both 1-shot and 5-shot
        print(f"query_num={args.query_num} for {args.shot_num}-shot")
    
    if args.num_epochs is None:
        args.num_epochs = 100 if args.shot_num == 1 else 70
        print(f"num_epochs={args.num_epochs} for {args.shot_num}-shot")
    
    print(f"Configuration: {args}")
    
    seed_func(args.seed)
    if not os.path.exists(args.path_weights):
        os.makedirs(args.path_weights)
    if not os.path.exists(args.path_results):
        os.makedirs(args.path_results)
        
    print("Loading dataset...")
    dataset = PDScalogram(args.dataset_path, total_training_samples=args.training_samples)
    
    def prep_data(X, y):
        X = torch.from_numpy(X.astype(np.float32))
        if X.dim() == 4 and X.shape[3] == 3: 
             X = X.permute(0, 3, 1, 2)
        y = torch.from_numpy(y)
        return X, y
        
    train_X_full, train_y_full = prep_data(dataset.X_train, dataset.y_train)
    val_X, val_y = prep_data(dataset.X_val, dataset.y_val)
    test_X, test_y = prep_data(dataset.X_test, dataset.y_test)

    # If user specified a limited number of training samples, sample evenly per class
    if args.training_samples is not None:
        per_class = args.training_samples // args.way_num
        train_X_list = []
        train_y_list = []
        for cls in range(args.way_num):
            idxs = (train_y_full == cls).nonzero(as_tuple=False).view(-1)
            if idxs.numel() < per_class:
                raise ValueError(f"Not enough training samples for class {cls}: required {per_class}, available {idxs.numel()}")
            # deterministic selection (shuffle with seed)
            g = torch.Generator()
            g.manual_seed(args.seed)
            perm = idxs[torch.randperm(idxs.numel(), generator=g)][:per_class]
            train_X_list.append(train_X_full[perm])
            train_y_list.append(train_y_full[perm])

        train_X = torch.cat(train_X_list, dim=0)
        train_y = torch.cat(train_y_list, dim=0)
    else:
        train_X, train_y = train_X_full, train_y_full
    
    # Episode generators
    train_ds = FewshotDataset(train_X, train_y, 
                              episode_num=args.episode_num_train,
                              way_num=args.way_num,
                              shot_num=args.shot_num,
                              query_num=args.query_num) 
    
    # Validation uses the reserved eval pool (val_X,val_y)
    val_ds = FewshotDataset(val_X, val_y,
                            episode_num=args.episode_num_test,
                            way_num=args.way_num,
                            shot_num=args.shot_num,
                            query_num=args.query_num)
                              
    # Test dataset used for final evaluation (may be overridden later for 1-shot final test)
    test_ds = FewshotDataset(test_X, test_y,
                             episode_num=args.episode_num_test,
                             way_num=args.way_num,
                             shot_num=args.shot_num,
                             query_num=args.query_num)
                             
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    net = get_model(args.model, args.device, use_classifier=args.covamnet_classifier if args.model == 'covamnet' else None)
    
    if args.mode == 'train':
        # Train with validation-based model selection
        train_loop(net, train_loader, val_loader, args)
        
        # Final Test Phase: evaluate best model on 150 one-shot episodes
        classifier_suffix = get_classifier_suffix(args)
        # Try new naming scheme first
        best_path = os.path.join(args.path_weights, f"{args.model}_{args.shot_num}shot{classifier_suffix}_best.pth")
        if not os.path.exists(best_path):
            # Fallback to old naming scheme for backward compatibility
            best_path = os.path.join(args.path_weights, f"{args.model}_{args.shot_num}shot_best.pth")
        if os.path.exists(best_path):
            print(f"Loading best model for final test evaluation: {best_path}")
            state_dict = torch.load(best_path)
            # Use strict=False to handle classifier mode mismatches
            net.load_state_dict(state_dict, strict=False)
            print("Weights loaded successfully")
            # Create final test episodes: 150 episodes, 1-shot 1-query per class
            print(f"Creating final test with: episode_num=150, way_num={args.way_num}, shot_num=1, query_num=1")
            final_test_ds = FewshotDataset(test_X, test_y,
                                           episode_num=150,
                                           way_num=args.way_num,
                                           shot_num=1,
                                           query_num=1,
                                           seed=args.seed)
            final_test_loader = DataLoader(final_test_ds, batch_size=args.batch_size, shuffle=False)
            # Final test: 150 episodes × way_num queries = 450 total predictions
            # Confusion matrix rows sum to 150 (one query per class per episode)
            print(f"Final test dataset length: {len(final_test_ds)}")
            test_full(net, final_test_loader, args, shot_num=1, query_num=1, generate_plots=True)
            
    elif args.mode == 'test':
        if args.weights is None:
            classifier_suffix = get_classifier_suffix(args)
            # Try new naming scheme first
            best_path = os.path.join(args.path_weights, f"{args.model}_{args.shot_num}shot{classifier_suffix}_best.pth")
            if os.path.exists(best_path):
                args.weights = best_path
            else:
                # Fallback to old naming scheme for backward compatibility
                old_path = os.path.join(args.path_weights, f"{args.model}_{args.shot_num}shot_best.pth")
                if os.path.exists(old_path):
                    args.weights = old_path
                    print(f"Using legacy checkpoint: {old_path}")
                else:
                    print("Warning: No weights provided. Testing with random init.")
        
        if args.weights:
            print(f"Loading weights: {args.weights}")
            state_dict = torch.load(args.weights)
            # Use strict=False to handle classifier mode mismatches
            net.load_state_dict(state_dict, strict=False)
            print("Weights loaded successfully")
            
        test_full(net, test_loader, args, generate_plots=False)

if __name__ == '__main__':
    main()
