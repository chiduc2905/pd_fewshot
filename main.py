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
from sklearn.metrics import precision_recall_fscore_support

# Local imports
from dataset import PDScalogram
from dataloader.dataloader import FewshotDataset
from function.function import ContrastiveLoss, seed_func, plot_confusion_matrix, plot_tsne

# Models
from net.cosine import CosineNet
from net.protonet import ProtoNet
from net.covamnet import CovaMNet

def get_args():
    parser = argparse.ArgumentParser(description='PD Scalogram Fewshot Training & Testing')
    
    # Dataset / Paths
    parser.add_argument('--dataset_path', type=str, default='./scalogram_images/', help='Path to dataset')
    parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--path_results', type=str, default='results/', help='Directory to save results')
    parser.add_argument('--weights', type=str, default=None, help='Path to specific weight file for testing')
    
    # Model
    parser.add_argument('--model', type=str, default='covamnet', choices=['cosine', 'protonet', 'covamnet'], help='Model architecture')
    
    # Fewshot settings
    parser.add_argument('--way_num', type=int, default=3, help='Number of classes per episode')
    parser.add_argument('--shot_num', type=int, default=1, help='Number of support samples per class')
    parser.add_argument('--query_num', type=int, default=1, help='Number of query samples per class')
    
    # Training settings
    parser.add_argument('--training_samples', type=int, default=None, help='Number of training samples per class (e.g. 30, 60). If None, use all.')
    parser.add_argument('--episode_num_train', type=int, default=100, help='Number of episodes per epoch (train)')
    parser.add_argument('--episode_num_test', type=int, default=75, help='Number of episodes per epoch (test)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Episodes per batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (init value)')
    parser.add_argument('--step_size', type=int, default=10, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Scheduler gamma (factor of 2 reduction = 0.5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Execution mode')
    
    return parser.parse_args()

def get_model(model_name, device):
    if model_name == 'cosine':
        model = CosineNet(use_gpu=(device=='cuda'))
    elif model_name == 'protonet':
        model = ProtoNet(use_gpu=(device=='cuda'))
    elif model_name == 'covamnet':
        model = CovaMNet(use_gpu=(device=='cuda'))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)

def train_loop(net, train_loader, val_loader, args):
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
        
        print(f"Validation Accuracy: {acc:.4f}")
        
        # Save Best Model
        if acc > best_acc:
            best_acc = acc
            save_name = f"{args.model}_{args.shot_num}shot_best.pth"
            save_path = os.path.join(args.path_weights, save_name)
            torch.save(net.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
            
    return history

def evaluate(net, loader, args):
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
    # Simple one-sample t-test approximation or binomial test for classification
    # Here we use a simplified z-test for proportions
    if n <= 0: return 1.0
    p = acc
    p0 = baseline
    if p * (1 - p) == 0: return 0.0
    z = (p - p0) / np.sqrt(p0 * (1 - p0) / n)
    # Two-tailed p-value from z-score (simplified)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return p_value

def test_full(net, loader, args):
    print(f"\n{'='*20} Testing Model: {args.model} ({args.shot_num}-shot) {'='*20}")
    
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
            s = support_imgs.view(B, args.way_num, args.shot_num, C, H, W).to(device)
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
    print(f"RESULTS SUMMARY: {args.model} {args.shot_num}-shot")
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
        
    res_file = os.path.join(res_dir, f"results_{args.model}_{args.shot_num}shot.txt")
    with open(res_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Shot: {args.shot_num}\n")
        f.write(f"Accuracy: {final_acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"p-value: {p_value:.4e}\n")
    print(f"Results saved to {res_file}")

    # Plotting
    # Confusion Matrix
    save_path_cm = os.path.join(res_dir, f"confusion_matrix_{args.model}_{args.shot_num}shot.png")
    try:
        plot_confusion_matrix(all_targets, all_preds, num_classes=args.way_num, save_path=save_path_cm)
    except Exception as e:
        print(f"Skipping confusion matrix: {e}")
    
    # t-SNE
    if all_features:
        all_features = np.vstack(all_features)
        all_targets_arr = np.array(all_targets)
        save_path_tsne = os.path.join(res_dir, f"tsne_{args.model}_{args.shot_num}shot.png")
        try:
            # Use only a subset if points > 2000 for speed, or full set for accuracy
            # Warning: t-SNE on small dataset (225 points) is unstable but will run.
            plot_tsne(all_features, all_targets_arr, num_classes=args.way_num, save_path=save_path_tsne)
        except Exception as e:
            print(f"Skipping t-SNE: {e}")

def main():
    args = get_args()
    print(f"Configuration: {args}")
    
    seed_func(args.seed)
    if not os.path.exists(args.path_weights):
        os.makedirs(args.path_weights)
    if not os.path.exists(args.path_results):
        os.makedirs(args.path_results)
        
    print("Loading Dataset...")
    dataset = PDScalogram(args.dataset_path, samples_per_class=args.training_samples)
    
    def prep_data(X, y):
        X = torch.from_numpy(X.astype(np.float32))
        if X.dim() == 4 and X.shape[3] == 3: 
             X = X.permute(0, 3, 1, 2)
        y = torch.from_numpy(y)
        return X, y
        
    train_X, train_y = prep_data(dataset.X_train, dataset.y_train)
    test_X, test_y = prep_data(dataset.X_test, dataset.y_test)
    
    # Create Episode Generators
    train_ds = FewshotDataset(train_X, train_y, 
                              episode_num=args.episode_num_train,
                              way_num=args.way_num,
                              shot_num=args.shot_num,
                              query_num=args.query_num) 
                              
    test_ds = FewshotDataset(test_X, test_y,
                             episode_num=args.episode_num_test,
                             way_num=args.way_num,
                             shot_num=args.shot_num,
                             query_num=args.query_num)
                             
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    net = get_model(args.model, args.device)
    
    if args.mode == 'train':
        train_loop(net, train_loader, test_loader, args)
        
        # Final Test Phase
        best_path = os.path.join(args.path_weights, f"{args.model}_{args.shot_num}shot_best.pth")
        if os.path.exists(best_path):
            print(f"Loading best model for final test evaluation: {best_path}")
            net.load_state_dict(torch.load(best_path))
            test_full(net, test_loader, args)
            
    elif args.mode == 'test':
        if args.weights is None:
            best_path = os.path.join(args.path_weights, f"{args.model}_{args.shot_num}shot_best.pth")
            if os.path.exists(best_path):
                args.weights = best_path
            else:
                print("Warning: No weights provided. Testing with random init.")
        
        if args.weights:
            print(f"Loading weights: {args.weights}")
            net.load_state_dict(torch.load(args.weights))
            
        test_full(net, test_loader, args)

if __name__ == '__main__':
    main()
