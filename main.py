import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
import function.function as function
import time
from tqdm import tqdm
from function.function import ContrastiveLoss, seed_func, cal_accuracy_fewshot_1shot, cal_accuracy_fewshot_5shot, predicted_fewshot_1shot, predicted_fewshot_5shot, plot_confusion_matrix, plot_tsne, get_features_for_tsne
from dataset import PDScalogram
import os
from dataloader.dataloader import FewshotDataset
from torch.utils.data import DataLoader
from net.cosine import CosineNet
from net.protonet import ProtoNet
from net.covamnet import CovaMNet

def get_args():
    parser = argparse.ArgumentParser(description='PD Scalogram Fewshot Configuration')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--dataset_path', type=str, default='./scalogram_images/', help='Path to scalogram dataset')
    parser.add_argument('--training_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--model', type=str, default='cosine', choices=['cosine', 'protonet', 'covamnet'], help='Model to use')
    parser.add_argument('--episode_num_train', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--episode_num_test', type=int, default=75, help='Number of testing episodes')
    parser.add_argument('--way_num', type=int, default=3, help='Number of classes (Corona, NoPD, Surface)')
    parser.add_argument('--shot_num', type=int, default=1, help='Number of samples per class')
    parser.add_argument('--query_num', type=int, default=1, help='Number of query samples')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Path to weights')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for scheduler')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights for testing')
    return parser.parse_args()

def get_model(model_name):
    if model_name == 'cosine':
        return CosineNet()
    elif model_name == 'protonet':
        return ProtoNet()
    elif model_name == 'covamnet':
        return CovaMNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_and_test_model(net, train_dataloader, test_loader, args):
    device = args.device
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = ContrastiveLoss().to(device)
    
    full_loss = []
    full_acc = []
    pred_acc = 0
    
    model_save_name = f'{args.model}_{args.shot_num}shot'
    
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        running_loss = 0
        num_batches = 0
        net.train()
        optimizer.zero_grad()
        
        print('='*50, f'Epoch: {epoch}', '='*50)
        with tqdm(train_dataloader, desc=f'Epoch {epoch}/{args.num_epochs}', unit='batch') as t:
            for query_images, query_targets, support_images, support_targets in t:
                q = query_images.permute(1, 0, 2, 3, 4).to(device)
                s = support_images.permute(1, 0, 2, 3, 4).to(device)
                targets = query_targets.to(device)
                targets = targets.permute(1, 0)
                
                # For CovaMNet and ProtoNet, we might need to adjust how we pass data if batch_size > 1
                # But current dataloader setup with batch_size=1 and permute logic suggests
                # we iterate over 'way' dimension? No, wait.
                # In train_1shot.py:
                # for i in range(len(q)):
                #   scores = net(q[i], s)
                # q[i] is (batch, query_num, C, H, W)
                # s is (way, batch, shot, C, H, W)
                
                # If we use my new models, they expect:
                # query: (batch, query_num, C, H, W)
                # support: (way, batch, shot, C, H, W)
                # So the loop structure is compatible.
                
                for i in range(len(q)):
                    scores = net(q[i], s).float()
                    target = targets[i].long()
                    loss = loss_fn(scores, target)
                    loss.backward()
                    running_loss += loss.detach().item()
                    num_batches += 1
                
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss=running_loss / num_batches if num_batches > 0 else 0)

        elapsed_time = time.time() - start_time
        scheduler.step()

        with torch.no_grad():
            total_loss = running_loss / num_batches if num_batches > 0 else 0
            full_loss.append(total_loss)
            print('Testing on validation set...')
            net.eval()
            if args.shot_num == 1:
                acc = cal_accuracy_fewshot_1shot(test_loader, net, device)
            else:
                acc = cal_accuracy_fewshot_5shot(test_loader, net, device)
                
            full_acc.append(acc)
            print(f'Accuracy: {acc:.4f}')
            
            if acc > pred_acc:
                best_path = os.path.join(args.path_weights, f'{model_save_name}_best.pth')
                if epoch >= 2 and os.path.exists(best_path):
                    try:
                        os.remove(best_path)
                    except:
                        pass
                pred_acc = acc
                torch.save(net.state_dict(), best_path) # Save state dict for flexibility
                print(f'Best model saved: {best_path}')
        
        torch.cuda.empty_cache()

    return full_loss, full_acc

def test_model(net, test_loader, args):
    device = args.device
    net.eval()
    print(f'Testing {args.model} {args.shot_num}-shot...')
    
    if args.shot_num == 1:
        acc = cal_accuracy_fewshot_1shot(test_loader, net, device)
        true_labels, predictions = predicted_fewshot_1shot(test_loader, net, device)
    else:
        acc = cal_accuracy_fewshot_5shot(test_loader, net, device)
        true_labels, predictions = predicted_fewshot_5shot(test_loader, net, device)
        
    print(f'Test Accuracy: {acc:.4f}')
    
    # Plot Confusion Matrix
    cm_path = f'confusion_matrix_{args.model}_{args.shot_num}shot.png'
    plot_confusion_matrix(true_labels, predictions, num_classes=args.way_num, save_path=cm_path)
    
    # Plot t-SNE if supported (CosineNet has encoder/fc structure, others too)
    try:
        features, labels = get_features_for_tsne(test_loader, net, device)
        tsne_path = f'tsne_{args.model}_{args.shot_num}shot.png'
        plot_tsne(features, labels, num_classes=args.way_num, save_path=tsne_path)
    except Exception as e:
        print(f"Could not plot t-SNE: {e}")

def main():
    args = get_args()
    print(args)
    seed_func()
    
    if not os.path.exists(args.path_weights):
        os.makedirs(args.path_weights)

    # Load dataset
    print('Loading PD Scalogram dataset...')
    data = PDScalogram(args.dataset_path)
    
    # Preprocess
    data.X_train = data.X_train.astype(np.float32)
    data.X_test = data.X_test.astype(np.float32)
    
    train_data = torch.from_numpy(data.X_train)
    train_label = torch.from_numpy(data.y_train)
    test_data = torch.from_numpy(data.X_test)
    test_label = torch.from_numpy(data.y_test)
    
    if len(train_data.shape) == 4:
        train_data = train_data.permute(0, 3, 1, 2)
        test_data = test_data.permute(0, 3, 1, 2)
        
    # Create datasets
    train_dataset = FewshotDataset(train_data, train_label, 
                                   episode_num=args.episode_num_train,
                                   way_num=args.way_num, 
                                   shot_num=args.shot_num,
                                   query_num=args.query_num)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dataset = FewshotDataset(test_data, test_label,
                                  episode_num=args.episode_num_test,
                                  way_num=args.way_num,
                                  shot_num=args.shot_num,
                                  query_num=args.query_num)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    net = get_model(args.model)
    net = net.to(args.device)
    
    if args.mode == 'train':
        train_and_test_model(net, train_dataloader, test_dataloader, args)
    elif args.mode == 'test':
        if args.weights:
            print(f"Loading weights from {args.weights}")
            net.load_state_dict(torch.load(args.weights))
        else:
            print("No weights provided for testing. Using random initialization (or provide --weights).")
        test_model(net, test_dataloader, args)

if __name__ == '__main__':
    main()
