import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import function.function as function
import time
from tqdm import tqdm
from function.function import ContrastiveLoss, seed_func, cal_accuracy_fewshot_5shot
from dataset import PDScalogram
import os
from dataloader.dataloader import FewshotDataset
from torch.utils.data import DataLoader
from net.pam_mamba import CovarianceNet

parser = argparse.ArgumentParser(description='PD Scalogram 5-shot Configuration')
parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
parser.add_argument('--dataset_path', type=str, default='./scalogram_images/', help='Path to scalogram dataset')
parser.add_argument('--training_samples', type=int, default=100, help='Number of training samples')
parser.add_argument('--model_name', type=str, default='pd_scalogram', help='Model name')
parser.add_argument('--episode_num_train', type=int, default=100, help='Number of training episodes')
parser.add_argument('--episode_num_test', type=int, default=75, help='Number of testing episodes')
parser.add_argument('--way_num', type=int, default=3, help='Number of classes')
parser.add_argument('--shot_num', type=int, default=5, help='Number of samples per class')
parser.add_argument('--query_num', type=int, default=1, help='Number of query samples')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Path to weights')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--step_size', type=int, default=10, help='Step size for scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for scheduler')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
args = parser.parse_args()

print(args)

# Load dataset
print('Loading PD Scalogram dataset...')
data = PDScalogram(args.dataset_path)
print(f'Dataset loaded: {data.nclasses} classes')
print(f'Train samples: {len(data.X_train)}, Test samples: {len(data.X_test)}')

# Convert to torch tensors
data.X_train = data.X_train.astype(np.float32)
data.X_test = data.X_test.astype(np.float32)

train_data = torch.from_numpy(data.X_train)
train_label = torch.from_numpy(data.y_train)
test_data = torch.from_numpy(data.X_test)
test_label = torch.from_numpy(data.y_test)

# Reshape: (N, H, W, C) -> (N, C, H, W)
train_data = train_data.permute(0, 3, 1, 2)
test_data = test_data.permute(0, 3, 1, 2)

print(f'Train data shape: {train_data.shape}')
print(f'Test data shape: {test_data.shape}')

# Create fewshot datasets
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

# Training function
def train_and_test_model(net, train_dataloader, test_loader, num_epochs=args.num_epochs, lr=args.lr):
    device = args.device
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = ContrastiveLoss().to(device)

    full_loss = []
    full_acc = []
    pred_acc = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        running_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        print('='*50, f'Epoch: {epoch}', '='*50)
        with tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as t:
            for query_images, query_targets, support_images, support_targets in t:
                q = query_images.permute(1, 0, 2, 3, 4).to(device)
                s = support_images.permute(1, 0, 2, 3, 4).to(device)
                targets = query_targets.to(device)
                targets = targets.permute(1, 0)

                for i in range(len(q)):
                    scores = net(q[i], s).float()
                    target = targets[i].long()
                    loss = loss_fn(scores, target)
                    loss.backward()
                    running_loss += loss.detach().item()
                    num_batches += 1

                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss=running_loss / num_batches)

        elapsed_time = time.time() - start_time
        scheduler.step()

        with torch.no_grad():
            total_loss = running_loss / num_batches
            full_loss.append(total_loss)
            print('Testing on validation set...')
            acc = cal_accuracy_fewshot_5shot(test_loader, net, device)
            full_acc.append(acc)
            print(f'Accuracy: {acc:.4f}')

            if acc > pred_acc:
                if epoch >= 2 and os.path.exists(args.path_weights + 'best_model.pth'):
                    os.remove(args.path_weights + 'best_model.pth')
                pred_acc = acc
                model_name = f'{args.model_name}_5shot_{acc:.4f}.pth'
                torch.save(net, args.path_weights + model_name)
                print(f'Best model saved: {model_name}')

        torch.cuda.empty_cache()

    return full_loss, full_acc

# Training
print('Starting training...')
seed_func()
net = CovarianceNet()
net = net.to(args.device)

train_and_test_model(net, train_dataloader, test_dataloader)
print('Training completed!')
