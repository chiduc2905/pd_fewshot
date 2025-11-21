import torch
import numpy as np
import argparse
import function.function as function
from dataset import PDScalogram
from dataloader.dataloader import FewshotDataset
from torch.utils.data import DataLoader
from net.pam_mamba import CovarianceNet
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

parser = argparse.ArgumentParser(description='PD Scalogram 5-shot Testing')
parser.add_argument('--dataset_path', type=str, default='./scalogram_images/', help='Path to scalogram dataset')
parser.add_argument('--model_path', type=str, help='Path to trained model', required=True)
parser.add_argument('--episode_num_test', type=int, default=75, help='Number of testing episodes')
parser.add_argument('--way_num', type=int, default=3, help='Number of classes')
parser.add_argument('--shot_num', type=int, default=5, help='Number of samples per class')
parser.add_argument('--query_num', type=int, default=1, help='Number of query samples')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
args = parser.parse_args()

print(args)

# Load dataset
print('Loading dataset...')
data = PDScalogram(args.dataset_path)

data.X_test = data.X_test.astype(np.float32)
test_data = torch.from_numpy(data.X_test)
test_label = torch.from_numpy(data.y_test)

# Reshape: (N, H, W, C) -> (N, C, H, W)
test_data = test_data.permute(0, 3, 1, 2)

print(f'Test data shape: {test_data.shape}')

# Create test dataset
test_dataset = FewshotDataset(test_data, test_label,
                              episode_num=args.episode_num_test,
                              way_num=args.way_num,
                              shot_num=args.shot_num,
                              query_num=args.query_num)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Load model
print(f'Loading model from {args.model_path}...')
net = torch.load(args.model_path, weights_only=False)
net = net.to(args.device)
net.eval()

# Test
print('Testing...')
true_labels, predictions = function.predicted_fewshot_5shot(test_dataloader, net, args.device)

# Calculate detailed metrics
accuracy, f1_score, recall, precision = function.cal_metrics_5shot(test_dataloader, net, args.device, args.way_num)

print(f'\nAccuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1_score:.4f}')

# Plot confusion matrix
print('\nPlotting Confusion Matrix...')
function.plot_confusion_matrix(true_labels, predictions, 
                              save_path='checkpoints/confusion_matrix_5shot.png')

# Plot t-SNE
print('\nPlotting t-SNE...')
features, labels = function.get_features_for_tsne(test_dataloader, net, args.device)
function.plot_tsne(features, labels, 
                  save_path='checkpoints/tsne_5shot.png')

torch.cuda.empty_cache()
