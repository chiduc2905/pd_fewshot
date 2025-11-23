import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import PDScalogram
from dataloader.dataloader import FewshotDataset
from torch.utils.data import DataLoader
from net.cosine import CosineNet
from net.protonet import ProtoNet
from net.covamnet import CovaMNet
from function.function import cal_accuracy_fewshot_1shot, cal_accuracy_fewshot_5shot, seed_func
import os

def get_args():
    parser = argparse.ArgumentParser(description='Compare Fewshot Models')
    parser.add_argument('--dataset_path', type=str, default='./scalogram_images/', help='Path to scalogram dataset')
    parser.add_argument('--episode_num_test', type=int, default=100, help='Number of testing episodes')
    parser.add_argument('--way_num', type=int, default=3, help='Number of classes')
    parser.add_argument('--shot_num', type=int, default=1, help='Number of samples per class')
    parser.add_argument('--query_num', type=int, default=1, help='Number of query samples')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Path to weights')
    return parser.parse_args()

def load_model(model_name, weights_path, device):
    if model_name == 'cosine':
        net = CosineNet()
    elif model_name == 'protonet':
        net = ProtoNet()
    elif model_name == 'covamnet':
        net = CovaMNet()
    else:
        return None
    
    net = net.to(device)
    if os.path.exists(weights_path):
        print(f"Loading {model_name} from {weights_path}")
        net.load_state_dict(torch.load(weights_path))
        net.eval()
        return net
    else:
        print(f"Weights not found for {model_name} at {weights_path}")
        return None

def main():
    args = get_args()
    seed_func()
    
    # Load dataset
    print('Loading PD Scalogram dataset...')
    data = PDScalogram(args.dataset_path)
    data.X_test = data.X_test.astype(np.float32)
    test_data = torch.from_numpy(data.X_test)
    test_label = torch.from_numpy(data.y_test)
    
    if len(test_data.shape) == 4:
        test_data = test_data.permute(0, 3, 1, 2)
        
    test_dataset = FewshotDataset(test_data, test_label,
                                  episode_num=args.episode_num_test,
                                  way_num=args.way_num,
                                  shot_num=args.shot_num,
                                  query_num=args.query_num)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    models = ['cosine', 'protonet', 'covamnet']
    results = {}
    
    for model_name in models:
        weights_path = os.path.join(args.path_weights, f'{model_name}_{args.shot_num}shot_best.pth')
        net = load_model(model_name, weights_path, args.device)
        
        if net:
            print(f"Testing {model_name}...")
            if args.shot_num == 1:
                acc = cal_accuracy_fewshot_1shot(test_dataloader, net, args.device)
            else:
                acc = cal_accuracy_fewshot_5shot(test_dataloader, net, args.device)
            results[model_name] = acc
            print(f"{model_name} Accuracy: {acc:.4f}")
        else:
            results[model_name] = 0.0
            
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
    plt.title(f'Few-Shot Model Comparison ({args.shot_num}-shot)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    save_path = f'model_comparison_{args.shot_num}shot.png'
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

if __name__ == '__main__':
    main()
