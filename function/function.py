import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from tsnecuda import TSNE
from sklearn.metrics import confusion_matrix

def seed_func():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, output, target):
        # output: (way,) - similarity scores for each class
        # target: scalar - true class label
        if output.dim() == 1:
            # output is 1D: (way,)
            upper = torch.exp(output[target])
            lower = torch.exp(output).sum()
            loss = -torch.log(upper / lower)
        else:
            # output is 2D: (batch, way)
            upper = torch.exp(output[:, target])
            lower = torch.exp(output).sum(1)
            loss = -torch.log(upper / lower)
        return loss

def cal_accuracy_fewshot_1shot(loader, net, device):
    true_label = 0
    num_batches = 0

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = support_images.permute(1, 0, 2, 3, 4).to(device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)

        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            true_label += 1 if torch.argmax(scores) == target else 0
            num_batches += 1

    return true_label / num_batches

def convert_for_5shots(support_images, support_targets, device):
    support_targets = support_targets.cpu()
    labels = torch.unique(support_targets)
    new_support_images = []

    for label in labels:
        label_images = support_images[:, support_targets[0] == label]
        padded_label_images = torch.zeros((5, 3, 224, 224), dtype=label_images.dtype)
        padded_label_images[:label_images.shape[1]] = label_images.squeeze(0)
        new_support_images.append(padded_label_images.to(device))

    return new_support_images

def cal_accuracy_fewshot_5shot(loader, net, device):
    true_label = 0
    num_batches = 0

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = convert_for_5shots(support_images, support_targets, device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)

        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            true_label += 1 if torch.argmax(scores) == target else 0
            num_batches += 1

    return true_label / num_batches

def predicted_fewshot_1shot(loader, net, device):
    predicted = []
    true_labels = []

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = support_images.permute(1, 0, 2, 3, 4).to(device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)

        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            predicted.append(torch.argmax(scores).cpu().detach().numpy())
            true_labels.append(target.cpu().detach().numpy())

    return np.array(true_labels), np.array(predicted)

def predicted_fewshot_5shot(loader, net, device):
    predicted = []
    true_labels = []

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = convert_for_5shots(support_images, support_targets, device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)

        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            predicted.append(torch.argmax(scores).cpu().detach().numpy())
            true_labels.append(target.cpu().detach().numpy())

    return np.array(true_labels), np.array(predicted)

def get_features_for_tsne(loader, net, device):
    """Extract features and labels for t-SNE visualization"""
    features = []
    labels = []

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = support_images.permute(1, 0, 2, 3, 4).to(device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)

        for i in range(len(q)):
            # Get features from encoder (before similarity computation)
            q_feat = net.encoder(q[i])
            q_feat = q_feat.view(q_feat.size(0), -1)
            q_feat = net.fc(q_feat)
            
            features.append(q_feat.cpu().detach().numpy())
            labels.append(targets[i].cpu().detach().numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)
    return features, labels

def plot_confusion_matrix(true_labels, predictions, num_classes=3, save_path=None):
    """
    Plot confusion matrix with style matching the reference image.
    Designed for 3-class classification.
    """
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Calculate percentages
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create annotations with count and percentage
    annotations = np.empty_like(conf_matrix, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            count = conf_matrix[i, j]
            percent = conf_matrix_percent[i, j]
            annotations[i, j] = f'{count}\n({percent:.1f}%)'
    
    # Plot heatmap with green colormap
    sns.heatmap(conf_matrix, annot=annotations, fmt='', cmap='Greens', 
                cbar_kws={'label': ''}, 
                linewidths=2, linecolor='white', ax=ax,
                annot_kws={'size': 11, 'weight': 'bold'},
                vmin=0, square=True)
    
    # Customize labels and title
    ax.set_xlabel('Predicted Labels', fontsize=12, fontweight='normal')
    ax.set_ylabel('True Labels', fontsize=12, fontweight='normal')
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='normal', pad=15)
    
    # Set tick labels
    ax.set_xticklabels(range(num_classes), fontsize=10)
    ax.set_yticklabels(range(num_classes), fontsize=10, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Confusion matrix saved to: {save_path}')
    plt.show()
    plt.close()

def plot_tsne(features, labels, num_classes=3, save_path=None, use_pca=True, n_pca_components=50):
    """
    Plot t-SNE visualization using tsnecuda (GPU-accelerated t-SNE).
    Designed for 3-class classification with distinct colors.
    
    Args:
        features: Feature vectors
        labels: Class labels
        num_classes: Number of classes
        save_path: Path to save the plot
        use_pca: Whether to use PCA before t-SNE (recommended for high-dim data)
        n_pca_components: Number of PCA components (default: 50)
    """
    print('Computing t-SNE...')
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    if use_pca and features_normalized.shape[1] > n_pca_components:
        from sklearn.decomposition import PCA
        print(f'Applying PCA: {features_normalized.shape[1]} -> {n_pca_components} dimensions')
        pca = PCA(n_components=n_pca_components, random_state=42)
        features_normalized = pca.fit_transform(features_normalized)
        print(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}')
    
    n_samples = features_normalized.shape[0]
    perplexity = min(5, max(1, (n_samples - 1) // 30))
    print(f'Number of samples: {n_samples}, perplexity: {perplexity}')
    
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, n_iter=1000, verbose=1)
    features_tsne = tsne.fit_transform(features_normalized)
    
    if np.any(np.isnan(features_tsne)) or np.any(np.isinf(features_tsne)):
        print('Warning: t-SNE produced NaN or Inf values. Using PCA 2D instead.')
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        features_tsne = pca.fit_transform(features_normalized)
        print(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    colors = ['#2E8B57', '#FF8C00', '#DC143C']
    class_names = ['Corona', 'No PD', 'Surface']
    
    for label in range(num_classes):
        mask = labels == label
        ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                  c=colors[int(label)], 
                  label=class_names[int(label)] if label < len(class_names) else f'Class {int(label)}', 
                  s=80, alpha=0.7, edgecolors='none')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
    
    for label in range(num_classes):
        mask = labels == label
        if np.sum(mask) > 0:
            centroid_x = np.mean(features_tsne[mask, 0])
            centroid_y = np.mean(features_tsne[mask, 1])
            ax.text(centroid_x, centroid_y, str(label), 
                   fontsize=24, fontweight='bold', color='white',
                   ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='none', edgecolor='none'))
    
    margin = 5
    x_min, x_max = np.nanmin(features_tsne[:, 0]), np.nanmax(features_tsne[:, 0])
    y_min, y_max = np.nanmin(features_tsne[:, 1]), np.nanmax(features_tsne[:, 1])
    
    if np.isfinite(x_min) and np.isfinite(x_max):
        ax.set_xlim(x_min - margin, x_max + margin)
    if np.isfinite(y_min) and np.isfinite(y_max):
        ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.tick_params(labelsize=10, colors='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f't-SNE plot saved to: {save_path}')
    plt.show()
    plt.close()

def print_model_layers(model):
    print("Model Layers:")
    print("=" * 70)
    for name, module in model.named_modules():
        if isinstance(module, nn.Module):
            print(f"{name}:")
            print(module)
            print("-" * 70)
    print("=" * 70)

def cal_metrics_1shot(loader, net, device, num_classes):
    dict_tp = {i: 0 for i in range(num_classes)}
    dict_fp = {i: 0 for i in range(num_classes)}
    dict_fn = {i: 0 for i in range(num_classes)}
    
    num_batches = 0
    
    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = support_images.permute(1, 0, 2, 3, 4).to(device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)
        
        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            pred = torch.argmax(scores)
            
            if pred == target:
                dict_tp[int(target)] += 1
            else:
                dict_fp[int(target)] += 1
                dict_fn[int(pred)] += 1
            num_batches += 1
    
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    
    print("\n" + "="*70)
    print("Detailed Metrics per Class:")
    print("="*70)
    print(f"TP: {dict_tp}")
    print(f"FP: {dict_fp}")
    print(f"FN: {dict_fn}")
    print("="*70)
    
    for i in dict_tp.keys():
        precision_dict[i] = dict_tp[i] / (dict_tp[i] + dict_fp[i] + 1e-6)
        recall_dict[i] = dict_tp[i] / (dict_tp[i] + dict_fn[i] + 1e-6)
        f1_dict[i] = 2 * (precision_dict[i] * recall_dict[i]) / (precision_dict[i] + recall_dict[i] + 1e-6)
    
    print(f"Precision per class: {precision_dict}")
    print(f"Recall per class: {recall_dict}")
    print(f"F1-Score per class: {f1_dict}")
    print("="*70)
    
    precision = sum(precision_dict.values()) / len(precision_dict)
    recall = sum(recall_dict.values()) / len(recall_dict)
    accuracy = sum(dict_tp.values()) / num_batches
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return accuracy, f1_score, recall, precision

def cal_metrics_5shot(loader, net, device, num_classes):
    dict_tp = {i: 0 for i in range(num_classes)}
    dict_fp = {i: 0 for i in range(num_classes)}
    dict_fn = {i: 0 for i in range(num_classes)}
    
    num_batches = 0
    
    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = convert_for_5shots(support_images, support_targets, device)
        targets = query_targets.to(device)
        targets = targets.permute(1, 0)
        
        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            pred = torch.argmax(scores)
            
            if pred == target:
                dict_tp[int(target)] += 1
            else:
                dict_fp[int(target)] += 1
                dict_fn[int(pred)] += 1
            num_batches += 1
    
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    
    print("\n" + "="*70)
    print("Detailed Metrics per Class:")
    print("="*70)
    print(f"TP: {dict_tp}")
    print(f"FP: {dict_fp}")
    print(f"FN: {dict_fn}")
    print("="*70)
    
    for i in dict_tp.keys():
        precision_dict[i] = dict_tp[i] / (dict_tp[i] + dict_fp[i] + 1e-6)
        recall_dict[i] = dict_tp[i] / (dict_tp[i] + dict_fn[i] + 1e-6)
        f1_dict[i] = 2 * (precision_dict[i] * recall_dict[i]) / (precision_dict[i] + recall_dict[i] + 1e-6)
    
    print(f"Precision per class: {precision_dict}")
    print(f"Recall per class: {recall_dict}")
    print(f"F1-Score per class: {f1_dict}")
    print("="*70)
    
    precision = sum(precision_dict.values()) / len(precision_dict)
    recall = sum(recall_dict.values()) / len(recall_dict)
    accuracy = sum(dict_tp.values()) / num_batches
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return accuracy, f1_score, recall, precision