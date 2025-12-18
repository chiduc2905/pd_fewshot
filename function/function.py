"""Utility functions: loss, seeding, and visualization."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def seed_func(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ContrastiveLoss(nn.Module):
    """Softmax cross-entropy loss for few-shot classification.
    
    (User requested name: ContrastiveLoss)
    Mathematically equivalent to: -log(exp(score_target) / sum(exp(scores)))
    """
    
    def forward(self, scores, targets):
        """
        Args:
            scores: (N, way_num) similarity scores
            targets: (N,) class labels
        """
        log_probs = torch.log_softmax(scores, dim=1)
        loss = -log_probs.gather(1, targets.view(-1, 1)).mean()
        return loss


class RelationLoss(nn.Module):
    """MSE loss for Relation Networks.
    
    From: Sung et al. "Learning to Compare: Relation Network for Few-Shot Learning" (CVPR 2018)
    
    Relation scores are in [0, 1], target is 1 for correct class, 0 for others.
    """
    
    def __init__(self):
        super(RelationLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, scores, targets):
        """
        Args:
            scores: (N, Way) relation scores (should be in [0,1] from sigmoid)
            targets: (N,) class labels
        Returns:
            MSE loss
        """
        N, Way = scores.size()
        
        # Create one-hot targets
        one_hot = torch.zeros(N, Way).to(scores.device)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        
        # MSE between scores and one-hot targets
        loss = self.mse(scores, one_hot)
        return loss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        use_gpu (bool): use gpu or not.
    """
    def __init__(self, num_classes=3, feat_dim=1600, use_gpu=True, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.device = device

        # Use device if provided, otherwise fallback to use_gpu flag
        if self.device:
             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        elif self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        
        # Normalize centers to unit sphere to match normalized features
        centers_norm = torch.nn.functional.normalize(self.centers, p=2, dim=1)
        
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(centers_norm, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, centers_norm.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long()
        if self.device:
            classes = classes.to(self.device)
        elif self.use_gpu:
            classes = classes.cuda()
            
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. 2017.
    
    Args:
        margin (float): margin for triplet loss
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss


def plot_confusion_matrix(targets, preds, num_classes=3, save_path=None, class_names=None):
    """
    Plot confusion matrix.
    
    For 150-episode test with 1-query/class: each row sums to 150.
    
    Args:
        targets: Ground truth labels
        preds: Predicted labels
        num_classes: Number of classes
        save_path: Path to save the figure
        class_names: List of class names (default: ['surface', 'corona', 'nopd'])
    """
    # Default class names
    if class_names is None:
        class_names = ['Surface', 'Corona', 'NoPD']
    
    # Set font properties globally for this plot (1.5x scale: 14->21, 16->24, 18->27)
    plt.rcParams.update({'font.size': 21, 'font.family': 'serif'})
    
    cm = confusion_matrix(targets, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100
    
    samples_per_class = int(cm.sum(axis=1)[0])
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Annotations: count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Greens',
                linewidths=2, linecolor='white', ax=ax,
                annot_kws={'size': 21, 'weight': 'bold'},
                vmin=0, square=True,
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_xlabel('Predicted Labels', fontsize=24, fontweight='bold')
    ax.set_ylabel('Actual Labels', fontsize=24, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=27, fontweight='bold')
    ax.set_xticklabels(class_names, fontsize=21)
    ax.set_yticklabels(class_names, rotation=0, fontsize=21)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
    plt.close()


def plot_tsne(features, labels, num_classes=3, save_path=None):
    """
    t-SNE visualization of query features.
    
    For 150-episode test: 450 points (150 per class).
    """
    # Set font properties globally for this plot
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    n = len(features)
    unique_n = len(np.unique(features, axis=0))
    print(f"t-SNE: Plotting {n} points (Unique: {unique_n})")
    
    # 1. StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 2. PCA (reduce to 30 dims or less)
    n_components = min(30, n, features.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    print(f"  PCA reduced to {n_components} dimensions")
    
    perp = min(30, max(5, n // 3))
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca')
    embedded = tsne.fit_transform(features_pca)
    
    # Rescale to fit within [-50, 50]
    max_val = np.abs(embedded).max()
    if max_val > 0:
        embedded = embedded / max_val * 45  # Scale to max 45 to leave margin
    
    plt.figure(figsize=(12, 10))
    sns.set_style('white')
    
    scatter = sns.scatterplot(
        x=embedded[:, 0], y=embedded[:, 1],
        hue=labels, palette='bright',
        s=80, alpha=0.8, legend='full'
    )
    
    sns.despine()
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
    plt.title(f't-SNE ({n} samples)', fontsize=20, fontweight='bold')
    plt.xlabel('Dim 1', fontsize=16, fontweight='bold')
    plt.ylabel('Dim 2', fontsize=16, fontweight='bold')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
    plt.close()


def plot_tsne_comparison(original_features, encoded_features, labels, num_classes=3, save_path=None):
    """
    t-SNE visualization comparing original (raw pixels) vs encoded features side-by-side.
    
    Args:
        original_features: Raw image features flattened (N, H*W*C)
        encoded_features: Features after encoder (N, feat_dim)
        labels: Class labels (N,)
        num_classes: Number of classes
        save_path: Path to save the figure
    """
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    
    n = len(labels)
    
    # Process original features
    scaler_orig = StandardScaler()
    orig_scaled = scaler_orig.fit_transform(original_features)
    n_comp_orig = min(50, n-1, original_features.shape[1])
    pca_orig = PCA(n_components=n_comp_orig, random_state=42)
    orig_pca = pca_orig.fit_transform(orig_scaled)
    perp = min(30, max(5, n // 3))
    tsne_orig = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca')
    orig_embedded = tsne_orig.fit_transform(orig_pca)
    
    # Process encoded features
    scaler_enc = StandardScaler()
    enc_scaled = scaler_enc.fit_transform(encoded_features)
    n_comp_enc = min(30, n-1, encoded_features.shape[1])
    pca_enc = PCA(n_components=n_comp_enc, random_state=42)
    enc_pca = pca_enc.fit_transform(enc_scaled)
    tsne_enc = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca')
    enc_embedded = tsne_enc.fit_transform(enc_pca)
    
    # Rescale both to [-45, 45]
    for embedded in [orig_embedded, enc_embedded]:
        max_val = np.abs(embedded).max()
        if max_val > 0:
            embedded[:] = embedded / max_val * 45
    
    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.set_style('white')
    
    # Original (Raw) t-SNE
    ax1 = axes[0]
    sns.scatterplot(
        x=orig_embedded[:, 0], y=orig_embedded[:, 1],
        hue=labels, palette='bright',
        s=80, alpha=0.8, legend=False, ax=ax1
    )
    ax1.set_title(f'Original Data (Raw Pixels)\n{n} samples', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Dim 1', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Dim 2', fontsize=16, fontweight='bold')
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    sns.despine(ax=ax1)
    
    # Encoded t-SNE
    ax2 = axes[1]
    scatter = sns.scatterplot(
        x=enc_embedded[:, 0], y=enc_embedded[:, 1],
        hue=labels, palette='bright',
        s=80, alpha=0.8, legend='full', ax=ax2
    )
    ax2.set_title(f'After Encoder\n{n} samples', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Dim 1', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Dim 2', fontsize=16, fontweight='bold')
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
    plt.close()


def plot_model_comparison_bar(model_results, training_samples, save_path=None):
    """
    Plot horizontal bar chart comparing model performance for 1-shot and 5-shot.
    
    Args:
        model_results: dict with model names as keys and dict {'1shot': acc, '5shot': acc} as values
                      Example: {'CosineNet': {'1shot': 0.9667, '5shot': 0.9800}, ...}
        training_samples: Number of training samples (for title)
        save_path: Path to save the figure
    
    Returns:
        fig: matplotlib figure object
    """
    # Set font properties
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    
    models = list(model_results.keys())
    acc_1shot = [model_results[m]['1shot'] * 100 for m in models]
    acc_5shot = [model_results[m]['5shot'] * 100 for m in models]
    
    # Sort by 5-shot accuracy (descending)
    sorted_indices = np.argsort(acc_5shot)[::-1]
    models = [models[i] for i in sorted_indices]
    acc_1shot = [acc_1shot[i] for i in sorted_indices]
    acc_5shot = [acc_5shot[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(models) * 0.8 + 2))
    
    y = np.arange(len(models))
    height = 0.35
    
    # Bars
    bars_5shot = ax.barh(y - height/2, acc_5shot, height, label='5 Shot', color='#5DA5DA', edgecolor='white')
    bars_1shot = ax.barh(y + height/2, acc_1shot, height, label='1 Shot', color='#FAA43A', edgecolor='white')
    
    # Add value labels on bars
    for bar, val in zip(bars_5shot, acc_5shot):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', ha='left', fontsize=11, color='#5DA5DA', fontweight='bold')
    
    for bar, val in zip(bars_1shot, acc_1shot):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', ha='left', fontsize=11, color='#FAA43A', fontweight='bold')
    
    # Customize
    ax.set_xlabel('Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Models', fontsize=16, fontweight='bold')
    ax.set_title(f'Performance distribution table for the case of {training_samples} samples', 
                 fontsize=18, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlim(50, 100)
    ax.legend(loc='lower right', fontsize=12)
    
    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
    
    return fig