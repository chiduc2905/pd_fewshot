"""Utility functions: loss, seeding, and visualization."""
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


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


def plot_confusion_matrix(targets, preds, num_classes=3, save_path=None):
    """
    Plot confusion matrix.
    
    For 150-episode test with 1-query/class: each row sums to 150.
    """
    cm = confusion_matrix(targets, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100
    
    samples_per_class = int(cm.sum(axis=1)[0])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Annotations: count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Greens',
                linewidths=2, linecolor='white', ax=ax,
                annot_kws={'size': 11, 'weight': 'bold'},
                vmin=0, square=True)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix ({samples_per_class} samples/class)', fontsize=13)
    ax.set_xticklabels(range(num_classes))
    ax.set_yticklabels(range(num_classes), rotation=0)
    
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
    n = len(features)
    unique_n = len(np.unique(features, axis=0))
    print(f"t-SNE: Plotting {n} points (Unique: {unique_n})")
    
    perp = min(30, max(5, n // 3))
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca')
    embedded = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    sns.set_style('white')
    
    scatter = sns.scatterplot(
        x=embedded[:, 0], y=embedded[:, 1],
        hue=labels, palette='bright',
        s=50, alpha=0.8, legend='full'
    )
    
    sns.despine()
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f't-SNE ({n} samples)', fontsize=15)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
    plt.close()