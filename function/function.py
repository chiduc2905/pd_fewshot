import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def seed_func(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, output, target):
        # output: (batch_size, way) or (way,)
        # target: (batch_size,) or scalar
        if output.dim() == 1:
            upper = torch.exp(output[target])
            lower = torch.exp(output).sum()
            loss = -torch.log(upper / lower)
        else:
            # Gather scores for target classes
            # output[i, target[i]]
            upper = torch.exp(output.gather(1, target.view(-1, 1)).squeeze())
            lower = torch.exp(output).sum(dim=1)
            loss = -torch.log(upper / lower)
            loss = loss.mean()
        return loss

def plot_confusion_matrix(true_labels, predictions, num_classes=3, save_path=None):
    """
    Plot confusion matrix for few-shot evaluation.
    
    For final test (150 episodes, 1-query/class): each row sums to 150.
    Shows count and percentage per cell.
    """
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Calculate percentages (row-wise)
    row_sums = conf_matrix.sum(axis=1)[:, np.newaxis]
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    conf_matrix_percent = conf_matrix.astype('float') / row_sums * 100
    
    # Row sum for title (should be 150 for final test)
    samples_per_class = int(conf_matrix.sum(axis=1)[0]) if conf_matrix.shape[0] > 0 else 0
    
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
    ax.set_title(f'Confusion Matrix ({samples_per_class} samples/class)', fontsize=13, fontweight='normal', pad=15)
    
    # Set tick labels
    ax.set_xticklabels(range(num_classes), fontsize=10)
    ax.set_yticklabels(range(num_classes), fontsize=10, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Confusion matrix saved to: {save_path}')
    plt.close()

def plot_tsne(features, labels, num_classes=3, save_path=None):
    """
    Plot t-SNE visualization of query features from few-shot test.

    For final test (150 episodes, 1-query/class): 150 * num_classes = 450 points.
    """
    # Run t-SNE
    n_samples = len(features)
    # Perplexity must be less than n_samples/3 and at least 5
    perp = min(30, max(5, n_samples // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        early_exaggeration=12.0,
        learning_rate=200,
        init='pca',
        random_state=42,
        verbose=0
    )
    features_embedded = tsne.fit_transform(features)

    # Setup plot
    plt.figure(figsize=(10, 8))
    sns.set_style("white")  # White background, no grid

    # Create scatter plot - use hue for colors, different markers per class
    marker_styles = ['o']  
    markers_dict = {i: marker_styles[i % len(marker_styles)] for i in range(num_classes)}

    scatter = sns.scatterplot(
        x=features_embedded[:, 0],
        y=features_embedded[:, 1],
        hue=labels,
        style=labels,
        palette=sns.color_palette("bright", n_colors=num_classes),
        markers=markers_dict,
        s=50,              # Slightly larger size for better visibility
        alpha=0.8,         # Slightly more opaque
        legend='full'
    )

    # Remove grid and spines
    sns.despine()
    plt.grid(False)

    # Improve legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Class')

    plt.title(f"t-SNE Visualization ({n_samples} samples)", fontsize=15)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f't-SNE plot saved to: {save_path}')
    plt.close()