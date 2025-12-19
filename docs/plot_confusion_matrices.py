"""
Script to recreate 5 confusion matrices with green colormap style
Layout: 3 on top, 2 centered on bottom
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Define the 5 confusion matrices data (from the original image)
matrices_data = [
    ("(1)", np.array([[138, 8, 4], [5, 143, 2], [0, 5, 145]])),
    ("(2)", np.array([[145, 0, 5], [10, 127, 13], [0, 5, 145]])),
    ("(3)", np.array([[145, 3, 2], [17, 112, 21], [1, 33, 116]])),
    ("(4)", np.array([[143, 6, 1], [25, 70, 55], [3, 7, 140]])),
    ("(5)", np.array([[146, 3, 1], [40, 77, 33], [8, 28, 114]]))
]

# Class labels
class_labels = ['Surface', 'Corona', 'NoPD']
samples_per_class = 150

# Create figure with GridSpec for custom layout
fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.4)

# Top row: 3 matrices (spanning 2 columns each)
axes_top = [
    fig.add_subplot(gs[0, 0:2]),  # (1)
    fig.add_subplot(gs[0, 2:4]),  # (2)
    fig.add_subplot(gs[0, 4:6]),  # (3)
]

# Bottom row: 2 matrices centered (offset by 1 column each side)
axes_bottom = [
    fig.add_subplot(gs[1, 1:3]),  # (4) - centered
    fig.add_subplot(gs[1, 3:5]),  # (5) - centered
]

axes = axes_top + axes_bottom

# Green colormap
cmap = 'Greens'

for idx, (title, cm) in enumerate(matrices_data):
    ax = axes[idx]
    
    # Calculate percentages
    cm_percent = cm / samples_per_class * 100
    
    # Create annotations with count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
    
    # Plot heatmap
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, 
                xticklabels=class_labels, yticklabels=class_labels,
                ax=ax, cbar=True, vmin=0, vmax=150,
                annot_kws={'size': 10, 'weight': 'bold'})
    
    ax.set_title(f'{title} Confusion Matrix (150 samples/class)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)

plt.tight_layout()
plt.savefig('confusion_matrices_5x.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('confusion_matrices_5x.pdf', bbox_inches='tight', facecolor='white')
plt.show()

print("Saved: confusion_matrices_5x.png and confusion_matrices_5x.pdf")
