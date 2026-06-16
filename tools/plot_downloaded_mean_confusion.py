from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


CLASS_NAMES = ["Surface", "Internal", "Corona", "NotPD"]


MATRICES = {
    "deepemd_160train_1shot_mean_confusion": [
        [[123, 12, 0, 15], [35, 96, 18, 1], [0, 14, 136, 0], [5, 0, 0, 145]],
        [[142, 3, 0, 5], [16, 125, 9, 0], [0, 1, 149, 0], [1, 0, 0, 149]],
        [[130, 10, 0, 10], [20, 105, 24, 1], [1, 14, 135, 0], [4, 2, 0, 144]],
    ],
    "deepemd_160train_5shot_mean_confusion": [
        [[137, 10, 0, 3], [16, 106, 28, 0], [0, 2, 148, 0], [4, 5, 0, 141]],
        [[150, 0, 0, 0], [18, 116, 16, 0], [0, 1, 149, 0], [2, 0, 0, 148]],
        [[142, 1, 0, 7], [20, 111, 19, 0], [0, 12, 138, 0], [0, 0, 0, 150]],
    ],
    "dn4_160train_1shot_mean_confusion": [
        [[115, 25, 1, 9], [38, 84, 23, 5], [0, 15, 133, 2], [10, 3, 2, 135]],
        [[125, 8, 0, 17], [16, 112, 22, 0], [0, 4, 146, 0], [8, 3, 1, 138]],
        [[125, 14, 1, 10], [28, 102, 17, 3], [1, 14, 133, 2], [15, 2, 1, 132]],
    ],
    "dn4_160train_5shot_mean_confusion": [
        [[142, 4, 0, 4], [13, 116, 20, 1], [0, 6, 142, 2], [6, 0, 0, 144]],
        [[140, 6, 0, 4], [12, 121, 16, 1], [0, 10, 137, 3], [6, 6, 0, 138]],
        [[139, 3, 0, 8], [14, 108, 27, 1], [0, 5, 145, 0], [7, 6, 1, 136]],
    ],
}


def row_percent(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums * 100.0


def plot_mean_confusion(mean_pct: np.ndarray, save_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": 17,
        }
    )

    width = 7.16
    fig, ax = plt.subplots(figsize=(width, width))
    annot = np.empty_like(mean_pct, dtype=object)
    for i in range(mean_pct.shape[0]):
        for j in range(mean_pct.shape[1]):
            annot[i, j] = f"{mean_pct[i, j]:.1f}"

    sns.heatmap(
        mean_pct,
        annot=annot,
        fmt="",
        cmap="Greens",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 20.5},
        vmin=0,
        vmax=100,
        square=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontsize=17)
    ax.set_ylabel("True Label", fontsize=17)
    ax.set_xticklabels(CLASS_NAMES, fontsize=17, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES, fontsize=17, rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    out_dir = Path(r"C:\Users\Admin\Downloads\mean_confusion_pdfs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, seed_matrices in MATRICES.items():
        matrices = np.array(seed_matrices, dtype=float)
        mean_pct = np.stack([row_percent(m) for m in matrices], axis=0).mean(axis=0)
        out_path = out_dir / f"{name}.pdf"
        plot_mean_confusion(mean_pct, out_path)
        print(out_path)
        print(np.array2string(mean_pct, precision=1, suppress_small=False))


if __name__ == "__main__":
    main()
