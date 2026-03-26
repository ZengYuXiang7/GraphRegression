"""分析 cluster 和 random 采样数据的 acc 分布"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def to_float(v):
    return v.item() if hasattr(v, 'item') else float(v)

sizes = [100, 172, 424]
fig, axes = plt.subplots(len(sizes), 3, figsize=(18, 5 * len(sizes)))

for row, size in enumerate(sizes):
    cluster_data = torch.load(f"data/nasbench101/rounds0/train_{size}_cluster_onehot_op_cached_data.pth",
                              map_location="cpu", weights_only=False)
    random_data = torch.load(f"data/nasbench101/rounds0/train_{size}_random_onehot_op_cached_data.pth",
                             map_location="cpu", weights_only=False)

    cluster_acc = np.array([to_float(d["validation_accuracy"]) for d in cluster_data])
    random_acc = np.array([to_float(d["validation_accuracy"]) for d in random_data])

    print(f"=== {size} 样本 ===")
    print(f"  cluster: mean={cluster_acc.mean():.4f}, std={cluster_acc.std():.4f}, "
          f"min={cluster_acc.min():.4f}, max={cluster_acc.max():.4f}, median={np.median(cluster_acc):.4f}")
    print(f"  random:  mean={random_acc.mean():.4f}, std={random_acc.std():.4f}, "
          f"min={random_acc.min():.4f}, max={random_acc.max():.4f}, median={np.median(random_acc):.4f}")
    print()

    # 直方图
    axes[row, 0].hist(cluster_acc, bins=50, alpha=0.6, label="cluster", color="steelblue", edgecolor="white")
    axes[row, 0].hist(random_acc, bins=50, alpha=0.6, label="random", color="salmon", edgecolor="white")
    axes[row, 0].set_title(f"Histogram (n={size})")
    axes[row, 0].set_xlabel("acc")
    axes[row, 0].set_ylabel("Count")
    axes[row, 0].legend()

    # KDE
    x_range = np.linspace(min(cluster_acc.min(), random_acc.min()) - 0.01,
                           max(cluster_acc.max(), random_acc.max()) + 0.01, 500)
    axes[row, 1].plot(x_range, gaussian_kde(cluster_acc)(x_range), label="cluster", color="steelblue", linewidth=2)
    axes[row, 1].plot(x_range, gaussian_kde(random_acc)(x_range), label="random", color="salmon", linewidth=2)
    axes[row, 1].set_title(f"KDE (n={size})")
    axes[row, 1].set_xlabel("acc")
    axes[row, 1].set_ylabel("Density")
    axes[row, 1].legend()

    # Boxplot
    bp = axes[row, 2].boxplot(
        [cluster_acc, random_acc],
        labels=["cluster", "random"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["steelblue", "salmon"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[row, 2].set_title(f"Boxplot (n={size})")
    axes[row, 2].set_ylabel("Accuracy")

plt.tight_layout()
plt.savefig("acc_distribution_comparison.png", dpi=150)
print("图片已保存到 acc_distribution_comparison.png")
plt.show()
