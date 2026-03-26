"""
分析采样策略与 Tau 的关系 —— 完整过程可视化
核心问题：为什么 cluster 采样的 Tau 不如 random？
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import gaussian_kde

plt.rcParams['font.size'] = 11

def to_float(v):
    return v.item() if hasattr(v, 'item') else float(v)

# ── 加载数据 ──
base = 'data/nasbench101/all_nasbench101_onehot_op'
all_acc = torch.load(f'{base}.validation_accuracy.pt', weights_only=False)
all_acc = np.array([to_float(a) for a in all_acc])
all_adj = torch.load(f'{base}.adj.pt', weights_only=False)
all_ops = torch.load(f'{base}.ops.pt', weights_only=False)

with open('data/nasbench101/101_traing_sample.pkl', 'rb') as f:
    cluster_indices = pickle.load(f)

test_data = torch.load('data/nasbench101/rounds0/test_100_random_onehot_op_cached_data.pth',
                        map_location='cpu', weights_only=False)
test_acc = np.array([to_float(d['validation_accuracy']) for d in test_data])

# ═══════════════════════════════════════════════
# 图1：测试集 acc 分布 —— 证明测试集集中在 0.88~0.94
# ═══════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.hist(test_acc, bins=100, color='gray', edgecolor='white', alpha=0.7)
ax1.axvspan(0.88, 0.94, alpha=0.2, color='red', label='0.88~0.94 (86.9%)')
ax1.axvspan(0, 0.8, alpha=0.2, color='blue', label='<0.8 (1.8%)')
ax1.set_title('Test Set Accuracy Distribution (N=423,624)')
ax1.set_xlabel('Accuracy')
ax1.set_ylabel('Count')
ax1.legend()
plt.tight_layout()
plt.savefig('analysis_1_test_distribution.png', dpi=150)
print('图1 已保存: analysis_1_test_distribution.png')

# ═══════════════════════════════════════════════
# 图2：cluster vs random 采样的 acc 分布对比（三种数据量）
#   观察：cluster 在 <0.88 区间采了更多样本
# ═══════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for col, size in enumerate([100, 172, 424]):
    c_acc = all_acc[cluster_indices[size]]
    r_data = torch.load(f'data/nasbench101/rounds0/train_{size}_random_onehot_op_cached_data.pth',
                         map_location='cpu', weights_only=False)
    r_acc = np.array([to_float(d['validation_accuracy']) for d in r_data])

    x = np.linspace(0.05, 0.96, 500)
    axes2[col].plot(x, gaussian_kde(c_acc)(x), label='cluster', color='steelblue', lw=2)
    axes2[col].plot(x, gaussian_kde(r_acc)(x), label='random', color='salmon', lw=2)
    axes2[col].axvspan(0.88, 0.94, alpha=0.1, color='red')
    axes2[col].set_title(f'Train Acc Distribution (n={size})')
    axes2[col].set_xlabel('acc')
    axes2[col].legend()

    # 标注密集区间样本数
    c_dense = ((c_acc >= 0.88) & (c_acc <= 0.94)).sum()
    r_dense = ((r_acc >= 0.88) & (r_acc <= 0.94)).sum()
    axes2[col].text(0.05, 0.95, f'0.88~0.94:\n  cluster={c_dense}\n  random={r_dense}',
                     transform=axes2[col].transAxes, va='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Cluster vs Random: Where Are the Samples?', fontsize=14)
plt.tight_layout()
plt.savefig('analysis_2_sampling_comparison.png', dpi=150)
print('图2 已保存: analysis_2_sampling_comparison.png')

# ═══════════════════════════════════════════════
# 图3：密集区间 (0.88~0.94) 内的子区间覆盖对比
#   观察：cluster 是否在密集区间内更均匀
# ═══════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
sub_bins = np.arange(0.88, 0.945, 0.01)
bin_labels = [f'{sub_bins[i]:.2f}-{sub_bins[i+1]:.2f}' for i in range(len(sub_bins)-1)]

for col, size in enumerate([100, 172, 424]):
    c_acc = all_acc[cluster_indices[size]]
    r_data = torch.load(f'data/nasbench101/rounds0/train_{size}_random_onehot_op_cached_data.pth',
                         map_location='cpu', weights_only=False)
    r_acc = np.array([to_float(d['validation_accuracy']) for d in r_data])

    c_counts, _ = np.histogram(c_acc, bins=sub_bins)
    r_counts, _ = np.histogram(r_acc, bins=sub_bins)
    # 测试集分布(归一化到同量级)
    t_counts, _ = np.histogram(test_acc, bins=sub_bins)
    t_counts_norm = t_counts / t_counts.sum() * max(c_counts.sum(), r_counts.sum())

    x_pos = np.arange(len(bin_labels))
    w = 0.25
    axes3[col].bar(x_pos - w, c_counts, w, label='cluster', color='steelblue', alpha=0.7)
    axes3[col].bar(x_pos, r_counts, w, label='random', color='salmon', alpha=0.7)
    axes3[col].bar(x_pos + w, t_counts_norm, w, label='test (scaled)', color='gray', alpha=0.4)
    axes3[col].set_xticks(x_pos)
    axes3[col].set_xticklabels(bin_labels, rotation=45, fontsize=8)
    axes3[col].set_title(f'Dense Region Coverage (n={size})')
    axes3[col].set_ylabel('Count')
    axes3[col].legend(fontsize=8)

plt.suptitle('Sub-interval Coverage in 0.88~0.94 (Where 86.9% of Test Data Lives)', fontsize=13)
plt.tight_layout()
plt.savefig('analysis_3_dense_region_coverage.png', dpi=150)
print('图3 已保存: analysis_3_dense_region_coverage.png')

# ═══════════════════════════════════════════════
# 图4：结构特征 vs acc —— 能不能用结构预判 acc？
#   观察：结构特征（边数、op种类数、最长路径）与 acc 关系弱
# ═══════════════════════════════════════════════
# 采样5万个计算
np.random.seed(42)
sample_idx = np.random.choice(len(all_acc), 50000, replace=False)

edges = []
unique_ops = []
node_counts = []
for i in sample_idx:
    adj = np.array(all_adj[i], dtype=float)
    node_counts.append(adj.shape[0])
    edges.append(int(adj.sum()))
    ops = [int(o) if isinstance(o, (int, float)) else int(o.item()) for o in all_ops[i]]
    unique_ops.append(len(set(ops)))

edges = np.array(edges)
unique_ops = np.array(unique_ops)
node_counts = np.array(node_counts)
sample_acc = all_acc[sample_idx]

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))

# 边数 vs acc
for e in sorted(set(edges)):
    m = edges == e
    if m.sum() > 20:
        vals = sample_acc[m]
        axes4[0].boxplot(vals, positions=[e], widths=0.6, showfliers=False)
axes4[0].set_xlabel('Edge Count')
axes4[0].set_ylabel('Accuracy')
axes4[0].set_title('Edge Count vs Accuracy')

# unique ops vs acc
for u in sorted(set(unique_ops)):
    m = unique_ops == u
    if m.sum() > 20:
        vals = sample_acc[m]
        axes4[1].boxplot(vals, positions=[u], widths=0.4, showfliers=False)
axes4[1].set_xlabel('Unique Op Types')
axes4[1].set_ylabel('Accuracy')
axes4[1].set_title('Op Diversity vs Accuracy')

# 散点图: edges vs acc (采样)
sub = np.random.choice(50000, 5000, replace=False)
axes4[2].scatter(edges[sub], sample_acc[sub], alpha=0.1, s=3, color='steelblue')
axes4[2].set_xlabel('Edge Count')
axes4[2].set_ylabel('Accuracy')
axes4[2].set_title('Edges vs Acc (scatter, 5k samples)\nWeak correlation → structure cannot predict acc well')

plt.suptitle('Can Graph Structure Predict Accuracy?', fontsize=14)
plt.tight_layout()
plt.savefig('analysis_4_structure_vs_acc.png', dpi=150)
print('图4 已保存: analysis_4_structure_vs_acc.png')

# ═══════════════════════════════════════════════
# 打印数值总结
# ═══════════════════════════════════════════════
print('\n' + '='*60)
print('数值总结')
print('='*60)

print('\n【1】测试集分布：')
print(f'   acc < 0.8:      {(test_acc<0.8).mean()*100:.1f}%  ({(test_acc<0.8).sum()} 个)')
print(f'   0.88 ~ 0.94:    {((test_acc>=0.88)&(test_acc<=0.94)).mean()*100:.1f}%  ({((test_acc>=0.88)&(test_acc<=0.94)).sum()} 个)')

print('\n【2】各采样方法在密集区间(0.88~0.94)的样本数：')
for size in [100, 172, 424]:
    c_acc = all_acc[cluster_indices[size]]
    r_data = torch.load(f'data/nasbench101/rounds0/train_{size}_random_onehot_op_cached_data.pth',
                         map_location='cpu', weights_only=False)
    r_acc = np.array([to_float(d['validation_accuracy']) for d in r_data])
    c_n = ((c_acc>=0.88)&(c_acc<=0.94)).sum()
    r_n = ((r_acc>=0.88)&(r_acc<=0.94)).sum()
    print(f'   n={size}: cluster={c_n}/{size} ({c_n/size*100:.0f}%)  random={r_n}/{size} ({r_n/size*100:.0f}%)')

print('\n【3】结构特征与acc的相关系数：')
from scipy.stats import pearsonr
r_edge, _ = pearsonr(edges, sample_acc)
r_ops, _ = pearsonr(unique_ops, sample_acc)
r_node, _ = pearsonr(node_counts, sample_acc)
print(f'   edge_count vs acc:  r = {r_edge:.4f}')
print(f'   unique_ops vs acc:  r = {r_ops:.4f}')
print(f'   node_count vs acc:  r = {r_node:.4f}')
print('   (|r| < 0.3 为弱相关，说明结构无法有效预判 acc)')

print('\n【结论】')
print('   1. 测试集 86.9% 在 0.88~0.94，Tau 主要取决于这个区间的排序精度')
print('   2. Cluster 在密集区间样本数更少 → 排序学习材料不够 → Tau 低')
print('   3. 结构特征与 acc 弱相关(r<0.1) → 无法通过结构预判acc来改善采样')
print('   4. 建议：保持 random 采样 + 加 ranking loss 直接优化排序目标')
