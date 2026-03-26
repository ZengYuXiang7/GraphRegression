# coding : utf-8
import pickle
import numpy as np
import torch
import os
from collections import Counter

def load_data():
    """加载数据：返回原始索引、填充后的算子序列、以及每个图的节点规模。"""
    base = "data/nasbench101/all_nasbench101_onehot_op"
    if not os.path.exists(f"{base}.ops.pt"):
        raise FileNotFoundError(f"未找到数据文件: {base}.ops.pt")

    all_ops = torch.load(f"{base}.ops.pt", weights_only=False)
    
    # 过滤节点数 2-4 的图
    valid_idx = [i for i in range(len(all_ops)) if not (2 <= len(all_ops[i]) <= 4)]
    print(f"池化区准备就绪：共 {len(valid_idx)} 个候选架构。")

    max_len = max(len(all_ops[i]) for i in valid_idx)
    op_seqs = np.full((len(valid_idx), max_len), -1, dtype=np.int8)
    sizes = np.zeros(len(valid_idx), dtype=np.int8)

    for j, i in enumerate(valid_idx):
        ops = np.array(all_ops[i], dtype=np.int8)
        op_seqs[j, :len(ops)] = ops
        sizes[j] = len(ops)
        
    return np.array(valid_idx), op_seqs, sizes

def get_entropy(counts):
    """计算离散概率分布的香农熵。"""
    total = np.sum(counts)
    if total == 0: return 0
    p = counts[counts > 0] / total
    return -np.sum(p * np.log2(p))

def joint_entropy_sampling(valid_idx, op_seqs, sizes, target_num, cand_pool=5000):
    """
    联合熵最大化算法：同时考虑算子位置熵和节点规模熵。
    """
    n_pool, seq_len = op_seqs.shape
    
    # 映射特征到索引，加速计算
    u_ops = np.unique(op_seqs)
    op_map = {v: i for i, v in enumerate(u_ops)}
    u_sizes = np.unique(sizes)
    sz_map = {v: i for i, v in enumerate(u_sizes)}

    # 初始化
    selected_local = [np.random.randint(n_pool)]
    op_counts = np.zeros((seq_len, len(u_ops)), dtype=int)
    sz_counts = np.zeros(len(u_sizes), dtype=int)

    def add_to_stats(idx, op_c, sz_c):
        for p in range(seq_len):
            op_c[p, op_map[op_seqs[idx, p]]] += 1
        sz_c[sz_map[sizes[idx]]] += 1

    add_to_stats(selected_local[0], op_counts, sz_counts)

    for step in range(1, target_num):
        # 排除已选，随机候选加速
        remain = np.delete(np.arange(n_pool), selected_local)
        candidates = np.random.choice(remain, min(len(remain), cand_pool), replace=False)
        
        best_idx = -1
        max_total_gain = -1.0
        
        # 当前基准熵
        base_op_ent = sum(get_entropy(op_counts[p]) for p in range(seq_len))
        base_sz_ent = get_entropy(sz_counts)

        # 核心评估：寻找总熵增最大的样本
        for cand in candidates:
            # 模拟算子熵增
            new_op_ent = 0
            for p in range(seq_len):
                tmp_c = op_counts[p].copy()
                tmp_c[op_map[op_seqs[cand, p]]] += 1
                new_op_ent += get_entropy(tmp_c)
            
            # 模拟规模熵增
            tmp_sz = sz_counts.copy()
            tmp_sz[sz_map[sizes[cand]]] += 1
            new_sz_ent = get_entropy(tmp_sz)
            
            # 联合增益 (这里认为位置算子和节点数量权重相等)
            total_gain = (new_op_ent - base_op_ent) + (new_sz_ent - base_sz_ent)
            
            if total_gain > max_total_gain:
                max_total_gain = total_gain
                best_idx = cand
        
        selected_local.append(best_idx)
        add_to_stats(best_idx, op_counts, sz_counts)

        if (step + 1) % 100 == 0 or (step + 1) == target_num:
            avg_op_ent = sum(get_entropy(op_counts[p]) for p in range(seq_len)) / seq_len
            print(f"  进度 {step+1}/{target_num} | 平均算子熵: {avg_op_ent:.4f} | 规模分布: {dict(Counter(sizes[selected_local]))}")

    return valid_idx[selected_local]

if __name__ == "__main__":
    # 1. 环境准备
    save_dir = "./data/nasbench101"
    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, "101_joint_max_entropy.pkl")

    # 2. 加载全量池化数据 (包含算子序列和节点规模)
    try:
        valid_idx, op_seqs, sizes = load_data()
    except Exception as e:
        print(f"数据加载失败: {e}")
        exit()

    all_cluster_idx = {}
    # 3. 设置采样场景：100, 172, 424, 4236
    scenarios = [100, 172, 424, 4236]
    
    for num in scenarios:
        print(f"\n" + "="*40)
        print(f"执行【联合最大熵采样】: 目标 {num} 个样本")
        print(f"="*40)
        
        # 针对不同规模调整候选池大小，保证 4236 场景也能精准均衡
        current_cand_pool = 5000 if num < 1000 else 10000
        
        # 调用联合采样算法 (同时考虑 OP 分布和 Size 分布)
        # 注意：这里返回的是映射后的全局索引
        selected_global_idx = joint_entropy_sampling(
            valid_idx, 
            op_seqs, 
            sizes, 
            target_num=num, 
            cand_pool=current_cand_pool
        )
        
        # 存入字典
        all_cluster_idx[num] = selected_global_idx
        
        # 打印该场景下的最终分布报告
        final_sizes = [sizes[np.where(valid_idx == i)[0][0]] for i in selected_global_idx]
        print(f"-> 场景 {num} 完成。规模分布: {dict(Counter(final_sizes))}")

    # 4. 序列化保存结果
    with open(pkl_path, "wb") as fp:
        pickle.dump(all_cluster_idx, fp)
    
    print(f"\n" + "-"*40)
    print(f"所有任务已成功对齐并完成！")
    print(f"索引字典已保存至: {pkl_path}")
    print(f"包含场景: {list(all_cluster_idx.keys())}")