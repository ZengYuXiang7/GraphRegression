# coding : utf-8
# Author : Yang Wang
# 最大熵采样：用离散 op 编码作为特征，最远点采样(汉明距离)保证训练集多样性

import pickle
import numpy as np
import torch


def load_ops(dataset="101_acc"):
    """直接加载 op 离散编码。"""
    base = "data/nasbench101/all_nasbench101_onehot_op"
    all_ops = torch.load(f"{base}.ops.pt", weights_only=False)

    # 过滤节点数 2-4 的图
    valid_idx = [i for i in range(len(all_ops)) if not (2 <= len(all_ops[i]) <= 4)]
    print(f"过滤前: {len(all_ops)} 个图，过滤后: {len(valid_idx)} 个图")

    # padding 到统一长度
    max_len = max(len(all_ops[i]) for i in valid_idx)
    op_seqs = np.full((len(valid_idx), max_len), -1, dtype=np.int8)
    for j, i in enumerate(valid_idx):
        ops = np.array(all_ops[i], dtype=np.int8)
        op_seqs[j, :len(ops)] = ops

    return valid_idx, op_seqs


def hamming_distance_batch(seq, seqs):
    """一个序列与一批序列的汉明距离（忽略 padding=-1）。"""
    valid = (seq != -1) & (seqs != -1)
    diff = (seq != seqs) & valid
    return diff.sum(axis=1)


def farthest_point_sampling(op_seqs, sample_num):
    """最远点采样（汉明距离）。"""
    n = len(op_seqs)
    min_dist = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)

    first = np.random.randint(n)
    selected = [first]
    min_dist = np.minimum(min_dist, hamming_distance_batch(op_seqs[first], op_seqs))

    for step in range(1, sample_num):
        next_idx = np.argmax(min_dist)
        selected.append(next_idx)
        min_dist = np.minimum(min_dist, hamming_distance_batch(op_seqs[next_idx], op_seqs))

        if (step + 1) % 100 == 0 or step == sample_num - 1:
            print(f"  step {step+1}/{sample_num}, max_min_dist={min_dist.max()}")

    return selected


if __name__ == "__main__":
    import os
    os.makedirs("data/nasbench101", exist_ok=True)

    valid_idx, op_seqs = load_ops()

    pkl_path = "./data/nasbench101/101_max_entropy_sample.pkl"
    all_cluster_idx = {}

    for sanerio in [100, 172, 424, 4236]:
        print(f"\n=== Sanerio {sanerio} ===")
        selected_local = farthest_point_sampling(op_seqs, sanerio)
        selected_idx = np.array([valid_idx[i] for i in selected_local], dtype=int)
        print(f"采样 {len(selected_idx)} 个图")
        all_cluster_idx[sanerio] = selected_idx

    with open(pkl_path, "wb") as fp:
        pickle.dump(all_cluster_idx, fp)
    print(f"\nDone! Saved to {pkl_path}")
