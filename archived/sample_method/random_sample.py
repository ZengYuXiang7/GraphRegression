# coding : utf-8
# Author : Yang Wang
# 纯随机采样：先过滤小图(节点数2-5)，再生成多轮随机采样索引并保存为 pkl

import os
import pickle
import numpy as np
import torch
import argparse


def get_valid_indices(dataset="101_acc", min_nodes=6):
    """
    加载数据集，返回节点数 >= min_nodes 的样本的原始索引。
    过滤掉节点数为 2~5 的小图。
    """
    if dataset == "101_acc":
        ops_path = "data/nasbench101/all_nasbench101_onehot_op.ops.pt"
        all_ops = torch.load(ops_path, weights_only=False)
        total = len(all_ops)
        valid_idx = []
        for i in range(total):
            n_nodes = len(all_ops[i])
            if n_nodes >= min_nodes:
                valid_idx.append(i)
        print(f"总样本: {total}, 过滤后(节点数>={min_nodes}): {len(valid_idx)}, "
              f"过滤掉: {total - len(valid_idx)}")
        return np.array(valid_idx, dtype=int)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")


def random_sample_from_pool(valid_idx, sample_num, n_rounds=5, seed_base=42):
    """
    从 valid_idx 中做多轮随机采样，每轮选 sample_num 个。

    Returns
    -------
    rounds_idx : list[np.ndarray]
        长度为 n_rounds，每个元素是该轮采样的原始索引数组。
    """
    pool_size = len(valid_idx)
    rounds_idx = []
    for r in range(n_rounds):
        rng = np.random.RandomState(seed_base + r)
        chosen = rng.choice(pool_size, sample_num, replace=False)
        idx = np.sort(valid_idx[chosen])
        rounds_idx.append(idx)
    return rounds_idx


def generate_random_samples(n_rounds=5, seed_base=42, dataset="101_acc"):
    """
    为所有 scenario 生成多轮随机采样索引，保存到 pkl。

    pkl 结构: {scenario: list[np.ndarray]}
        scenario 是采样数量 (int)，list 长度 = n_rounds
    """
    valid_idx = get_valid_indices(dataset=dataset)

    all_scenarios = [100, 172, 424, 4236]
    result = {}

    for scenario in all_scenarios:
        rounds_idx = random_sample_from_pool(valid_idx, scenario, n_rounds, seed_base)
        result[scenario] = rounds_idx
        print(f"scenario={scenario}: {n_rounds} 轮, 每轮 {scenario} 个样本")

    os.makedirs("data/nasbench101", exist_ok=True)
    pkl_path = f"data/nasbench101/101_random_sample_{n_rounds}rounds.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    print(f"已保存到 {pkl_path}")
    return pkl_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_rounds", type=int, default=5,
                   help="随机采样轮数")
    p.add_argument("--seed_base", type=int, default=42,
                   help="随机种子基数，第 r 轮种子 = seed_base + r")
    args = p.parse_args()

    generate_random_samples(
        n_rounds=args.n_rounds,
        seed_base=args.seed_base,
    )
