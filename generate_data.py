import os
import torch
import random
import argparse
import json
import tqdm
from nnformer.models.encoders import tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="nasbench101", help="dataset type"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/nasbench101/nasbench101.json",
        help="path of json file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/nasbench101/",
        help="path of generated pt files",
    )
    parser.add_argument(
        "--n_percent", type=float, default=0.01, help="train proportion"
    )
    parser.add_argument(
        "--load_all", type=bool, default=True, help="load total dataset"
    )
    parser.add_argument(
        "--enc_dim", type=int, default=32, help="dim of operation encoding"
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        default="onehot_op",
        help="Type of position embedding: onehot_op|onehot_oppos|nape|nerf|trans",
    )
    parser.add_argument("--split_type", type=str, default="GATES", help="GATES|TNASP")
    args = parser.parse_args()
    return args


import numpy as np
from collections import deque

import numpy as np


def _get_split_meta_path(base_path: str) -> str:
    stem = base_path[:-3] if base_path.endswith(".pt") else base_path
    return f"{stem}.meta.pt"


def _get_split_field_path(base_path: str, field: str) -> str:
    stem = base_path[:-3] if base_path.endswith(".pt") else base_path
    return f"{stem}.{field}.pt"


def compute_rw_pe(A: np.ndarray, rw_steps: int = 3, pr: float = 0.05) -> np.ndarray:
    """
    方向性随机游走位置编码 (基于 Geisler et al., ICML 2023)
    修正：使用列聚合(落地概率分布)替代对角线(返回概率)，并补充 Personalized PageRank。
    Returns: (n, 2*rw_steps + 2) float32
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 2 * rw_steps + 2), dtype=np.float32)

    A = A.astype(np.float64)

    def _get_transition_matrix(M):
        # 计算出度，为 Sink 节点添加自环以保证概率转移矩阵合法 [cite: 208]
        out_deg = M.sum(axis=1, keepdims=True)
        sinks = out_deg.flatten() == 0

        M_walk = M.copy()
        M_walk[sinks, sinks] = 1.0
        out_deg[sinks] = 1.0

        return M_walk / out_deg  # Row-stochastic matrix P

    P_fwd = _get_transition_matrix(A)  # T
    P_rev = _get_transition_matrix(A.T)  # R

    features = []

    # 1. 前向有限步游走 (T^1 到 T^k)
    curr_P_fwd = P_fwd.copy()
    for _ in range(rw_steps):
        # 沿着列求和：假设初始在所有节点均匀分布，经过 k 步后落入节点 v 的概率
        features.append(curr_P_fwd.sum(axis=0))
        curr_P_fwd = curr_P_fwd @ P_fwd

    # 2. 反向有限步游走 (R^1 到 R^k)
    curr_P_rev = P_rev.copy()
    for _ in range(rw_steps):
        features.append(curr_P_rev.sum(axis=0))
        curr_P_rev = curr_P_rev @ P_rev

    # 3. Personalized PageRank (无限步带重启) [cite: 214, 215]
    # PPR = pr * (I - (1 - pr) * P)^-1
    I = np.eye(n)
    PPR_fwd = pr * np.linalg.inv(I - (1 - pr) * P_fwd)
    PPR_rev = pr * np.linalg.inv(I - (1 - pr) * P_rev)

    features.append(PPR_fwd.sum(axis=0))
    features.append(PPR_rev.sum(axis=0))

    return np.stack(features, axis=-1).astype(np.float32)


def compute_magnetic_laplacian_pe(
    A: np.ndarray, k: int = 5, q: float = 0.25
) -> np.ndarray:
    """
    Magnetic Laplacian 特征向量位置编码
    修正：相位基准点判定、k 的边界处理
    Returns: (n, k, 2) float32
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, k, 2), dtype=np.float32)

    A = np.array(A, dtype=np.float64)

    # 计算自适应势能 q_abs [cite: 144, 145]
    m_tilde = np.sum((A > 0) & (A.T == 0))
    d_G = max(min(m_tilde, n), 1)
    q_abs = q / d_G

    # 对称化及度矩阵
    A_s = np.maximum(A, A.T)
    d_s = A_s.sum(axis=1)

    # 构造归一化的 Magnetic Laplacian
    d_s_inv_sqrt = np.zeros_like(d_s)
    mask = d_s > 0
    d_s_inv_sqrt[mask] = 1.0 / np.sqrt(d_s[mask])
    D_inv_sqrt = np.diag(d_s_inv_sqrt)

    exp_iTheta = np.exp(1j * 2 * np.pi * q_abs * (A - A.T))
    L = np.eye(n, dtype=complex) - D_inv_sqrt @ A_s @ D_inv_sqrt * exp_iTheta

    # 提取特征向量，最大数量可达 n
    k_actual = min(k, n)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues.real)[:k_actual]]

    Gamma = eigenvectors.copy()

    # Algorithm D.1: Sign Normalization (解决正负号歧义)
    j_sign = np.argmax(np.abs(Gamma.real), axis=0)
    signs = np.sign(Gamma.real[j_sign, np.arange(k_actual)])
    signs[signs == 0] = 1
    Gamma *= signs[np.newaxis, :]

    # Algorithm D.1: Phase Normalization (旋转对齐)
    # 修正：寻找最大相位节点 (而不是最大虚部)，并对所有特征向量进行旋转对齐 [cite: 1183, 1184]
    root_node = np.argmax(np.angle(Gamma[:, 0]))
    alpha = np.angle(Gamma[root_node, :])
    Gamma *= np.exp(-1j * alpha)[np.newaxis, :]

    # 填充结果 (自动处理 n < k 的情况)
    result = np.zeros((n, k, 2), dtype=np.float32)
    result[:, :k_actual, 0] = Gamma.real.astype(np.float32)
    result[:, :k_actual, 1] = Gamma.imag.astype(np.float32)

    return result


def bfs_depth_from_start(adj: np.ndarray, start: int = 0) -> np.ndarray:
    """
    adj: [N, N] numpy array (0/1), directed by default (u->v if adj[u,v]==1)
    start: 起点节点编号
    return: depth [N], start=0 depth=0, unreachable=-1
    """
    adj = np.asarray(adj)
    N = adj.shape[0]
    depth = np.full(N, -1, dtype=int)
    depth[start] = 0

    q = deque([start])
    while q:
        u = q.popleft()
        # 找到 u 的所有出邻居
        nbrs = np.where(adj[u] != 0)[0]
        for v in nbrs:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                q.append(v)
    return list(depth)


def compute_shortest_path_distance(adj: np.ndarray) -> torch.Tensor:
    """
    计算点到点的最短路径距离矩阵（Floyd-Warshall 算法）

    adj: [N, N] numpy array (0/1), 有向图
    return: distance [N, N] torch.Tensor, distance[i][j] 是从节点 i 到节点 j 的最短路径长度
            如果不可达，设置为 -1
    """
    adj = np.asarray(adj)
    N = adj.shape[0]
    INF = N + 1  # 内部用N+1表示不可达，最后转换为-1

    # 初始化距离矩阵
    distance = np.full((N, N), INF, dtype=int)  # 默认距离为INF（不可达）

    # 对角线为0（自己到自己）
    np.fill_diagonal(distance, 0)

    # 直接相连的节点距离为1
    distance[adj != 0] = 1

    # Floyd-Warshall 算法
    for k in range(N):
        for i in range(N):
            for j in range(N):
                distance[i, j] = min(distance[i, j], distance[i, k] + distance[k, j])

    # 将不可达的距离设为-1
    distance[distance >= INF] = -1

    return torch.tensor(distance, dtype=torch.long)


def compute_node_degrees(adj: np.ndarray) -> tuple:
    """
    计算节点的入度和出度

    adj: [N, N] numpy array (0/1), 有向图
    return: (in_degree [N], out_degree [N])
    """
    adj = np.asarray(adj)
    # 入度：每列的和（有多少条边指向该节点）
    in_degree = np.sum(adj, axis=0)
    # 出度：每行的和（该节点指向多少条边）
    out_degree = np.sum(adj, axis=1)
    return in_degree, out_degree


def get_nasbench101_item(archs, i: int, enc_dim, embed_type):
    index = str(i)
    ops = archs[index]["module_operations"]
    adj = archs[index]["module_adjacency"]
    depth = len(ops)
    op_depth_raw = bfs_depth_from_start(adj)
    code, rel_pos, code_depth, op_depth = tokenizer(
        ops, adj, depth, op_depth_raw, enc_dim, embed_type
    )
    distance = compute_shortest_path_distance(np.array(adj))
    in_degree, out_degree = compute_node_degrees(np.array(adj))
    adj_np = np.array(adj)
    dir_pe_rw = compute_rw_pe(adj_np)  # (n, 16) float32
    dir_pe_ml = compute_magnetic_laplacian_pe(adj_np)  # (n, 5, 2) float32
    return {
        "index": i,
        "adj": adj,
        "ops": ops,
        "validation_accuracy": archs[index]["validation_accuracy"],
        "test_accuracy": archs[index]["test_accuracy"],
        "code": code,
        "code_rel_pos": rel_pos,
        "code_depth": code_depth,
        "op_depth_raw": op_depth_raw,
        "op_depth": op_depth,
        "distance": distance,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "dir_pe_rw": dir_pe_rw,
        "dir_pe_ml": dir_pe_ml,
    }


def get_nasbench201_item(archs, i: int, enc_dim, embed_type):
    index = str(i)
    ops = archs[index]["module_operations"]
    adj = archs[index]["module_adjacency"]
    depth = len([op for op in ops if op != 5])  # `op == 5` indicates `none`
    op_depth_raw = bfs_depth_from_start(adj)
    code, rel_pos, code_depth, op_depth = tokenizer(
        ops, adj, depth, op_depth_raw, enc_dim, embed_type
    )
    distance = compute_shortest_path_distance(np.array(adj))
    in_degree, out_degree = compute_node_degrees(np.array(adj))
    adj_np = np.array(adj)
    dir_pe_rw = compute_rw_pe(adj_np)  # (n, 16) float32
    dir_pe_ml = compute_magnetic_laplacian_pe(adj_np)  # (n, 5, 2) float32
    return {
        "index": i,
        "adj": adj,
        "ops": ops,
        "test_accuracy": archs[index]["test_accuracy"],
        "test_accuracy_avg": archs[index]["test_accuracy_avg"],
        "valid_accuracy": archs[index]["validation_accuracy"],
        "valid_accuracy_avg": archs[index]["validation_accuracy_avg"],
        "code": code,
        "code_rel_pos": rel_pos,
        "code_depth": code_depth,
        "op_depth": op_depth,
        "op_depth_raw": op_depth_raw,
        "distance": distance,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "dir_pe_rw": dir_pe_rw,
        "dir_pe_ml": dir_pe_ml,
    }


def main():
    args = parse_args()

    random.seed(args.seed)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.load_all:
        with open(args.data_path) as f:
            archs = json.load(f)

        id_list = list(range(0, len(archs)))
        # random.shuffle(id_list)

        data_columns = {}
        for i in tqdm.trange(len(archs)):
            if args.dataset == "nasbench101":
                item = get_nasbench101_item(archs, i, args.enc_dim, args.embed_type)
            elif args.dataset == "nasbench201":
                item = get_nasbench201_item(archs, i, args.enc_dim, args.embed_type)

            for key, value in item.items():
                if key not in data_columns:
                    data_columns[key] = []
                data_columns[key].append(value)

        all_file_path = os.path.join(
            save_dir, f"all_{args.dataset}_{args.embed_type}.pt"
        )
        meta = {
            "format": "split_fields_v1",
            "fields": list(data_columns.keys()),
            "length": len(archs),
        }
        torch.save(meta, _get_split_meta_path(all_file_path))
        for field, values in data_columns.items():
            torch.save(values, _get_split_field_path(all_file_path, field))

    if not args.load_all:
        if args.dataset == "nasbench101":
            torch.set_num_threads(1)
            train_data = {}
            test_data = {}
            val_data = {}
            with open(args.data_path) as f:
                archs = json.load(f)
            print(len(archs))
            id_list = list(range(0, len(archs)))
            # Split dataset following TNASP
            if args.split_type == "TNASP":
                random.shuffle(id_list)
                train_list = id_list
                l1 = int(len(archs) * args.n_percent)
                lv = int(len(archs) * (args.n_percent + 0.0005))
                l2 = int(len(archs) * (args.n_percent + 0.0005))  # val 0.05%

            # Split dataset following GATES
            if args.split_type == "GATES":
                train_list = id_list[: int(len(archs) * 0.9)]
                random.shuffle(train_list)
                l1 = int(len(archs) * 0.9 * args.n_percent)
                lv = int(len(archs) * (0.9 * args.n_percent + 0.0005))
                l2 = int(len(archs) * 0.9)

            for i in train_list[:l1]:
                idx = len(train_data)
                train_data[idx] = get_nasbench101_item(
                    archs, i, args.enc_dim, args.embed_type
                )
            torch.save(train_data, os.path.join(save_dir, "train.pt"))

            # for i in id_list[l1:l2]:
            for i in train_list[l1:lv]:
                idx = len(val_data)
                val_data[idx] = get_nasbench101_item(
                    archs, i, args.enc_dim, args.embed_type
                )
            torch.save(val_data, os.path.join(save_dir, "val.pt"))

            for i in id_list[l2:]:
                idx = len(test_data)
                test_data[idx] = get_nasbench101_item(
                    archs, i, args.enc_dim, args.embed_type
                )
            torch.save(test_data, os.path.join(save_dir, "test.pt"))

        elif args.dataset == "nasbench201":
            with open(args.data_path) as f:
                archs = json.load(f)
            print(len(archs))
            id_list = list(range(0, len(archs)))
            # Split dataset following TNASP
            if args.split_type == "TNASP":
                random.shuffle(id_list)
                train_list = id_list
                l1 = int(len(archs) * args.n_percent)
                lv = int(len(archs) * args.n_percent) + 200
                l2 = int(len(archs) * args.n_percent) + 200

            # Split dataset following GATES
            if args.split_type == "GATES":
                train_list = id_list[: int(len(archs) * 0.5)]
                random.shuffle(train_list)
                l1 = int(len(archs) * 0.5 * args.n_percent)
                # lv = int(len(archs)*(0.9*args.n_percent + 0.0005))
                l2 = int(len(archs) * 0.5)

            train_data, test_data = {}, {}
            for i in train_list[:l1]:
                idx = len(train_data)
                train_data[idx] = get_nasbench201_item(
                    archs, i, args.enc_dim, args.embed_type
                )
            torch.save(train_data, os.path.join(save_dir, "train.pt"))

            if args.split_type == "TNASP":
                val_data = {}
                for i in train_list[l1:lv]:
                    print(i)
                    idx = len(val_data)
                    val_data[idx] = get_nasbench201_item(
                        archs, i, args.enc_dim, args.embed_type
                    )
                torch.save(val_data, os.path.join(save_dir, "val.pt"))

            for i in id_list[l2:]:
                print(i)
                idx = len(test_data)
                test_data[idx] = get_nasbench201_item(
                    archs, i, args.enc_dim, args.embed_type
                )
            torch.save(test_data, os.path.join(save_dir, "test.pt"))


if __name__ == "__main__":
    main()
