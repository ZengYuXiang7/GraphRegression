import os
import onnx
import torch
import random
import numpy as np
from collections import deque
from torch.utils.data import Dataset
from .feature.graph_feature import extract_graph_feature




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



def get_torch_data(onnx_file, batch_size, cost_time, model):
    adjacent, node_features, static_features = extract_graph_feature(onnx_file, batch_size)
    edge_index = torch.from_numpy(np.array(np.where(adjacent > 0))).type(torch.long)
    node_features = np.array(node_features, dtype=np.float32)
    x = torch.from_numpy(node_features).type(torch.float)
    sf = torch.from_numpy(static_features).type(torch.float)
    y = torch.FloatTensor([cost_time])
    ops = x[:, :32]
    ops = torch.argmax(ops, dim=-1).unsqueeze(-1).float()

    # Compute graph structural features
    adj_np = np.array(adjacent)
    in_degree, out_degree = compute_node_degrees(adj_np)
    op_depth_raw = bfs_depth_from_start(adj_np)
    distance = compute_shortest_path_distance(adj_np)
    dir_pe_rw = compute_rw_pe(adj_np)
    dir_pe_ml = compute_magnetic_laplacian_pe(adj_np)

    data = {
        "ops": ops,
        "x": x,
        "code_adj": adjacent,
        "cost": y,
        "sf": sf,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "op_depth": torch.tensor(op_depth_raw, dtype=torch.long),
        "distance": distance,
        "dir_pe_rw": dir_pe_rw,
        "dir_pe_ml": dir_pe_ml,
    }
    return data


class GraphLatencyDataset(Dataset):
    # specific a platform
    def __init__(
        self,
        root,
        onnx_dir,
        latency_file,
        override_data=False,
        model_types=None,
        train_test_stage=None,
        n_finetuning=0,
        sample_num=-1,
        config=None,
    ):
        super(GraphLatencyDataset, self).__init__()
        self.config = config
        self.data_root = root
        self.onnx_dir = onnx_dir  # .../unseen_structre/
        self.latency_file = latency_file  # gt.txt
        self.latency_ids = []
        self.override_data = override_data
        self.model_types = model_types
        self.train_test_stage = train_test_stage  # =default=None
        self.device = None
        print("Extract input data from onnx...")
        self.custom_process()
        print("Done.")

        if sample_num > 0:
            random.seed(1234)
            random.shuffle(self.latency_ids)
            self.latency_ids = self.latency_ids[:sample_num]
        random.seed(1234)
        random.shuffle(self.latency_ids)

        if n_finetuning > 0:
            self.latency_ids = self.latency_ids[:n_finetuning]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []


    def custom_process(self):
        with open(self.latency_file) as f:  # gt.txt
            for line in f.readlines():

                line = line.rstrip()
                items = line.split(" ")
                speed_id = str(items[0])
                graph_id = str(items[1])
                batch_size = int(items[2])
                cost_time = float(items[3])
                plt_id = int(items[5])

                if self.model_types and items[4] not in self.model_types:
                    continue

                # if self.train_test_stage and items[6] != self.train_test_stage:
                #     continue

                onnx_file = os.path.join(
                    self.onnx_dir, graph_id
                )  # self.onnx_dir: ../..dataset/unseen_structure/${graph_id}
                if os.path.exists(onnx_file):
                    data_file = os.path.join(
                        self.data_root, "{}_{}_data.pt".format(speed_id, plt_id)
                    )
                    graph_name = "{}_{}_{}".format(graph_id, batch_size, plt_id)
                    self.latency_ids.append((data_file, None, graph_name, plt_id))
                    
                    # Example: speed_id=00428, plt_id=1, batch_size=1
                    # data_file: ../../dataset/unseen_structure/data/00428_1_data.pt
                    # graph_name: onnx/nnmeter_alexnet/nnmeter_alexnet_transform_0427.onnx_1_1

                    if (
                        (not self.override_data)
                        and os.path.exists(data_file)
                    ):
                        continue

                    if len(self.latency_ids) % 1000 == 0:
                        print(len(self.latency_ids))

                    GG = onnx.load(onnx_file)
                    data= get_torch_data(
                        GG, batch_size, cost_time, self.config.model
                    )  # process onnx file，cose_time:latency

                    torch.save(data, data_file)

    def __len__(self):
        return len(self.latency_ids)

    def __getitem__(self, idx):
        data_file, sf_file, graph_name, plt_id = self.latency_ids[idx]
        data = torch.load(data_file)
        return data
