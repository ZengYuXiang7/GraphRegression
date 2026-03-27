# 请使用python 3.10版本运行此代码
# coding : utf-8
# Author : Yang Wang

import pickle
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from karateclub import Graph2Vec
from karateclub import FeatherGraph
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def graph_clustering(
    adjs,
    features,
    model_type="Graph2Vec",
    dimensions=64,
    workers=2,
    epochs=10,
    n_clusters=3,
    random_state=42,
):
    """
    根据模型类型执行不同的图聚类方法。
    :param adjs: 邻接矩阵, 形状为 (m, n, n), m是图的数量，n是每个图的节点数
    :param features: 节点特征, 形状为 (m, n, d), m是图的数量，n是节点数，d是节点特征的维度
    :param model_type: 使用的模型类型 ('Graph2Vec' 或 'FeatherGraph')
    :param dimensions: 嵌入维度
    :param workers: Graph2Vec 的工作线程数
    :param epochs: 训练轮数
    :param n_clusters: KMeans 的聚类簇数
    :param random_state: KMeans 的随机种子
    :return: 聚类标签 (kmeans.labels_)
    """
    if model_type == "Graph2Vec":
        return graph_clustering_Graph2Vec(
            adjs,
            features,
            dimensions=dimensions,
            workers=workers,
            epochs=epochs,
            n_clusters=n_clusters,
            random_state=random_state,
        )
    elif model_type == "FeatherGraph":
        return graph_clustering_FeatherGraph(
            adjs,
            features,
            n_clusters=n_clusters,
            dimensions=dimensions,
            epochs=epochs,
            random_state=random_state,
        )
    else:
        raise ValueError("模型类型无效!")


def graph_clustering_Graph2Vec(
    adjs, features, dimensions=64, workers=2, epochs=10, n_clusters=3, random_state=42
):
    """
    对图进行聚类
    :param adjs: 邻接矩阵, 形状为 (m, n, n), m是图的数量，n是每个图的节点数
    :param features: 节点特征, 形状为 (m, n, d), m是图的数量，n是节点数，d是节点特征的维度
    """
    m, n, d = adjs.shape
    graphs = []
    for i in range(m):
        adj = adjs[i]
        G = nx.from_numpy_array(adj)
        for j in range(n):
            G.nodes[j]["feature"] = features[i][j].tolist()
        graphs.append(G)

    graph2vec = Graph2Vec(dimensions=dimensions, workers=workers, epochs=epochs)
    graph2vec.fit(graphs)
    embeddings = graph2vec.get_embedding()

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embeddings)
    return kmeans.labels_


def graph_clustering_FeatherGraph(
    adjs, features, n_clusters=3, dimensions=64, epochs=10, random_state=42
):
    """
    对图进行聚类
    :param adjs: 邻接矩阵, 形状为 (m, n, n), m是图的数量，n是每个图的节点数
    :param features: 节点特征, 形状为 (m, n, d), m是图的数量，n是节点数，d是节点特征的维度
    """
    m, n, d = adjs.shape
    graphs = []
    for i in range(m):
        adj = adjs[i]
        G = nx.from_numpy_array(adj)
        for j in range(n):
            G.nodes[j]["feature"] = features[i][j].tolist()
        graphs.append(G)

    feather_graph = FeatherGraph()
    feather_graph.fit(graphs)
    embeddings = feather_graph.get_embedding()

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, verbose=True)
    kmeans.fit(embeddings)
    return kmeans.labels_


def get_dataset_graph(dataset="101_acc"):
    """
    从数据集加载图，返回 networkx.Graph 列表。
    每个节点的 "feature" 属性为该节点操作的 one-hot 编码（整数列表）。
    """
    all_graphs = []

    if dataset == "101_acc":
        # 从 nasbench101 分列 .pt 文件加载
        base = "data/nasbench101/all_nasbench101_onehot_op"
        all_adj = torch.load(f"{base}.adj.pt", weights_only=False)
        all_ops = torch.load(f"{base}.ops.pt", weights_only=False)
        for i in tqdm(range(len(all_adj)), desc="构建图 (101_acc)"):
            adj = np.array(all_adj[i], dtype=np.float32)        # [L, L]
            ops = all_ops[i]                                     # [L]

            graph = nx.from_numpy_array(adj)
            for j in range(len(ops)):
                graph.nodes[j]["feature"] = int(ops[j])
            all_graphs.append(graph)

    elif dataset == "201_acc":
        raw_data = pickle.load(open("data/201_acc_data.pkl", "rb"))
        adj_matrix = raw_data["adj_matrix"]
        all_features = raw_data["features"]
        for i in tqdm(range(len(adj_matrix)), desc="构建图 (201_acc)"):
            graph = nx.from_numpy_array(np.array(adj_matrix[i]))
            for j in range(graph.number_of_nodes()):
                node_id = int(np.array(all_features[i][j]))
                graph.nodes[j]["feature"] = np.eye(5, dtype=int)[node_id].tolist()
            all_graphs.append(graph)

    elif dataset == "nnlqp":
        pass

    return all_graphs


def get_graph2vec_clusters(graphs, dimensions=128, epochs=100):
    """
    使用 Graph2Vec 生成图嵌入。

    Returns
    -------
    embeddings : np.ndarray  shape [N, dimensions]
    """
    model = Graph2Vec(dimensions=dimensions, workers=4, epochs=epochs)
    model.fit(graphs)
    embeddings = model.get_embedding()
    return embeddings


def select_diverse_by_kmeans(embeddings: np.ndarray, n_clusters: int = 100):
    """
    对嵌入做 KMeans，每个 cluster 选最近中心的代表索引。
    返回 selected_idx: np.ndarray shape [n_clusters]
    """
    X = normalize(embeddings)   # L2-normalize，让距离更稳定
    km = KMeans(n_clusters=n_clusters, n_init="auto")
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    selected_idx = []
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        dist = np.linalg.norm(X[idx] - centers[c], axis=1)
        rep = idx[np.argmin(dist)]
        selected_idx.append(rep)

    return np.array(selected_idx, dtype=int)


def _cluster_once(n_clusters, dataset="101_acc"):
    """聚类一次，返回 (valid_idx, cluster_members)。"""
    all_graphs = get_dataset_graph(dataset=dataset)
    valid_idx = [i for i, g in enumerate(all_graphs) if not (2 <= g.number_of_nodes() <= 4)]
    filtered_graphs = [all_graphs[i] for i in valid_idx]
    print(f"过滤前: {len(all_graphs)} 个图，过滤后: {len(filtered_graphs)} 个图")

    embeddings = get_graph2vec_clusters(filtered_graphs, dimensions=128, epochs=100)
    X = normalize(embeddings)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(X)

    cluster_members = {}
    for i, label in enumerate(labels):
        cluster_members.setdefault(label, []).append(i)
    return valid_idx, cluster_members


def balanced_cluster_sample(sample_num, valid_idx, cluster_members):
    """
    给定已聚类的结果，从每个簇中均匀采样，总数严格等于 sample_num。
    余数按簇大小降序分配。
    """
    n_clusters = len(cluster_members)
    sorted_clusters = sorted(cluster_members.keys(), key=lambda c: len(cluster_members[c]), reverse=True)

    base = sample_num // n_clusters
    remainder = sample_num % n_clusters
    quota = {}
    for i, c in enumerate(sorted_clusters):
        quota[c] = base + (1 if i < remainder else 0)

    print(f"n_clusters={n_clusters}, sample_num={sample_num}")
    for c in sorted_clusters:
        print(f"  簇 {c}: {len(cluster_members[c])} 个图, 采样 {quota[c]} 个")

    selected_local = []
    for c in sorted_clusters:
        members = np.array(cluster_members[c])
        n_select = min(quota[c], len(members))
        chosen = np.random.choice(len(members), n_select, replace=False)
        selected_local.extend(members[chosen].tolist())

    selected_idx = np.array([valid_idx[i] for i in selected_local], dtype=int)
    print(f"最终采样 {len(selected_idx)} 个图")
    return selected_idx


def sample_method(method="ours", sample_num=100, dataset="101_acc"):
    """
    :param method: 'random' 或 'ours'
    :param sample_num: 选取的样本数（= n_clusters）
    :param dataset: '101_acc' 或 '201_acc'
    :return: 选出的图原始索引数组
    """
    all_graphs = get_dataset_graph(dataset=dataset)

    # 过滤掉节点数为 2-4 的图，同时保留原始索引映射
    valid_idx = [i for i, g in enumerate(all_graphs) if not (2 <= g.number_of_nodes() <= 4)]
    filtered_graphs = [all_graphs[i] for i in valid_idx]
    print(f"过滤前: {len(all_graphs)} 个图，过滤后: {len(filtered_graphs)} 个图（移除了 {len(all_graphs) - len(filtered_graphs)} 个节点数为 2-4 的图）")

    if method == "random":
        chosen = np.random.choice(len(filtered_graphs), sample_num, replace=False)
        cluster_idx = np.array([valid_idx[c] for c in chosen])
    elif method == "ours":
        embeddings = get_graph2vec_clusters(filtered_graphs, dimensions=128, epochs=100)
        local_idx = select_diverse_by_kmeans(embeddings, n_clusters=sample_num)
        cluster_idx = np.array([valid_idx[i] for i in local_idx])
    elif method == "op_filtered":
        # 过滤掉不含 conv3x3 (op=2) 的图，这些图 acc 普遍低且分散
        # 过滤后候选池密集区 (0.88~0.94) 比例从 86.9% 提升到 93.5%
        op_valid_idx = []
        for i, g in enumerate(filtered_graphs):
            ops = [g.nodes[j]["feature"] for j in g.nodes]
            if 2 in ops:
                op_valid_idx.append(i)
        print(f"op_filtered: 过滤不含conv3x3的图后剩余 {len(op_valid_idx)}/{len(filtered_graphs)} 个")
        chosen = np.random.choice(len(op_valid_idx), sample_num, replace=False)
        cluster_idx = np.array([valid_idx[op_valid_idx[c]] for c in chosen])
    return cluster_idx

def kmeans_method():
    # ---- nasbench101 ----
    with open("./data/nasbench101/101_traing_sample.pkl", "wb") as fp:
        all_cluster_idx = {}
        all_sanerios = [100, 172, 424, 4236]
        for sanerio in all_sanerios:
            print(f"=== Sanerio {sanerio} ===")
            cluster_idx = sample_method("ours", sample_num=sanerio, dataset="101_acc")
            print(len(cluster_idx), cluster_idx[:10])
            all_cluster_idx[sanerio] = cluster_idx
        pickle.dump(all_cluster_idx, fp)
    print("Done! (101)")
    
def balanced_cluster_method(n_clusters=5):
    """聚类一次，对所有 scenario 复用同一聚类结果。"""
    import os
    os.makedirs("data/nasbench101", exist_ok=True)

    # 只聚类一次
    print(f"=== 聚类 n_clusters={n_clusters} ===")
    valid_idx, cluster_members = _cluster_once(n_clusters, dataset="101_acc")

    pkl_path = f"./data/nasbench101/101_balanced_cluster_k{n_clusters}_sample.pkl"
    all_cluster_idx = {}
    all_sanerios = [100, 172, 424, 4236]
    for sanerio in all_sanerios:
        print(f"\n=== Sanerio {sanerio} ===")
        cluster_idx = balanced_cluster_sample(sanerio, valid_idx, cluster_members)
        print(len(cluster_idx), cluster_idx[:10])
        all_cluster_idx[sanerio] = cluster_idx

    with open(pkl_path, "wb") as fp:
        pickle.dump(all_cluster_idx, fp)
    print(f"\nDone! Saved to {pkl_path}")


if __name__ == "__main__":
    import os
    import argparse as _ap

    os.makedirs("data/nasbench101", exist_ok=True)

    p = _ap.ArgumentParser()
    p.add_argument("--method", default="balanced_cluster",
                    choices=["balanced_cluster", "op_filtered", "kmeans"])
    p.add_argument("--n_clusters", type=int, default=5,
                    help="balanced_cluster 的簇数量")
    cli = p.parse_args()

    if cli.method == "balanced_cluster":
        balanced_cluster_method(n_clusters=cli.n_clusters)
    elif cli.method == "op_filtered":
        with open("./data/nasbench101/101_op_filtered_sample.pkl", "wb") as fp:
            all_cluster_idx = {}
            all_sanerios = [100, 172, 424, 4236]
            for sanerio in all_sanerios:
                print(f"=== Sanerio {sanerio} ===")
                cluster_idx = sample_method("op_filtered", sample_num=sanerio, dataset="101_acc")
                print(len(cluster_idx), cluster_idx[:10])
                all_cluster_idx[sanerio] = cluster_idx
            pickle.dump(all_cluster_idx, fp)
        print("Done! (101)")
    elif cli.method == "kmeans":
        kmeans_method()
