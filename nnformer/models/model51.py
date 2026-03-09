import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGATConv
from mytools.registry import register_model
from typing import List, Optional
from torch import Tensor
import math
from timm.models.layers import DropPath, to_2tuple

from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


class DenseGraphSAGEBlock(nn.Module):
    """
    Dense GraphSAGE block for dense adjacency:
        x:   [B, N, F]
        adj: [B, N, N]
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3,
                 dropout=0.0, concat=True, use_bn=True):
        super().__init__()
        assert num_layers >= 1

        self.concat = concat
        self.dropout = dropout
        self.use_bn = use_bn

        dims = [in_dim]
        if num_layers == 1:
            dims.append(out_dim)
        else:
            dims += [hidden_dim] * (num_layers - 1)
            dims.append(out_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(DenseSAGEConv(dims[i], dims[i + 1]))
            if i != num_layers - 1 and use_bn:
                # 使用 LayerNorm 做归一化
                self.bns.append(nn.LayerNorm(dims[i + 1]))

    def forward(self, x, adj):
        outs = []

        for i, conv in enumerate(self.convs):
            x = conv(x, adj)

            if i != len(self.convs) - 1:
                x = F.relu(x)

                if self.use_bn:
                    # 直接对特征维做 LayerNorm
                    x = self.bns[i](x)

                x = F.dropout(x, p=self.dropout, training=self.training)

            outs.append(x)

        if self.concat:
            return torch.cat(outs, dim=-1)
        return outs[-1]


class DiffPoolLayer(nn.Module):
    """
    One dense DiffPool layer.
    """
    def __init__(self, input_dim, embed_hidden_dim, embed_dim,
                 assign_hidden_dim, assign_dim,
                 gnn_layers=3, dropout=0.0, concat=True):
        super().__init__()

        self.embed_gnn = DenseGraphSAGEBlock(
            in_dim=input_dim,
            hidden_dim=embed_hidden_dim,
            out_dim=embed_dim,
            num_layers=gnn_layers,
            dropout=dropout,
            concat=concat
        )

        self.assign_gnn = DenseGraphSAGEBlock(
            in_dim=input_dim,
            hidden_dim=assign_hidden_dim,
            out_dim=assign_dim,
            num_layers=gnn_layers,
            dropout=dropout,
            concat=False
        )

    def forward(self, x, adj):
        """
        x:    [B, N, F]
        adj:  [B, N, N]
        """
        z = self.embed_gnn(x, adj)       # node embedding
        s = self.assign_gnn(x, adj)      # assignment logits

        x_next, adj_next, lp_loss, ent_loss = dense_diff_pool(z, adj, s)
        return x_next, adj_next, lp_loss, ent_loss, z, s


class DiffPoolEncoder(nn.Module):
    """
    Dense adjacency + PyG DiffPool encoder.

    输入:
        x:    [B, N, F]  连续节点特征
        adj:  [B, N, N]
        mask: [B, N] 可选

    输出:
        graph_emb: [B, D]   图级嵌入
        aux_loss:  scalar   link pred + entropy
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        embedding_dim=128,
        base_layers=4,           # backbone GNN 层数（StableGNN: 5 层）
        pool_gnn_layers=3,       # DiffPool 内部 GNN 层数（StableGNN: 3 层）
        num_pooling=1,           # DiffPool 层数（StableGNN: 1 层）
        assign_dim=7,
        pool_ratio=0.25,
        dropout=0.0,
        concat=True,
        final_readout="sum",  # "sum" | "mean" | "max" | "cls"
    ):
        super().__init__()

        assert final_readout in ["sum", "mean", "max", "cls"]

        self.concat = concat
        self.final_readout = final_readout
        self.num_pooling = num_pooling

        # 第一段 backbone GNN（StableGNN 中的基础 5 层 GNN）
        self.pre_gnn = DenseGraphSAGEBlock(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=embedding_dim,
            num_layers=base_layers,
            dropout=dropout,
            concat=concat,
        )

        # 计算 pre_gnn 和每个 DiffPool 层输出的特征维度，用于确定最终 graph_emb 维度
        if concat:
            # pre_gnn 输出维度：hidden_dim * (base_layers - 1) + embedding_dim
            self.pre_out_dim = hidden_dim * (base_layers - 1) + embedding_dim
            # 每个 DiffPool 中 embed_gnn 输出维度：hidden_dim * (pool_gnn_layers - 1) + embedding_dim
            self.pool_out_dim = hidden_dim * (pool_gnn_layers - 1) + embedding_dim
        else:
            self.pre_out_dim = embedding_dim
            self.pool_out_dim = embedding_dim

        pool_input_dim = self.pre_out_dim

        # 多层 diffpool
        self.pool_layers = nn.ModuleList()
        cur_assign_dim = assign_dim

        for _ in range(num_pooling):
            self.pool_layers.append(
                DiffPoolLayer(
                    input_dim=pool_input_dim,
                    embed_hidden_dim=hidden_dim,
                    embed_dim=embedding_dim,
                    assign_hidden_dim=hidden_dim,
                    assign_dim=cur_assign_dim,
                    gnn_layers=pool_gnn_layers,  # StableGNN: 每个 DiffPool 内部 3 层 GNN
                    dropout=dropout,
                    concat=concat,
                )
            )
            cur_assign_dim = max(1, int(cur_assign_dim * pool_ratio))

        # 每一层 pooling 都会做一次 graph readout，然后拼接：
        # 总维度 = pre_gnn_readout_dim + num_pooling * pool_readout_dim
        self.output_dim = self.pre_out_dim + num_pooling * self.pool_out_dim

    def _graph_readout(self, x):
        """
        x: [B, N, F]
        """
        # CLS 聚合：直接取第一个节点向量
        if self.final_readout == "cls":
            return x[:, 0, :]

        if self.final_readout == "sum":
            out = x.sum(dim=1)
        elif self.final_readout == "mean":
            out = x.mean(dim=1)
        else:  # "max"
            out = x.max(dim=1).values

        return out

    def forward(self, x, adj, return_all=False):
        """
        x:   [B, N, F]  连续节点特征
        adj: [B, N, N]
        """

        # 第一段图编码
        x = self.pre_gnn(x, adj)

        readouts = [self._graph_readout(x)]

        total_lp_loss = x.new_zeros(())
        total_ent_loss = x.new_zeros(())

        all_assign = []
        all_node_emb = [x]

        # 多层 diffpool
        for pool_layer in self.pool_layers:
            x, adj, lp_loss, ent_loss, z, s = pool_layer(x, adj)

            total_lp_loss = total_lp_loss + lp_loss
            total_ent_loss = total_ent_loss + ent_loss

            readouts.append(self._graph_readout(x))
            all_assign.append(s)
            all_node_emb.append(x)

        graph_emb = torch.cat(readouts, dim=-1)
        aux_loss = total_lp_loss + total_ent_loss

        if return_all:
            return {
                "graph_emb": graph_emb,
                "aux_loss": aux_loss,
                "link_loss": total_lp_loss,
                "entropy_loss": total_ent_loss,
                "node_embeds": all_node_emb,
                "assign_mats": all_assign,
            }

        return graph_emb, aux_loss


@register_model("model51")
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_node_features = 32
        self.d_model = int(getattr(config, "d_model", 192))  # Model dimension
        self.dropout = float(getattr(config, "dropout", 0.10))  # Dropout rate

        # StableGNN 风格硬编码配置
        self.backbone_layers = 4       # 基础 GNN 5 层
        self.pool_gnn_layers = 2       # DiffPool 内部 GNN 3 层
        self.num_pooling = 1           # 1 层 DiffPool

        # 使用 CLS 节点作为图级聚合锚点
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.op_embeds = nn.Linear(self.num_node_features, self.d_model)
        self.depth_embed = nn.Linear(32, self.d_model)

        # 使用 DiffPoolEncoder 作为图级 encoder，配合 CLS 聚合
        self.encoder = DiffPoolEncoder(
            input_dim=self.d_model,
            hidden_dim=self.d_model,
            embedding_dim=self.d_model,
            base_layers=self.backbone_layers,
            pool_gnn_layers=self.pool_gnn_layers,
            num_pooling=self.num_pooling,
            assign_dim=7,
            pool_ratio=0.25,
            dropout=self.dropout,
            concat=True,
            final_readout="cls",
        )

        # DiffPoolEncoder 的输出维度为 encoder.output_dim
        self.encoder_out_dim = self.encoder.output_dim

        self.predictor = nn.Linear(self.encoder_out_dim, 1)

        # Weights initialization
        self.init_weights()

    def get_data(self, sample, static_feature):
        x = sample["code"]
        adj = sample["code_adj"]  # Adjacency matrix for graph structure
        adj = adj + torch.eye(adj.size(1), device=adj.device)
        return x, adj

    def forward(self, sample, static_feature):
        x, adj = self.get_data(sample, static_feature)
        depth = sample["op_depth"]

        # 节点特征编码
        x = self.op_embeds(x) + self.depth_embed(depth)

        # 加 CLS 节点：节点 0 为 CLS，和所有节点全连
        num_nodes = adj.size(1)
        new_adj = torch.ones(
            adj.size(0), num_nodes + 1, num_nodes + 1, device=adj.device
        )
        new_adj[:, 1:, 1:] = adj
        adj = new_adj
        cls_token = self.cls_token.expand(x.size(0), 1, self.d_model)
        x = torch.cat([cls_token, x], dim=1)

        # DiffPoolEncoder 直接输出图级 CLS 表示
        graph_emb, aux_loss = self.encoder(x, adj, return_all=False)

        # Final latency prediction
        latency = self.predictor(graph_emb)  # Predict latency

        # 同时返回 DiffPool 的辅助正则项，供外部 loss 使用
        return latency, aux_loss

    def init_weights(self):
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.constant_(m.weight, 0.02)
            # nn.init.trunc_normal_(m.weight, std=0.02)
