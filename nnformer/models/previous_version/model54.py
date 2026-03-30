import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseSAGEConv
from mytools.registry import register_model
from typing import Optional
from torch import Tensor
from timm.models.layers import to_2tuple


class BatchedMoEGraphFFN(nn.Module):
    """
    MoE-FFN with 3 experts:
      e0: self expert        -> W0 x
      e1: in-neighbor expert -> A  (W1 x)
      e2: out-neighbor expert-> A^T(W2 x)

    x:   [B, L, C]
    adj: [B, L, L]  (建议是 0/1 或归一化邻接；你上游已经加了自环)
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: int | None = None,
        act_layer: str = "relu",
        drop: float = 0.0,
        gate_hidden_ratio: float = 0.5,
        temperature: float = 1.0,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.self_expert = nn.Linear(in_features, hidden_features, bias=False)
        self.in_expert = nn.Linear(in_features, hidden_features, bias=False)
        self.out_expert = nn.Linear(in_features, hidden_features, bias=False)

        gate_hidden = int(in_features * gate_hidden_ratio)
        if gate_hidden_ratio > 0:
            self.gate = nn.Sequential(
                nn.Linear(in_features, gate_hidden, bias=False),
                nn.GELU(),
                nn.Linear(gate_hidden, 3, bias=False),
            )
        else:
            self.gate = nn.Linear(in_features, 3, bias=False)

        self.temperature = float(temperature)

        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")

        self.drop1 = nn.Dropout(drop_probs[0])
        self.mix_norm = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor, adj: Tensor) -> tuple[Tensor, Tensor]:
        B, L, C = x.shape
        adj = adj.float()
        adj[:, 0, :] = adj[:, :, 0] = 0
        adj = adj.masked_fill(torch.eye(L, device=adj.device, dtype=torch.bool), 0)

        e0 = self.self_expert(x)

        in_msg = self.in_expert(x)
        out_msg = self.out_expert(x)

        e1 = torch.bmm(adj, in_msg)
        e2 = torch.bmm(adj.transpose(1, 2), out_msg)

        logits = self.gate(x) / self.temperature
        w = F.softmax(logits, dim=-1)

        out = w[..., 0:1] * e0 + w[..., 1:2] * e1 + w[..., 2:3] * e2
        # out = self.mix_norm(out + e0)
        out = self.mix_norm(out) + e0

        out = self.act(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)

        # 路由负熵：最小化 = 防止塌缩到某个专家；值域 [log(1/3), 0]，越接近log(1/3)越均匀
        routing_penalty = (w * (w + 1e-8).log()).sum(dim=-1).mean()

        return out, routing_penalty


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float,
        dropout: float,
        activation: str = "relu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True, bias=False
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_first = norm_first
        self.ffn = BatchedMoEGraphFFN(
            d_model, mlp_ratio=mlp_ratio, act_layer=activation, drop=dropout
        )

    def _sa_block(self, x: Tensor, src_mask: Optional[Tensor]) -> Tensor:
        x, _ = self.self_attn(x, x, x, attn_mask=src_mask, need_weights=False)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor, adj: Tensor) -> tuple[Tensor, Tensor]:
        return self.ffn(x, adj)

    def forward(self, x: Tensor, adj: Tensor, src_mask: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask)
            ffn_out, penalty = self._ff_block(self.norm2(x), adj)
            x = x + ffn_out
            return x, penalty

        x = self.norm1(x + self._sa_block(x, src_mask))
        ffn_out, penalty = self._ff_block(x, adj)
        x = self.norm2(x + ffn_out)
        return x, penalty


class TransformerEncoder(nn.Module):
    def __init__(
        self, d_model, d_ff_ratio, gcn_layers, n_head=2, try_exp=-1, activation="relu"
    ):
        super().__init__()
        self.n_head = n_head
        self.try_exp = try_exp
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    d_model=d_model,
                    n_head=self.n_head,
                    mlp_ratio=d_ff_ratio,
                    dropout=0.1,
                    activation=activation,
                    norm_first=True,
                )
                for _ in range(gcn_layers)
            ]
        )

    def forward(self, x, adj, distance=None):
        L = x.size(1)
        src_mask = None

        if adj is not None:
            adj = adj.clone()
            adj = adj.masked_fill(torch.logical_and(adj > 1, adj < 9), 0)
            adj = adj.masked_fill(adj != 0, 1)
            adj = adj.float()

            # 高阶结构感知: A@A^T → 共同后继, A^T@A → 共同前驱
            # 屏蔽全局节点(index 0)的行列，避免其全连接污染结构相似度
            adj_local = adj.clone()
            adj_local[:, 0, :] = 0
            adj_local[:, :, 0] = 0
            common_succ = (torch.bmm(adj_local, adj_local.mT) > 0).float()   # Head 5
            common_pred = (torch.bmm(adj_local.mT, adj_local) > 0).float()   # Head 6
            
            if distance is not None:
                # 6个头: [adj, adj.mT, dist_fwd, dist_bwd, common_succ, common_pred]
                dist_fwd = (distance > 0).float()
                dist_bwd = (distance.mT > 0).float()
                pe = torch.stack(
                    [adj, adj.mT, dist_fwd, dist_bwd, common_succ, common_pred], dim=1
                )
            else:
                # 退化为4个头
                pe = torch.stack([adj, adj.mT, common_succ, common_pred], dim=1)

            pe = pe + torch.eye(L, dtype=adj.dtype, device=adj.device)
            pe = pe.int()

            src_mask = torch.zeros_like(pe, dtype=x.dtype)
            src_mask = src_mask.masked_fill(pe == 0, torch.finfo(x.dtype).min)
            # reshape成 (B*n_head, L, L)，n_head由pe第1维决定
            src_mask = src_mask.reshape(-1, L, L)

        penalties = []
        for layer in self.layers:
            x, penalty = layer(x, adj, src_mask)
            penalties.append(penalty)

        # 对所有层的路由惩罚取均值，作为辅助损失
        aux_loss = torch.stack(penalties).mean()
        return x, aux_loss


class GraphConvolutionLayer(nn.Module):
    """2层GraphSAGE用于更新节点嵌入"""

    def __init__(self, d_model: int, dropout: float = 0.1, act_layer: str = "relu"):
        super().__init__()
        self.sage1 = DenseSAGEConv(d_model, d_model, normalize=False)
        self.sage2 = DenseSAGEConv(d_model, d_model, normalize=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        x: [B, L, d_model]
        adj: [B, L, L] 邻接矩阵
        """
        # 给邻接矩阵加自环
        B, L = adj.shape[0], adj.shape[1]
        adj = adj + torch.eye(L, device=adj.device, dtype=adj.dtype).unsqueeze(0)

        # 第一层GraphSAGE (Pre-Norm)
        x = x + self.dropout(self.act(self.sage1(self.norm1(x), adj)))

        # 第二层GraphSAGE (Pre-Norm)
        x = x + self.dropout(self.act(self.sage2(self.norm2(x), adj)))

        return x


@register_model("model54")
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_node_features = 32
        self.d_model = int(getattr(config, "d_model", 192))
        self.dropout = float(getattr(config, "dropout", 0.10))
        self.gcn_layers = int(getattr(config, "gcn_layers", 2))
        self.graph_readout = str(getattr(config, "graph_readout", "cls"))
        self.d_ff_ratio = float(getattr(config, "d_ff_ratio", 4))
        self.graph_n_head = int(getattr(config, "graph_n_head", 2))
        self.encoder_type = str(getattr(config, "encoder_type", "nn"))
        self.try_exp = int(getattr(config, "try_exp", -1))
        self.act_layer = str(getattr(config, "act_layer", "relu"))
        self.use_aux_loss = bool(getattr(config, "use_aux_loss", False))

        if self.graph_readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        # 用 Linear + one-hot 统一嵌入
        # - op: [B, L]，范围 [0, 31]，one-hot后32维
        # - in/out degree: [B, L]，范围 [0, 31]，one-hot后32维
        # - depth: [B] 或 [B, 1]，范围 [0, 31]
        self.op_embed = nn.Linear(self.num_node_features, self.d_model)
        self.in_degree_embed = nn.Linear(32, self.d_model)
        self.out_degree_embed = nn.Linear(32, self.d_model)
        self.depth_embed = nn.Linear(32, self.d_model)
        self.embed_proj = nn.Linear(4 * self.d_model, self.d_model)

        # 2层GNN用于更新节点嵌入
        self.gnn_layer = GraphConvolutionLayer(
            self.d_model, dropout=self.dropout, act_layer=self.act_layer
        )

        # Transformer编码器
        self.encoder = TransformerEncoder(
            self.d_model,
            self.d_ff_ratio,
            self.gcn_layers,
            self.graph_n_head,
            self.try_exp,
            self.act_layer,
        )

        # 可学习融合权重：alpha in (0,1)，x = alpha*trans + (1-alpha)*gnn
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

        self.post_norm = nn.LayerNorm(self.d_model)
        self.predictor = nn.Linear(self.d_model, 1)

        self.init_weights()

    def get_data(self, sample, static_feature):
        x = sample["ops"]
        adj = sample["code_adj"]
        in_degree = sample["in_degree"]
        out_degree = sample["out_degree"]
        depth = sample["op_depth"]
        distance = sample["distance"]
        return x, adj, in_degree, out_degree, depth, distance

    def _add_cls_token(
        self, x: Tensor, adj: Tensor, distance: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """添加cls token并扩展邻接矩阵和distance矩阵"""
        if self.graph_readout == "cls":
            num_nodes = adj.size(1)
            new_adj = torch.ones(
                adj.size(0), num_nodes + 1, num_nodes + 1, device=adj.device
            )
            new_adj[:, 1:, 1:] = adj
            adj = new_adj
            # cls token 行/列视为与所有节点距离为1
            new_dist = torch.ones(
                distance.size(0), num_nodes + 1, num_nodes + 1, device=distance.device
            )
            new_dist[:, 1:, 1:] = distance
            distance = new_dist
            cls_token = self.cls_token.expand(x.size(0), 1, self.d_model)
            x = torch.cat([cls_token, x], dim=1)
        return x, adj, distance

    def _embed_features(
        self, x: Tensor, in_degree: Tensor, out_degree: Tensor, depth: Tensor
    ) -> Tensor:
        x = self.op_embed(
            F.one_hot(x.long(), num_classes=32).float()
        )  # [B, L, d_model]
        in_deg = self.in_degree_embed(
            F.one_hot(in_degree.long(), num_classes=32).float()
        )
        out_deg = self.out_degree_embed(
            F.one_hot(out_degree.long(), num_classes=32).float()
        )
        dep = self.depth_embed(depth.float())
        return self.embed_proj(
            torch.cat([x, in_deg, out_deg, dep], dim=-1)
        )  # [B, L, d_model]

    def forward(self, sample, static_feature):
        x, adj, in_degree, out_degree, depth, distance = self.get_data(sample, static_feature)

        # 节点特征嵌入
        x = self._embed_features(x, in_degree, out_degree, depth)

        # 添加cls token
        x, adj, distance = self._add_cls_token(x, adj, distance)

        # 经过Transformer编码器
        x, aux_loss = self.encoder(x, adj, distance)

        x = self.post_norm(x)

        # 图级聚合：由类成员变量 self.graph_readout 决定
        if self.graph_readout == "cls":
            graph_feat = x[:, 0, :]
        elif self.graph_readout == "sum":
            graph_feat = x.sum(dim=1)
        elif self.graph_readout == "att":
            att_logit = self.predictor(x).squeeze(-1)
            att_weight = torch.softmax(att_logit, dim=1).unsqueeze(-1)
            graph_feat = (x * att_weight).sum(dim=1)
        else:
            raise ValueError(f"Unsupported graph_readout: {self.graph_readout}")

        x = graph_feat

        predict = self.predictor(x)

        return predict, aux_loss if self.use_aux_loss else 0

    def init_weights(self):
        self.apply(self._init_weights)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.constant_(m.weight, 0.02)
