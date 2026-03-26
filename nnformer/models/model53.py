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

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
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
        out = self.mix_norm(out)

        out = self.act(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        return out


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

    def _ff_block(self, x: Tensor, adj: Tensor) -> Tensor:
        return self.ffn(x, adj)

    def forward(self, x: Tensor, adj: Tensor, src_mask: Optional[Tensor]) -> Tensor:
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask)
            x = x + self._ff_block(self.norm2(x), adj)
            return x

        x = self.norm1(x + self._sa_block(x, src_mask))
        x = self.norm2(x + self._ff_block(x, adj))
        return x


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

    def forward(self, x, adj):
        L = x.size(1)
        src_mask = None

        if adj is not None:
            adj = adj.clone()
            adj = adj.masked_fill(torch.logical_and(adj > 1, adj < 9), 0)
            adj = adj.masked_fill(adj != 0, 1)
            adj = adj.float()

            # 使用2项堆栈: [adj, adj.mT]
            pe = torch.stack([adj, adj.mT], dim=1)
            pe = pe + torch.eye(L, dtype=adj.dtype, device=adj.device)
            pe = pe.int()

            src_mask = torch.zeros_like(pe, dtype=x.dtype)
            src_mask = src_mask.masked_fill(pe == 0, torch.finfo(x.dtype).min)
            # 保留2个mask头: reshape成 (B*2, L, L)
            src_mask = src_mask.reshape(-1, L, L)

        for layer in self.layers:
            x = layer(x, adj, src_mask)

        return x


class GraphConvolutionLayer(nn.Module):
    """2层GraphSAGE用于更新节点嵌入"""

    def __init__(self, d_model: int, dropout: float = 0.1, act_layer: str = "relu"):
        super().__init__()
        self.sage1 = DenseSAGEConv(d_model, d_model, normalize=True)
        self.sage2 = DenseSAGEConv(d_model, d_model, normalize=True)
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


@register_model("model53")
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

        if self.graph_readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        # 用 Embedding 统一嵌入（输入都是纯数字 id）
        # - op: [B, L]，范围 [0, 31]
        # - in/out degree: [B, L]，范围 [0, 31]
        # - depth: [B] 或 [B, 1]，范围 [0, 31]
        self.op_embed = nn.Embedding(self.num_node_features, self.d_model)
        self.in_degree_embed = nn.Embedding(32, self.d_model)
        self.out_degree_embed = nn.Embedding(32, self.d_model)
        self.depth_embed = nn.Embedding(32, self.d_model)

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

        self.post_norm = nn.LayerNorm(self.d_model)
        self.predictor = nn.Linear(self.d_model, 1)

        self.init_weights()

    def get_data(self, sample, static_feature):
        x = sample["ops"]
        adj = sample["code_adj"]
        in_degree = sample["in_degree"]
        out_degree = sample["out_degree"]
        return x, adj, in_degree, out_degree

    def _add_cls_token(self, x: Tensor, adj: Tensor) -> tuple[Tensor, Tensor]:
        """添加cls token并扩展邻接矩阵"""
        if self.graph_readout == "cls":
            num_nodes = adj.size(1)
            new_adj = torch.ones(
                adj.size(0), num_nodes + 1, num_nodes + 1, device=adj.device
            )
            new_adj[:, 1:, 1:] = adj
            adj = new_adj
            cls_token = self.cls_token.expand(x.size(0), 1, self.d_model)
            x = torch.cat([cls_token, x], dim=1)
        return x, adj

    def _add_depth_token(
        self, x: Tensor, adj: Tensor, depth_feat: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """添加depth token并扩展邻接矩阵"""
        if self.depth_embed is not None and depth_feat is not None:
            depth_ids = depth_feat.squeeze(1) if depth_feat.dim() == 2 else depth_feat
            depth_ids = depth_ids.long().clamp_(0, 31)  # [B]
            depth_emb = self.depth_embed(depth_ids)  # [B, d_model]

            # 添加到序列末尾：[B, L, d_model] + [B, 1, d_model] = [B, L+1, d_model]
            x = torch.cat([x, depth_emb.unsqueeze(1)], dim=1)

            # 扩展邻接矩阵（添加到末尾）
            new_adj = torch.zeros(
                adj.shape[0], adj.shape[1] + 1, adj.shape[2] + 1, device=adj.device
            )
            new_adj[:, :-1, :-1] = adj
            adj = new_adj
        return x, adj

    def _embed_features(
        self,
        x: Tensor,
        in_degree: Tensor,
        out_degree: Tensor,
        depth: Optional[Tensor] = None,
    ) -> Tensor:
        """节点特征嵌入

        Args:
            x: [B, L] op id
            in_degree: [B, L] 入度
            out_degree: [B, L] 出度
            depth: [B] or [B, 1] 深度 id（可选）

        Returns:
            [B, L, d_model] 嵌入后的特征
        """
        B, L = x.shape
        x = self.op_embed(x.long())  # [B, L, d_model]

        # degree embedding
        x = (
            x
            + self.in_degree_embed(in_degree.long())
            + self.out_degree_embed(out_degree.long())
        )  # [B, L, d_model]

        # depth embedding：全局条件，广播到每个节点
        if depth is not None:
            depth_ids = depth.squeeze(1) if depth.dim() == 2 else depth
            depth_ids = depth_ids.long().clamp_(0, 31)
            x = x + self.depth_embed(depth_ids).unsqueeze(1).expand(B, L, -1)

        return x

    def forward(self, sample, static_feature):
        x, adj, in_degree, out_degree = self.get_data(sample, static_feature)

        # 提取并处理depth特征
        depth_feat = None
        if "code_depth" in sample:
            depth_feat = sample["code_depth"]

        # 节点特征嵌入
        x = self._embed_features(x, in_degree, out_degree, depth_feat)

        # 添加cls token
        x, adj = self._add_cls_token(x, adj)

        # 添加depth token
        x, adj = self._add_depth_token(x, adj, depth_feat)

        # 经过2层GraphSAGE更新节点嵌入
        x = self.gnn_layer(x, adj)

        # 经过Transformer编码器
        x = self.encoder(x, adj)

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

        return predict

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
