import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGATConv
from nnformer.models.encoders.neuralformer import EncoderBlock
from nnformer.models.registry import register_model
from typing import List, Optional
from torch import Tensor
import math
from timm.models.layers import DropPath, to_2tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        gate_hidden_ratio: float = 0.5,  # gate 的小 MLP 宽度比例（可设 0 表示只用一层线性）
        temperature: float = 1.0,  # softmax 温度，<1 更尖锐，>1 更平滑
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        # --- 3 个 experts，都输出同一维度 hidden_features ---
        self.self_expert = nn.Linear(in_features, hidden_features, bias=False)
        self.in_expert = nn.Linear(in_features, hidden_features, bias=False)
        self.out_expert = nn.Linear(in_features, hidden_features, bias=False)

        # --- gate：对每个 token 输出 3 个权重 ---
        gate_hidden = int(in_features * gate_hidden_ratio)
        if gate_hidden_ratio > 0:
            self.gate = nn.Sequential(
                nn.Linear(in_features, gate_hidden, bias=True),
                nn.ReLU(),
                nn.Linear(gate_hidden, 3, bias=True),
            )
        else:
            self.gate = nn.Linear(in_features, 3, bias=True)

        self.temperature = float(temperature)

        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")

        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # x: [B,L,C], adj: [B,L,L]
        B, L, C = x.shape
        adj = adj.float()
        
        # --- experts ---
        e0 = self.self_expert(x)  # [B,L,H]

        in_msg = self.in_expert(x)  # [B,L,H]
        out_msg = self.out_expert(x)  # [B,L,H]

        e1 = torch.bmm(adj, in_msg)  # [B,L,H]
        e2 = torch.bmm(adj.transpose(1, 2), out_msg)  # [B,L,H]

        # --- gate weights ---
        logits = self.gate(x) / self.temperature  # [B,L,3]
        w = F.softmax(logits, dim=-1)  # [B,L,3]

        # --- mixture (按 expert 维度加权求和) ---
        out = w[..., 0:1] * e0 + w[..., 1:2] * e1 + w[..., 2:3] * e2  # [B,L,H]

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
            d_model, n_head, dropout=dropout, batch_first=True
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
    def __init__(self, d_model, d_ff_ratio, gcn_layers, n_head=2, try_exp=-1):
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
                    activation="relu",
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
            adj = adj.bool()
            
            if self.try_exp == 1:
                bias = adj.unsqueeze(1)
            elif self.try_exp == 2:
                bias = torch.stack([adj, adj.mT], dim=1)
            elif self.try_exp == 3:
                bias = torch.stack([adj, adj.mT, adj.mT & adj, adj & adj.mT], dim=1)
            else:
                raise ValueError(f"Unsupported try_exp: {self.try_exp}")
            
            src_mask = torch.zeros_like(bias, dtype=x.dtype)
            src_mask = src_mask.masked_fill(~bias, torch.finfo(x.dtype).min)
            src_mask = src_mask.reshape(-1, L, L)

        for layer in self.layers:
            x = layer(x, adj, src_mask)
        return x


class PredHead(nn.Module):
    def __init__(self, d_model):
        super(PredHead, self).__init__()
        self.fc_1 = nn.Linear(d_model, d_model)
        self.fc_2 = nn.Linear(d_model, d_model)
        self.fc_relu1 = nn.ReLU()
        self.fc_relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc_relu1(self.fc_1(x))
        x = self.fc_relu2(self.fc_2(x))
        return x


@register_model("model49")
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_node_features = 32
        self.d_model = int(getattr(config, "d_model", 192))  # Model dimension
        self.dropout = float(getattr(config, "dropout", 0.10))  # Dropout rate
        self.gcn_layers = int(getattr(config, "gcn_layers", 2))  # Number of GCN layers
        self.graph_readout = str(
            getattr(config, "graph_readout", "cls")
        )  # Graph readout type
        self.d_ff_ratio = float(
            getattr(config, "d_ff_ratio", 4)
        )  # FFN width control via ratio
        self.graph_n_head = int(getattr(config, "graph_n_head", 1))
        self.encoder_type = str(getattr(config, "encoder_type", "nn"))
        self.try_exp = int(getattr(config, "try_exp", -1))

        if self.graph_readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        # self.op_embeds = nn.Linear(self.num_node_features, self.d_model)
        # self.depth_embed = nn.Linear(32, self.d_model)

        self.op_embeds = nn.Embedding(self.num_node_features, self.d_model)
        self.depth_embed = nn.Embedding(32, self.d_model)

        self.prev_norm = nn.LayerNorm(self.d_model)

        # self.encoder = nnEncoder(self.d_model, self.d_ff_ratio, self.gcn_layers)
        self.encoder = TransformerEncoder(
            self.d_model,
            self.d_ff_ratio,
            self.gcn_layers,
            self.graph_n_head,
            self.try_exp,
        )

        self.post_norm = nn.LayerNorm(self.d_model)

        if config.use_head:
            self.pred_head = PredHead(self.d_model)  # Number of GCN layers

        self.predictor = nn.Linear(self.d_model, 1)

        # Weights initialization
        self.init_weights()

    def get_data(self, sample, static_feature):
        x = sample["ops"].long()
        adj = sample["code_adj"]  # Adjacency matrix for graph structure
        adj = adj + torch.eye(adj.size(1), device=adj.device)
        return x, adj

    def forward(self, sample, static_feature):
        x, adj = self.get_data(sample, static_feature)

        depth = sample["op_depth"].long()
        # depth = F.one_hot(depth, num_classes=32).float()

        # 首先是不是编码的问题
        x = self.op_embeds(x) + self.depth_embed(depth)

        # 然后才是encoder的问题
        if self.graph_readout == "cls":
            num_nodes = adj.size(1)
            new_adj = torch.ones(
                adj.size(0), num_nodes + 1, num_nodes + 1, device=adj.device
            )
            new_adj[:, 1:, 1:] = adj
            adj = new_adj
            cls_token = self.cls_token.expand(x.size(0), 1, self.d_model)
            x = torch.cat([cls_token, x], dim=1)

        x = self.prev_norm(x)

        x = self.encoder(x, adj)

        x = self.post_norm(x)

        # Global pooling to obtain a fixed-size graph representation
        if self.graph_readout == "sum":
            x = torch.sum(x, dim=1)  # Global sum pooling
        elif self.graph_readout == "mean":
            x = torch.mean(x, dim=1)
        elif self.graph_readout == "max":
            x = torch.max(x, dim=1)[0]  # Get only the max values
        elif self.graph_readout == "cls":
            x = x[:, 0, :]

        # Final latency prediction
        if self.config.use_head:
            x = self.pred_head(x)

        latency = self.predictor(x)  # Predict latency

        return latency

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
            nn.init.constant_(m.weight, 0)
            # nn.init.trunc_normal_(m.weight, std=0.02)
