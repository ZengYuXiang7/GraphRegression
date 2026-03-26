import torch
import torch.nn as nn
import torch.nn.functional as F
from mytools.registry import register_model
from typing import Optional
from torch import Tensor
from timm.models.layers import to_2tuple


# ============================================================
# 可学习的有向图位置编码器
# ============================================================


class SignInvariantEncoder(nn.Module):
    """SignNet: f(x) + f(-x)，处理特征向量的符号歧义"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + self.net(-x)

class MagLapEncoder(nn.Module):
    def __init__(self, k=25, hidden_dim=64, output_dim=128, dropout=0.15, use_signnet=False):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.use_signnet = use_signnet

        self.f_elem = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.drop = nn.Dropout(dropout)

        self.f_mix = nn.Sequential(
            nn.Linear(k * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        B, N, K, _ = x.shape

        if self.use_signnet:
            h0 = self.f_elem(x[:, :, 0, :]).unsqueeze(2)
            h_rest = self.f_elem(x[:, :, 1:, :]) + self.f_elem(-x[:, :, 1:, :])
            h = torch.cat([h0, h_rest], dim=2)
        else:
            h = self.f_elem(x)   # 论文主线更接近这个

        h = self.norm(h)
        h = h.reshape(B * N, K, self.hidden_dim)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.drop(h)

        h = h.reshape(B, N, K * self.hidden_dim)
        return self.f_mix(h)


class RWEncoder(nn.Module):
    """
    方向性随机游走编码器。
    输入: (B, n, rw_dim)  →  输出: (B, n, output_dim)
    """

    def __init__(self, rw_dim: int = 16, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rw_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        in_dim = self.net[0].in_features
        cur_dim = x.size(-1)
        if cur_dim < in_dim:
            x = F.pad(x, (0, in_dim - cur_dim))
        elif cur_dim > in_dim:
            x = x[..., :in_dim]
        return self.net(x)


# ============================================================
# 模型组件
# ============================================================


class BatchedMoEGraphFFN(nn.Module):
    """
    MoE-FFN，3个专家：
      e0: self expert         -> W0 x
      e1: in-neighbor expert  -> A   (W1 x)
      e2: out-neighbor expert -> A^T (W2 x)
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
        e1 = torch.bmm(adj, self.in_expert(x))
        e2 = torch.bmm(adj.transpose(1, 2), self.out_expert(x))

        w = F.softmax(self.gate(x) / self.temperature, dim=-1)
        out = w[..., 0:1] * e0 + w[..., 1:2] * e1 + w[..., 2:3] * e2
        out = self.mix_norm(out) + e0

        out = self.act(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)

        # 路由负熵辅助损失，防止专家塌缩
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

    def forward(
        self, x: Tensor, adj: Tensor, src_mask: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask)
            ffn_out, penalty = self.ffn(self.norm2(x), adj)
            x = x + ffn_out
            return x, penalty

        x = self.norm1(x + self._sa_block(x, src_mask))
        ffn_out, penalty = self.ffn(x, adj)
        x = self.norm2(x + ffn_out)
        return x, penalty


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff_ratio, gcn_layers, n_head=6, activation="relu"):
        super().__init__()
        self.n_head = n_head
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

            # 高阶结构感知：屏蔽全局节点(index 0)避免全连接污染结构相似度
            adj_local = adj.clone()
            adj_local[:, 0, :] = 0
            adj_local[:, :, 0] = 0
            common_succ = (torch.bmm(adj_local, adj_local.mT) > 0).float()
            common_pred = (torch.bmm(adj_local.mT, adj_local) > 0).float()

            if distance is not None:
                # 6个注意力头: [adj, adj^T, dist_fwd, dist_bwd, common_succ, common_pred]
                dist_fwd = (distance > 0).float()
                dist_bwd = (distance.mT > 0).float()
                pe = torch.stack(
                    [adj, adj.mT, dist_fwd, dist_bwd, common_succ, common_pred], dim=1
                )
            else:
                pe = torch.stack([adj, adj.mT, common_succ, common_pred], dim=1)

            pe = (pe + torch.eye(L, dtype=adj.dtype, device=adj.device)).int()
            src_mask = torch.zeros_like(pe, dtype=x.dtype)
            src_mask = src_mask.masked_fill(pe == 0, torch.finfo(x.dtype).min)
            src_mask = src_mask.reshape(-1, L, L)

        penalties = []
        for layer in self.layers:
            x, penalty = layer(x, adj, src_mask)
            penalties.append(penalty)

        aux_loss = torch.stack(penalties).mean()
        return x, aux_loss


@register_model("model55")
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
        self.graph_n_head = int(getattr(config, "graph_n_head", 6))
        self.act_layer = str(getattr(config, "act_layer", "relu"))
        self.use_aux_loss = bool(getattr(config, "use_aux_loss", False))
        pe_rw_steps = int(getattr(config, "pe_rw_steps", 8))  # → rw_dim = 16
        pe_k = int(getattr(config, "pe_k", 5))  # → ml shape (n, 5, 2)
        pe_hidden = min(self.d_model, 128)

        if self.graph_readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        self.op_embed = nn.Linear(self.num_node_features, self.d_model)
        self.in_degree_embed = nn.Linear(32, self.d_model)
        self.out_degree_embed = nn.Linear(32, self.d_model)
        self.depth_embed = nn.Linear(32, self.d_model)
        self.embed_proj = nn.Linear(4 * self.d_model, self.d_model)

        # 两路有向图位置编码
        # RW：捕获局部方向可达性（forward/reverse 随机游走回访概率）
        self.rw_encoder = RWEncoder(
            rw_dim=2 * pe_rw_steps + 2,
            hidden_dim=pe_hidden,
            output_dim=self.d_model,
        )
        # ML：捕获全局有向拓扑结构（Magnetic Laplacian 特征向量）
        self.ml_encoder = MagLapEncoder(
            k=pe_k,
            hidden_dim=pe_hidden,
            output_dim=self.d_model,
        )
        # 门控融合：可学习地控制 PE 对嵌入的贡献
        self.pe_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid(),
        )

        self.encoder = TransformerEncoder(
            self.d_model,
            self.d_ff_ratio,
            self.gcn_layers,
            self.graph_n_head,
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
        depth = sample["op_depth"]
        distance = sample["distance"]
        return x, adj, in_degree, out_degree, depth, distance

    def _add_cls_token(
        self, x: Tensor, adj: Tensor, distance: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.graph_readout == "cls":
            num_nodes = adj.size(1)
            new_adj = torch.ones(
                adj.size(0), num_nodes + 1, num_nodes + 1, device=adj.device
            )
            new_adj[:, 1:, 1:] = adj
            adj = new_adj
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
        x = self.op_embed(F.one_hot(x.long(), num_classes=32).float())
        in_deg = self.in_degree_embed(
            F.one_hot(in_degree.long(), num_classes=32).float()
        )
        out_deg = self.out_degree_embed(
            F.one_hot(out_degree.long(), num_classes=32).float()
        )
        dep = self.depth_embed(depth.float())
        return self.embed_proj(torch.cat([x, in_deg, out_deg, dep], dim=-1))

    def forward(self, sample, static_feature):
        x, adj, in_degree, out_degree, depth, distance = self.get_data(
            sample, static_feature
        )

        # 节点特征嵌入：[B, L, d_model]
        x = self._embed_features(x, in_degree, out_degree, depth)

        # 仅保留 ML 一路有向图 PE（均来自预计算，在 cls token 加入前计算）
        # ML: [B, L, k, 2]        →  [B, L, d_model]
        # pe_rw = self.rw_encoder(sample["dir_pe_rw"])  # 注释掉 RW
        pe_ml = self.ml_encoder(sample["dir_pe_ml"])
        pe = pe_ml  # 仅使用 ML 信息

        # 门控融合：g = sigmoid(W[x; pe])，让模型自己决定 PE 的贡献比例
        # gate = self.pe_gate(torch.cat([x, pe], dim=-1))
        # x = x + gate * pe
        x = x + pe

        # 添加 cls token
        x, adj, distance = self._add_cls_token(x, adj, distance)

        # Transformer 编码器
        x, aux_loss = self.encoder(x, adj, distance)

        x = self.post_norm(x)

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

        predict = self.predictor(graph_feat)
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
