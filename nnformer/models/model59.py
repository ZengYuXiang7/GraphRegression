import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mytools.registry import register_model
from typing import Optional
from torch import Tensor
from timm.models.layers import to_2tuple


# ============================================================
# 方案3: Attention MoE (专家选择式)
#
# 核心思路:
#   K=3 个 attention 专家, 共享 QKV 投影, 各专家结构偏置不同:
#     expert 0: 纯自由 attention
#     expert 1: forward-biased (adj)
#     expert 2: backward-biased (adj^T)
#
#   gate: 从节点特征 x 出发, 输出每节点 (B, L, K) 的 softmax 权重
#   输出: h_final[b, l] = sum_k( gate[b, l, k] * h_k[b, l] )
#
#   与 FFN MoE 不同: gate 是 per-node 的, 决定每个节点从哪个专家接收信息
#   熵正则 (路由惩罚) 可选, 鼓励负载均衡
# ============================================================


class MagLapEncoder(nn.Module):
    def __init__(self, k=25, hidden_dim=64, output_dim=128, dropout=0.15):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim

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
        h = self.f_elem(x)
        h = self.norm(h)
        h = h.reshape(B * N, K, self.hidden_dim)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.drop(h)
        h = h.reshape(B, N, K * self.hidden_dim)
        return self.f_mix(h)


class AttentionMoE(nn.Module):
    """
    Attention MoE: K=3 attention 专家, 每节点软门控.

    专家结构:
      expert 0: 无结构偏置 (自由)
      expert 1: score += bias_scale[0] * B_fwd
      expert 2: score += bias_scale[1] * B_bwd

    门控: Linear(dim -> gate_hidden -> K), 输入节点特征 x, 输出 per-node 专家权重.
    输出: h = sum_k( gate[b,l,k] * h_k[b,l] )  -- per-node 专家融合
    """

    N_EXPERTS = 3

    def __init__(
        self, dim: int, n_head: int, dropout: float = 0.0, has_cls: bool = True
    ):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_size = dim // n_head
        self.scale = math.sqrt(self.head_size)
        self.has_cls = has_cls

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 每个结构专家的偏置强度 [adj_fwd, adj_bwd, dist_fwd, dist_bwd]
        # 初始 3.0: 有边的位置获得明显正向引导
        self.expert_bias_scale = nn.Parameter(torch.tensor([3.0, 3.0, 3.0, 3.0]))

        # 门控网络: dim -> gate_hidden -> K
        gate_hidden = max(dim // 4, self.N_EXPERTS * 4)
        self.gate = nn.Sequential(
            nn.Linear(dim, gate_hidden, bias=False),
            nn.GELU(),
            nn.Linear(gate_hidden, self.N_EXPERTS, bias=False),
        )

    def _expert_attn(
        self, Q: Tensor, K: Tensor, V: Tensor, struct_bias: Optional[Tensor] = None
    ) -> Tensor:
        """单专家 attention, 可选加性结构偏置."""
        S = torch.matmul(Q, K.mT) / self.scale  # (B, H, L, L)
        if struct_bias is not None:
            S = S + struct_bias
        attn = F.softmax(S, dim=-1)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, V)  # (B, H, L, d)

    def forward(
        self, x: Tensor, adj: Tensor, distance: Optional[Tensor] = None
    ) -> Tensor:
        B, L, C = x.shape

        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, L, self.n_head, self.head_size).transpose(1, 2)  # (B,H,L,d)
        K = K.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        V = V.view(B, L, self.n_head, self.head_size).transpose(1, 2)

        # 结构矩阵

        adj_local = adj.clone()
        if self.has_cls:
            adj_local[:, 0, :] = 0
            adj_local[:, :, 0] = 0
        B_fwd = adj_local.unsqueeze(1)  # (B,1,L,L)
        B_bwd = adj_local.mT.unsqueeze(1)  # (B,1,L,L)
        B_dist_fwd = (
            (distance > 0).float()
            if distance is not None
            else (adj_local @ adj_local > 0).float()
        ).unsqueeze(
            1
        )  # (B,1,L,L)
        B_dist_bwd = (
            (distance.mT > 0).float()
            if distance is not None
            else (adj_local.mT @ adj_local.mT > 0).float()
        ).unsqueeze(
            1
        )  # (B,1,L,L)

        # K 个专家的 attention 输出
        s = self.expert_bias_scale
        h0 = self._expert_attn(Q, K, V)  # (B, H, L, d)
        h1 = self._expert_attn(
            Q, K, V, s[0] * B_fwd + s[2] * B_dist_fwd
        )  # (B, H, L, d)
        h2 = self._expert_attn(
            Q, K, V, s[1] * B_bwd + s[3] * B_dist_bwd
        )  # (B, H, L, d)

        # Per-node 门控: (B, L, K)
        gate_logits = self.gate(x)  # (B, L, K)
        gate_weight = F.softmax(gate_logits, dim=-1)  # (B, L, K)

        # 专家输出融合:
        #   h_stack: (B, H, L, K, d)
        #   gate_w:  (B, 1, L, K, 1)  -> broadcast over H and d
        h_stack = torch.stack([h0, h1, h2], dim=-2)  # (B, H, L, K, d)
        gate_w = gate_weight.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, K, 1)
        out = (h_stack * gate_w).sum(dim=-2)  # (B, H, L, d)

        out = out.transpose(1, 2).reshape(B, L, C)
        return self.resid_dropout(self.proj(out))


class GINMlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: Optional[int] = None,
        act_layer: str = "relu",
        drop: float = 0.0,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, False)
        self.gcn = nn.Linear(in_features, hidden_features, False)
        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, False)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        out = self.fc1(x)
        gcn_x1, gcn_x2 = self.gcn(x).chunk(2, dim=-1)
        out = out + torch.cat([adj @ gcn_x1, adj.mT @ gcn_x2], dim=-1)
        out = self.act(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        return out


class BatchedMoEGraphFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: int | None = None,
        act_layer: str = "relu",
        drop: float = 0.0,
        gate_hidden_ratio: float = 0.5,
        temperature: float = 1.0,
        has_cls: bool = True,
    ):
        super().__init__()
        self.has_cls = has_cls
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
        L = x.size(1)
        adj = adj.float()
        if self.has_cls:
            adj[:, 0, :] = adj[:, :, 0] = 0
        adj = adj.masked_fill(torch.eye(L, device=adj.device, dtype=torch.bool), 0)

        e0 = self.self_expert(x)
        e1 = torch.bmm(adj, self.in_expert(x))
        e2 = torch.bmm(adj.transpose(1, 2), self.out_expert(x))

        w = F.softmax(self.gate(x) / self.temperature, dim=-1)
        out = w[..., 0:1] * e0 + w[..., 1:2] * e1 + w[..., 2:3] * e2

        out = self.act(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)

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
        has_cls=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = AttentionMoE(d_model, n_head, dropout, has_cls=has_cls)
        self.ffn = BatchedMoEGraphFFN(
            d_model,
            mlp_ratio=mlp_ratio,
            act_layer=activation,
            drop=dropout,
            has_cls=has_cls,
        )

    def forward(
        self, x: Tensor, adj: Tensor, distance: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        x = x + self.self_attn(self.norm1(x), adj, distance)
        ffn_out, penalty = self.ffn(self.norm2(x), adj)
        x = x + ffn_out
        return x, torch.zeros((), device=x.device)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff_ratio,
        gcn_layers,
        n_head=4,
        activation="relu",
        graph_readout="cls",
    ):
        super().__init__()
        has_cls = graph_readout == "cls"
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    d_model=d_model,
                    n_head=n_head,
                    mlp_ratio=d_ff_ratio,
                    dropout=0.1,
                    activation=activation,
                    has_cls=has_cls,
                )
                for _ in range(gcn_layers)
            ]
        )

    def forward(self, x: Tensor, adj: Tensor, distance: Optional[Tensor] = None):
        adj = adj.clone()
        adj = adj.masked_fill(torch.logical_and(adj > 1, adj < 9), 0)
        adj = adj.masked_fill(adj != 0, 1).float()
        penalties = []
        for layer in self.layers:
            x, penalty = layer(x, adj, distance)
            penalties.append(penalty)
        aux_loss = torch.stack(penalties).mean()
        return x, aux_loss


@register_model("model59")
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
        pe_k = int(getattr(config, "pe_k", 5))
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

        self.ml_encoder = MagLapEncoder(
            k=pe_k,
            hidden_dim=pe_hidden,
            output_dim=self.d_model,
        )

        self.encoder = TransformerEncoder(
            self.d_model,
            self.d_ff_ratio,
            self.gcn_layers,
            self.graph_n_head,
            self.act_layer,
            self.graph_readout,
        )

        self.post_norm = nn.LayerNorm(self.d_model)
        self.att_pool = nn.Linear(self.d_model, 1)
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

        x = self._embed_features(x, in_degree, out_degree, depth)

        pe_ml = self.ml_encoder(sample["dir_pe_ml"])
        x = x + pe_ml

        x, adj, distance = self._add_cls_token(x, adj, distance)

        x, aux_loss = self.encoder(x, adj, distance)

        x = self.post_norm(x)

        if self.graph_readout == "cls":
            graph_feat = x[:, 0, :]
        elif self.graph_readout == "att":
            att_logit = self.att_pool(x).squeeze(-1)
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
