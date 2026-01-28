# coding: utf-8
# model29 (refactor): Input -> (LocalGNNBranch, GlobalTransformerBranch) -> Fuse -> PredictHead

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_scatter import scatter
from torch_geometric.utils import (
    to_dense_batch,
    to_dense_adj,
    to_undirected,
    degree,
)


from nnformer.models.registry import register_model
from nnformer.models.layer_init import init_tensor


# ============================================================
# 1) Local GNN Branch
# ============================================================
class LocalGNNBranch(nn.Module):
    """
    Local/GNN branch:
      raw(in_dim) -> SAGEConv K layers (first: in_dim->d_model; rest: d_model->d_model)
      readout: mean | sum | sum_ln | cls

    Notes:
      - If readout=cls: we add one CLS node per graph AFTER the first conv
        (because CLS token must be in d_model space).
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        gcn_layers: int,
        dropout: float = 0.05,
        readout: str = "sum",
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.gcn_layers = int(gcn_layers)

        self.readout = str(readout).lower()
        if self.readout not in {"mean", "sum", "sum_ln", "cls"}:
            raise ValueError(f"Unknown GNN readout: {self.readout}")

        # CLS token for GNN (in d_model space, because we add after first conv)
        if self.readout == "cls":
            self.gnn_cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.trunc_normal_(self.gnn_cls, std=0.02)
        else:
            self.gnn_cls = None

        # Convs
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.drops = nn.ModuleList()

        for i in range(self.gcn_layers):
            in_c = self.in_dim if i == 0 else self.d_model
            self.convs.append(tgnn.SAGEConv(in_c, self.d_model, normalize=True))
            self.acts.append(nn.ReLU())
            self.drops.append(nn.Dropout(dropout))

    @staticmethod
    def _scatter_readout(
        x_nodes: torch.Tensor, batch: torch.Tensor, mode: str
    ) -> torch.Tensor:
        if mode == "mean":
            return scatter(x_nodes, batch, dim=0, reduce="mean")
        if mode == "sum":
            return scatter(x_nodes, batch, dim=0, reduce="sum")
        if mode == "sum_ln":
            g = scatter(x_nodes, batch, dim=0, reduce="sum")
            return F.layer_norm(g, (g.size(-1),))
        raise ValueError(mode)

    def _add_cls_after_first_conv(
        self, h_dmodel: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ):
        """
        h_dmodel: [N, d_model]
        Add one CLS node per graph, connect CLS <-> all nodes in that graph.
        Returns:
          h2:        [N+B, d_model]
          edge2:     [2, E + 2N]
          batch2:    [N+B]
          cls_index: [B]
        """
        device = h_dmodel.device
        B = int(batch.max().item()) + 1
        N = h_dmodel.size(0)

        cls = self.gnn_cls.expand(B, 1, self.d_model).squeeze(1)  # [B,d_model]
        cls_index = torch.arange(N, N + B, device=device)

        h2 = torch.cat([h_dmodel, cls], dim=0)
        batch2 = torch.cat([batch, torch.arange(B, device=device)], dim=0)

        node_idx = torch.arange(N, device=device)
        cls_for_node = cls_index[batch]  # [N]

        e1 = torch.stack([node_idx, cls_for_node], dim=0)  # node -> cls
        e2 = torch.stack([cls_for_node, node_idx], dim=0)  # cls -> node
        edge2 = torch.cat([edge_index, e1, e2], dim=1)

        return h2, edge2, batch2, cls_index

    def forward(
        self, x_raw: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns:
          g_local: [B, d_model]
        """
        if self.gcn_layers <= 0:
            raise ValueError("gcn_layers must be >= 1")

        # first conv: raw -> d_model
        h = self.drops[0](self.acts[0](self.convs[0](x_raw, edge_index)))  # [N,d_model]

        if self.readout == "cls":
            # add CLS in d_model space
            h2, edge2, batch2, cls_index = self._add_cls_after_first_conv(
                h, edge_index, batch
            )

            # remaining layers on augmented graph
            for i in range(1, self.gcn_layers):
                h2 = self.drops[i](self.acts[i](self.convs[i](h2, edge2)))

            g_local = h2[cls_index]  # [B,d_model]
            return g_local

        # no CLS: just keep passing on original graph
        for i in range(1, self.gcn_layers):
            h = self.drops[i](self.acts[i](self.convs[i](h, edge_index)))

        g_local = self._scatter_readout(h, batch, self.readout)  # [B,d_model]
        return g_local


# ============================================================
# 2) Global Transformer Branch
# ============================================================
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn.dense.linear import Linear


class GNN_LinearAttn(nn.Module):
    """
    原样照搬（只把命名 degree -> degree_gate 避免遮蔽）
    x: (B, N, D)
    A: (B, N, N) 0/1 adjacency (pad 行列应为 0)
    return: (B, N, out_dim)
    """

    def __init__(self, in_dim, out_dim, normalize, degree=False, bias=True):
        super(GNN_LinearAttn, self).__init__()
        self.normalize = normalize
        self.degree = degree

        self.lin_l = Linear(in_dim, out_dim, bias=bias)
        self.lin_r = Linear(in_dim, out_dim, bias=False)

        if degree:
            self.lin_d = Linear(1, in_dim, bias=True)
            self.sigmoid_d = nn.Sigmoid()

        self.lin_qk = Linear(in_dim, in_dim, bias=True)
        self.sigmoid_qk = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_qk.reset_parameters()
        if self.degree:
            self.lin_d.reset_parameters()

    def forward(self, x, A):
        # x (B, N, D)
        # A (B, N, N)
        if self.degree:
            deg = A.sum(dim=-1, keepdim=True)  # (B, N, 1)
            deg = self.sigmoid_d(self.lin_d(deg))
            x = x * deg

        QK = self.sigmoid_qk(self.lin_qk(x))  # (B,N,D)
        scores = torch.matmul(QK, QK.transpose(-2, -1)) / math.sqrt(
            x.size(-1)
        )  # (B,N,N)
        scores = scores * A  # only neighbors

        attn = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.matmul(attn, x)  # (B,N,D)
        out = self.lin_l(out)
        out = out + self.lin_r(x)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out


class GroupLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_groups: int = 4,
        use_shuffle: bool = False,
        norm_type: Optional[str] = None,
        use_bias: bool = False,
    ):
        super(GroupLinear, self).__init__()

        if in_features % n_groups != 0:
            raise Exception(
                f"Input dim {in_features} must be divisible by n_groups {n_groups}"
            )
        if out_features % n_groups != 0:
            raise Exception(
                f"Output dim {out_features} must be divisible by n_groups {n_groups}"
            )

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        self.weights = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        self.bias = (
            nn.Parameter(torch.Tensor(n_groups, 1, out_groups)) if use_bias else None
        )

        if norm_type is not None:
            if "ln" in norm_type.lower():
                self.normalization_fn = nn.LayerNorm(out_groups)
            else:
                raise NotImplementedError
        else:
            self.normalization_fn = None

        self.n_groups = n_groups
        self.feature_shuffle = True if use_shuffle else False
        self.use_bias = use_bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)

    def process_input_bmm(self, x):
        bsz = x.size(0)
        x = x.contiguous().view(bsz, self.n_groups, -1)  # [B,g,N/g]
        x = x.transpose(0, 1)  # [g,B,N/g]
        x = torch.bmm(x, self.weights)  # [g,B,M/g]

        if self.use_bias:
            x = torch.add(x, self.bias)

        if self.feature_shuffle:
            x = x.permute(1, 2, 0).contiguous().view(bsz, self.n_groups, -1)
        else:
            x = x.transpose(0, 1)  # [B,g,M/g]

        if self.normalization_fn is not None:
            x = self.normalization_fn(x)

        x = x.contiguous().view(bsz, -1)
        return x

    def forward(self, x):
        if x.dim() == 2:
            return self.process_input_bmm(x)
        elif x.dim() == 3:
            # 支持 [B,T,N] 或 [T,B,N] 都行，这里按你原作者写法处理
            T, B, N = x.size()
            x = x.contiguous().view(B * T, -1)
            x = self.process_input_bmm(x)
            x = x.contiguous().view(T, B, -1)
            return x
        else:
            raise NotImplementedError


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GroupedFeedForward(nn.Module):
    def __init__(
        self, dim, hidden_dim, feat_shuffle, norm_type=None, num_groups=8, dropout=0.0
    ):
        super().__init__()
        glt_up = GroupLinear(
            dim, hidden_dim, num_groups, feat_shuffle, norm_type, use_bias=True
        )
        glt_down = GroupLinear(
            hidden_dim, dim, num_groups, feat_shuffle, norm_type, use_bias=True
        )

        self.net = nn.Sequential(
            glt_up, nn.ReLU(), nn.Dropout(dropout), glt_down, nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class GlobalNarFormerBackbone(nn.Module):
    """
    输入：x[N,F], edge_index[2,E], batch[N]
    输出：g[B,H] (sum/mean)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        ffn_ratio: int = 4,
        dropout: float = 0.05,
        normalize: bool = True,
        degree: bool = True,  # 对应原作者的 degree gate
        feat_shuffle: bool = False,
        glt_norm: str = "LN",
        num_groups: int = 8,
        init_values: float = 1e-4,  # layer scale
        pool: str = "sum",  # "sum" | "mean"
        add_self_loop: bool = True,  # 邻接加 self-loop（通常更稳）
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.pool = str(pool).lower()
        if self.pool not in {"sum", "mean"}:
            raise ValueError(f"Unknown pool: {self.pool}")
        self.add_self_loop = bool(add_self_loop)

        self.gnn_layers = nn.ModuleList()
        self.relu = nn.ModuleList()
        self.drop = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.gamma = nn.ParameterList()

        for _ in range(int(n_layers)):
            self.gnn_layers.append(
                GNN_LinearAttn(hidden_dim, hidden_dim, normalize, degree)
            )
            self.relu.append(nn.ReLU())
            self.drop.append(nn.Dropout(dropout))
            self.ffn_layers.append(
                PreNorm(
                    hidden_dim,
                    GroupedFeedForward(
                        hidden_dim,
                        hidden_dim * int(ffn_ratio),
                        feat_shuffle,
                        glt_norm,
                        num_groups=num_groups,
                        dropout=dropout,
                    ),
                )
            )
            self.gamma.append(
                nn.Parameter(init_values * torch.ones(hidden_dim), requires_grad=True)
            )

    @staticmethod
    def _mask_adj(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # A: [B,N,N] float, mask: [B,N] bool
        m = mask.to(dtype=A.dtype)
        A = A * m.unsqueeze(1) * m.unsqueeze(2)  # 去掉 pad 行/列
        return A

    def forward(self, x, edge_index, batch):
        # 1) sparse -> dense
        x_dense, mask = to_dense_batch(x, batch=batch)  # [B,Nmax,F], [B,Nmax] bool
        B, Nmax, _ = x_dense.shape

        # 2) dense adj
        A = to_dense_adj(edge_index=edge_index, batch=batch, max_num_nodes=Nmax).to(
            torch.float32
        )  # [B,N,N]
        A = self._mask_adj(A, mask)
        if self.add_self_loop:
            eye = torch.eye(Nmax, device=A.device, dtype=A.dtype).unsqueeze(0)
            A = A + eye * mask.to(A.dtype).unsqueeze(1)
            A = (A > 0).to(A.dtype)

        # 3) proj
        h = self.in_proj(x_dense)  # [B,N,H]

        # 4) stacked blocks (原作者逻辑：gnn->relu->drop->ffn residual)
        for gnn, relu, drop, ffn, gamma in zip(
            self.gnn_layers, self.relu, self.drop, self.ffn_layers, self.gamma
        ):
            h = gnn(h, A)
            h = relu(h)
            h = drop(h)
            h2 = ffn(h)
            h = h + gamma * h2

        # 5) pool (只对 valid nodes)
        h = h * mask.unsqueeze(-1).to(h.dtype)
        if self.pool == "sum":
            g = h.sum(dim=1)  # [B,H]
        else:
            denom = mask.sum(dim=1).clamp_min(1).to(h.dtype).unsqueeze(-1)
            g = h.sum(dim=1) / denom

        return g


# ============================================================
# 3) Predict Head
# ============================================================
class PredictHead(nn.Module):
    """
    static_feature (4) -> proj(d_model) -> concat with graph emb (d_model) -> MLP -> pred
    """

    def __init__(
        self,
        d_model: int,
        fc_hidden: int,
        dropout: float = 0.05,
        norm_sf: bool = True,
        with_sf=True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.norm_sf = bool(norm_sf)
        self.with_sf = with_sf
        
        if self.norm_sf:
            self.sf_linear = nn.Linear(4, self.d_model)
            self.sf_drop = nn.Dropout(dropout)
            self.sf_act = nn.ReLU()
            sf_hidden = self.d_model
        else:
            if with_sf:
                sf_hidden = 4
            else:
                sf_hidden = 0
                

        in_dim = self.d_model + sf_hidden
        self.fc_1 = nn.Linear(in_dim, fc_hidden)
        self.fc_2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_drop_1 = nn.Dropout(dropout)
        self.fc_drop_2 = nn.Dropout(dropout)
        self.fc_relu1 = nn.ReLU()
        self.fc_relu2 = nn.ReLU()
        self.predictor = nn.Linear(fc_hidden, 1)

    def forward(self, g: torch.Tensor, static_feature: torch.Tensor):
        if self.norm_sf and self.with_sf:
            static_feature = self.sf_act(self.sf_drop(self.sf_linear(static_feature)))
            h = torch.cat([g, static_feature], dim=1)
        else:
            h = g
        h = self.fc_drop_1(self.fc_relu1(self.fc_1(h)))
        h = self.fc_drop_2(self.fc_relu2(self.fc_2(h)))
        out = self.predictor(h)
        pred = -F.logsigmoid(out)
        return pred


# ============================================================
# 4) Model32 = orchestrator
# ============================================================
@register_model("model32")
class Net(nn.Module):
    """
    Orchestrator:
      - parse graph input
      - optionally compute local/global branches
      - fuse
      - PredictHead
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_node_features = 5
        self.d_model = int(getattr(config, "d_model", 128))
        self.dropout = float(getattr(config, "dropout", 0.05))
        self.gcn_layers = int(getattr(config, "gcn_layers", 2))

        # fuse
        self.fuse = str(getattr(config, "fuse_method", "sum")).lower()
        if self.fuse not in {"local_only", "global_only", "sum", "weighted"}:
            raise ValueError(f"Unknown fuse_method: {self.fuse}")

        # local readout
        self.gnn_readout = str(getattr(config, "readout", "sum")).lower()
        if self.gnn_readout not in {"mean", "sum", "sum_ln", "cls"}:
            raise ValueError(f"Unknown readout: {self.gnn_readout}")

        # Local branch
        self.local_branch = LocalGNNBranch(
            in_dim=self.num_node_features,
            d_model=self.d_model,
            gcn_layers=self.gcn_layers,
            dropout=self.dropout,
            readout=self.gnn_readout,
        )

        # Global TF core + branch wrapper
        self.global_tf = GlobalNarFormerBackbone(
            in_dim=self.num_node_features,
            hidden_dim=self.d_model,
            n_layers=int(getattr(config, "tf_layers", 2)),
            ffn_ratio=int(getattr(config, "tf_ffn_ratio", 4)),
            dropout=float(getattr(config, "tf_dropout", self.dropout)),
            normalize=True,
            degree=bool(int(getattr(config, "tf_degree", 0))),  # 默认关
            pool=str(getattr(config, "tf_pool", "sum")),  # sum/mean
        )

        # fuse stabilizer (optional)
        if self.fuse == "weighted":
            self.fuse_logit = nn.Parameter(torch.tensor(0.0))

        # head
        self.head = PredictHead(
            d_model=self.d_model,
            fc_hidden=self.d_model,
            dropout=self.dropout,
            norm_sf=True if self.config.dataset == "nnlqp" else False,
            with_sf=True if self.config.dataset == "nnlqp" else False,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_tensor(m.weight, "thomas", "relu")
                init_tensor(m.bias, "thomas", "relu")

    def forward(self, data, static_feature):
        # ---- parse inputs ----
        x_raw = data.x
        edge_index = to_undirected(data.edge_index)
        batch = data.batch

        # ---- compute skipping ----
        need_local = self.fuse in {"local_only", "sum", "weighted"}
        need_global = self.fuse in {"global_only", "sum", "weighted"}

        g_local = None
        g_global = None

        if need_local:
            g_local = self.local_branch(x_raw, edge_index, batch)  # [B,d_model]

        if need_global:
            g_global = self.global_tf(x_raw, edge_index, batch)

        # ---- fuse ----
        if self.fuse == "local_only":
            g = g_local
        elif self.fuse == "global_only":
            g = g_global
        elif self.fuse == "sum":
            g = g_local + g_global
        elif self.fuse == "weighted":
            alpha = torch.sigmoid(self.fuse_logit)
            g = alpha * g_local + (1.0 - alpha) * g_global
        else:
            raise ValueError(self.fuse)

        # ---- head ----
        pred = self.head(g, static_feature)
        return pred
