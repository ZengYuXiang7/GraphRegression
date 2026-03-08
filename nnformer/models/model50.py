import torch
import torch.nn as nn
import torch.nn.functional as F
from nnformer.models.registry import register_model


# =========================
# Dense graph operators (input: x, adj)
# adj: [N, N] (can be 0/1 or weighted), dense
# =========================


def _add_self_loops(adj: torch.Tensor):
    n = adj.size(0)
    return adj + torch.eye(n, device=adj.device, dtype=adj.dtype)


def _gcn_norm_adj(adj: torch.Tensor, eps: float = 1e-12):
    # A_hat = A + I
    a = _add_self_loops(adj)
    deg = a.sum(dim=1)  # [N]
    deg_inv_sqrt = (deg + eps).pow(-0.5)
    d_inv_sqrt = torch.diag(deg_inv_sqrt)
    return d_inv_sqrt @ a @ d_inv_sqrt  # [N, N]


def _row_norm_adj(adj: torch.Tensor, eps: float = 1e-12):
    # Row-normalize (mean aggregator style)
    deg = adj.sum(dim=1, keepdim=True)  # [N,1]
    return adj / (deg + eps)


class DenseGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, adj):
        a_norm = _gcn_norm_adj(adj)
        x = a_norm @ x
        x = self.lin(x)
        return x


class DenseSAGEConv(nn.Module):
    # simple mean aggregation: out = W_self x + W_neigh (A_row @ x)
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.lin_self.reset_parameters()
        self.lin_neigh.reset_parameters()

    def forward(self, x, adj):
        a_row = _row_norm_adj(adj)
        neigh = a_row @ x
        return self.lin_self(x) + self.lin_neigh(neigh)


class DenseGATConv(nn.Module):
    # Dense multi-head GAT (O(N^2)), masked by adj > 0
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_l = nn.Parameter(torch.empty(heads, out_channels))
        self.att_r = nn.Parameter(torch.empty(heads, out_channels))

        if bias:
            if concat:
                self.bias = nn.Parameter(torch.zeros(heads * out_channels))
            else:
                self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: [N, Fin], adj: [N, N]
        N = x.size(0)
        h = self.lin(x).view(N, self.heads, self.out_channels)  # [N,H,C]

        # e_ij = LeakyReLU( a_l^T h_i + a_r^T h_j )
        el = (h * self.att_l.unsqueeze(0)).sum(dim=-1)  # [N,H]
        er = (h * self.att_r.unsqueeze(0)).sum(dim=-1)  # [N,H]
        e = el.unsqueeze(1) + er.unsqueeze(0)  # [N,N,H]
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # mask: only edges where adj > 0
        mask = adj > 0
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        alpha = F.softmax(e, dim=1)  # softmax over j neighbors, [N,N,H]
        out = torch.einsum("ijh,jhc->ihc", alpha, h)  # [N,H,C]

        if self.concat:
            out = out.reshape(N, self.heads * self.out_channels)  # [N,H*C]
        else:
            out = out.mean(dim=1)  # [N,C]

        if self.bias is not None:
            out = out + self.bias
        return out


# =========================
# Experts (Dense)
# =========================


class PTNorm(nn.Module):
    def __init__(self, in_channel, out_channel, dropout, type):
        super(PTNorm, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.type = type
        self.norm = nn.LayerNorm(in_channel)

        if self.type == "SAGE":
            self.convs.append(DenseSAGEConv(in_channel, out_channel))
        elif self.type == "GCN":
            self.convs.append(DenseGCNConv(in_channel, out_channel))
        elif self.type == "GAT":
            self.convs.append(
                DenseGATConv(in_channel, out_channel // 4, heads=4, concat=True)
            )

        self.convs.append(nn.Linear(in_channel, out_channel))

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

    def forward(self, x, adj):
        if self.type in ["SAGE", "GCN", "GAT"]:
            x = self.convs[0](x, adj)
        x = self.convs[1](x)
        F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = self.norm(x)
        return x


class TPNorm(nn.Module):
    def __init__(self, in_channel, out_channel, dropout, type):
        super(TPNorm, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.type = type
        self.norm = nn.LayerNorm(in_channel)

        self.convs.append(nn.Linear(in_channel, out_channel))

        if self.type == "SAGE":
            self.convs.append(DenseSAGEConv(in_channel, out_channel))
        elif self.type == "GCN":
            self.convs.append(DenseGCNConv(in_channel, out_channel))
        elif self.type == "GAT":
            self.convs.append(
                DenseGATConv(in_channel, out_channel // 4, heads=4, concat=True)
            )

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

    def forward(self, x, adj):
        x = self.convs[0](x)
        F.dropout(F.relu(x), p=self.dropout, training=self.training)

        if self.type in ["SAGE", "GCN", "GAT"]:
            x = self.convs[1](x, adj)

        x = self.norm(x)
        return x


class TTNorm(nn.Module):
    def __init__(self, in_channel, out_channel, dropout, type):
        super(TTNorm, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.type = type
        self.convs.append(nn.Linear(in_channel, out_channel))
        self.convs.append(nn.Linear(in_channel, out_channel))
        self.norm = nn.LayerNorm(in_channel)

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

    def forward(self, x, adj):
        x = self.convs[0](x)
        F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = self.convs[1](x)
        F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = self.norm(x)
        return x


class PPNorm(nn.Module):
    def __init__(self, in_channel, out_channel, dropout, type):
        super(PPNorm, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.type = type
        self.norm = nn.LayerNorm(in_channel)

        if self.type == "SAGE":
            self.convs.append(DenseSAGEConv(in_channel, out_channel))
            self.convs.append(DenseSAGEConv(in_channel, out_channel))
        elif self.type == "GCN":
            self.convs.append(DenseGCNConv(in_channel, out_channel))
            self.convs.append(DenseGCNConv(in_channel, out_channel))
        elif self.type == "GAT":
            self.convs.append(
                DenseGATConv(in_channel, out_channel // 4, heads=4, concat=True)
            )
            self.convs.append(
                DenseGATConv(in_channel, out_channel // 4, heads=4, concat=True)
            )

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

    def forward(self, x, adj):
        if self.type in ["SAGE", "GCN", "GAT"]:
            x = self.convs[0](x, adj)
            x = self.convs[1](x, adj)
        x = self.norm(x)
        return x


# =========================
# Gating / GLU / MoE
# =========================
class GatingNetwork(nn.Module):
    def __init__(self, in_channels, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=-1),
        )

    def reset_parameters(self):
        for layer in self.gate:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        return self.gate(x)


class GLULayer(nn.Module):
    def __init__(self, hidden, activate):
        super(GLULayer, self).__init__()
        self.w1 = nn.Linear(hidden, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.w3 = nn.Linear(hidden, hidden)
        self.activate = activate

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()

    def forward(self, x):
        if self.activate == "SwishGLU":
            x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        elif self.activate == "GEGLU":
            x = self.w2(F.gelu(self.w1(x)) * self.w3(x))
        elif self.activate == "ReGLU":
            x = self.w2(F.relu(self.w1(x)) * self.w3(x))
        return x


class MoELayer(nn.Module):
    def __init__(self, hidden, dropout, expert_type, num_experts, gamma):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList(
            [
                PTNorm(hidden, hidden, dropout, expert_type),
                TPNorm(hidden, hidden, dropout, expert_type),
                TTNorm(hidden, hidden, dropout, expert_type),
                PPNorm(hidden, hidden, dropout, expert_type),
            ]
        )
        self.gating = GatingNetwork(hidden, num_experts)

    def reset_parameters(self):
        for expert in self.experts:
            expert.reset_parameters()
        self.gating.reset_parameters()

    def forward(self, x, adj):
        gating_weights = self.gating(x)

        expert_outputs = [expert(x, adj) for expert in self.experts]
        combined_output = torch.zeros_like(expert_outputs[0])
        for i, output in enumerate(expert_outputs):
            combined_output += gating_weights[..., i].unsqueeze(-1) * output

        return combined_output


class MoEFFNLayer(nn.Module):
    def __init__(self, hidden):
        super(MoEFFNLayer, self).__init__()
        self.GLUList = nn.ModuleList(
            [
                GLULayer(hidden, "SwishGLU"),
                GLULayer(hidden, "GEGLU"),
                GLULayer(hidden, "ReGLU"),
            ]
        )
        self.FFNGate = nn.Linear(hidden, 3)

    def reset_parameters(self):
        self.FFNGate.reset_parameters()
        for layer in self.GLUList:
            layer.reset_parameters()

    def forward(self, x):
        gate = F.gumbel_softmax(self.FFNGate(x), tau=1, hard=True)
        x = torch.stack([layer(x) for layer in self.GLUList], dim=2)
        x = torch.sum(x * gate.unsqueeze(-1), dim=2)
        return x


class MoE(nn.Module):
    def __init__(
        self, in_dim, hidden, n_layers, dropout, num_experts, expert_type, gamma
    ):
        super(MoE, self).__init__()
        self.dropout = dropout

        self.MoELayers = nn.ModuleList(
            [
                MoELayer(hidden, dropout, expert_type, num_experts, gamma)
                for _ in range(n_layers)
            ]
        )
        self.MoEFFNLayers = MoEFFNLayer(hidden)

        self.start = nn.Linear(in_dim, hidden)

        self.theta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.norm = nn.LayerNorm(hidden)

    def reset_parameters(self):
        self.start.reset_parameters()
        for layer in self.MoELayers:
            layer.reset_parameters()
        self.MoEFFNLayers.reset_parameters()

    # ======= 改这里：输入只有 x 和 adj =======
    def forward(self, x, adj):
        # Start-Linear
        x = self.start(x)
        initial_x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        res = initial_x

        # MoE layers
        for i, layer in enumerate(self.MoELayers):
            if i == 0:
                x = layer(initial_x, adj)
            else:
                x = layer(x, adj)

        # MoE FFN
        x = self.MoEFFNLayers(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm((1 - self.theta) * x + self.theta * res)

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


@register_model("model50")
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

        self.op_embeds = nn.Linear(self.num_node_features, self.d_model)
        self.depth_embed = nn.Linear(32, self.d_model)

        # self.op_embeds = nn.Embedding(self.num_node_features, self.d_model)
        # self.depth_embed = nn.Embedding(32, self.d_model)

        self.prev_norm = nn.LayerNorm(self.d_model)

        # self.encoder = nnEncoder(self.d_model, self.d_ff_ratio, self.gcn_layers)
        self.encoder = MoE(
            in_dim=self.d_model,
            hidden=self.d_model,
            n_layers=self.gcn_layers,
            dropout=0.1,
            num_experts=4,
            expert_type="SAGE",
            gamma=1.0,
        )

        self.post_norm = nn.LayerNorm(self.d_model)

        if config.use_head:
            self.pred_head = PredHead(self.d_model)  # Number of GCN layers

        self.predictor = nn.Linear(self.d_model, 1)

        # Weights initialization
        self.init_weights()

    def get_data(self, sample, static_feature):
        # x = sample["ops"].long()
        # 2026年02月28日17:23:56，先尝试各种编码方案
        x = sample["code"]
        adj = sample["code_adj"]  # Adjacency matrix for graph structure
        adj = adj + torch.eye(adj.size(1), device=adj.device)
        return x, adj

    def forward(self, sample, static_feature):
        x, adj = self.get_data(sample, static_feature)
        depth = sample["op_depth"]

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
