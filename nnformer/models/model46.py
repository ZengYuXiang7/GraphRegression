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

class FFN(nn.Module):
    def __init__(self, d_model, d_ff_ratio):
        super(FFN, self).__init__()
        # Use d_ff_ratio to define the width of FFN layers
        d_ff = int(d_model * d_ff_ratio)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)  # First layer (Up-sampling)
        self.fc2 = nn.Linear(d_ff, d_model)  # Second layer (Down-sampling)
        self.act = nn.ReLU()  # GELU activation
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x):
        x_ = self.norm(x)

        x_ = self.fc1(x_)
        x_ = self.act(x_)  # Up-sampling with activation
        x_ = self.dropout1(x_)
        x_ = self.fc2(x_)  # Down-sampling to match original dimension
        x_ = self.dropout2(x_)

        return x_ + x


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


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff_ratio, gcn_layers):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.gcn_layers = gcn_layers
        self.d_ff_ratio = d_ff_ratio

        # GAT layers (Graph Attention Network)
        self.gat = nn.ModuleList(
            [DenseGATConv(self.d_model, self.d_model, heads=4, concat=False)] +  # Initial GAT layer
            [DenseGATConv(self.d_model, self.d_model, heads=4, concat=False) for _ in range(self.gcn_layers - 1)]  # Intermediate GAT layers
        )

        self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.gcn_layers)])

        self.ffn = nn.ModuleList(
            [FFN(self.d_model, self.d_ff_ratio) for _ in range(self.gcn_layers)]
        )

    def forward(self, x, adj):
        # GAT + FFN (ResNet style)
        for gcn_layer, norm, ffn in zip(self.gat, self.norms, self.ffn):
            x_ = norm(x)
            x = gcn_layer(x_, adj) + x

            x_ = F.relu(x)
            x = ffn(x_) + x
        return x




class SquareReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * F.relu(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float = 0.0,
        rel_pos_bias: bool = False,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_size = dim // n_head
        self.scale = math.sqrt(self.head_size)

        self.qkv = nn.Linear(dim, 3 * dim, False)
        self.proj = nn.Linear(dim, dim, False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rel_pos_bias = rel_pos_bias
        # if rel_pos_bias:
        #     self.rel_pos_forward = nn.Embedding(10, self.n_head, padding_idx=9)
        #     self.rel_pos_backward = nn.Embedding(10, self.n_head, padding_idx=9)

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        B, L, C = x.shape

        query, key, value = self.qkv(x).chunk(3, -1)
        query = query.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        score = torch.matmul(query, key.mT) / self.scale
        if self.rel_pos_bias:
            adj = adj.masked_fill(torch.logical_and(adj > 1, adj < 9), 0)
            adj = adj.masked_fill(adj != 0, 1)
            adj = adj.float()
            # pe = torch.stack([adj], dim=1).repeat(1, self.n_head // 1, 1, 1)
            # pe = torch.stack([adj.mT], dim=1).repeat(1, self.n_head // 1, 1, 1)
            # pe = torch.stack([adj, adj.mT], dim=1).repeat(1, self.n_head // 2, 1, 1)
            # pe = torch.stack([adj, adj.mT, adj @ adj, adj.mT @ adj.mT], dim=1)
            pe = torch.stack([adj, adj.mT, adj.mT @ adj, adj @ adj.mT], dim=1)
            pe = pe + torch.eye(L, dtype=adj.dtype, device=adj.device)
            pe = pe.int()

            # pe = (
            #     self.rel_pos_forward(rel_pos) + self.rel_pos_backward(rel_pos.mT)
            # ).permute(0, 3, 1, 2)
            # score = score * (1 + pe)
            score = score.masked_fill(pe == 0, -torch.inf)
        attn = F.softmax(score, dim=-1)
        attn = self.attn_dropout(attn)  # (b, n_head, l_q, l_k)
        x = torch.matmul(attn, value)

        x = x.transpose(1, 2).reshape(B, L, C)
        return self.resid_dropout(self.proj(x))

    def extra_repr(self) -> str:
        return f"n_head={self.n_head}"


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float,
        droppath: float,
        rel_pos_bias: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # The larger the dataset, the better rel_pos_bias works
        # probably due to the overfitting of rel_pos_bias
        self.attn = MultiHeadAttention(dim, n_head, dropout, rel_pos_bias=rel_pos_bias)

    def forward(self, x: Tensor, rel_pos: Optional[Tensor] = None) -> Tensor:
        x_ = self.norm(x)
        x_ = self.attn(x_, rel_pos)
        return x_ + x


class Mlp(nn.Module):
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
        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        elif act_layer.lower() == "square_relu":
            self.act = SquareReLU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, False)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        act_layer: str,
        dropout: float,
        droppath: float,
        gcn: bool = False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, act_layer=act_layer, drop=dropout)

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        x_ = self.norm(x)
        x_ = self.mlp(x_, adj)
        return x_ + x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        mlp_ratio: float,
        act_layer: str,
        dropout: float,
        droppath: float,
    ):
        super().__init__()
        self.self_attn = SelfAttentionBlock(
            dim, n_head, dropout, droppath, rel_pos_bias=True
        )
        self.feed_forward = FeedForwardBlock(
            dim, mlp_ratio, act_layer, dropout, droppath, gcn=True
        )

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        x = self.self_attn(x, adj)
        x = self.feed_forward(x, adj)
        return x


class nnEncoder(nn.Module):
    def __init__(self, d_model, d_ff_ratio, gcn_layers):
        super().__init__()
        # Encoder stage
        self.layers = nn.ModuleList()
        for i in range(12):
            self.layers.append(
                EncoderBlock(d_model, 4, 4, 'relu', 0.1, 0)
            )

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
        return x


@register_model("model46")
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_node_features = 32
        self.d_model = int(getattr(config, "d_model", 192))  # Model dimension
        self.dropout = float(getattr(config, "dropout", 0.10))  # Dropout rate
        self.gcn_layers = int(getattr(config, "gcn_layers", 2))  # Number of GCN layers
        self.graph_readout = str(getattr(config, "graph_readout", 'cls'))  # Graph readout type
        self.d_ff_ratio = float(getattr(config, "d_ff_ratio", 4))  # FFN width control via ratio

        if self.graph_readout == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        self.op_embeds = nn.Linear(self.num_node_features, self.d_model)
        self.depth_embed = nn.Linear(32, self.d_model)
        self.prev_norm = nn.LayerNorm(self.d_model)

        self.encoder = Encoder(self.d_model, self.d_ff_ratio, self.gcn_layers)
        # self.encoder = nnEncoder(self.d_model, self.d_ff_ratio, self.gcn_layers)

        self.post_norm = nn.LayerNorm(self.d_model)

        # Output head (PredHead) and final predictor
        if config.use_head:
            self.pred_head = PredHead(self.d_model)  # Number of GCN layers

        # Final predictor (output layer)
        self.predictor = nn.Linear(self.d_model, 1)

        # Weights initialization
        self.init_weights()

    def get_data(self, sample, static_feature):
        # x = sample['ops'].long()
        # x = F.one_hot(x, num_classes=self.num_node_features).float()  # One-hot encoding for operations
        x = sample['code']
        adj = sample['code_adj']  # Adjacency matrix for graph structure
        x = x.to(device=adj.device)  # Move the one-hot encoded tensor to the same device as adj
        return x, adj

    def forward(self, sample, static_feature):
        x, adj = self.get_data(sample, static_feature)

        depth = sample['op_depth'].long()
        depth = F.one_hot(depth, num_classes=32).float()

        # 首先是不是编码的问题
        # x = self.op_embeds(x)
        x = self.op_embeds(x) + self.depth_embed(depth)

        # 然后才是encoder的问题

        if self.graph_readout == 'cls':
            num_nodes = adj.size(1)  # Assuming adj is of shape [batch_size, num_nodes, num_nodes]

            # Create a new adjacency matrix with the CLS token added (expand to 2D first)
            new_adj = torch.zeros(adj.size(0), num_nodes + 1, num_nodes + 1, device=adj.device)

            # Copy the original adjacency matrix into the top-left corner of the new adjacency matrix
            new_adj[:, 1:, 1:] = adj

            # Update the adjacency matrix to include the CLS token's connections
            new_adj[:, 0, :num_nodes] = 1  # CLS token connects to all nodes
            # new_adj[:, :num_nodes, 0] = 1  # All nodes connect to CLS token

            # Optionally, add a self-loop to the CLS token
            # new_adj[:, 0, 0] = 1  # CLS token has a self-loop

            # Update adj to be the new adjacency matrix
            adj = new_adj

            # Create the CLS token (a learnable parameter)
            cls_token = self.cls_token.expand(x.size(0), 1, self.d_model)  # Expand CLS token to match batch size

            # Insert the CLS token at the first position in the feature matrix
            x = torch.cat([cls_token, x], dim=1)  # Add the CLS token as the first node

        x = self.prev_norm(x)

        x = self.encoder(x, adj)

        x = self.post_norm(x)

        # Global pooling to obtain a fixed-size graph representation
        if self.graph_readout == 'sum':
            x = torch.sum(x, dim=1)  # Global sum pooling
        elif self.graph_readout == 'mean':
            x = torch.mean(x, dim=1)
        elif self.graph_readout == 'max':
            x = torch.max(x, dim=1)[0]  # Get only the max values
        elif self.graph_readout == 'cls':
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
