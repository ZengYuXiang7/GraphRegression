from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mytools.registry import register_model
from torch import Tensor
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


class DenseGraphSAGEBlock(nn.Module):
    """Dense GraphSAGE block for dense adjacency matrices.

    Input:
        x: [B, N, F]
        adj: [B, N, N]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
        concat: bool = True,
        use_bn: bool = True,
    ) -> None:
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
                self.bns.append(nn.LayerNorm(dims[i + 1]))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        outs = []

        for i, conv in enumerate(self.convs):
            x = conv(x, adj)

            if i != len(self.convs) - 1:
                x = F.relu(x)

                if self.use_bn:
                    x = self.bns[i](x)

                x = F.dropout(x, p=self.dropout, training=self.training)

            outs.append(x)

        if self.concat:
            return torch.cat(outs, dim=-1)
        return outs[-1]


class DiffPoolLayer(nn.Module):
    """One dense DiffPool layer."""

    def __init__(
        self,
        input_dim: int,
        embed_hidden_dim: int,
        embed_dim: int,
        assign_hidden_dim: int,
        assign_dim: int,
        gnn_layers: int = 3,
        dropout: float = 0.0,
        concat: bool = True,
    ) -> None:
        super().__init__()

        self.embed_gnn = DenseGraphSAGEBlock(
            in_dim=input_dim,
            hidden_dim=embed_hidden_dim,
            out_dim=embed_dim,
            num_layers=gnn_layers,
            dropout=dropout,
            concat=concat,
        )

        self.assign_gnn = DenseGraphSAGEBlock(
            in_dim=input_dim,
            hidden_dim=assign_hidden_dim,
            out_dim=assign_dim,
            num_layers=gnn_layers,
            dropout=dropout,
            concat=False,
        )

    def forward(
        self, x: Tensor, adj: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return pooled graph states and diffpool auxiliary outputs."""
        z = self.embed_gnn(x, adj)
        s = self.assign_gnn(x, adj)

        x_next, adj_next, lp_loss, ent_loss = dense_diff_pool(z, adj, s)
        return x_next, adj_next, lp_loss, ent_loss, z, s


class DiffPoolEncoder(nn.Module):
    """Dense adjacency + PyG DiffPool encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        base_layers: int = 4,
        pool_gnn_layers: int = 3,
        num_pooling: int = 1,
        assign_dim: int = 7,
        pool_ratio: float = 0.25,
        dropout: float = 0.0,
        concat: bool = True,
        final_readout: str = "sum",  # "sum" | "mean" | "max" | "cls"
    ) -> None:
        super().__init__()

        assert final_readout in ["sum", "mean", "max", "cls"]

        self.concat = concat
        self.final_readout = final_readout
        self.num_pooling = num_pooling

        self.pre_gnn = DenseGraphSAGEBlock(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=embedding_dim,
            num_layers=base_layers,
            dropout=dropout,
            concat=concat,
        )

        if concat:
            self.pre_out_dim = hidden_dim * (base_layers - 1) + embedding_dim
            self.pool_out_dim = hidden_dim * (pool_gnn_layers - 1) + embedding_dim
        else:
            self.pre_out_dim = embedding_dim
            self.pool_out_dim = embedding_dim

        pool_input_dim = self.pre_out_dim

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
                    gnn_layers=pool_gnn_layers,
                    dropout=dropout,
                    concat=concat,
                )
            )
            cur_assign_dim = max(1, int(cur_assign_dim * pool_ratio))

        self.output_dim = self.pre_out_dim + num_pooling * self.pool_out_dim

    def _graph_readout(self, x: Tensor) -> Tensor:
        if self.final_readout == "cls":
            return x[:, 0, :]

        if self.final_readout == "sum":
            out = x.sum(dim=1)
        elif self.final_readout == "mean":
            out = x.mean(dim=1)
        else:  # "max"
            out = x.max(dim=1).values

        return out

    def forward(
        self, x: Tensor, adj: Tensor, return_all: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Dict[str, Any]]:
        x = self.pre_gnn(x, adj)

        readouts = [self._graph_readout(x)]

        total_lp_loss = x.new_zeros(())
        total_ent_loss = x.new_zeros(())

        all_assign = []
        all_node_emb = [x]

        for pool_layer in self.pool_layers:
            x, adj, lp_loss, ent_loss, _z, s = pool_layer(x, adj)

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
    """Graph latency predictor with Dense GraphSAGE + DiffPool encoder."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

        self.num_node_features = 32
        self.d_model = self._cfg("d_model", 192, int)
        self.dropout = self._cfg("dropout", 0.10, float)
        self.backbone_layers = self._cfg("gcn_layers", 4, int)
        self.pool_gnn_layers = self._cfg("pool_gnn_layers", 2, int)
        self.num_pooling = self._cfg("num_pooling", 1, int)
        self.pool_ratio = self._cfg("pool_ratio", 0.25, float)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.op_embeds = nn.Linear(self.num_node_features, self.d_model)
        self.depth_embed = nn.Linear(32, self.d_model)

        self.encoder = DiffPoolEncoder(
            input_dim=self.d_model,
            hidden_dim=self.d_model,
            embedding_dim=self.d_model,
            base_layers=self.backbone_layers,
            pool_gnn_layers=self.pool_gnn_layers,
            num_pooling=self.num_pooling,
            assign_dim=4,
            pool_ratio=self.pool_ratio,
            dropout=self.dropout,
            concat=True,
            final_readout="cls",
        )

        self.encoder_out_dim = self.encoder.output_dim
        self.predictor = nn.Linear(self.encoder_out_dim, 1)
        self.init_weights()

    def _cfg(self, key: str, default: Any, cast: Callable[[Any], Any]) -> Any:
        return cast(getattr(self.config, key, default))

    def _append_cls_node(self, x: Tensor, adj: Tensor) -> Tuple[Tensor, Tensor]:
        """Prepend a learned CLS node and connect it to all nodes."""
        num_nodes = adj.size(1)
        full_adj = torch.ones(adj.size(0), num_nodes + 1, num_nodes + 1, device=adj.device)
        full_adj[:, 1:, 1:] = adj

        cls_token = self.cls_token.expand(x.size(0), 1, self.d_model)
        x_with_cls = torch.cat([cls_token, x], dim=1)
        return x_with_cls, full_adj

    def get_data(
        self, sample: Dict[str, Tensor], static_feature: Any
    ) -> Tuple[Tensor, Tensor]:
        del static_feature
        x = sample["code"]
        adj = sample["code_adj"]
        adj = adj + torch.eye(adj.size(1), device=adj.device)
        return x, adj

    def forward(
        self, sample: Dict[str, Tensor], static_feature: Any
    ) -> Tuple[Tensor, Tensor]:
        x, adj = self.get_data(sample, static_feature)
        depth = sample["op_depth"]

        x = self.op_embeds(x) + self.depth_embed(depth)
        x, adj = self._append_cls_node(x, adj)

        graph_emb, aux_loss = self.encoder(x, adj, return_all=False)
        latency = self.predictor(graph_emb)
        return latency, aux_loss

    def init_weights(self) -> None:
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.constant_(m.weight, 0.02)
