# coding : utf-8
# Author : Yuxiang Zeng
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGATConv
from nnformer.models.registry import register_model
import math
import torch
import torch.nn.functional as F


class GraphConvolution(torch.torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            weight_init="thomas",
            bias_init="thomas",
    ):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.torch.nn.Parameter(
            torch.FloatTensor(in_features, out_features)
        )
        if bias:
            self.bias = torch.torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self):
        self.init_tensor(self.weight, self.weight_init, "act")
        self.init_tensor(self.bias, self.bias_init, "act")

    def forward(self, adjacency, features):
        # print(features.shape)
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )

    @staticmethod
    def init_tensor(tensor, init_type, nonlinearity):
        if tensor is None or init_type is None:
            return
        if init_type == "thomas":
            size = tensor.size(-1)
            stdv = 1.0 / math.sqrt(size)
            torch.nn.init.uniform_(tensor, -stdv, stdv)
        elif init_type == "kaiming_normal_in":
            torch.nn.init.kaiming_normal_(
                tensor, mode="fan_in", nonlinearity=nonlinearity
            )
        elif init_type == "kaiming_normal_out":
            torch.nn.init.kaiming_normal_(
                tensor, mode="fan_out", nonlinearity=nonlinearity
            )
        elif init_type == "kaiming_uniform_in":
            torch.nn.init.kaiming_uniform_(
                tensor, mode="fan_in", nonlinearity=nonlinearity
            )
        elif init_type == "kaiming_uniform_out":
            torch.nn.init.kaiming_uniform_(
                tensor, mode="fan_out", nonlinearity=nonlinearity
            )
        elif init_type == "orthogonal":
            torch.nn.init.orthogonal_(
                tensor, gain=torch.nn.init.calculate_gain(nonlinearity)
            )
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, args):
        super(GCN, self).__init__()
        self.args = args
        self.nfeat = input_dim
        self.nlayer = num_layers
        self.nhid = hidden_dim
        self.dropout_ratio = dropout
        weight_init = "thomas"
        bias_init = "thomas"

        self.gcn = torch.nn.ModuleList()
        self.norm = torch.nn.ModuleList()
        self.act = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()

        # 初始化第一个图卷积层
        self.gcn.append(
            GraphConvolution(
                self.nfeat,
                self.nhid,
                bias=True,
                weight_init=weight_init,
                bias_init=bias_init,
            )
        )
        self.norm.append(torch.nn.LayerNorm(self.nhid))
        self.act.append(torch.nn.ReLU())
        self.dropout.append(torch.nn.Dropout(self.dropout_ratio))

        # 对后续层使用相同的隐藏层维度
        for i in range(1, self.nlayer):
            self.gcn.append(
                GraphConvolution(
                    self.nhid,
                    self.nhid,
                    bias=True,
                    weight_init=weight_init,
                    bias_init=bias_init,
                )
            )
            self.norm.append(torch.nn.LayerNorm(self.nhid))
            self.act.append(torch.nn.ReLU())
            self.dropout.append(torch.nn.Dropout(self.dropout_ratio))
        self.fc = torch.nn.Linear(self.nhid, 1)

    def forward(self, adjacency, features):
        x = features
        for i in range(0, self.nlayer):
            x = self.act[i](self.norm[i](self.gcn[i](adjacency, x)))
            x = self.dropout[i](x)
        x = x[:, 0]  # use global node
        y = self.fc(x)
        return y


@register_model("brpnas")
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_node_features = 5
        self.GCN = GCN(self.num_node_features + 1, 600, 4, 0.10, config)

    def get_data(self, sample, static_feature):
        x = sample['ops'].long()
        x = F.one_hot(x, num_classes=self.num_node_features).float()  # One-hot encoding for operations
        adj = sample['code_adj']  # Adjacency matrix for graph structure
        x = x.to(device=adj.device)  # Move the one-hot encoded tensor to the same device as adj
        return x, adj

    def forward(self, sample, static_feature):
        # x -> [batch_size, graph_nodes, one-hot]
        x, adj = self.get_data(sample, static_feature)

        additional_feature = torch.zeros(x.size(0), x.size(1) + 1, self.num_node_features + 1, device=x.device)  # [batch_size, nodes+1, features+1]
        additional_feature[:, 0, -1] = 1  # Set the 6th feature (index 5) for the first node
        additional_feature[:, 1:, :-1] = x  # Insert original features into the first 5 columns (for other nodes)
        x = additional_feature

        # Adjust adjacency matrix (expand it to accommodate the new node)
        new_adj = torch.zeros(adj.size(0), adj.size(1) + 1, adj.size(2) + 1, device=adj.device)
        new_adj[:, 1:, 1:] = adj
        new_adj[:, 0, :adj.size(1)] = 1  # New node (index 0) connects to all other nodes
        new_adj[:, :adj.size(1), 0] = 1  # All other nodes connect to the new node
        new_adj[:, 0, 0] = 0

        # Update adj to be the new adjacency matrix
        adj = new_adj

        # Pass the modified adjacency and feature matrices to the GCN
        output = self.GCN(adj, x)
        return output