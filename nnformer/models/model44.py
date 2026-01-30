import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGATConv
from nnformer.models.registry import register_model


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


@register_model("model44")
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_node_features = 5
        self.d_model = int(getattr(config, "d_model", 192))  # Model dimension
        self.dropout = float(getattr(config, "dropout", 0.10))  # Dropout rate
        self.gcn_layers = int(getattr(config, "gcn_layers", 2))  # Number of GCN layers
        self.graph_readout = str(getattr(config, "graph_readout", 'sum'))  # Graph readout type
        self.d_ff_ratio = float(getattr(config, "d_ff_ratio", 4))  # FFN width control via ratio

        if self.graph_readout == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.op_embeds = nn.Linear(self.num_node_features, self.d_model)
        self.prev_norm = nn.LayerNorm(self.d_model)

        # GAT layers (Graph Attention Network)
        self.gat = nn.ModuleList(
            [DenseGATConv(self.d_model, self.d_model, heads=4, concat=False)] +  # Initial GAT layer
            [DenseGATConv(self.d_model, self.d_model, heads=4, concat=False) for _ in range(self.gcn_layers - 1)]  # Intermediate GAT layers
        )

        self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.gcn_layers)])

        self.ffn = nn.ModuleList(
            [FFN(self.d_model, self.d_ff_ratio) for _ in range(self.gcn_layers)]
        )

        self.post_norm = nn.LayerNorm(self.d_model)

        # Output head (PredHead) and final predictor
        if config.use_head:
            self.pred_head = PredHead(self.d_model)  # Number of GCN layers

        # Final predictor (output layer)
        self.predictor = nn.Linear(self.d_model, 1)

        # Weights initialization
        self.init_weights()

    def get_data(self, sample, static_feature):
        x = sample['ops'].long()
        x = F.one_hot(x, num_classes=self.num_node_features).float()  # One-hot encoding for operations
        adj = sample['code_adj']  # Adjacency matrix for graph structure
        x = x.to(device=adj.device)  # Move the one-hot encoded tensor to the same device as adj
        return x, adj

    def forward(self, sample, static_feature):
        """
        Forward pass that calculates latency prediction.
        """
        # Get the input data
        x, adj = self.get_data(sample, static_feature)

        x = self.op_embeds(x)

        if self.graph_readout == 'cls':
            num_nodes = adj.size(1)  # Assuming adj is of shape [batch_size, num_nodes, num_nodes]

            # Create a new adjacency matrix with the CLS token added (expand to 2D first)
            new_adj = torch.zeros(adj.size(0), num_nodes + 1, num_nodes + 1, device=adj.device)

            # Copy the original adjacency matrix into the top-left corner of the new adjacency matrix
            new_adj[:, 1:, 1:] = adj

            # Update the adjacency matrix to include the CLS token's connections
            new_adj[:, 0, :num_nodes] = 1  # CLS token connects to all nodes
            new_adj[:, :num_nodes, 0] = 1  # All nodes connect to CLS token

            # Optionally, add a self-loop to the CLS token
            # new_adj[:, 0, 0] = 1  # CLS token has a self-loop

            # Update adj to be the new adjacency matrix
            adj = new_adj

            # Create the CLS token (a learnable parameter)
            cls_token = self.cls_token.expand(x.size(0), 1, self.d_model)  # Expand CLS token to match batch size

            # Insert the CLS token at the first position in the feature matrix
            x = torch.cat([cls_token, x], dim=1)  # Add the CLS token as the first node

        x = self.prev_norm(x)

        # GAT + FFN (ResNet style)
        for gcn_layer, norm, ffn in zip(self.gat, self.norms, self.ffn):
            x_ = norm(x)
            x = gcn_layer(x_, adj) + x

            x_ = F.relu(x)
            x = ffn(x_) + x

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