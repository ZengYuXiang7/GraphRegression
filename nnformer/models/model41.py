import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseSAGEConv
from nnformer.models.registry import register_model


class PredHead(nn.Module):
    def __init__(self, d_model):
        super(PredHead, self).__init__()
        self.fc_1 = nn.Linear(d_model, d_model)
        self.fc_2 = nn.Linear(d_model, d_model)
        self.fc_drop_1 = nn.Dropout(p=0.05)
        self.fc_drop_2 = nn.Dropout(p=0.05)
        self.fc_relu1 = nn.ReLU()
        self.fc_relu2 = nn.ReLU()
    def forward(self, x):
        x = self.fc_relu1(self.fc_1(x))
        x = self.fc_relu2(self.fc_2(x))
        return x

@register_model("model41")
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_node_features = 5
        self.d_model = int(getattr(config, "d_model", 192))  # Model dimension
        self.dropout = float(getattr(config, "dropout", 0.05))  # Dropout rate
        self.gcn_layers = int(getattr(config, "gcn_layers", 2))  # Number of GCN layers

        # GCN layers
        self.gcn = nn.ModuleList(
            [DenseSAGEConv(self.num_node_features, self.d_model, normalize=True)] +  # Initial GCN layer
            [DenseSAGEConv(self.d_model, self.d_model, normalize=True) for _ in range(self.gcn_layers - 1)]  # Intermediate GCN layers
        )

        # Output layer
        if config.use_ffn:
            self.pred_head = PredHead(self.d_model)  # Number of GCN layers

        self.predictor = nn.Linear(self.d_model, 1)

        # Weights initialization
        self._initialize_weights()

    def get_data(self, sample, static_feature):
        x = sample['ops'].long()
        x = F.one_hot(x, num_classes=self.num_node_features).float()  # One-hot encoding for operations
        adj = sample['code_adj']  # Adjacency matrix for graph structure
        x = x.to(device=adj.device)  # Move the one-hot encoded tensor to the same device as x
        return x, adj

    def forward(self, sample, static_feature):
        """
        Forward pass that calculates latency prediction.
        """
        # Get the input data
        x, adj = self.get_data(sample, static_feature)

        # GCN processing
        for gcn_layer in self.gcn:
            x = gcn_layer(x, adj)  # Forward pass through each GCN layer
            x = F.relu(x)  # Apply ReLU activation after each GCN layer
            x = F.dropout(x, self.dropout, training=self.training)  # Apply dropout

        # Global pooling to obtain a fixed-size graph representation
        x = torch.sum(x, dim=1)  # Global mean pooling (or use sum pooling)

        if self.config.use_ffn:
            x = self.pred_head(x)

        # Final latency prediction
        latency = self.predictor(x)  # Predict latency

        return latency

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)