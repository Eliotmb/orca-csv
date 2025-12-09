import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.dropout = dropout

    def forward(self, x):
        x = self.net(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ORCAGNNMultiTask(nn.Module):
    """GNN for multi-property prediction: energy + dipole moment."""

    def __init__(self, node_in_dim: int = 4, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINConv(MLP(hidden_dim, hidden_dim, hidden_dim)))
        
        # Shared graph representation
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.dipole_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.proj(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.shared(x)
        
        energy = self.energy_head(x).view(-1)
        dipole = self.dipole_head(x).view(-1)
        return energy, dipole

    def get_embedding(self, data):
        """Extract shared graph embedding."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.proj(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.shared(x)


class ORCAGNN(nn.Module):
    """Simple GIN-style GNN for energy regression."""

    def __init__(self, node_in_dim: int = 4, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINConv(MLP(hidden_dim, hidden_dim, hidden_dim)))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.proj(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        out = self.head(x)
        return out.view(-1)
