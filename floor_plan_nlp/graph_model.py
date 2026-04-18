import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, SAGEConv, global_max_pool, global_mean_pool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required for Person 2 graph model. "
        "Install with: pip install torch-geometric"
    ) from exc


class GraphPlanEncoder(nn.Module):
    """
    Lightweight 3-layer graph encoder for plan-level embedding export.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 256,
        dropout: float = 0.2,
        conv_type: str = "sage",
    ):
        super().__init__()
        conv_cls = SAGEConv if conv_type == "sage" else GCNConv
        self.conv1 = conv_cls(in_dim, hidden_dim)
        self.conv2 = conv_cls(hidden_dim, hidden_dim)
        self.conv3 = conv_cls(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch):
        x, edge_index, graph_batch = batch.x, batch.edge_index, batch.batch

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)

        pooled = torch.cat(
            [global_mean_pool(x, graph_batch), global_max_pool(x, graph_batch)],
            dim=-1,
        )
        embedding = self.readout(pooled)
        return F.normalize(embedding, p=2, dim=-1)

