import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
        num_layers: int = 3,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        conv_cls = SAGEConv if conv_type == "sage" else GCNConv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(conv_cls(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(conv_cls(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _forward_block(self, x, edge_index, conv, norm, use_dropout):
        x = conv(x, edge_index)
        x = norm(x)
        x = F.relu(x)
        if use_dropout:
            x = self.dropout(x)
        return x

    def forward(self, batch):
        x, edge_index, graph_batch = batch.x, batch.edge_index, batch.batch

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            use_dropout = layer_idx < (len(self.convs) - 1)
            if self.use_gradient_checkpointing and self.training and x.requires_grad:
                x = checkpoint(
                    lambda x_: self._forward_block(x_, edge_index, conv, norm, use_dropout),
                    x,
                    use_reentrant=False,
                )
            else:
                x = self._forward_block(x, edge_index, conv, norm, use_dropout)

        pooled = torch.cat(
            [global_mean_pool(x, graph_batch), global_max_pool(x, graph_batch)],
            dim=-1,
        )
        embedding = self.readout(pooled)
        return F.normalize(embedding, p=2, dim=-1)

