import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, GATConv

class RGCNEncoder(nn.Module):
    def __init__(self, n_drugs, emb_dim, num_rels, num_layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_drugs, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.convs = nn.ModuleList([
            RGCNConv(emb_dim, emb_dim, num_rels) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        # If x is indices (LongTensor), do embedding lookup
        if x.dtype == torch.long:
            h = self.emb(x)
        else:
            # Already dense features (e.g. RotatE initialisation)
            h = x

        for conv in self.convs:
            h = conv(h, edge_index, edge_type)
            h = torch.relu(h)
            h = self.dropout(h)
        return h


class RGATEncoder(nn.Module):
    def __init__(self, n_drugs, emb_dim, num_rels=None, heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_drugs, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.convs = nn.ModuleList([
            GATConv(emb_dim, emb_dim // heads, heads=heads) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Same logic: handle dense features too
        if x.dtype == torch.long:
            h = self.emb(x)
        else:
            h = x

        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)
        return h