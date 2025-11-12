"""
A compact GRU-based forecaster for multivariate time-series.
Used in Centralized, FedAvg, and AEFL (GRU backbone) experiments.
"""

import torch
import torch.nn as nn


class SimpleGRU(nn.Module):
    """Minimal GRU forecaster for multivariate sequences."""
    def __init__(self, num_nodes, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.gru = nn.GRU(
            input_size=num_nodes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, num_nodes)
        for name, p in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)

    def forward(self, x):
        """Forward pass: [B, L, N] → [B, N]."""
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)
