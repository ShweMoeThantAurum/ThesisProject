"""
Evaluates models, logs metrics, and computes optional energy-efficiency scores.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.utils.metrics import eval_loader_metrics


def make_eval_loader(X_np, y_np, batch=128):
    """Creates a DataLoader from numpy arrays."""
    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).float()
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=False)


def init_csv(path, header):
    """Initializes a CSV log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def log_row(path, row):
    """Appends one row to a CSV log."""
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def eval_model(model, loader, device, adj=None):
    """Evaluates metrics on a loader."""
    if adj is None:
        return eval_loader_metrics(model, loader, device)
    class Wrapper(torch.nn.Module):
        def __init__(self, m, A): super().__init__(); self.m, self.A = m, A
        def forward(self, x): return self.m(x, self.A)
    return eval_loader_metrics(Wrapper(model, adj), loader, device)


def ees_from_table(test_mae_vec, energy_vec):
    """Computes Energy Efficiency Score from arrays (lower is better)."""
    best_mae = np.min(test_mae_vec)
    energy_norm = energy_vec / (energy_vec[0] + 1e-8) if energy_vec.size else energy_vec
    return 0.5 * (test_mae_vec / (best_mae + 1e-8)) + 0.5 * energy_norm
