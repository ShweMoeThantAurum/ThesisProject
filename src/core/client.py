"""
Handles local client training and dataset loading during FL rounds.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


def client_indices(num_nodes, num_clients, i):
    """Splits node indices among clients deterministically."""
    parts = np.array_split(np.arange(num_nodes), num_clients)
    return parts[i]


class ClientPaddedDataset(Dataset):
    """Loads a client's subset and pads missing nodes."""
    def __init__(self, X_path, y_path, idxs, num_nodes):
        cX, cY = np.load(X_path), np.load(y_path)
        S, L, k = cX.shape
        X = np.zeros((S, L, num_nodes), np.float32)
        Y = np.zeros((S, num_nodes), np.float32)
        X[:, :, idxs] = cX
        Y[:, idxs] = cY
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def make_loader_for_client(proc_dir, cid, idxs, num_nodes, batch=64):
    """Creates a DataLoader for one client's local subset."""
    Xp = f"{proc_dir}/clients/client{cid}_X.npy"
    yp = f"{proc_dir}/clients/client{cid}_y.npy"
    ds = ClientPaddedDataset(Xp, yp, idxs, num_nodes)
    return DataLoader(ds, batch_size=batch, shuffle=True)


def train_local(model, loader, device, epochs=1, lr=1e-3, adj=None):
    """Trains model locally and returns (state_dict, num_samples)."""
    model.train()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_samples = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            yhat = model(x, adj) if adj is not None else model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            n_samples += x.size(0)

    return {k: v.detach().clone() for k, v in model.state_dict().items()}, n_samples
