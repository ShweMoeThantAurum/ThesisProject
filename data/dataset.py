"""
Defines PyTorch Dataset class to load pre-windowed arrays (X_train.npy, etc.) for model training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class WindowedArray(Dataset):
    """Loads pre-windowed SZ-Taxi arrays and exposes them as tensors."""
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return x, y
