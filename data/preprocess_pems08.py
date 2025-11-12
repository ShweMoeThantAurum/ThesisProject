"""
Preprocesses PeMSD8 .npz file into standardized arrays and client splits.
"""

import os
import numpy as np
from data.dataloader import build_client_datasets

def preprocess_and_split(raw_dir="data/raw/pems08",
                         out_dir="data/processed/pems08/prepared",
                         num_clients=5,
                         noniid=False,
                         imbalance=0.4,
                         seed=42):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Preprocessing PeMSD8 data into {out_dir}...")

    # Load and auto-handle shape
    data = np.load(os.path.join(raw_dir, "pems08.npz"))
    key = "data" if "data" in data.files else list(data.files)[0]
    arr = data[key]
    print("Original shape:", arr.shape)

    # Typical shapes:
    # (N, T, C) or (T, N, C) or (T, N)
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[1]:  # likely (N, T, C)
            arr = arr.transpose(1, 0, 2)
        else:  # (T, N, C)
            pass
        # take first channel if multi-dimensional
        arr = arr[..., 0]
    elif arr.ndim == 2:
        pass  # already [T, N]
    else:
        raise ValueError(f"Unexpected array shape {arr.shape}")

    print("After reshape:", arr.shape)  # should be [T, N]

    # Normalization
    mean, std = np.mean(arr), np.std(arr)
    arr = (arr - mean) / std

    # Sliding window (12 input → 1 output)
    seq_len, pred_len = 12, 1
    X, y = [], []
    for i in range(len(arr) - seq_len - pred_len):
        X.append(arr[i:i + seq_len])
        y.append(arr[i + seq_len])
    X, y = np.array(X), np.array(y)
    print("Windowed:", X.shape, y.shape)

    # Train/valid/test split (70/15/15)
    n = len(X)
    n_train, n_val = int(0.7 * n), int(0.85 * n)
    np.save(os.path.join(out_dir, "X_train.npy"), X[:n_train])
    np.save(os.path.join(out_dir, "y_train.npy"), y[:n_train])
    np.save(os.path.join(out_dir, "X_valid.npy"), X[n_train:n_val])
    np.save(os.path.join(out_dir, "y_valid.npy"), y[n_train:n_val])
    np.save(os.path.join(out_dir, "X_test.npy"), X[n_val:])
    np.save(os.path.join(out_dir, "y_test.npy"), y[n_val:])

    print(f"Train={n_train}, Val={n_val-n_train}, Test={n-n_val}, Nodes={X.shape[-1]}")

    # Federated client partitioning
    build_client_datasets(out_dir, num_clients, noniid, imbalance, seed)
    print("Completed preprocessing of PeMSD8 dataset.")

if __name__ == "__main__":
    preprocess_and_split()
