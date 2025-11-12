"""
Preprocesses Los-Loop CSVs into the same federated data format.
"""

import os
import numpy as np
import pandas as pd
from data.dataloader import build_client_datasets

def preprocess_and_split(raw_dir="data/raw/los",
                         out_dir="data/processed/los/prepared",
                         num_clients=5,
                         noniid=False,
                         imbalance=0.4,
                         seed=42):
    """Processes los_speed.csv + los_adj.csv into numpy arrays."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Preprocessing Los-Loop data into {out_dir}...")

    # Load data
    speed = pd.read_csv(os.path.join(raw_dir, "los_speed.csv"), header=None).values
    adj = pd.read_csv(os.path.join(raw_dir, "los_adj.csv"), header=None).values

    # Normalize speeds
    mean, std = np.mean(speed), np.std(speed)
    speed = (speed - mean) / std

    # Sliding window: 12 input → 1 output
    seq_len, pred_len = 12, 1
    X, y = [], []
    for i in range(len(speed) - seq_len - pred_len):
        X.append(speed[i:i + seq_len])
        y.append(speed[i + seq_len])
    X, y = np.array(X), np.array(y)

    # Train/valid/test split (70/15/15)
    n = len(X)
    n_train, n_val = int(0.7 * n), int(0.85 * n)
    np.save(os.path.join(out_dir, "X_train.npy"), X[:n_train])
    np.save(os.path.join(out_dir, "y_train.npy"), y[:n_train])
    np.save(os.path.join(out_dir, "X_valid.npy"), X[n_train:n_val])
    np.save(os.path.join(out_dir, "y_valid.npy"), y[n_train:n_val])
    np.save(os.path.join(out_dir, "X_test.npy"), X[n_val:])
    np.save(os.path.join(out_dir, "y_test.npy"), y[n_val:])
    np.save(os.path.join(out_dir, "adj.npy"), adj)

    print(f"Train={n_train}, Val={n_val-n_train}, Test={n-n_val}, Nodes={X.shape[-1]}")
    build_client_datasets(out_dir, num_clients, noniid, imbalance, seed)
    print("Completed preprocessing of Los-Loop dataset.")

if __name__ == "__main__":
    preprocess_and_split()
