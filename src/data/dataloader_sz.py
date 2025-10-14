"""
Prepares windowed sequences and simple FL client splits for SZ-Taxi.
"""
import numpy as np
import json
import os

PROC_DIR = "data/processed/sz"

def load_processed():
    """Loads processed adjacency and normalized speed arrays."""
    adj = np.load(f"{PROC_DIR}/adj.npy")
    speed = np.load(f"{PROC_DIR}/speed_norm.npy")
    return adj, speed

def make_windows(data, lookback, horizon, stride):
    """Builds X,y sliding windows from [T, N] -> X[T', lookback, N], y[T', N]."""
    T, N = data.shape
    X, y = [], []
    for t in range(lookback, T - horizon + 1, stride):
        X.append(data[t - lookback:t])
        y.append(data[t + horizon - 1])
    X = np.stack(X).astype(np.float32)
    y = np.stack(y).astype(np.float32)
    return X, y

def split_train_valid_test(T, train_ratio, valid_ratio):
    """Computes index splits over time dimension."""
    train_end = int(T * train_ratio)
    valid_end = int(T * (train_ratio + valid_ratio))
    return slice(0, train_end), slice(train_end, valid_end), slice(valid_end, T)

def split_clients_by_nodes(X, y, num_clients):
    """Splits node dimension across clients for simple FL."""
    num_nodes = X.shape[-1]
    parts = np.array_split(np.arange(num_nodes), num_clients)
    client_sets = []
    for idx in parts:
        client_sets.append((X[..., idx], y[..., idx]))
    return client_sets

def save_numpy(path, name, arr):
    """Saves numpy array to disk."""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, f"{name}.npy"), arr)

def run():
    """Runs windowing, splits, and client partitioning for SZ."""
    adj, speed = load_processed()
    lookback = 12      # example: past 12 steps
    horizon = 1        # predict next step
    stride = 1
    num_clients = 5
    train_ratio, valid_ratio = 0.6, 0.2

    X, y = make_windows(speed, lookback, horizon, stride)
    tr, va, te = split_train_valid_test(X.shape[0], train_ratio, valid_ratio)

    out_root = os.path.join(PROC_DIR, "prepared")
    # Full sets
    save_numpy(out_root, "X_train", X[tr])
    save_numpy(out_root, "y_train", y[tr])
    save_numpy(out_root, "X_valid", X[va])
    save_numpy(out_root, "y_valid", y[va])
    save_numpy(out_root, "X_test", X[te])
    save_numpy(out_root, "y_test", y[te])
    np.save(os.path.join(out_root, "adj.npy"), adj.astype(np.float32))

    # Client partitions from training set
    clients = split_clients_by_nodes(X[tr], y[tr], num_clients)
    client_dir = os.path.join(out_root, "clients")
    os.makedirs(client_dir, exist_ok=True)
    meta = {"lookback": lookback, "horizon": horizon, "num_clients": num_clients}
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f)

    for i, (cX, cY) in enumerate(clients):
        save_numpy(client_dir, f"client{i}_X", cX)
        save_numpy(client_dir, f"client{i}_y", cY)

    print("Prepared:")
    print("  X:", X.shape, "y:", y.shape)
    print("  Train/Valid/Test:", X[tr].shape, X[va].shape, X[te].shape)
    print("  Clients:", num_clients, "example client0:", clients[0][0].shape, clients[0][1].shape)

if __name__ == "__main__":
    run()
