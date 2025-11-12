"""
Splits dataset among clients (IID or non-IID) and saves per-client .npy data and metadata.
"""

import os
import json
import numpy as np


def load_prepared_data(proc_dir):
    """Loads preprocessed SZ-Taxi arrays from disk."""
    data = {}
    for split in ["train", "valid", "test"]:
        Xp, yp = os.path.join(proc_dir, f"X_{split}.npy"), os.path.join(proc_dir, f"y_{split}.npy")
        if not os.path.exists(Xp) or not os.path.exists(yp):
            raise FileNotFoundError(f"Missing {split} arrays in {proc_dir}")
        data[f"X_{split}"], data[f"y_{split}"] = np.load(Xp), np.load(yp)
    return data


def split_clients_by_nodes(num_nodes, num_clients, noniid=False, imbalance_factor=0.4, seed=42):
    """Splits node indices across clients with optional imbalance."""
    rng = np.random.default_rng(seed)
    all_nodes = np.arange(num_nodes)
    if not noniid:
        return np.array_split(all_nodes, num_clients)

    shares = rng.dirichlet(np.ones(num_clients))
    sizes = np.maximum(1, (shares * num_nodes).astype(int))
    diff = num_nodes - sizes.sum()
    sizes[-1] += diff
    rng.shuffle(all_nodes)

    splits, start = [], 0
    for s in sizes:
        splits.append(all_nodes[start:start + s])
        start += s
    return splits


def build_client_datasets(proc_dir, num_clients=5, noniid=False, imbalance_factor=0.4,
                          seed=42, save_dir=None):
    """Builds per-client datasets and saves to disk."""
    print(f"Building client datasets | Non-IID={noniid} | imbalance={imbalance_factor}")
    data = load_prepared_data(proc_dir)
    Xtr, ytr = data["X_train"], data["y_train"]
    num_nodes = Xtr.shape[-1]
    splits = split_clients_by_nodes(num_nodes, num_clients, noniid, imbalance_factor, seed)

    out_dir = save_dir or os.path.join(proc_dir, "clients")
    os.makedirs(out_dir, exist_ok=True)

    for i, idxs in enumerate(splits):
        np.save(os.path.join(out_dir, f"client{i}_X.npy"), Xtr[:, :, idxs])
        np.save(os.path.join(out_dir, f"client{i}_y.npy"), ytr[:, idxs])

    meta = {
        "num_clients": num_clients,
        "num_nodes": num_nodes,
        "noniid": noniid,
        "imbalance_factor": imbalance_factor,
        "seed": seed,
        "clients": [len(x) for x in splits],
    }
    with open(os.path.join(proc_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {num_clients} client datasets in {out_dir}")
    print(f"Node allocation per client: {meta['clients']}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Split SZ-Taxi data for FL")
    p.add_argument("--proc_dir", type=str, default="data/processed/sz/prepared")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--noniid", action="store_true")
    p.add_argument("--imbalance", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_client_datasets(args.proc_dir, args.clients, args.noniid, args.imbalance, args.seed)
