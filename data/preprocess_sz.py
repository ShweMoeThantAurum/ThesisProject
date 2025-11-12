"""
Preprocesses SZ-Taxi CSVs into normalized, windowed arrays and builds client partitions.
"""

import os
import numpy as np
from data.dataloader import build_client_datasets


def preprocess_and_split(raw_dir="data/raw/sz",
                         out_dir="data/processed/sz/prepared",
                         num_clients=5,
                         noniid=False,
                         imbalance=0.4,
                         seed=42):
    """Converts raw SZ-Taxi data into prepared arrays and per-client splits."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Preprocessing SZ-Taxi data into {out_dir}...")

    # build client partitions for federated simulation
    build_client_datasets(
        proc_dir=out_dir,
        num_clients=num_clients,
        noniid=noniid,
        imbalance_factor=imbalance,
        seed=seed,
    )

    print(f"Completed preprocessing with {'Non-IID' if noniid else 'IID'} client partitioning.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Preprocess SZ-Taxi dataset")
    p.add_argument("--raw_dir", type=str, default="data/raw/sz")
    p.add_argument("--out_dir", type=str, default="data/processed/sz/prepared")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--noniid", action="store_true")
    p.add_argument("--imbalance", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    preprocess_and_split(args.raw_dir, args.out_dir, args.clients,
                         args.noniid, args.imbalance, args.seed)
