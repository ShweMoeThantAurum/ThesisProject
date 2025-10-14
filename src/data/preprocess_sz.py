"""
Preprocesses SZ-Taxi: loads CSVs, cleans, normalizes, and saves arrays.
"""
import os, json
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_DIR = "data/processed/sz"
os.makedirs(OUT_DIR, exist_ok=True)

def load_raw():
    """Loads raw SZ adjacency and speed CSVs."""
    adj = pd.read_csv(f"{DATA_DIR}/sz_adj.csv", header=None)
    speed = pd.read_csv(f"{DATA_DIR}/sz_speed.csv")
    return adj, speed

def clean_speed(df):
    """Cleans speed data with numeric casting and forward-fill."""
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.ffill().bfill().fillna(0.0)
    return df

def normalize_speed(df):
    """Normalizes per-sensor with z-score and returns stats."""
    vals = df.values.astype(np.float32)
    mean = vals.mean(axis=0, keepdims=True)
    std = vals.std(axis=0, keepdims=True) + 1e-6
    norm = (vals - mean) / std
    stats = {"mean": mean.flatten().tolist(), "std": std.flatten().tolist()}
    return norm, stats

def save_processed(adj, norm, stats):
    """Saves processed arrays and stats."""
    np.save(f"{OUT_DIR}/adj.npy", adj.values.astype(np.float32))
    np.save(f"{OUT_DIR}/speed_norm.npy", norm.astype(np.float32))
    with open(f"{OUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f)

def preview_shapes(adj, norm):
    """Prints basic shape info for quick verification."""
    print("Adjacency:", adj.shape)
    print("Speed normalized:", norm.shape)

def run():
    """Runs the preprocessing pipeline."""
    adj, speed = load_raw()
    speed = clean_speed(speed)
    norm, stats = normalize_speed(speed)
    save_processed(adj, norm, stats)
    preview_shapes(adj, norm)

if __name__ == "__main__":
    run()
