# scripts/energy_all_datasets.py
"""
Generate a high-quality thesis-ready energy comparison plot
across all methods (AEFL, FedAvg, FedProx) and datasets (SZ, PeMS08, LOS).

Metric visualised: Total Energy (J)
Saves figure locally + uploads to:
    s3://<BUCKET>/outputs/energy/energy_all_datasets.png
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Ensure project root in PYTHONPATH
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.s3_helpers import upload_to_s3

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
INPUT_CSV = "energy_all_datasets.csv"
BUCKET = os.environ.get("RESULTS_BUCKET", os.environ.get("S3_BUCKET", "aefl"))

MODE_LABELS = {
    "aefl": "AEFL (Ours)",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
}

DATASET_LABELS = {
    "sz": "SZ",
    "pems08": "PEMS08",
    "los": "LOS"
}

BAR_COLORS = {
    "aefl": "#1f77b4",     # blue
    "fedavg": "#ff7f0e",   # orange
    "fedprox": "#2ca02c",  # green
}


# -----------------------------------------------------------------------------
# Main plot function
# -----------------------------------------------------------------------------
def main():
    df = pd.read_csv(INPUT_CSV)
    df["mode"] = df["mode"].str.lower()
    df["dataset"] = df["dataset"].str.lower()

    # Pivot table: dataset rows, mode columns
    pivot = df.pivot(index="dataset", columns="mode", values="total_energy_j")
    pivot = pivot[["aefl", "fedavg", "fedprox"]]  # consistent ordering

    datasets = [DATASET_LABELS[d] for d in pivot.index]
    values = pivot.values

    x = np.arange(len(datasets))
    width = 0.25

    # Figure setup
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title(
        "Energy Consumption Across Methods and Datasets",
        fontsize=20, fontweight="bold"
    )

    # Bars
    for i, mode in enumerate(["aefl", "fedavg", "fedprox"]):
        ax.bar(
            x + (i - 1) * width,
            values[:, i],
            width,
            label=MODE_LABELS[mode],
            color=BAR_COLORS[mode],
            edgecolor="black",
            linewidth=0.4,
        )

    # Axes formatting
    ax.set_ylabel("Energy (J)", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=13)

    fig.tight_layout()

    # Save locally
    out_path = Path("outputs/energy/energy_all_datasets.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    print(f"[OK] Saved energy plot → {out_path}")

    # Upload to S3
    s3_key = "outputs/energy/energy_all_datasets.png"
    upload_to_s3(str(out_path), BUCKET, s3_key)
    print(f"[OK] Uploaded to S3 → s3://{BUCKET}/{s3_key}")


if __name__ == "__main__":
    main()

