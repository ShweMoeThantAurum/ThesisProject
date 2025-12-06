# scripts/privacy_dp_plots.py
"""
Plots for DP Ablation:
1) Sigma vs MAE (accuracy degradation)
2) Sigma vs Total Energy (DP energy cost)
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.s3_helpers import upload_to_s3

CSV_PATH = "privacy_dp_results.csv"
BUCKET = os.environ.get("RESULTS_BUCKET", os.environ.get("S3_BUCKET", "aefl"))

DATASET_LABELS = {
    "sz": "SZ",
    "pems08": "PEMS08",
    "los": "LOS"
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def plot_accuracy(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    for (dataset, color) in zip(df["dataset"].unique(), COLORS):
        sub = df[df["dataset"] == dataset]
        ax.plot(sub["sigma"], sub["MAE"], marker="o", color=color, label=DATASET_LABELS[dataset])

    ax.set_xlabel("DP Noise σ")
    ax.set_ylabel("MAE")
    ax.set_title("Privacy–Accuracy Tradeoff (DP Noise)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    out = Path("outputs/privacy/dp_accuracy.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    upload_to_s3(str(out), BUCKET, "outputs/privacy/dp_accuracy.png")

    print(f"[OK] Saved DP accuracy plot → {out}")


def plot_energy(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    for (dataset, color) in zip(df["dataset"].unique(), COLORS):
        sub = df[df["dataset"] == dataset]
        ax.plot(sub["sigma"], sub["total_energy_j"], marker="o", color=color, label=DATASET_LABELS[dataset])

    ax.set_xlabel("DP Noise σ")
    ax.set_ylabel("Energy (J)")
    ax.set_title("DP Noise vs Energy Consumption")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    out = Path("outputs/privacy/dp_energy.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    upload_to_s3(str(out), BUCKET, "outputs/privacy/dp_energy.png")

    print(f"[OK] Saved DP energy plot → {out}")


def main():
    df = pd.read_csv(CSV_PATH)
    df["dataset"] = df["dataset"].str.lower()

    plot_accuracy(df)
    plot_energy(df)


if __name__ == "__main__":
    main()

