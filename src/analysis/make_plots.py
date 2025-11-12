"""
AEFL Evaluation Visualization
Generates:
    - MAE comparison plot
    - Energy comparison plot
    - summary_results.csv
"""

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3

from src.analysis.load_logs import load_experiment


# ======================================================
# OUTPUT FOLDERS
# ======================================================
OUT_FIGS = "outputs/figs"
os.makedirs(OUT_FIGS, exist_ok=True)

sns.set_style("whitegrid")


# ======================================================
# FINAL 5-BASELINE COLOR PALETTE
# ======================================================
COLORS = {
    "Centralized": "#6C757D",
    "Local-Only": "#999999",
    "FedAvg": "#1E88E5",
    "FedProx": "#00796B",
    "AEFL": "#E53935",
}


# ======================================================
# Helpers
# ======================================================

def normalize_method_name(name: str) -> str:
    n = name.strip().lower()
    if "aefl" in n: return "AEFL"
    if "central" in n: return "Centralized"
    if "local" in n: return "Local-Only"
    if "fedprox" in n: return "FedProx"
    if "fedavg" in n: return "FedAvg"
    return name.title()


def normalize_energy(df):
    df = df.copy()
    norms = []
    for _, row in df.iterrows():
        d = row["dataset"]
        fedavg_e = df.loc[
            (df["dataset"] == d) & (df["method"] == "FedAvg"),
            "total_energy_j",
        ]
        fedavg_e = fedavg_e.values[0] if not fedavg_e.empty else np.nan
        norms.append(row["total_energy_j"] / fedavg_e if fedavg_e > 0 else 1.0)
    df["energy_norm"] = norms
    return df


def plot_comparison(df, metric, ylabel, filename, title):
    plt.figure(figsize=(8, 4.5))
    
    order = [
        "Centralized",
        "Local-Only",
        "FedAvg",
        "FedProx",
        "AEFL",
    ]

    sns.barplot(
        data=df,
        x="dataset",
        y=metric,
        hue="method",
        hue_order=[m for m in order if m in df["method"].unique()],
        palette=COLORS
    )

    plt.title(title, fontsize=12, weight="bold")
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.legend(ncol=3, fontsize=8, loc="upper right", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(OUT_FIGS, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")
    return out_path


# ======================================================
# S3 Upload helper
# ======================================================

def upload_to_s3(local_file, bucket, prefix):
    if not os.path.exists(local_file):
        print(f"[WARN] File not found, skip upload: {local_file}")
        return

    key = prefix + os.path.basename(local_file)

    try:
        s3 = boto3.client("s3")
        s3.upload_file(local_file, bucket, key)
        print(f"[S3] Uploaded → s3://{bucket}/{key}")
    except Exception as e:
        print(f"[S3 ERROR] Could not upload {local_file}: {e}")


# ======================================================
# MAIN
# ======================================================

def run():
    print("[make_plots] Generating Test MAE and Energy comparison plots...")

    dirs = [
        d for d in os.listdir("outputs")
        if os.path.isdir(os.path.join("outputs", d))
    ]

    exps = [load_experiment(d, d) for d in dirs]
    exps = [e for e in exps if e["final"]]

    rows = []
    for e in exps:
        f = e["final"]
        rows.append({
            "experiment": e["name"],
            "test_mae": f.get("test_mae"),
            "test_rmse": f.get("test_rmse"),
            "total_energy_j": f.get("total_energy_j"),
            "total_bytes_mb": f.get("total_bytes_mb"),
            "avg_clients": f.get("avg_clients_per_round"),
        })

    df = pd.DataFrame(rows)

    # Dataset detection
    df["dataset"] = df["experiment"].apply(
        lambda x: "SZ" if "sz" in x.lower()
        else "Los" if "los" in x.lower()
        else "PeMS08" if "pems" in x.lower()
        else "Unknown"
    )

    # Normalize method names
    df["method"] = df["experiment"].apply(
        lambda x: normalize_method_name(re.split(r"[\\s\\(]", x.strip())[0])
    )

    # Filter only the 5 baselines we care about
    df = df[df["method"].isin(["Centralized", "Local-Only", "FedAvg", "FedProx", "AEFL"])]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["test_mae", "total_energy_j"])
    df = df[df["test_mae"] < 10]
    df = normalize_energy(df)

    # Summary CSV
    summary_csv = os.path.join(OUT_FIGS, "summary_results.csv")
    df.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")

    # Plots
    mae_png = plot_comparison(
        df, "test_mae", "Test MAE ↓ (Lower is Better)",
        "figure_mae_comparison.png",
        "Model Accuracy Comparison Across 5 Baselines"
    )

    energy_png = plot_comparison(
        df, "total_energy_j", "Total Energy Consumption (J) ↓",
        "figure_energy_comparison.png",
        "Energy Consumption Comparison Across 5 Baselines"
    )

    # ======================================================
    # S3 upload
    # ======================================================
    bucket = "aefl-results"
    prefix = f"analysis/{time.strftime('%Y%m%d_%H%M%S')}/"

    print("\n[Uploading analysis outputs to S3...]")
    upload_to_s3(summary_csv, bucket, prefix)
    upload_to_s3(mae_png, bucket, prefix)
    upload_to_s3(energy_png, bucket, prefix)

    print(f"\n[S3] All analysis files uploaded to s3://{bucket}/{prefix}\n")
    print("[make_plots] Completed successfully.")


if __name__ == "__main__":
    run()
