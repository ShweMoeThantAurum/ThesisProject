"""
AEFL Evaluation Visualization
Generates only a figure for MAE Comparison Across Baselines and Datasets 
and a figure for Total Energy Consumption Comparison Across Baselines and Datasets
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.analysis.load_logs import load_experiment

# === Output setup ===
OUT_FIGS = "outputs/figs"
os.makedirs(OUT_FIGS, exist_ok=True)
sns.set_style("whitegrid")

# === Consistent color palette ===
COLORS = {
    "Centralized": "#6C757D",
    "Local-Only": "#999999",
    "FedAvg": "#1E88E5",
    "FedProx": "#00796B",
    "SCAFFOLD": "#AB47BC",
    "Periodic2": "#009688",
    "TopK": "#FBC02D",
    "Q8": "#43A047",
    "AEFL": "#E53935",
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def normalize_method_name(name: str) -> str:
    """Standardize baseline names for consistency."""
    n = name.strip().lower()
    if "aefl" in n:
        return "AEFL"
    if "fedprox" in n:
        return "FedProx"
    if "scaffold" in n:
        return "SCAFFOLD"
    if "fedavg" in n:
        return "FedAvg"
    if "local" in n:
        return "Local-Only"
    if "central" in n:
        return "Centralized"
    if "periodic" in n:
        return "Periodic2"
    if "topk" in n:
        return "TopK"
    if "q8" in n:
        return "Q8"
    return name.title()


def normalize_energy(df):
    """Compute normalized energy relative to FedAvg."""
    df = df.copy()
    norms = []
    for _, row in df.iterrows():
        d = row["dataset"]
        fedavg_e = df.loc[
            (df["dataset"] == d) & (df["method"] == "FedAvg"),
            "total_energy_j",
        ]
        fedavg_e = fedavg_e.values[0] if not fedavg_e.empty else np.nan
        if fedavg_e > 0:
            norms.append(row["total_energy_j"] / fedavg_e)
        else:
            norms.append(1.0)
    df["energy_norm"] = norms
    return df


def plot_comparison(df, metric, ylabel, filename, title):
    """Generic grouped bar plot for MAE or Energy comparison."""
    plt.figure(figsize=(8, 4.5))
    order = [
        "Centralized", "Local-Only", "FedAvg", "FedProx",
        "SCAFFOLD", "Periodic2", "TopK", "Q8", "AEFL"
    ]
    sns.barplot(
        data=df,
        x="dataset", y=metric, hue="method",
        hue_order=[m for m in order if m in df["method"].unique()],
        palette=COLORS, alpha=0.9
    )
    plt.title(title, fontsize=12, weight="bold")
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.legend(ncol=4, fontsize=8, loc="upper right", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, filename), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run():
    print("[make_plots] Generating Test MAE and Energy comparison plots...")
    dirs = [d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))]
    exps = []
    for d in dirs:
        name = d.replace("_", " ").title()
        exps.append(load_experiment(name, d))
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

    # infer dataset and method
    df["dataset"] = df["experiment"].apply(
        lambda x: "SZ" if "sz" in x.lower()
        else "Los" if "los" in x.lower()
        else "PeMS08" if "pems" in x.lower() else "Unknown"
    )
    df["method"] = df["experiment"].apply(
        lambda x: normalize_method_name(re.split(r"[\s\(]", x.strip())[0])
    )

    # clean invalid rows and outliers
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["test_mae", "total_energy_j"])
    df = df[df["test_mae"] < 10]  # remove SCAFFOLD divergence
    df = normalize_energy(df)

    # export cleaned summary
    df.to_csv(os.path.join(OUT_FIGS, "summary_results.csv"), index=False)
    print("Saved summary -> outputs/figs/summary_results.csv")

    # === PLOT 1: Test MAE Comparison ===
    plot_comparison(
        df,
        metric="test_mae",
        ylabel="Test MAE ↓ (Lower is Better)",
        filename="figure_mae_comparison.png",
        title="Model Accuracy Comparison Across Federated Learning Baselines"
    )

    # === PLOT 2: Energy Consumption Comparison ===
    plot_comparison(
        df,
        metric="total_energy_j",
        ylabel="Total Energy Consumption (J) ↓",
        filename="figure_energy_comparison.png",
        title="Energy Consumption Comparison Across Federated Learning Baselines"
    )

    print("Generated:")
    print("  - outputs/figs/figure_mae_comparison.png")
    print("  - outputs/figs/figure_energy_comparison.png")


if __name__ == "__main__":
    run()
