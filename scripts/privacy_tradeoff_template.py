# scripts/privacy_tradeoff_template.py
"""
Plot privacy / DP / compression trade-offs for AEFL variants.

Assumes you have a CSV like:
  outputs/sz/privacy_ablation.csv

with columns:
  variant, sigma, MAE, RMSE, total_energy_j, total_comm_mb

Example usage:
  python -m scripts.privacy_tradeoff_template \
      --csv outputs/sz/privacy_ablation.csv \
      --outdir outputs/sz/privacy_plots
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_variant_bars(df: pd.DataFrame, outdir: Path):
    """Bar plots: MAE and total_energy_j by variant."""
    # Ensure variant is categorical in original order
    df["variant"] = pd.Categorical(df["variant"], categories=df["variant"].tolist(), ordered=True)

    # MAE bar
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["variant"], df["MAE"])
    ax.set_xlabel("Variant")
    ax.set_ylabel("MAE")
    ax.set_title("Accuracy vs AEFL Variant (Privacy / Compression)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig_path = outdir / "mae_by_variant.png"
    plt.savefig(fig_path, dpi=300)
    print(f"[OK] Saved MAE-by-variant plot to {fig_path}")
    plt.close(fig)

    # Energy bar
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["variant"], df["total_energy_j"])
    ax.set_xlabel("Variant")
    ax.set_ylabel("Total Energy (J)")
    ax.set_title("Energy vs AEFL Variant (Privacy / Compression)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig_path = outdir / "energy_by_variant.png"
    plt.savefig(fig_path, dpi=300)
    print(f"[OK] Saved energy-by-variant plot to {fig_path}")
    plt.close(fig)


def plot_sigma_curve(df: pd.DataFrame, outdir: Path):
    """
    If there are multiple rows with different sigma (DP noise),
    plot MAE vs sigma as a privacy–utility curve.
    """
    if "sigma" not in df.columns:
        print("[INFO] No 'sigma' column found; skipping sigma curve plot.")
        return

    # Only keep rows with numeric sigma
    df_sigma = df.dropna(subset=["sigma"]).copy()
    if df_sigma.empty:
        print("[INFO] No sigma values to plot; skipping sigma curve.")
        return

    # Sort by sigma
    df_sigma = df_sigma.sort_values("sigma")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df_sigma["sigma"], df_sigma["MAE"], marker="o")
    ax.set_xlabel("DP Noise Level σ")
    ax.set_ylabel("MAE")
    ax.set_title("Privacy–Utility Trade-off (MAE vs σ)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    fig_path = outdir / "mae_vs_sigma.png"
    plt.savefig(fig_path, dpi=300)
    print(f"[OK] Saved MAE vs sigma plot to {fig_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot privacy / DP trade-offs for AEFL variants.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to privacy ablation CSV (variant,sigma,MAE,RMSE,total_energy_j,total_comm_mb).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save plots (default: same directory as CSV).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    outdir = Path(args.outdir) if args.outdir else csv_path.parent / "privacy_plots"

    plot_variant_bars(df, outdir)
    plot_sigma_curve(df, outdir)

    print("\n[DONE] Privacy / DP trade-off plots generated.")


if __name__ == "__main__":
    main()
