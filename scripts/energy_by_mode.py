# scripts/energy_by_mode.py
"""
Aggregate total energy per method from client_energy_summary.log and
generate:
  - energy_summary.csv
  - a bar chart of total energy (J) by method

Assumes log file:
  run_logs/client_energy_summary.log

Each line is JSON with keys:
  - role
  - rounds
  - total_energy_j
  - mode
  - lambda_offload
  - ts  (timestamp, auto-added by logger)

Example usage:
  python -m scripts.energy_by_mode
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LOG_PATH = Path("run_logs") / "client_energy_summary.log"
OUT_DIR = Path("outputs") / "combined"


def load_energy_log():
    """Load all entries from client_energy_summary.log into a DataFrame."""
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing log file: {LOG_PATH}")

    rows = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Skipping bad line: {e}")

    if not rows:
        raise RuntimeError("client_energy_summary.log is empty.")

    df = pd.DataFrame(rows)
    return df


def latest_per_role_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the latest entry per (mode, role) using the 'ts' timestamp.

    This handles multiple runs where the same mode/role appears more than once.
    """
    if "ts" not in df.columns:
        # If ts is somehow missing, just keep the last occurrence per (mode, role)
        df["_row_id"] = range(len(df))
        idx = (
            df.sort_values("_row_id")
            .groupby(["mode", "role"], as_index=False)["_row_id"]
            .last()["_row_id"]
        )
        clean = df.loc[idx].drop(columns=["_row_id"])
    else:
        idx = (
            df.sort_values("ts")
            .groupby(["mode", "role"], as_index=False)["ts"]
            .last()
            .index
        )
        clean = df.loc[idx]

    return clean


def aggregate_by_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total energy across roles per method.

    Returns DataFrame with:
      mode, total_energy_j, mean_energy_per_client, num_clients
    """
    grouped = (
        df.groupby("mode")
        .agg(
            total_energy_j=("total_energy_j", "sum"),
            mean_energy_per_client=("total_energy_j", "mean"),
            num_clients=("role", "nunique"),
        )
        .reset_index()
    )
    return grouped


def save_energy_table(df: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "energy_summary_by_mode.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved energy summary to {csv_path}")


def plot_energy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(df["mode"], df["total_energy_j"])

    ax.set_xlabel("Method")
    ax.set_ylabel("Total Energy (J)")
    ax.set_title("Total Client Energy by Method (Sum over All Roles)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / "energy_by_mode.png"
    plt.savefig(fig_path, dpi=300)
    print(f"[OK] Saved energy plot to {fig_path}")
    plt.close(fig)


def main():
    df_raw = load_energy_log()
    df_latest = latest_per_role_mode(df_raw)
    df_mode = aggregate_by_mode(df_latest)
    save_energy_table(df_mode)
    plot_energy(df_mode)
    print("\n[DONE] Energy aggregation and plot generated.")


if __name__ == "__main__":
    main()
