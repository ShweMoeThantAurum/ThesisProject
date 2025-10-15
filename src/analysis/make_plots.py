"""
Creates plots and a summary table across experiments.
Now includes AEFL round CSV and % deltas vs. FedAvg and Centralized.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis.load_logs import load_experiment

OUT_FIGS = "outputs/figs"
os.makedirs(OUT_FIGS, exist_ok=True)

def line_plot_rounds(exps, metric_col, ylabel, fname):
    """Plots a line per experiment for a given round metric, skipping missing CSVs."""
    plt.figure()
    has_any = False
    xlabel_text = "Round"
    for exp in exps:
        df = exp["rounds"]
        if df is None or metric_col not in df.columns:
            continue
        xcol = "round" if "round" in df.columns else ("epoch" if "epoch" in df.columns else None)
        if xcol is None:
            continue
        plt.plot(df[xcol].values, df[metric_col].astype(float).values, label=exp["name"])
        xlabel_text = "Round" if xcol == "round" else "Epoch"
        has_any = True
    if not has_any:
        plt.close()
        return None
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel)
    plt.legend()
    out_path = os.path.join(OUT_FIGS, fname)
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    return out_path

def bar_plot_final(exps, key, ylabel, fname):
    """Bar plot using final results (test metrics)."""
    names, vals = [], []
    for exp in exps:
        fin = exp["final"]
        if fin and key in fin:
            names.append(exp["name"])
            vals.append(fin[key])
    if not names:
        return None
    idx = np.arange(len(names))
    plt.figure()
    plt.bar(idx, vals)
    plt.xticks(idx, names, rotation=15)
    plt.ylabel(ylabel)
    out_path = os.path.join(OUT_FIGS, fname)
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    return out_path

def make_summary_table(exps, out_csv):
    """Builds a summary CSV with test metrics and averages of round metrics."""
    rows = []
    for exp in exps:
        row = {"experiment": exp["name"]}
        fin = exp["final"] or {}
        row["test_mae"] = fin.get("test_mae", np.nan)
        row["test_rmse"] = fin.get("test_rmse", np.nan)
        df = exp["rounds"]
        if df is not None:
            for col in ["val_mae", "val_rmse", "energy_proxy", "avg_comm_ratio", "round_secs", "cpu_percent", "mem_mb"]:
                row[f"avg_{col}"] = df[col].astype(float).mean() if col in df.columns else np.nan
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out_csv, index=False)
    return tab

def add_percent_deltas(tab, out_csv):
    """Adds % deltas vs FedAvg and Centralized baselines."""
    t = tab.copy()
    def pick_row(name):
        m = t["experiment"] == name
        return t.loc[m].iloc[0] if m.any() else None
    cen = pick_row("Centralized")
    fed = pick_row("FedAvg")
    for i in range(len(t)):
        # Accuracy deltas (lower is better)
        if pd.notna(t.loc[i, "test_mae"]) and fed is not None:
            t.loc[i, "%_MAE_vs_FedAvg"] = 100.0 * (t.loc[i, "test_mae"] - fed["test_mae"]) / fed["test_mae"]
        if pd.notna(t.loc[i, "test_mae"]) and cen is not None:
            t.loc[i, "%_MAE_vs_Centralized"] = 100.0 * (t.loc[i, "test_mae"] - cen["test_mae"]) / cen["test_mae"]
        # Energy deltas (lower is better) – use avg_energy_proxy if available
        if "avg_energy_proxy" in t.columns and pd.notna(t.loc[i, "avg_energy_proxy"]) and fed is not None and pd.notna(fed.get("avg_energy_proxy", np.nan)):
            t.loc[i, "%_Energy_vs_FedAvg"] = 100.0 * (t.loc[i, "avg_energy_proxy"] - fed["avg_energy_proxy"]) / fed["avg_energy_proxy"]
    t.to_csv(out_csv, index=False)
    return t

def run():
    """Loads experiments, makes plots, saves summary and deltas CSV."""
    exps = [
        load_experiment("Centralized",       "centralized_sz"),
        load_experiment("FedAvg",            "fedavg_sz"),
        load_experiment("AEFL Framework",    "aefl_framework_sz"),
    ]

    mae_round = line_plot_rounds(exps, "val_mae",  "Validation MAE",  "rounds_val_mae.png")
    rmse_round= line_plot_rounds(exps, "val_rmse", "Validation RMSE", "rounds_val_rmse.png")
    eproxy    = line_plot_rounds(exps, "energy_proxy", "Energy Proxy", "rounds_energy_proxy.png")
    comm      = line_plot_rounds(exps, "avg_comm_ratio", "Comm Ratio", "rounds_comm_ratio.png")

    bar_mae = bar_plot_final(exps, "test_mae",  "Test MAE",  "test_mae_bar.png")
    bar_rmse= bar_plot_final(exps, "test_rmse", "Test RMSE", "test_rmse_bar.png")

    summary_csv = os.path.join(OUT_FIGS, "summary.csv")
    tab = make_summary_table(exps, summary_csv)

    deltas_csv = os.path.join(OUT_FIGS, "summary_with_deltas.csv")
    tab2 = add_percent_deltas(tab, deltas_csv)

    print("Saved figures:")
    for p in [mae_round, rmse_round, eproxy, comm, bar_mae, bar_rmse]:
        if p:
            print(" -", p)
    print("Saved summary:", summary_csv)
    print("Saved deltas:", deltas_csv)
    print(tab2.fillna("-").to_string(index=False))

if __name__ == "__main__":
    run()
