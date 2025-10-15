"""
Loads round logs and final results for multiple experiments.
"""
import os
import re
import pandas as pd

def read_round_csv(path):
    """Reads a round CSV if it exists; returns DataFrame or None."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def read_results_txt(path):
    """Parses TEST MAE/RMSE from results.txt; returns dict with floats or None."""
    if not os.path.exists(path):
        return None
    txt = open(path, "r").read()
    mae_m = re.search(r"TEST MAE:\s*([0-9.]+)", txt)
    rmse_m = re.search(r"TEST RMSE:\s*([0-9.]+)", txt)
    if not mae_m or not rmse_m:
        return None
    return {"test_mae": float(mae_m.group(1)), "test_rmse": float(rmse_m.group(1))}

def load_experiment(name, folder):
    """Loads round CSV and final results for an experiment."""
    base = os.path.join("outputs", folder)
    rounds_df = read_round_csv(os.path.join(base, "round_log.csv"))
    results   = read_results_txt(os.path.join(base, "results.txt"))
    return {"name": name, "folder": folder, "rounds": rounds_df, "final": results}
