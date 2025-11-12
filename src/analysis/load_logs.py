"""
Loads experiment logs and final results for Centralized, FedAvg, and AEFL experiments
across multiple datasets (SZ-Taxi, Los-Loop, PeMSD8).
"""

import os
import pandas as pd


def read_results(path):
    """Parses key-value metrics from results.txt into a dictionary."""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            key = k.strip().lower().replace(" ", "_")
            try:
                out[key] = float(v.strip())
            except ValueError:
                pass
    return out


def read_rounds(path):
    """Reads per-round or per-epoch CSV logs."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_experiment(name, folder):
    """
    Loads both round_log.csv and results.txt for a single experiment.
    Returns dict containing:
        - name   : experiment display name
        - rounds : per-round dataframe
        - final  : final test metrics dict
    """
    base = os.path.join("outputs", folder)
    return {
        "name": name,
        "rounds": read_rounds(os.path.join(base, "round_log.csv")),
        "final": read_results(os.path.join(base, "results.txt")),
    }
