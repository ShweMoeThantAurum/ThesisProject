import os
import json
import matplotlib.pyplot as plt

DATASETS = ["sz", "los", "pems08"]
MODES = ["aefl", "fedavg", "fedprox", "localonly"]
BASE_DIR = "outputs/cloud_summary"
OUT_DIR = "outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ["MAE", "RMSE", "MAPE", "sMAPE", "R2"]

def load_metrics(dataset, mode):
    path = f"{BASE_DIR}/{dataset}/{mode}/final_metrics_{mode}.json"
    if not os.path.exists(path):
        print(f"[WARN] Missing metrics: {path}")
        return None
    with open(path) as f:
        return json.load(f)

def plot_final_metrics(dataset):
    results = {}
    for mode in MODES:
        metrics = load_metrics(dataset, mode)
        if metrics:
            results[mode] = metrics

    if not results:
        print(f"[SKIP] No metrics for {dataset}")
        return

    for metric in METRICS:
        plt.figure(figsize=(8,5))
        values = [results[m][metric] for m in results]
        labels = [m.upper() for m in results]

        plt.bar(labels, values)
        plt.title(f"{metric} Comparison — {dataset.upper()}")
        plt.ylabel(metric)
        plt.grid(axis="y", linestyle="--", alpha=0.4)

        out = f"{OUT_DIR}/{dataset}_{metric}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[OK] Saved {out}")

for ds in DATASETS:
    plot_final_metrics(ds)
