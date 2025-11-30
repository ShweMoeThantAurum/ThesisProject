"""Generate per-round cloud summary CSV + plots and upload results to S3."""

import os
import json
import boto3
import pandas as pd
import matplotlib.pyplot as plt

from src.fl.logger import LOG_DIR

RESULTS_BUCKET = os.environ.get("RESULTS_BUCKET", "aefl")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
_s3 = boto3.client("s3", region_name=AWS_REGION)


# ------------------------------------------------------------
# Load log entries
# ------------------------------------------------------------
def _load_log_file(name: str):
    """Load JSONL event log entries from run_logs."""
    path = os.path.join(LOG_DIR, name)
    if not os.path.exists(path):
        return []

    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    return entries


# ------------------------------------------------------------
# Build summary DataFrames
# ------------------------------------------------------------
def _build_round_dataframe(final_metrics: dict, num_rounds: int):
    """Build latency summary per round."""
    uploads = _load_log_file("server_s3_upload.log")
    downloads = _load_log_file("server_s3_download.log")

    rows = []
    for r in range(1, num_rounds + 1):
        up = [u for u in uploads if u.get("round") == r]
        dn = [d for d in downloads if d.get("round") == r]

        mean_up_lat = sum(u.get("latency_sec", 0.0) for u in up) / max(len(up), 1)
        mean_dn_lat = sum(d.get("latency_sec", 0.0) for d in dn) / max(len(dn), 1)
        mean_dn_mb = (
            sum(d.get("size_bytes", 0.0) for d in dn) / max(len(dn), 1)
        ) / (1024 * 1024)

        rows.append({
            "round": r,
            "upload_latency_sec": mean_up_lat,
            "download_latency_sec": mean_dn_lat,
            "download_mb": mean_dn_mb,
        })

    df = pd.DataFrame(rows)
    df.attrs["final_metrics"] = final_metrics
    return df


def _build_client_energy_dataframe():
    """Build DataFrame of per-role energy consumption."""
    entries = _load_log_file("client_energy.log")
    if not entries:
        return pd.DataFrame([])

    rows = []
    for e in entries:
        rows.append({
            "round": e.get("round"),
            "role": e.get("role"),
            "compute_j": e.get("compute_j"),
            "comm_j": e.get("comm_j"),
            "total_j": e.get("total_j"),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Plotting utilities
# ------------------------------------------------------------
def _make_latency_plot(df: pd.DataFrame, out_path: str, mode: str):
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["upload_latency_sec"], label="Upload")
    plt.plot(df["round"], df["download_latency_sec"], label="Download")
    plt.xlabel("Round")
    plt.ylabel("Latency (s)")
    plt.title(f"Latency per Round ({mode})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _make_energy_plot(energy_df: pd.DataFrame, out_path: str, mode: str):
    if energy_df.empty:
        return

    grouped = energy_df.groupby("round")["total_j"].sum().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["round"], grouped["total_j"])
    plt.xlabel("Round")
    plt.ylabel("Total Energy (J)")
    plt.title(f"Client Energy per Round ({mode})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------------
# S3 upload helper (NEW STRUCTURE)
# ------------------------------------------------------------
def _upload_to_s3_dataset(local_path: str, dataset: str, mode: str):
    """Upload artifact using new S3 prefix structure."""
    if not os.path.exists(local_path):
        return

    key = f"experiments/{dataset}/{mode.lower()}/{os.path.basename(local_path)}"

    try:
        _s3.upload_file(local_path, RESULTS_BUCKET, key)
        print(f"[SERVER] Uploaded to s3://{RESULTS_BUCKET}/{key}")
    except Exception as e:
        print(f"[SERVER] WARNING: Failed upload {local_path}: {e}")


# ------------------------------------------------------------
# Main summary generator (UPDATED)
# ------------------------------------------------------------
def generate_cloud_summary(final_metrics: dict, num_rounds: int, mode: str):
    """
    Save summary CSV + plots and upload results to S3.

    Local output dir:
        outputs/cloud_summary/<dataset>/<mode>/

    S3 output dir:
        experiments/<dataset>/<mode>/
    """

    mode_lower = mode.lower()
    dataset = os.environ.get("DATASET", "unknown").lower()

    # Updated output directory structure
    out_dir = os.path.join("outputs", "cloud_summary", dataset, mode_lower)
    os.makedirs(out_dir, exist_ok=True)

    # Build CSV
    df = _build_round_dataframe(final_metrics, num_rounds)
    csv_path = os.path.join(out_dir, f"summary_{mode_lower}.csv")
    df.to_csv(csv_path, index=False)

    # Latency plot
    latency_plot = os.path.join(out_dir, f"latency_per_round_{mode_lower}.png")
    _make_latency_plot(df, latency_plot, mode)

    # Energy plot
    energy_df = _build_client_energy_dataframe()
    energy_plot = os.path.join(out_dir, f"client_energy_per_round_{mode_lower}.png")
    _make_energy_plot(energy_df, energy_plot, mode)

    # Final metrics JSON
    metrics_path = os.path.join(out_dir, f"final_metrics_{mode_lower}.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"[SERVER] Summary saved | dataset={dataset} | mode={mode}")
    print(f"  {csv_path}")
    print(f"  {metrics_path}")
    print(f"  {latency_plot}")
    if not energy_df.empty:
        print(f"  {energy_plot}")

    # Upload all to S3 using updated structure
    for path in [csv_path, metrics_path, latency_plot]:
        _upload_to_s3_dataset(path, dataset, mode)

    if not energy_df.empty:
        _upload_to_s3_dataset(energy_plot, dataset, mode)

    return df
