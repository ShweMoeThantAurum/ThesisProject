"""
Generate server-side CSV summaries and essential metrics only.
 - summary_<mode>.csv
 - final_metrics_<mode>.json

Everything is stored under: outputs/<dataset>/<mode>/
"""

import os
import json
import boto3
import pandas as pd

from src.fl.logger import LOG_DIR

RESULTS_BUCKET = os.environ.get("RESULTS_BUCKET", "aefl")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
_s3 = boto3.client("s3", region_name=AWS_REGION)


def _load_log_file(name):
    """Load JSONL events from run_logs."""
    path = os.path.join(LOG_DIR, name)
    if not os.path.exists(path):
        return []

    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


def _build_round_dataframe(final_metrics, num_rounds):
    """Build per-round latency + MB summary DataFrame."""
    uploads = _load_log_file("server_s3_upload.log")
    downloads = _load_log_file("server_s3_download.log")

    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = os.environ.get("FL_MODE", "AEFL").strip().lower()
    variant = os.environ.get("VARIANT_ID", "").strip()

    rows = []
    for r in range(1, num_rounds + 1):
        up = [u for u in uploads if u.get("round") == r]
        dn = [d for d in downloads if d.get("round") == r]

        mean_up = sum(u.get("latency_sec", 0.0) for u in up) / max(len(up), 1)
        mean_dn = sum(d.get("latency_sec", 0.0) for d in dn) / max(len(dn), 1)
        mean_mb = (
            sum(d.get("size_bytes", 0.0) for d in dn) / max(len(dn), 1)
        ) / (1024 * 1024)

        rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "variant": variant,
                "round": r,
                "upload_latency_sec": mean_up,
                "download_latency_sec": mean_dn,
                "download_mb": mean_mb,
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["final_metrics"] = final_metrics
    return df


def _upload_to_s3_dataset(local_path, dataset, mode):
    """Upload summary artifacts to S3."""
    if not os.path.exists(local_path):
        return

    key = f"experiments/{dataset}/{mode.lower()}/{os.path.basename(local_path)}"

    try:
        _s3.upload_file(local_path, RESULTS_BUCKET, key)
        print(f"[SERVER] Uploaded to s3://{RESULTS_BUCKET}/{key}")
    except Exception as e:
        print(f"[SERVER] WARNING: Failed upload {local_path}: {e}")


def generate_cloud_summary(final_metrics, num_rounds, mode):
    """
    Save ONLY the required artifacts:
      - summary_<mode>.csv
      - final_metrics_<mode>.json

    Additionally, if VARIANT_ID is set, create variant-specific
    filenames so multiple runs (e.g. different DP σ) do not overwrite
    each other:

      - summary_<mode>_<variant>.csv
      - final_metrics_<mode>_<variant>.json
    """
    mode_lower = mode.lower()
    dataset = os.environ.get("DATASET", "unknown").lower()
    variant = os.environ.get("VARIANT_ID", "").strip()
    variant_suffix = f"_{variant}" if variant else ""

    out_dir = os.path.join("outputs", dataset, mode_lower)
    os.makedirs(out_dir, exist_ok=True)

    # Build extended metrics with configuration metadata
    dp_enabled = os.environ.get("DP_ENABLED", "false").lower() == "true"
    dp_sigma = float(os.environ.get("DP_SIGMA", "0.0"))
    compression_enabled = (
        os.environ.get("COMPRESSION_ENABLED", "false").lower() == "true"
    )
    compression_mode = os.environ.get("COMPRESSION_MODE", "").lower()
    compression_sparsity = float(os.environ.get("COMPRESSION_SPARSITY", "0.0"))
    compression_k_frac = float(os.environ.get("COMPRESSION_K_FRAC", "0.0"))

    metrics_with_meta = dict(final_metrics)
    metrics_with_meta.update(
        {
            "dataset": dataset,
            "mode": mode_lower,
            "variant": variant,
            "dp_enabled": dp_enabled,
            "dp_sigma": dp_sigma,
            "compression_enabled": compression_enabled,
            "compression_mode": compression_mode,
            "compression_sparsity": compression_sparsity,
            "compression_k_frac": compression_k_frac,
        }
    )

    # Summary CSV (base + optional variant-specific)
    df = _build_round_dataframe(metrics_with_meta, num_rounds)
    csv_base = os.path.join(out_dir, f"summary_{mode_lower}.csv")
    df.to_csv(csv_base, index=False)

    csv_variant = None
    if variant:
        csv_variant = os.path.join(
            out_dir, f"summary_{mode_lower}{variant_suffix}.csv"
        )
        df.to_csv(csv_variant, index=False)

    # Final metrics JSON (base + optional variant-specific)
    metrics_base = os.path.join(out_dir, f"final_metrics_{mode_lower}.json")
    with open(metrics_base, "w") as f:
        json.dump(metrics_with_meta, f, indent=4)

    metrics_variant = None
    if variant:
        metrics_variant = os.path.join(
            out_dir, f"final_metrics_{mode_lower}{variant_suffix}.json"
        )
        with open(metrics_variant, "w") as f:
            json.dump(metrics_with_meta, f, indent=4)

    print(f"[SERVER] Summary saved | dataset={dataset} | mode={mode}")
    print(f"  {csv_base}")
    if csv_variant:
        print(f"  {csv_variant}")
    print(f"  {metrics_base}")
    if metrics_variant:
        print(f"  {metrics_variant}")

    # Upload only essentials (both base and variant if present)
    upload_paths = [csv_base, metrics_base]
    if csv_variant:
        upload_paths.append(csv_variant)
    if metrics_variant:
        upload_paths.append(metrics_variant)

    for path in upload_paths:
        _upload_to_s3_dataset(path, dataset, mode)

    return df
