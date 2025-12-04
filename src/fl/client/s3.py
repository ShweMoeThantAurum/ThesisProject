"""
Client-side S3 communication utilities.

Handles downloading global models, uploading processed updates,
and writing metadata to the correct S3 keys for each round.
"""

import os
import json
import time
from typing import Tuple

import boto3
import torch

from src.fl.logger import log_event, Timer
from src.fl.utils import get_bucket, get_prefix


BUCKET = get_bucket()
PREFIX = get_prefix()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------
def global_key(round_id):
    """S3 key for the global model of a given round."""
    return f"{PREFIX}/round_{round_id}/global.pt"


def raw_update_key(round_id, role):
    """Key for raw updates (if Lambda offload were enabled)."""
    return f"{PREFIX}/round_{round_id}/raw_updates/{role}.pt"


def processed_update_key(round_id, role):
    """Key for final processed (DP/compressed) updates."""
    return f"{PREFIX}/round_{round_id}/updates/{role}.pt"


def metadata_key(round_id, role):
    """Key for client metadata JSON."""
    return f"{PREFIX}/round_{round_id}/metadata/{role}.json"


# ---------------------------------------------------------------------
# Download global model
# ---------------------------------------------------------------------
def download_global(round_id, role):
    """Block until the server uploads the global model for this round."""
    key = global_key(round_id)
    local_path = f"/tmp/global_{role}_round_{round_id}.pt"

    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = os.environ.get("FL_MODE", "AEFL").strip().lower()
    variant = os.environ.get("VARIANT_ID", "").strip()

    while True:
        timer = Timer()
        timer.start()
        try:
            s3.download_file(BUCKET, key, local_path)
            latency = timer.stop()
            size_bytes = os.path.getsize(local_path)

            log_event(
                "client_s3_download.log",
                {
                    "role": role,
                    "round": round_id,
                    "dataset": dataset,
                    "mode": mode,
                    "variant": variant,
                    "latency_sec": latency,
                    "size_bytes": size_bytes,
                    "s3_key": key,
                },
            )

            print(
                f"[{role}] Downloaded global model for round {round_id} "
                f"(size={size_bytes / (1024**2):.3f} MB, latency={latency:.3f}s)"
            )
            return local_path, size_bytes

        except Exception:
            print(f"[{role}] Waiting for global model round {round_id}...")
            time.sleep(3.0)


# ---------------------------------------------------------------------
# Upload RAW update (unused in your pipeline)
# ---------------------------------------------------------------------
def upload_raw_update(round_id, role, state_dict):
    """Upload unprocessed raw update (Lambda mode only; unused here)."""
    local_path = f"/tmp/raw_update_{role}_round_{round_id}.pt"
    torch.save(state_dict, local_path)

    key = raw_update_key(round_id, role)

    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = os.environ.get("FL_MODE", "AEFL").strip().lower()
    variant = os.environ.get("VARIANT_ID", "").strip()

    timer = Timer()
    timer.start()
    s3.upload_file(local_path, BUCKET, key)
    latency = timer.stop()

    size_bytes = os.path.getsize(local_path)

    log_event(
        "client_raw_upload.log",
        {
            "role": role,
            "round": round_id,
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "latency_sec": latency,
            "size_bytes": size_bytes,
            "s3_key": key,
        },
    )

    print(
        f"[{role}] Uploaded RAW update for round {round_id} "
        f"(size={size_bytes / (1024**2):.3f} MB, latency={latency:.3f}s)"
    )

    return size_bytes, latency


# ---------------------------------------------------------------------
# Upload processed update
# ---------------------------------------------------------------------
def upload_processed_update(round_id, role, state_dict):
    """Upload the compressed/DP-processed update for this round."""
    local_path = f"/tmp/update_{role}_round_{round_id}.pt"
    torch.save(state_dict, local_path)

    key = processed_update_key(round_id, role)

    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = os.environ.get("FL_MODE", "AEFL").strip().lower()
    variant = os.environ.get("VARIANT_ID", "").strip()

    timer = Timer()
    timer.start()
    s3.upload_file(local_path, BUCKET, key)
    latency = timer.stop()

    size_bytes = os.path.getsize(local_path)

    log_event(
        "client_s3_upload.log",
        {
            "role": role,
            "round": round_id,
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "latency_sec": latency,
            "size_bytes": size_bytes,
            "s3_key": key,
        },
    )

    print(
        f"[{role}] Uploaded PROCESSED update for round {round_id} "
        f"(size={size_bytes / (1024**2):.3f} MB, latency={latency:.3f}s)"
    )

    return size_bytes, latency


# ---------------------------------------------------------------------
# Upload metadata
# ---------------------------------------------------------------------
def upload_metadata(round_id, role, meta):
    """Upload metadata JSON containing energy, communication and training stats."""
    key = metadata_key(round_id, role)
    body = json.dumps(meta).encode("utf-8")

    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = os.environ.get("FL_MODE", "AEFL").strip().lower()
    variant = os.environ.get("VARIANT_ID", "").strip()

    timer = Timer()
    timer.start()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)
    latency = timer.stop()

    log_event(
        "client_meta_upload.log",
        {
            "role": role,
            "round": round_id,
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "latency_sec": latency,
            "size_bytes": len(body),
            "s3_key": key,
        },
    )

    print(
        f"[{role}] Uploaded metadata for round {round_id}: "
        f"bandwidth={meta.get('bandwidth_mbps', 0):.3f} Mb/s, "
        f"total_energy={meta.get('total_energy_j', 0):.2f} J"
    )
