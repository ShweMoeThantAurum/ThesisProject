"""
Server-side S3 utilities.

Downloads client updates, loads metadata JSON, and returns model updates
as PyTorch state_dicts for aggregation.
"""

import json
import boto3
import torch
import os
import io

from src.fl.logger import log_event, Timer
from src.fl.utils import get_bucket, get_prefix

BUCKET = get_bucket()
PREFIX = get_prefix()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=AWS_REGION)


def load_client_update(round_id, role, prefix=PREFIX):
    """Download a client model update for the given round and role."""
    key = f"{prefix}/round_{round_id}/updates/{role}.pt"
    timer = Timer()

    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = os.environ.get("FL_MODE", "AEFL").strip().lower()
    variant = os.environ.get("VARIANT_ID", "").strip()

    try:
        timer.start()
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        latency = timer.stop()

        raw = obj["Body"].read()
        size_bytes = len(raw)
        state = torch.load(io.BytesIO(raw), map_location="cpu")

        log_event(
            "server_update_download.log",
            {
                "round": round_id,
                "role": role,
                "dataset": dataset,
                "mode": mode,
                "variant": variant,
                "size_bytes": size_bytes,
                "latency_sec": latency,
            },
        )

        print(
            f"[SERVER] Downloaded update from {role} r={round_id} "
            f"({size_bytes/1e6:.3f} MB, {latency:.3f}s)"
        )

        return state

    except Exception:
        return None


def load_round_metadata(round_id, prefix=PREFIX):
    """Load metadata for all clients for a given round."""
    meta_prefix = f"{prefix}/round_{round_id}/metadata/"

    try:
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=meta_prefix)
        if "Contents" not in listing:
            return {}

        results = {}
        for obj in listing["Contents"]:
            key = obj["Key"]
            role = key.split("/")[-1].replace(".json", "")
            raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            results[role] = json.loads(raw.decode("utf-8"))

        return results

    except Exception:
        return {}
