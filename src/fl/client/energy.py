"""
Energy estimation utilities for FL clients.
Produces clean per-round JSON entries for server aggregation.
"""

import os
from pathlib import Path
import json

from src.utils.flops import estimate_gru_flops

ENERGY_PER_FLOP_J = 1e-9  # academic assumption: 1 nJ per FLOP


def _log_energy(role, record):
    """Append a single JSON entry to run_logs/energy_<role>.jsonl"""
    log_dir = Path("run_logs")
    log_dir.mkdir(exist_ok=True)

    path = log_dir / f"energy_{role}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def estimate_round_energy(
    role,
    round_id,
    train_time_sec,
    download_bytes,
    upload_bytes,
    device_power_watts,
    net_j_per_mb,
    num_nodes=None,
    hidden_size=None,
    seq_len=12,
):
    """Estimate compute and communication energy for a single client round."""

    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = os.environ.get("FL_MODE", "aefl").strip().lower()
    variant = os.environ.get("VARIANT_ID", "").strip()

    # Compute energy (time-based)
    compute_j_time = device_power_watts * train_time_sec

    # Optional FLOPs-based energy
    if num_nodes is not None and hidden_size is not None:
        flops = estimate_gru_flops(num_nodes, hidden_size, seq_len)
        compute_j_flops = flops * ENERGY_PER_FLOP_J
    else:
        flops = 0
        compute_j_flops = 0.0

    # Communication energy
    download_mb = download_bytes / (1024 * 1024)
    upload_mb = upload_bytes / (1024 * 1024)

    comm_j_download = net_j_per_mb * download_mb
    comm_j_upload = net_j_per_mb * upload_mb
    comm_j_total = comm_j_download + comm_j_upload

    total_energy = compute_j_time + comm_j_total

    record = {
        "dataset": dataset,
        "mode": mode,
        "variant": variant,
        "role": role,
        "round": round_id,
        "compute_j_time": compute_j_time,
        "compute_j_flops": compute_j_flops,
        "comm_j_total": comm_j_total,
        "total_energy_j": total_energy,
    }

    _log_energy(role, record)

    print(
        f"[{role}] Energy r={round_id}: "
        f"compute={compute_j_time:.4f} J, "
        f"comm={comm_j_total:.4f} J, "
        f"TOTAL={total_energy:.4f} J"
    )

    return record
