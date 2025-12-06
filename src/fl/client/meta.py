"""
Build metadata records for client rounds.

Includes communication statistics, bandwidth, training statistics,
and per-round energy estimates (used by AEFL server selection).
"""


def compute_bandwidth_mbps(payload_bytes, upload_latency_sec):
    """Compute uplink bandwidth in megabits/sec."""
    if upload_latency_sec <= 0.0:
        return 0.0
    mbits = (payload_bytes * 8.0) / 1e6
    return mbits / upload_latency_sec


def build_round_metadata(
    role,
    round_id,
    energy_record,
    train_loss,
    train_samples,
    update_bytes,
    upload_latency_sec,
):
    """
    Assemble metadata dictionary for upload to the server.

    This is what AEFL uses for energy-aware client selection.
    """
    bw_mbps = compute_bandwidth_mbps(update_bytes, upload_latency_sec)

    return {
        "role": role,
        "round": round_id,
        "train_loss": float(train_loss),
        "train_samples": int(train_samples),

        # energy (from energy_record)
        "compute_j_time": float(energy_record.get("compute_j_time", 0.0)),
        "compute_j_flops": float(energy_record.get("compute_j_flops", 0.0)),
        "comm_j_download": float(energy_record.get("comm_j_download", 0.0)),
        "comm_j_upload": float(energy_record.get("comm_j_upload", 0.0)),
        "comm_j_total": float(energy_record.get("comm_j_total", 0.0)),
        "FLOPs": float(energy_record.get("FLOPs", 0.0)),
        "total_energy_j": float(energy_record.get("total_energy_j", 0.0)),

        # communication volumes
        "download_mb": float(energy_record.get("download_mb", 0.0)),
        "upload_mb": float(energy_record.get("upload_mb", 0.0)),

        # network condition
        "bandwidth_mbps": bw_mbps,
    }
