"""
Client metadata for AEFL selection (bandwidth + training stats only).
Energy is now handled separately in dedicated energy logs.
"""


def compute_bandwidth_mbps(payload_bytes, upload_latency_sec):
    if upload_latency_sec <= 0.0:
        return 0.0
    mbits = (payload_bytes * 8.0) / 1e6
    return mbits / upload_latency_sec


def build_round_metadata(
    role,
    round_id,
    energy_record,    # kept for backward compatibility, but not uploaded
    train_loss,
    train_samples,
    update_bytes,
    upload_latency_sec,
):
    bw_mbps = compute_bandwidth_mbps(update_bytes, upload_latency_sec)

    return {
        "role": role,
        "round": round_id,
        "train_loss": float(train_loss),
        "train_samples": int(train_samples),
        "bandwidth_mbps": bw_mbps,
    }
