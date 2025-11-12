"""
Estimates per-round energy consumption using CPU and network usage models.
"""

import psutil
import torch

DEVICE_PROFILES = {
    "edge":  {"cpu_watts": 3.5,  "net_j_per_mb": 0.6},
    "mid":   {"cpu_watts": 6.0,  "net_j_per_mb": 0.9},
    "cloud": {"cpu_watts": 45.0, "net_j_per_mb": 5.0},
}


def cpu_mem_snapshot():
    """Returns (cpu_percent, mem_mb)."""
    cpu_percent = psutil.cpu_percent(interval=None)
    mem_mb = psutil.virtual_memory().used / (1024 ** 2)
    return cpu_percent, mem_mb


def state_size_bytes(state_dict):
    """Computes size in bytes of a state_dict."""
    total = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.element_size() * v.numel()
    return int(total)


def _bytes_to_mb(nbytes):
    """Converts bytes to MB."""
    return float(nbytes) / (1024.0 * 1024.0)


def compute_round_energy_j(round_secs, cpu_percent, sent_bytes, device_kind="edge"):
    """Estimates round energy (J)."""
    prof = DEVICE_PROFILES.get(device_kind, DEVICE_PROFILES["edge"])
    cpu_active = max(0.0, min(1.0, cpu_percent / 100.0))
    e_cpu = prof["cpu_watts"] * cpu_active * round_secs
    e_net = prof["net_j_per_mb"] * _bytes_to_mb(sent_bytes)
    return e_cpu + e_net


def energy_proxy(duration_s, cpu_percent, comm_ratio):
    """Computes a simple proxy for energy."""
    alpha, beta = 0.7, 0.3
    return alpha * (cpu_percent / 100.0) * duration_s + beta * comm_ratio
