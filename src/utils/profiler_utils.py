"""
Provides simple CPU, memory, and energy-proxy measurements.
"""
import psutil, time

def now_s():
    """Returns current time in seconds."""
    return time.time()

def cpu_mem_snapshot():
    """Returns CPU percent and memory MB."""
    cpu = psutil.cpu_percent(interval=None)
    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    return cpu, mem_mb

def energy_proxy(round_secs, cpu_percent, comm_ratio):
    """
    Estimates an energy proxy for the round.
    Simple model combining compute time, CPU%, and comm volume.
    """
    cpu_factor = max(0.1, cpu_percent / 100.0)
    net_factor = max(0.05, comm_ratio)
    return round_secs * cpu_factor * (0.5 + 0.5 * net_factor)
