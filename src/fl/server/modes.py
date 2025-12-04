"""
Helpers for determining which FL mode the server is running in.

Provides simple boolean checks for AEFL, FedAvg, and FedProx.
"""

import os


def get_mode():
    """Return the active FL mode in lowercase."""
    return os.environ.get("FL_MODE", "AEFL").strip().lower()


def is_aefl(mode):
    return mode.lower() == "aefl"


def is_fedavg(mode):
    return mode.lower() == "fedavg"


def is_fedprox(mode):
    return mode.lower() == "fedprox"
