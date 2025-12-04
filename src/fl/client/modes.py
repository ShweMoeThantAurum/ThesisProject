"""
Client mode utilities for federated learning.

Determines the selected FL mode (AEFL, FedAvg, FedProx)
based on environment variables.
"""

import os

VALID_MODES = ["AEFL", "FedAvg", "FedProx"]


def get_client_mode():
    """Return the current FL mode from environment variables."""
    mode = os.environ.get("FL_MODE", "AEFL").strip()
    if mode not in VALID_MODES:
        print(f"[CLIENT] WARNING: invalid FL_MODE='{mode}', using AEFL.")
        return "AEFL"
    return mode


def client_allows_training(mode):
    """Return True if training should occur for the given mode."""
    return True
