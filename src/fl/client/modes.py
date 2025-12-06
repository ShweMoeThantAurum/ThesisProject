"""
Client-side FL mode helpers.

Clients must know:
 - which FL mode is active (FedAvg, FedProx, AEFL)
 - whether they should train locally (always true currently)
"""

import os

VALID_MODES = ["aefl", "fedavg", "fedprox"]


def get_client_mode():
    """
    Return the client FL mode in lowercase.

    Clients read FL_MODE from env, same as server.
    """
    mode = os.environ.get("FL_MODE", "AEFL").strip().lower()
    if mode not in VALID_MODES:
        print(f"[CLIENT] WARNING: Invalid FL_MODE='{mode}'. Defaulting to AEFL.")
        return "aefl"
    return mode


def client_allows_training(mode):
    """
    Return True if the client should perform local training.
    (In future you may disable training for specific modes.)
    """
    return True  
    
def is_aefl(mode):
    return mode == "aefl"


def is_fedavg(mode):
    return mode == "fedavg"


def is_fedprox(mode):
    return mode == "fedprox"
