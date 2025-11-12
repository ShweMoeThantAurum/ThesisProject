"""
Implements global model aggregation methods: FedAvg and AEFL (adaptive energy-weighted).
"""

import numpy as np
import torch


def fedavg_average(states, weights):
    """Computes FedAvg weighted average of model states."""
    avg = {}
    total = float(sum(weights))
    for k in states[0].keys():
        acc = None
        for s, w in zip(states, weights):
            val = s[k] * (w / (total + 1e-8))
            acc = val if acc is None else acc + val
        avg[k] = acc
    return avg


def adaptive_average(states, weights, scores):
    """Computes AEFL weighted average using sample weights and energy-bandwidth scores."""
    w = np.array(weights, dtype=np.float32) * np.array(scores, dtype=np.float32)
    w = (w + 1e-6) / (w.sum() + 1e-6)
    avg = {}
    for k in states[0].keys():
        acc = None
        for s, wi in zip(states, w):
            val = s[k] * float(wi)
            acc = val if acc is None else acc + val
        avg[k] = acc
    return avg


def clone_state(state):
    """Clones a PyTorch state_dict tensor-wise."""
    return {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in state.items()}


def set_state(model, state):
    """Loads a state_dict into a model."""
    model.load_state_dict(state)


def pick_clients(states_dict, alpha, min_energy, max_participants):
    """Selects clients by energy-bandwidth score."""
    scored = [(cid, alpha * s["energy"] + (1 - alpha) * s["bandwidth"])
              for cid, s in states_dict.items() if s["energy"] >= min_energy]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:max_participants]]
