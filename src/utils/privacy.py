"""
Adds differential privacy noise to model updates.
"""

import torch


def dp_add_noise(state_dict, sigma=0.05):
    """Adds Gaussian noise to model parameters."""
    noisy = {}
    for k, v in state_dict.items():
        noise = torch.randn_like(v) * sigma
        noisy[k] = v + noise
    return noisy
