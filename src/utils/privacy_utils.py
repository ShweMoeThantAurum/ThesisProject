"""
Adds simple Differential Privacy noise and simulates Secure Aggregation masks.
"""
import os, json, torch, random

def dp_add_noise(state, noise_std):
    """Adds Gaussian noise to each float tensor in a state dict."""
    if noise_std <= 0:
        return state
    out = {}
    for k, v in state.items():
        if v.dtype.is_floating_point:
            out[k] = v + torch.randn_like(v) * noise_std
        else:
            out[k] = v
    return out

def sa_init_pairwise_masks(path, num_clients, seed=123):
    """Creates deterministic pairwise masks for secure aggregation simulation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rnd = random.Random(seed)
    pairs = {}
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            key = f"{i}-{j}"
            pairs[key] = rnd.randint(1, 10)  # small scalar mask seed
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)

def sa_load_pairwise_masks(path):
    """Loads pairwise mask seeds."""
    with open(path) as f:
        return json.load(f)

def sa_apply_masks(state, client_id, masks, num_clients):
    """Applies pairwise additive masks; masks cancel out in the sum."""
    out = {}
    for k, v in state.items():
        if not v.dtype.is_floating_point:
            out[k] = v
            continue
        acc = v
        cid = int(client_id)
        for other in range(num_clients):
            if other == cid:
                continue
            i, j = sorted([cid, other])
            key = f"{i}-{j}"
            if key not in masks:
                continue
            s = masks[key]
            if cid == i:
                acc = acc + s
            else:
                acc = acc - s
        out[k] = acc
    return out

def sa_remove_masks(sum_state, num_clients, masks):
    """No-op for pairwise additive scheme since masks cancel in the sum."""
    return sum_state
