"""
Keeps a simple per-client energy state and cost model.
"""
import json, os, random

def init_client_states(path, num_clients):
    """Initializes client energy and bandwidth."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    states = {}
    for i in range(num_clients):
        states[str(i)] = {
            "energy": random.uniform(0.4, 1.0),
            "bandwidth": random.uniform(0.4, 1.0)
        }
    with open(path, "w") as f:
        json.dump(states, f, indent=2)

def load_states(path):
    """Loads client states."""
    with open(path) as f:
        return json.load(f)

def save_states(path, states):
    """Saves client states."""
    with open(path, "w") as f:
        json.dump(states, f, indent=2)

def estimate_round_cost(samples, model_scale=1.0):
    """Estimates a simple energy cost for local training."""
    base = 0.001
    return base * samples * model_scale

def drain_energy(state, cost):
    """Reduces client energy by cost and clamps."""
    e = max(0.0, state["energy"] - cost)
    state["energy"] = e

def recharge_idle(state, rate=0.02):
    """Recharges energy slightly if client is idle."""
    e = min(1.0, state["energy"] + rate)
    state["energy"] = e
