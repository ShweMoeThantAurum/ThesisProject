"""
Scores clients, selects participants, and weights model updates adaptively.
"""
import numpy as np

def score_client(energy, bandwidth, alpha=0.6):
    """Scores a client using energy and bandwidth."""
    return alpha * energy + (1 - alpha) * bandwidth

def pick_clients(states, max_clients, min_energy=0.15):
    """Selects clients by score while excluding very low energy."""
    items = []
    for cid, st in states.items():
        if st["energy"] < min_energy:
            continue
        s = score_client(st["energy"], st["bandwidth"])
        items.append((cid, s))
    if not items:
        return []
    items.sort(key=lambda x: x[1], reverse=True)
    chosen = [cid for cid, _ in items[:max_clients]]
    return chosen

def aggregation_weights(sample_counts, scores):
    """Builds normalized weights combining samples and scores."""
    sample_counts = np.array(sample_counts, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    sample_counts = sample_counts + 1e-6
    scores = scores + 1e-6
    w = sample_counts * scores
    w = w / w.sum()
    return w.tolist()
