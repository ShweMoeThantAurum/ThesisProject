"""
AEFL: Adaptive Energy-Aware Federated Learning (single-file framework).
Adds CLI flags and per-round CSV logging for AEFL comparisons.
"""
import os, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.utils.run_logger import init_csv, log_row
from src.utils.profiler_utils import cpu_mem_snapshot, energy_proxy

# -----------------------------
# Data helpers
# -----------------------------
PROC = "data/processed/sz/prepared"
OUT = "outputs/aefl_framework_sz"
os.makedirs(OUT, exist_ok=True)

class SimpleGRU(nn.Module):
    """A small GRU model for multivariate time series forecasting."""
    def __init__(self, num_nodes, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size=num_nodes, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_nodes)
    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        yhat = self.head(last)
        return yhat

class ClientPaddedDataset(Dataset):
    """Loads a client's subset and pads to full node dimension."""
    def __init__(self, X_client_path, y_client_path, idxs, num_nodes):
        cX = np.load(X_client_path); cY = np.load(y_client_path)
        S, L, k = cX.shape
        X = np.zeros((S, L, num_nodes), dtype=np.float32)
        Y = np.zeros((S, num_nodes), dtype=np.float32)
        X[:, :, idxs] = cX; Y[:, idxs] = cY
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

def load_global_sets():
    """Loads full train/valid/test sets."""
    Xtr = np.load(f"{PROC}/X_train.npy"); ytr = np.load(f"{PROC}/y_train.npy")
    Xva = np.load(f"{PROC}/X_valid.npy"); yva = np.load(f"{PROC}/y_valid.npy")
    Xte = np.load(f"{PROC}/X_test.npy");  yte = np.load(f"{PROC}/y_test.npy")
    return Xtr, ytr, Xva, yva, Xte, yte

def make_eval_loader(X, y, batch):
    """Creates a DataLoader for evaluation."""
    X = torch.from_numpy(X).float(); y = torch.from_numpy(y).float()
    ds = torch.utils.data.TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch, shuffle=False)

def client_indices(num_nodes, num_clients, i):
    """Returns node indices for client i."""
    parts = np.array_split(np.arange(num_nodes), num_clients)
    return parts[i]

# -----------------------------
# Energy and selection
# -----------------------------
def init_client_states(num_clients, seed=123):
    """Initializes client energy and bandwidth states."""
    rng = np.random.RandomState(seed)
    states = {}
    for i in range(num_clients):
        states[str(i)] = {
            "energy": float(rng.uniform(0.4, 1.0)),
            "bandwidth": float(rng.uniform(0.4, 1.0))
        }
    return states

def estimate_round_cost(samples, model_scale=1.0):
    """Estimates a simple energy cost for local training."""
    base = 0.001
    return base * samples * model_scale

def drain_energy(state, cost):
    """Reduces client energy by cost."""
    state["energy"] = max(0.0, state["energy"] - cost)

def recharge_idle(state, rate=0.02):
    """Recharges energy slightly when idle."""
    state["energy"] = min(1.0, state["energy"] + rate)

def score_client(energy, bandwidth, alpha):
    """Scores a client using energy and bandwidth."""
    return alpha * energy + (1 - alpha) * bandwidth

def pick_clients(states, max_clients, min_energy, alpha):
    """Selects clients by score while excluding very low energy."""
    items = []
    for cid, st in states.items():
        if st["energy"] < min_energy:
            continue
        s = score_client(st["energy"], st["bandwidth"], alpha)
        items.append((cid, s))
    if not items:
        return []
    items.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in items[:max_clients]]

# -----------------------------
# Compression (simple)
# -----------------------------
def topk_sparsify_tensor(t, sparsity):
    """Keeps top-(1-s) magnitude values, zeros rest."""
    if sparsity <= 0.0:
        return t, 1.0
    if sparsity >= 1.0:
        return torch.zeros_like(t), 0.0
    k = max(1, int(t.numel() * (1.0 - sparsity)))
    flat = t.view(-1).abs()
    if k >= flat.numel():
        return t, 1.0
    thresh = torch.topk(flat, k).values.min()
    mask = t.abs() >= thresh
    out = t * mask
    kept_frac = mask.float().mean().item()
    return out, kept_frac

def sparsify_state(state, sparsity):
    """Applies top-k sparsity to float tensors in a state dict."""
    out, kept = {}, []
    for k, v in state.items():
        if v.dtype.is_floating_point:
            sv, frac = topk_sparsify_tensor(v, sparsity)
            out[k] = sv; kept.append(frac)
        else:
            out[k] = v
    kept_ratio = sum(kept) / max(1, len(kept))
    return out, kept_ratio

def compress_for_energy(state, energy, use_compression):
    """Chooses compression based on energy and returns state and payload ratio."""
    if not use_compression:
        return state, 1.0
    if energy >= 0.6:
        return state, 1.0
    if 0.4 <= energy < 0.6:
        s_state, kept = sparsify_state(state, sparsity=0.5)
        return s_state, kept
    s_state, kept = sparsify_state(state, sparsity=0.8)
    bytes_ratio = max(1e-3, kept * 0.25)
    return s_state, bytes_ratio

# -----------------------------
# Privacy (DP)
# -----------------------------
def dp_add_noise(state, noise_std, use_dp):
    """Adds Gaussian noise to float tensors in a state dict."""
    if not use_dp or noise_std <= 0:
        return state
    out = {}
    for k, v in state.items():
        if v.dtype.is_floating_point:
            out[k] = v + torch.randn_like(v) * noise_std
        else:
            out[k] = v
    return out

# -----------------------------
# Train/Eval helpers
# -----------------------------
def get_device():
    """Selects MPS on Apple Silicon if available."""
    return "mps" if torch.backends.mps.is_available() else "cpu"

def clone_state(state):
    """Deep-copies a model state dict."""
    return {k: v.detach().clone() for k, v in state.items()}

def set_state(model, state):
    """Loads a state dict into model."""
    model.load_state_dict(state)

def train_local(model, loader, device, epochs, lr):
    """Trains one client locally and returns updated state and sample count."""
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n_samples = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            n_samples += x.size(0)
    return clone_state(model.state_dict()), n_samples

def evaluate(model, loader, device):
    """Evaluates MAE and RMSE."""
    model.eval()
    m_mae, m_rmse, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            m_mae += (yhat - y).abs().sum().item()
            m_rmse += ((yhat - y) ** 2).sum().item()
            n += y.numel()
    mae_val = m_mae / n
    rmse_val = math.sqrt(m_rmse / n)
    return mae_val, rmse_val

def aggregation_weights(counts, scores):
    """Builds normalized weights combining sample counts and scores."""
    counts = np.array(counts, dtype=np.float32) + 1e-6
    scores = np.array(scores, dtype=np.float32) + 1e-6
    w = counts * scores
    w = w / w.sum()
    return w.tolist()

# -----------------------------
# Main AEFL
# -----------------------------
def run(args):
    """Runs AEFL with logging and CLI flags."""
    device = get_device()
    meta = json.load(open(f"{PROC}/meta.json"))
    num_clients = meta["num_clients"]
    max_participating = args.max_participants or max(2, num_clients // 2)

    Xtr, ytr, Xva, yva, Xte, yte = load_global_sets()
    eval_valid = make_eval_loader(Xva, yva, batch=128)
    eval_test  = make_eval_loader(Xte, yte, batch=128)

    num_nodes = Xtr.shape[-1]
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=args.hidden).to(device)
    global_state = clone_state(model.state_dict())

    states = init_client_states(num_clients, seed=args.seed)

    csv_path = os.path.join(OUT, "round_log.csv")
    csv_header = [
        "round", "chosen", "val_mae", "val_rmse",
        "avg_comm_ratio", "cpu_percent", "mem_mb",
        "round_secs", "energy_proxy"
    ]
    init_csv(csv_path, csv_header)

    print(f"AEFL | rounds {args.rounds} | max_participants {max_participating} "
          f"| comp {args.use_compression} | dp {args.use_dp} (std={args.dp_std})")

    for r in range(1, args.rounds + 1):
        t0 = time.time()
        chosen = pick_clients(states, max_participating, args.min_energy, args.alpha_energy)
        if not chosen:
            chosen = [str(i) for i in range(num_clients)]

        part_states, counts, scores, comm_ratios = [], [], [], []

        for cid in range(num_clients):
            cid_str = str(cid)
            if cid_str not in chosen:
                recharge_idle(states[cid_str])
                continue

            idxs = client_indices(num_nodes, num_clients, cid)
            cX = f"{PROC}/clients/client{cid}_X.npy"
            cY = f"{PROC}/clients/client{cid}_y.npy"
            ds = ClientPaddedDataset(cX, cY, idxs, num_nodes)
            dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

            set_state(model, global_state)
            model.to(device)
            local_state, n_i = train_local(model, dl, device, args.local_epochs, args.lr)

            energy_before = states[cid_str]["energy"]
            comp_state, payload_ratio = compress_for_energy(local_state, energy_before, args.use_compression)
            comp_state = dp_add_noise(comp_state, args.dp_std, args.use_dp)

            part_states.append(comp_state)
            counts.append(n_i)
            comm_ratios.append(payload_ratio)

            cost = estimate_round_cost(n_i, model_scale=payload_ratio)
            drain_energy(states[cid_str], cost)

            s = score_client(states[cid_str]["energy"], states[cid_str]["bandwidth"], args.alpha_energy)
            scores.append(max(1e-6, s))

        if part_states:
            w = aggregation_weights(counts, scores)
            total = sum(w)
            weights = [wi / total for wi in w]
            avg = {}
            keys = part_states[0].keys()
            for k in keys:
                acc = None
                for st, wi in zip(part_states, weights):
                    val = st[k] * wi
                    acc = val if acc is None else acc + val
                avg[k] = acc
            global_state = avg
            set_state(model, global_state)

        v_mae, v_rmse = evaluate(model, eval_valid, device)
        round_secs = time.time() - t0
        avg_comm = sum(comm_ratios) / max(1, len(comm_ratios)) if comm_ratios else 1.0
        cpu_p, mem_mb = cpu_mem_snapshot()
        eproxy = energy_proxy(round_secs, cpu_p, avg_comm)

        print(
            f"Round {r:02d} | chosen {len(chosen)}/{num_clients} "
            f"| val_MAE {v_mae:.4f} | val_RMSE {v_rmse:.4f} "
            f"| comm_ratio~{avg_comm:.2f} | cpu {cpu_p:.1f}% | mem {mem_mb:.1f}MB "
            f"| {round_secs:.1f}s | eproxy {eproxy:.4f}"
        )

        log_row(csv_path, [
            r, len(chosen), f"{v_mae:.6f}", f"{v_rmse:.6f}",
            f"{avg_comm:.4f}", f"{cpu_p:.2f}", f"{mem_mb:.2f}",
            f"{round_secs:.3f}", f"{eproxy:.6f}"
        ])

    set_state(model, global_state)
    te_mae, te_rmse = evaluate(model, eval_test, device)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(f"TEST MAE: {te_mae:.6f}\nTEST RMSE: {te_rmse:.6f}\n")
    torch.save(global_state, os.path.join(OUT, "aefl_state.pt"))
    print(f"Saved {OUT}/results.txt, aefl_state.pt and {csv_path}")

def parse_args():
    """Parses command line flags for AEFL."""
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    p.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    p.add_argument("--batch", type=int, default=64, help="Client batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--hidden", type=int, default=64, help="GRU hidden size")
    p.add_argument("--max-participants", type=int, default=None, help="Max clients per round")
    p.add_argument("--use-compression", action="store_true", help="Enable compression")
    p.add_argument("--no-compression", dest="use_compression", action="store_false")
    p.set_defaults(use_compression=True)
    p.add_argument("--use-dp", action="store_true", help="Enable DP noise")
    p.add_argument("--no-dp", dest="use_dp", action="store_false")
    p.set_defaults(use_dp=False)
    p.add_argument("--dp-std", type=float, default=0.01, help="DP Gaussian std")
    p.add_argument("--alpha-energy", type=float, default=0.6, help="Energy weight in score")
    p.add_argument("--min-energy", type=float, default=0.15, help="Min energy to participate")
    p.add_argument("--seed", type=int, default=123, help="Seed for client states")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
