"""
Runs adaptive FL with energy-aware selection plus sparsification/quantization.
"""
import os, math, time, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.models.simple_gru import SimpleGRU
from src.fl.fedavg_baseline import clone_state, set_state, get_device, load_global_sets, make_eval_loader, client_indices
from src.fl.adaptive_aggregator import pick_clients, aggregation_weights, score_client
from src.utils.energy_utils import init_client_states, load_states, save_states, estimate_round_cost, drain_energy, recharge_idle
from src.utils.compression_utils import sparsify_state, quantize_state_8bit, dequantize_state_8bit
from src.utils.run_logger import init_csv, log_row
from src.utils.profiler_utils import cpu_mem_snapshot, energy_proxy

PROC = "data/processed/sz/prepared"
OUT = "outputs/adaptive_fl_comp_sz"
STATE_PATH = os.path.join(OUT, "client_states.json")
CSV_PATH = os.path.join(OUT, "round_log.csv")
CSV_HEADER = [
    "round", "chosen", "val_mae", "val_rmse",
    "avg_comm_ratio", "cpu_percent", "mem_mb",
    "round_secs", "energy_proxy"
]
os.makedirs(OUT, exist_ok=True)

class ClientPaddedDataset(Dataset):
    """
    Loads a client's subset and pads to full node dimension.
    """
    def __init__(self, X_client_path, y_client_path, idxs, num_nodes):
        cX = np.load(X_client_path)
        cY = np.load(y_client_path)
        S, L, k = cX.shape
        X = np.zeros((S, L, num_nodes), dtype=np.float32)
        Y = np.zeros((S, num_nodes), dtype=np.float32)
        X[:, :, idxs] = cX
        Y[:, idxs] = cY
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

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

def compress_for_energy(state, energy):
    """Chooses compression based on energy and returns decompressed-for-train state and stats."""
    if energy >= 0.6:
        return state, {"mode": "none", "sparsity": 0.0, "bytes_ratio": 1.0}
    if 0.4 <= energy < 0.6:
        s_state, kept = sparsify_state(state, sparsity=0.5)
        return s_state, {"mode": "sparse50", "sparsity": 0.5, "bytes_ratio": kept}
    s_state, kept = sparsify_state(state, sparsity=0.8)
    q_payload, q_meta = quantize_state_8bit(s_state)
    dq_state = dequantize_state_8bit(q_payload, q_meta)
    bytes_ratio = max(1e-3, kept * (1.0 / 4.0))
    return dq_state, {"mode": "sparse80_q8", "sparsity": 0.8, "bytes_ratio": bytes_ratio}

def run():
    """Runs adaptive FL with compression and logs metrics."""
    device = get_device()
    meta = json.load(open(f"{PROC}/meta.json"))
    num_clients = meta["num_clients"]
    max_participating = max(2, num_clients // 2)

    if not os.path.exists(STATE_PATH):
        init_client_states(STATE_PATH, num_clients)
    states = load_states(STATE_PATH)

    init_csv(CSV_PATH, CSV_HEADER)

    Xtr, ytr, Xva, yva, Xte, yte = load_global_sets()
    eval_valid = make_eval_loader(Xva, yva, batch=128)
    eval_test  = make_eval_loader(Xte, yte, batch=128)

    num_nodes = Xtr.shape[-1]
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=64).to(device)
    global_state = clone_state(model.state_dict())

    rounds = 5
    local_epochs = 1
    batch = 64
    lr = 1e-3

    print(f"Adaptive FL + Compression | rounds {rounds} | max_participants {max_participating}")

    for r in range(1, rounds + 1):
        t0 = time.time()
        chosen = pick_clients(states, max_participating)
        if not chosen:
            chosen = [str(i) for i in range(num_clients)]

        c_states, counts, scores, comm_ratios = [], [], [], []

        for cid in range(num_clients):
            cid_str = str(cid)
            if cid_str not in chosen:
                recharge_idle(states[cid_str])
                continue

            idxs = client_indices(num_nodes, num_clients, cid)
            cX = f"{PROC}/clients/client{cid}_X.npy"
            cY = f"{PROC}/clients/client{cid}_y.npy"
            ds = ClientPaddedDataset(cX, cY, idxs, num_nodes)
            dl = DataLoader(ds, batch_size=batch, shuffle=True)

            set_state(model, global_state)
            model.to(device)
            local_state, n_i = train_local(model, dl, device, local_epochs, lr)

            energy_before = states[cid_str]["energy"]
            comp_state, comp_info = compress_for_energy(local_state, energy_before)
            c_states.append(comp_state)
            counts.append(n_i)
            comm_ratios.append(comp_info["bytes_ratio"])

            cost = estimate_round_cost(n_i, model_scale=comp_info["bytes_ratio"])
            drain_energy(states[cid_str], cost)

            s = score_client(states[cid_str]["energy"], states[cid_str]["bandwidth"])
            scores.append(max(1e-6, s))

        if c_states:
            w = aggregation_weights(counts, scores)
            total = sum(w)
            weights = [wi / total for wi in w]
            avg = {}
            keys = c_states[0].keys()
            for k in keys:
                acc = None
                for st, wi in zip(c_states, weights):
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

        log_row(CSV_PATH, [
            r, len(chosen), f"{v_mae:.6f}", f"{v_rmse:.6f}",
            f"{avg_comm:.4f}", f"{cpu_p:.2f}", f"{mem_mb:.2f}",
            f"{round_secs:.3f}", f"{eproxy:.6f}"
        ])

        save_states(STATE_PATH, states)

    set_state(model, global_state)
    te_mae, te_rmse = evaluate(model, eval_test, device)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(f"TEST MAE: {te_mae:.6f}\nTEST RMSE: {te_rmse:.6f}\n")
    torch.save(global_state, os.path.join(OUT, "adaptive_comp_state.pt"))
    print(f"Saved {OUT}/results.txt, adaptive_comp_state.pt and {CSV_PATH}")

if __name__ == "__main__":
    run()
