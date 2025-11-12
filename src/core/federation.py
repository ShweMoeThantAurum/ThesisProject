"""
Main FL engine — coordinates client selection, local training, aggregation, and energy logging.
Supports: FedAvg, AEFL, Local-Only, Periodic Aggregation, FedProx, and SCAFFOLD.
"""

import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.client import client_indices, make_loader_for_client, train_local
from src.core.aggregation import (
    fedavg_average, adaptive_average, clone_state, set_state, pick_clients
)
from src.core.evaluator import init_csv, log_row, make_eval_loader, eval_model
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.compression import (
    sparsify_state,           # magnitude pruning (dense payload)
    topk_compress_state,      # indices + values
    quantize8_state,          # uint8/int8 per-tensor symmetric quantization
)
from src.utils.privacy import dp_add_noise
from src.core.energy import cpu_mem_snapshot, state_size_bytes, compute_round_energy_j

from src.models.simple_gru import SimpleGRU


# ----------------------------------------------------------
# Helper: local training for FedProx
# ----------------------------------------------------------
def _train_local_fedprox(model, loader, device, epochs, lr, mu, global_state):
    """FedProx: adds proximal penalty ||w - w_global||^2."""
    model.train()
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    global_params = {k: v.detach().clone().to(device) for k, v in global_state.items()}
    n_samples = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            prox_term = 0.0
            for name, param in model.named_parameters():
                prox_term += torch.norm(param - global_params[name]) ** 2
            loss = loss + (mu / 2.0) * prox_term
            loss.backward()
            opt.step()
            n_samples += x.size(0)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}, n_samples


# ----------------------------------------------------------
# Helper: local training for SCAFFOLD
# ----------------------------------------------------------
def _train_local_scaffold(model, loader, device, epochs, lr, c_global, c_local):
    """
    SCAFFOLD local update:
        grad <- grad - c_local + c_global
    Returns updated weights and delta control variate.
    """
    model.train()
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)  # SGD preferred for SCAFFOLD
    n_samples = 0
    initial_state = {k: v.detach().clone().to(device) for k, v in model.state_dict().items()}

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()

            # Apply correction: param.grad -= (c_local - c_global)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in c_local:
                        param.grad -= (c_local[name] - c_global[name])

            opt.step()
            n_samples += x.size(0)

    # Δc_i = (1 / (E * lr)) * (w_global - w_i)
    delta_c = {}
    for name, param in model.state_dict().items():
        delta_c[name] = (initial_state[name] - param.detach()) / (epochs * lr)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}, n_samples, delta_c


def _build_model(model_type, num_nodes, hidden, in_timesteps=None, device="cpu"):
    """Builds GRU backbone."""
    if model_type.lower() != "simplegru":
        raise ValueError("This thesis build uses GRU only. Set model_type=SimpleGRU.")
    return SimpleGRU(num_nodes=num_nodes, hidden_size=hidden).to(device)


def _maybe_flower(cfg):
    """Prints Flower hook status."""
    if cfg.get("federation", {}).get("use_flower", False):
        print("[Flower] Hook enabled (local simulation).")


def run_federated(config_path):
    """Runs a federated experiment and writes totals into results.txt."""
    cfg = load_config(config_path)
    set_global_seed(cfg.get("experiment", {}).get("seed", 42))

    exp = cfg["experiment"]
    data_cfg = cfg.get("data", {})
    fed = cfg.get("federation", {})
    train = cfg["training"]
    comp = cfg.get("compression", {})
    priv = cfg.get("privacy", {})
    eva = cfg["evaluation"]

    PROC = exp["proc_dir"]
    OUT = exp["output_dir"]
    os.makedirs(OUT, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() and eva.get("device_preference") == "mps" else "cpu"

    meta_path = os.path.join(PROC, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json at {meta_path}. Preprocess + split first.")
    meta = json.load(open(meta_path))
    num_clients = meta["num_clients"]

    Xva = np.load(f"{PROC}/X_valid.npy")
    yva = np.load(f"{PROC}/y_valid.npy")
    Xte = np.load(f"{PROC}/X_test.npy")
    yte = np.load(f"{PROC}/y_test.npy")
    num_nodes = Xva.shape[-1]

    model = _build_model(exp["model_type"], num_nodes, train.get("hidden_size", 64), device=device)
    global_state = clone_state(model.state_dict())

    eval_valid = make_eval_loader(Xva, yva, batch=128)
    eval_test = make_eval_loader(Xte, yte, batch=128)

    rng = np.random.RandomState(exp.get("seed", 42))
    states = {
        str(i): {"energy": float(rng.uniform(0.4, 1.0)), "bandwidth": float(rng.uniform(0.4, 1.0))}
        for i in range(num_clients)
    }

    csv_path = os.path.join(OUT, "round_log.csv")
    init_csv(
        csv_path,
        ["round","chosen","val_mae","val_rmse","round_secs","cpu_percent","mem_mb","energy_j","bytes_sent_mb","kept_ratio"]
    )

    rounds = fed.get("rounds", 20)
    local_epochs = fed.get("local_epochs", 1)
    max_part = fed.get("max_participants", num_clients)
    min_energy = fed.get("min_energy", 0.0)
    alpha = fed.get("alpha", 0.5)
    mu = fed.get("mu", 0.0)  # FedProx coefficient
    strategy = fed.get("strategy", "").lower()      # "fedavg" | "fedprox" | "scaffold" | "periodic"
    period_k = fed.get("period_k", 1)               # Periodic aggregation K
    mode = exp.get("mode", "fedavg").lower()        # "fedavg" | "aefl" | "localonly"

    # Compression config
    comp_enabled = comp.get("enabled", False)
    comp_strategy = (comp.get("strategy", "") or "").lower()   # "magnitude" | "topk" | "q8"
    sparsity = comp.get("sparsity", 0.0)
    k_frac = comp.get("k_frac", 0.1)

    # Privacy
    dp_enabled = priv.get("enabled", False)
    dp_sigma = priv.get("dp_sigma", 0.0)

    _maybe_flower(cfg)
    print(f"[{exp['name']}] rounds={rounds} | mode={mode} | clients={num_clients} | strategy={strategy}")

    # ----------------------------------------------------------
    # Local-Only baseline (no communication; ensemble at eval)
    # ----------------------------------------------------------
    if mode == "localonly":
        print(f"[{exp['name']}] Running Local-Only baseline...")
        client_states = []
        total_energy_j = 0.0

        for cid in range(num_clients):
            idxs = client_indices(num_nodes, num_clients, cid)
            dl = make_loader_for_client(PROC, cid, idxs, num_nodes, batch=train.get("batch_size", 64))

            # Fresh model per client
            m_i = _build_model(exp["model_type"], num_nodes, train.get("hidden_size", 64), device=device)
            state_i, _ = train_local(m_i, dl, device, epochs=train.get("local_epochs", 1), lr=train.get("lr", 1e-3))
            m_i.load_state_dict(state_i)
            client_states.append(state_i)

            # Approximate per-client compute energy (no comms)
            cpu_p, _ = cpu_mem_snapshot()
            e_i = compute_round_energy_j(round_secs=1.0, cpu_percent=cpu_p, sent_bytes=0, device_kind="edge")
            total_energy_j += e_i

        # Ensemble evaluation
        test_ld = DataLoader(
            TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(yte).float()),
            batch_size=128, shuffle=False
        )
        models = []
        for s in client_states:
            m = _build_model(exp["model_type"], num_nodes, train.get("hidden_size", 64), device=device)
            m.load_state_dict(s); m.eval(); models.append(m)

        @torch.no_grad()
        def ensemble_metrics(loader):
            maes, rmses = [], []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = [m(x) for m in models]
                yhat = torch.stack(preds, dim=0).mean(dim=0)
                maes.append(torch.mean(torch.abs(yhat - y)).item())
                rmses.append(torch.sqrt(torch.nn.functional.mse_loss(yhat, y)).item())
            return sum(maes)/len(maes), sum(rmses)/len(rmses)

        t_mae, t_rmse = ensemble_metrics(test_ld)
        with open(os.path.join(OUT, "results.txt"), "w") as f:
            f.write(f"TEST MAE: {t_mae:.6f}\n")
            f.write(f"TEST RMSE: {t_rmse:.6f}\n")
            f.write(f"TOTAL ENERGY_J: {total_energy_j:.6f}\n")
            f.write("TOTAL BYTES_MB: 0.000000\n")
            f.write("AVG CLIENTS PER ROUND: 0.000\n")
        print(f"[Local-Only] Results saved to {OUT}/results.txt")
        return

    # ----------------------------------------------------------
    # Initialize SCAFFOLD control variates if required
    # ----------------------------------------------------------
    if strategy == "scaffold":
        c_global = {k: torch.zeros_like(v) for k, v in global_state.items()}
        c_locals = [{k: torch.zeros_like(v) for k, v in global_state.items()} for _ in range(num_clients)]

    total_energy_j, total_bytes_mb, chosen_sum = 0.0, 0.0, 0

    for r in range(1, rounds + 1):
        t0 = time.time()
        chosen = pick_clients(states, alpha, min_energy, max_part) if mode == "aefl" else [str(i) for i in range(num_clients)]

        local_states, weights, scores, comm_kept = [], [], [], []
        deltas_c = []                  # list of (cid, delta_c) for SCAFFOLD
        bytes_per_client = []          # compressed payload per chosen client (in bytes)

        for cid in range(num_clients):
            cid_str = str(cid)
            if cid_str not in chosen:
                states[cid_str]["energy"] = min(1.0, states[cid_str]["energy"] + 0.02)
                continue

            idxs = client_indices(num_nodes, num_clients, cid)
            dl = make_loader_for_client(PROC, cid, idxs, num_nodes, batch=train.get("batch_size", 64))
            set_state(model, global_state)

            # --- Local training mode selector ---
            if strategy == "fedprox":
                upd_state, n_i = _train_local_fedprox(
                    model, dl, device, local_epochs, train.get("lr", 1e-3), mu, global_state
                )
            elif strategy == "scaffold":
                upd_state, n_i, delta_c = _train_local_scaffold(
                    model, dl, device, local_epochs, train.get("lr", 1e-3), c_global, c_locals[cid]
                )
                deltas_c.append((cid, delta_c))
            else:
                upd_state, n_i = train_local(model, dl, device, epochs=local_epochs, lr=train.get("lr", 1e-3))

            # ---- Privacy then Compression (payload reflects DP + compression) ----
            if dp_enabled and dp_sigma > 0:
                upd_state = dp_add_noise(upd_state, dp_sigma)

            kept_ratio = 1.0
            comp_bytes = None
            if comp_enabled:
                if comp_strategy in ("magnitude", "prune", "sparsify"):
                    upd_state, kept_ratio, comp_bytes = sparsify_state(upd_state, sparsity)
                elif comp_strategy in ("topk", "top-k", "tk"):
                    upd_state, kept_ratio, comp_bytes = topk_compress_state(upd_state, k_frac)
                elif comp_strategy in ("q8", "int8", "8bit", "quant8"):
                    upd_state, kept_ratio, comp_bytes = quantize8_state(upd_state)
                else:
                    # Unknown strategy -> fall back to dense size
                    comp_bytes = None

            if comp_bytes is None:
                comp_bytes = state_size_bytes(upd_state)  # dense fallback

            local_states.append(upd_state)
            weights.append(n_i)
            comm_kept.append(kept_ratio)
            bytes_per_client.append(comp_bytes)

            if mode == "aefl":
                states[cid_str]["energy"] = max(0.0, states[cid_str]["energy"] - 0.001 * kept_ratio)
                s_new = alpha * states[cid_str]["energy"] + (1 - alpha) * states[cid_str]["bandwidth"]
                scores.append(max(1e-6, s_new))

        # --- Aggregation logic (Periodic, SCAFFOLD) ---
        do_aggregate = True
        if strategy == "periodic":
            do_aggregate = (r % period_k == 0)

        if local_states and do_aggregate:
            if mode == "aefl":
                global_state = adaptive_average(local_states, weights, scores or [1.0] * len(weights))
            else:
                global_state = fedavg_average(local_states, weights)
            set_state(model, global_state)

            # Update control variates for SCAFFOLD
            if strategy == "scaffold" and deltas_c:
                # Average delta across participating clients
                mean_delta = {k: torch.zeros_like(v) for k, v in c_global.items()}
                for _, dc in deltas_c:
                    for k in mean_delta:
                        mean_delta[k] += dc[k] / len(deltas_c)
                for k in c_global:
                    c_global[k] += mean_delta[k]
                # Update each chosen client's local c_i using its own delta
                for cid, dc in deltas_c:
                    for k in c_locals[cid]:
                        c_locals[cid][k] += dc[k]
        elif local_states and not do_aggregate:
            print(f"Round {r:02d} | Periodic aggregation skipped (r mod {period_k} != 0)")

        # --- Validation, energy & bytes accounting ---
        v_mae, v_rmse, *_ = eval_model(model, eval_valid, device)
        secs = time.time() - t0
        cpu_p, mem_mb = cpu_mem_snapshot()

        # If no aggregation this round, no bytes transmitted.
        bytes_sent = sum(bytes_per_client) if do_aggregate else 0
        bytes_sent_mb = bytes_sent / (1024.0 * 1024.0)

        energy_j = compute_round_energy_j(secs, cpu_p, bytes_sent, device_kind="edge")

        total_energy_j += energy_j
        total_bytes_mb += bytes_sent_mb
        chosen_sum += len(chosen)

        print(f"Round {r:02d} | chosen {len(chosen)}/{num_clients} | val_MAE {v_mae:.4f} | val_RMSE {v_rmse:.4f}")
        log_row(
            csv_path,
            [
                r, len(chosen), f"{v_mae:.6f}", f"{v_rmse:.6f}",
                f"{secs:.3f}", f"{cpu_p:.2f}", f"{mem_mb:.2f}",
                f"{energy_j:.6f}", f"{bytes_sent_mb:.6f}",
                f"{(np.mean(comm_kept) if comm_kept else 1.0):.4f}"
            ]
        )

    t_mae, t_rmse, *_ = eval_model(model, eval_test, device)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(f"TEST MAE: {t_mae:.6f}\n")
        f.write(f"TEST RMSE: {t_rmse:.6f}\n")
        f.write(f"TOTAL ENERGY_J: {total_energy_j:.6f}\n")
        f.write(f"TOTAL BYTES_MB: {total_bytes_mb:.6f}\n")
        f.write(f"AVG CLIENTS PER ROUND: {chosen_sum / rounds:.3f}\n")

    torch.save(global_state, os.path.join(OUT, f"{exp['name'].lower().replace(' ', '_')}_state.pt"))
    print(f"Saved {OUT}/results.txt and {csv_path}")

