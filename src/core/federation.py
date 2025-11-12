"""
Main Federated Learning engine — cleaned version.
Supports:
    - AEFL (adaptive energy-aware federated learning)
    - FedAvg
    - FedProx
    - Local-Only
All other algorithms removed for clarity and reproducibility.
"""

import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.client import client_indices, make_loader_for_client, train_local
from src.core.aggregation import (
    fedavg_average,
    adaptive_average,
    clone_state,
    set_state,
)
from src.core.evaluator import (
    init_csv,
    log_row,
    make_eval_loader,
    eval_model,
)
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.privacy import dp_add_noise
from src.core.energy import cpu_mem_snapshot, state_size_bytes, compute_round_energy_j
from src.models.simple_gru import SimpleGRU


# ----------------------------------------------------------
# FedProx helper
# ----------------------------------------------------------
def _train_local_fedprox(model, loader, device, epochs, lr, mu, global_state):
    """FedProx: local update with proximal penalty."""
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

            # proximal term
            prox = 0.0
            for name, param in model.named_parameters():
                prox += torch.norm(param - global_params[name]) ** 2

            loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()
            n_samples += x.size(0)

    return {k: v.detach().clone() for k, v in model.state_dict().items()}, n_samples


# ----------------------------------------------------------
# Model builder
# ----------------------------------------------------------
def _build_model(model_type, num_nodes, hidden, device="cpu"):
    if model_type.lower() != "simplegru":
        raise ValueError("Only SimpleGRU is supported in this cleaned version.")
    return SimpleGRU(num_nodes=num_nodes, hidden_size=hidden).to(device)


# ----------------------------------------------------------
# Main Federated Procedure
# ----------------------------------------------------------
def run_federated(config_path):
    """Runs a federated experiment (AEFL, FedAvg, FedProx, Local-Only)."""

    cfg = load_config(config_path)
    set_global_seed(cfg["experiment"]["seed"])

    exp = cfg["experiment"]
    data_cfg = cfg["data"]
    fed = cfg["federation"]
    train = cfg["training"]
    priv = cfg["privacy"]
    eva = cfg["evaluation"]

    PROC = exp["proc_dir"]
    OUT = exp["output_dir"]
    os.makedirs(OUT, exist_ok=True)

    # device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        and eva.get("device_preference") == "mps"
        else "cpu"
    )

    # load metadata
    meta = json.load(open(os.path.join(PROC, "meta.json")))
    num_clients = meta["num_clients"]

    # load validation/test sets
    Xva = np.load(f"{PROC}/X_valid.npy")
    yva = np.load(f"{PROC}/y_valid.npy")
    Xte = np.load(f"{PROC}/X_test.npy")
    yte = np.load(f"{PROC}/y_test.npy")
    num_nodes = Xva.shape[-1]

    # build model
    model = _build_model(exp["model_type"], num_nodes, train["hidden_size"], device)
    global_state = clone_state(model.state_dict())

    eval_valid = make_eval_loader(Xva, yva)
    eval_test = make_eval_loader(Xte, yte)

    # simulate energy/bandwidth states
    rng = np.random.RandomState(exp["seed"])
    states = {
        str(i): {
            "energy": float(rng.uniform(0.4, 1.0)),
            "bandwidth": float(rng.uniform(0.4, 1.0)),
        }
        for i in range(num_clients)
    }

    # log file
    csv_path = os.path.join(OUT, "round_log.csv")
    init_csv(
        csv_path,
        [
            "round",
            "chosen",
            "val_mae",
            "val_rmse",
            "round_secs",
            "cpu_percent",
            "mem_mb",
            "energy_j",
            "bytes_sent_mb",
        ],
    )

    # federation params
    rounds = fed["rounds"]
    local_epochs = fed["local_epochs"]
    max_part = fed.get("max_participants", num_clients)
    min_energy = fed.get("min_energy", 0.0)
    alpha = fed.get("alpha", 0.5)
    mu = fed.get("mu", 0.0)  # FedProx
    mode = exp["mode"].lower()

    dp_enabled = priv.get("enabled", False)
    dp_sigma = priv.get("dp_sigma", 0.0)

    print(f"[{exp['name']}] mode={mode} | rounds={rounds}")

    # ----------------------------------------------------------
    # Local-Only baseline
    # ----------------------------------------------------------
    if mode == "localonly":
        print("[LocalOnly] Running local-only clients...")

        client_states = []
        total_energy_j = 0.0

        for cid in range(num_clients):
            idxs = client_indices(num_nodes, num_clients, cid)
            dl = make_loader_for_client(PROC, cid, idxs, num_nodes, train["batch_size"])

            m_i = _build_model(exp["model_type"], num_nodes, train["hidden_size"], device)
            s_i, _ = train_local(
                m_i, dl, device, epochs=train["local_epochs"], lr=train["lr"]
            )
            client_states.append(s_i)

            cpu_p, _ = cpu_mem_snapshot()
            total_energy_j += compute_round_energy_j(1.0, cpu_p, 0, "edge")

        # ensemble evaluation
        test_ld = DataLoader(
            TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(yte).float()),
            batch_size=128,
            shuffle=False,
        )

        @torch.no_grad()
        def ensemble_metrics(loader):
            maes, rmses = [], []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = []
                for s in client_states:
                    m = _build_model(exp["model_type"], num_nodes, train["hidden_size"])
                    m.load_state_dict(s)
                    m.eval()
                    preds.append(m(x).cpu())

                yhat = torch.stack(preds, dim=0).mean(dim=0)
                maes.append(torch.mean(torch.abs(yhat - y.cpu())).item())
                rmses.append(torch.sqrt(torch.nn.functional.mse_loss(yhat, y.cpu())).item())
            return np.mean(maes), np.mean(rmses)

        t_mae, t_rmse = ensemble_metrics(test_ld)

        with open(os.path.join(OUT, "results.txt"), "w") as f:
            f.write(f"TEST MAE: {t_mae:.6f}\n")
            f.write(f"TEST RMSE: {t_rmse:.6f}\n")
            f.write(f"TOTAL ENERGY_J: {total_energy_j:.6f}\n")
            f.write("TOTAL BYTES_MB: 0.000000\n")
            f.write("AVG CLIENTS PER ROUND: 0.000\n")

        print("[LocalOnly] Complete.")
        return

    # ----------------------------------------------------------
    # Federated Loop (FedAvg, FedProx, AEFL)
    # ----------------------------------------------------------
    total_energy_j = 0.0
    total_bytes_mb = 0.0
    chosen_sum = 0

    for r in range(1, rounds + 1):
        t0 = time.time()

        # AEFL: adaptive client selection
        if mode == "aefl":
            scored = [
                (cid, alpha * s["energy"] + (1 - alpha) * s["bandwidth"])
                for cid, s in states.items()
                if s["energy"] >= min_energy
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            chosen = [cid for cid, _ in scored[:max_part]]
        else:
            chosen = [str(i) for i in range(num_clients)]

        local_states = []
        weights = []
        bytes_per_client = []

        # local updates
        for cid_str in chosen:
            cid = int(cid_str)
            idxs = client_indices(num_nodes, num_clients, cid)
            dl = make_loader_for_client(PROC, cid, idxs, num_nodes, train["batch_size"])
            set_state(model, global_state)

            # training mode
            if mode == "fedprox":
                upd_state, n_i = _train_local_fedprox(
                    model, dl, device, local_epochs, train["lr"], mu, global_state
                )
            else:
                upd_state, n_i = train_local(
                    model, dl, device, epochs=local_epochs, lr=train["lr"]
                )

            # differential privacy
            if dp_enabled:
                upd_state = dp_add_noise(upd_state, dp_sigma)

            # payload size (no compression)
            comp_bytes = state_size_bytes(upd_state)

            local_states.append(upd_state)
            weights.append(n_i)
            bytes_per_client.append(comp_bytes)

            # AEFL energy decay
            if mode == "aefl":
                states[cid_str]["energy"] = max(
                    0.0, states[cid_str]["energy"] - 0.001
                )

        # aggregation
        if local_states:
            if mode == "aefl":
                scores = [alpha * states[cid]["energy"] + (1 - alpha) * states[cid]["bandwidth"] for cid in chosen]
                global_state = adaptive_average(local_states, weights, scores)
            else:
                global_state = fedavg_average(local_states, weights)

            set_state(model, global_state)

        # validation + logging
        v_mae, v_rmse, *_ = eval_model(model, eval_valid, device)
        secs = time.time() - t0
        cpu_p, mem_mb = cpu_mem_snapshot()

        bytes_sent = sum(bytes_per_client)
        energy_j = compute_round_energy_j(secs, cpu_p, bytes_sent, "edge")

        total_energy_j += energy_j
        total_bytes_mb += bytes_sent / (1024**2)
        chosen_sum += len(chosen)

        print(
            f"Round {r:02d} | chosen {len(chosen)}/{num_clients} | "
            f"MAE {v_mae:.4f} | RMSE {v_rmse:.4f}"
        )

        log_row(
            csv_path,
            [
                r,
                len(chosen),
                f"{v_mae:.6f}",
                f"{v_rmse:.6f}",
                f"{secs:.3f}",
                f"{cpu_p:.2f}",
                f"{mem_mb:.2f}",
                f"{energy_j:.6f}",
                f"{bytes_sent / (1024**2):.6f}",
            ],
        )

    # final test
    t_mae, t_rmse, *_ = eval_model(model, eval_test, device)

    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(f"TEST MAE: {t_mae:.6f}\n")
        f.write(f"TEST RMSE: {t_rmse:.6f}\n")
        f.write(f"TOTAL ENERGY_J: {total_energy_j:.6f}\n")
        f.write(f"TOTAL BYTES_MB: {total_bytes_mb:.6f}\n")
        f.write(f"AVG CLIENTS PER ROUND: {chosen_sum / rounds:.3f}\n")

    torch.save(
        global_state,
        os.path.join(OUT, f"{exp['name'].lower().replace(' ', '_')}_state.pt"),
    )

    print("Federated training complete.")
