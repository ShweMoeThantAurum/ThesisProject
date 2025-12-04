"""
Main server orchestration loop for adaptive federated learning.

Coordinates:
- S3 cleanup
- Global model initialisation
- FL rounds
- Aggregation
- Final evaluation
- Summary generation
"""

import os
import time
import torch

from src.fl.logger import log_event
from src.fl.utils import get_proc_dir, get_hidden_size, get_fl_rounds
from src.fl.server.init import (
    clear_round_data,
    infer_num_nodes,
    init_global_model,
    store_global_model,
)
from src.fl.server.s3 import load_client_update, load_round_metadata
from src.fl.server.selection import select_all_clients, select_clients_aefl
from src.fl.server.aggregation import (
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_aefl,
)
from src.fl.server.evaluate import evaluate_final_model
from src.fl.server.summary import generate_cloud_summary
from src.fl.server.modes import get_mode, is_aefl, is_fedavg, is_fedprox

ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]

PROC_DIR = get_proc_dir()
HIDDEN_SIZE = get_hidden_size()
FL_MODE = get_mode()
FL_ROUNDS = get_fl_rounds()

print(f"[SERVER] CONFIG: mode={FL_MODE.upper()}, rounds={FL_ROUNDS}, hidden={HIDDEN_SIZE}")


def main():
    """Run the main federated learning training loop on the server."""
    print(f"[SERVER] Starting server | mode={FL_MODE.upper()}")

    clear_round_data()

    num_nodes = infer_num_nodes(PROC_DIR)
    global_state = init_global_model(num_nodes, hidden=HIDDEN_SIZE)

    store_global_model(global_state, round_id=1)

    # FL rounds
    for r in range(1, FL_ROUNDS + 1):
        print(f"\n========== ROUND {r} ==========")

        # ===== CLIENT SELECTION =====
        if is_aefl(FL_MODE) and r > 1:
            prev_meta = load_round_metadata(r - 1)
            chosen, scores = select_clients_aefl(prev_meta, ROLES)
        else:
            chosen = select_all_clients(ROLES)
            scores = {r: 1.0 for r in chosen}

        print(f"[SERVER] Selected clients: {chosen}")

        updates = {}
        start_wait = time.time()
        timeout = 300

        print(f"[SERVER] Waiting for updates for round {r}...")

        while len(updates) < len(chosen):
            for role in chosen:
                if role not in updates:
                    upd = load_client_update(r, role)
                    if upd is not None:
                        updates[role] = upd
                        print(f"[SERVER] Received {role} ({len(updates)}/{len(chosen)})")

            if len(updates) == len(chosen):
                break

            if time.time() - start_wait > timeout:
                print("[SERVER] WARNING: Timeout waiting for updates.")
                break

            time.sleep(2)

        # ===== AGGREGATION =====
        start_aggr = time.time()

        if is_aefl(FL_MODE):
            global_state = aggregate_aefl(updates, scores)
            mode_label = "AEFL"
        elif is_fedavg(FL_MODE):
            global_state = aggregate_fedavg(updates)
            mode_label = "FedAvg"
        elif is_fedprox(FL_MODE):
            global_state = aggregate_fedprox(updates)
            mode_label = "FedProx"

        else:
            global_state = aggregate_fedavg(updates)
            mode_label = FL_MODE

        aggr_time = time.time() - start_aggr
        print(f"[SERVER] Aggregation complete | mode={mode_label}, time={aggr_time:.3f}s")

        # Store next round global
        next_round = r + 1
        if next_round <= FL_ROUNDS:
            store_global_model(global_state, next_round)

    # ==========================
    # FINAL EVALUATION
    # ==========================
    metrics = evaluate_final_model(global_state, PROC_DIR, num_nodes, hidden_size)

    print("\n[SERVER] Final Evaluation (TEST SET):")
    for k, v in metrics.items():
        print(f" {k} = {v:.6f}")

    generate_cloud_summary(metrics, fl_rounds, mode.upper())

    # ---------- NEW: Aggregate client energy ----------
    from src.fl.server.energy import aggregate_energy_logs
    aggregate_energy_logs()

    print(f"[SERVER] Training finished after {fl_rounds} rounds.")


if __name__ == "__main__":
    main()
