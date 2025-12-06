"""
Main client loop with adaptive local epochs and clean energy logging.

For AEFL:
 - high-power roles (camera, bus) use fewer local epochs to reduce energy
 - per-round energy is passed into metadata so the server can do
   energy-aware client selection.
"""

import os
import torch
import numpy as np

from src.models.simple_gru import SimpleGRU
from src.fl.logger import log_event
from src.fl.utils import (
    get_proc_dir,
    get_fl_rounds,
    get_batch_size,
    get_local_epochs,
    get_lr,
    get_hidden_size,
)

from src.fl.client.cleanup import cleanup_local_tmp
from src.fl.client.data import load_local_data
from src.fl.client.train import train_one_round
from src.fl.client.energy import estimate_round_energy
from src.fl.client.meta import build_round_metadata
from src.fl.client.modes import get_client_mode, client_allows_training
from src.fl.client.s3 import download_global, upload_processed_update, upload_metadata
from src.fl.client.privacy import maybe_add_dp_noise
from src.fl.client.compression import maybe_compress


# ---------------------------------------------------------------------------
# Helper: adaptive local epochs for AEFL
# ---------------------------------------------------------------------------
def get_effective_local_epochs(base_epochs: int, mode: str, role: str) -> int:
    """
    Return the number of local epochs this client should run.

    For AEFL we reduce local epochs for high-power devices to save energy.
    For FedAvg / FedProx we keep the configured value.
    """
    if base_epochs <= 0:
        return 0

    if mode != "aefl":
        # Baselines: no adaptation
        return base_epochs

    # Simple heuristic:
    #  - camera, bus: heavy nodes → half the epochs (at least 1)
    #  - others: unchanged
    if role in ("camera", "bus"):
        eff = max(1, base_epochs // 2)
    else:
        eff = base_epochs

    return eff


# ---------------------------------------------------------------------------
# Main client entrypoint
# ---------------------------------------------------------------------------
def main():
    role = os.environ.get("CLIENT_ROLE", "roadside")
    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = get_client_mode()  # "aefl", "fedavg", "fedprox"
    variant = os.environ.get("VARIANT_ID", "").strip()

    fl_rounds = get_fl_rounds()
    proc_dir = get_proc_dir()
    batch_size = get_batch_size()
    base_local_epochs = get_local_epochs()
    lr = get_lr()
    hidden_size = get_hidden_size()

    device_power_watts = float(os.environ.get("DEVICE_POWER_WATTS", "3.5"))
    net_j_per_mb = float(os.environ.get("NET_J_PER_MB", "0.6"))

    dp_enabled = os.environ.get("DP_ENABLED", "false").lower() == "true"
    dp_sigma = float(os.environ.get("DP_SIGMA", "0.0"))
    compression_enabled = (
        os.environ.get("COMPRESSION_ENABLED", "false").lower() == "true"
    )
    compression_mode = os.environ.get("COMPRESSION_MODE", "").lower()

    # Load graph dimensions
    x_train_path = os.path.join(proc_dir, "X_train.npy")
    if not os.path.exists(x_train_path):
        raise FileNotFoundError("Missing X_train.npy – did you run preprocessing?")

    num_nodes = np.load(x_train_path).shape[-1]
    device = "cpu"

    print(
        f"[{role}] Client start | dataset={dataset}, mode={mode}, "
        f"variant='{variant}', nodes={num_nodes}, rounds={fl_rounds}, "
        f"base_local_epochs={base_local_epochs}"
    )

    log_event(
        "client_config.log",
        {
            "role": role,
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "num_nodes": num_nodes,
            "rounds": fl_rounds,
            "batch_size": batch_size,
            "base_local_epochs": base_local_epochs,
            "lr": lr,
            "hidden_size": hidden_size,
            "device_power_watts": device_power_watts,
            "net_j_per_mb": net_j_per_mb,
            "dp_enabled": dp_enabled,
            "dp_sigma": dp_sigma,
            "compression_enabled": compression_enabled,
            "compression_mode": compression_mode,
        },
    )

    cleanup_local_tmp(role)

    loader = load_local_data(
        proc_dir=proc_dir,
        role=role,
        num_nodes=num_nodes,
        batch_size=batch_size,
        local_epochs=base_local_epochs,
        lr=lr,
    )

    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden_size).to(device)
    total_energy_j = 0.0

    # ===================================
    # Federated rounds
    # ===================================
    for r in range(1, fl_rounds + 1):
        print(f"\n[{role}] ===== ROUND {r} =====")

        # ----------------------------------------------------
        # Download latest global model
        # ----------------------------------------------------
        global_path, dl_bytes = download_global(r, role)
        global_state = torch.load(global_path, map_location=device)
        model.load_state_dict(global_state)

        # ----------------------------------------------------
        # Adaptive local epochs (AEFL only)
        # ----------------------------------------------------
        effective_local_epochs = get_effective_local_epochs(
            base_local_epochs, mode, role
        )
        print(
            f"[{role}] Round {r}: effective_local_epochs={effective_local_epochs} "
            f"(base={base_local_epochs})"
        )

        # ----------------------------------------------------
        # Local training
        # ----------------------------------------------------
        if client_allows_training(mode) and effective_local_epochs > 0:
            prox_ref = (
                {k: v.clone().to(device) for k, v in global_state.items()}
                if mode == "fedprox"
                else None
            )
            updated_state, train_time, train_loss, train_samples = train_one_round(
                model=model,
                loader=loader,
                role=role,
                round_id=r,
                device=device,
                local_epochs=effective_local_epochs,
                lr=lr,
                mode=mode,
                global_state=prox_ref,
            )
        else:
            updated_state = {k: v.cpu() for k, v in model.state_dict().items()}
            train_time = 0.0
            train_loss = 0.0
            train_samples = 0

        # ----------------------------------------------------
        # DP + Compression
        # ----------------------------------------------------
        dp_state = maybe_add_dp_noise(updated_state)
        comp_state, kept_ratio, modeled_bytes = maybe_compress(dp_state)

        log_event(
            "client_update_processing.log",
            {
                "role": role,
                "round": r,
                "dataset": dataset,
                "mode": mode,
                "variant": variant,
                "effective_local_epochs": effective_local_epochs,
                "dp_enabled": dp_enabled,
                "dp_sigma": dp_sigma,
                "compression_enabled": compression_enabled,
                "compression_mode": compression_mode,
                "compression_kept_ratio": float(kept_ratio),
                "compression_modeled_bytes": int(modeled_bytes),
            },
        )

        # ----------------------------------------------------
        # Upload update
        # ----------------------------------------------------
        processed_bytes, up_latency = upload_processed_update(r, role, comp_state)

        # ----------------------------------------------------
        # Energy estimation (per round)
        # ----------------------------------------------------
        energy_record = estimate_round_energy(
            role=role,
            round_id=r,
            train_time_sec=train_time,
            download_bytes=dl_bytes,
            upload_bytes=processed_bytes,
            device_power_watts=device_power_watts,
            net_j_per_mb=net_j_per_mb,
            num_nodes=num_nodes,
            hidden_size=hidden_size,
        )
        total_energy_j += float(energy_record.get("total_energy_j", 0.0))

        # ----------------------------------------------------
        # Upload AEFL metadata (includes energy + bandwidth)
        # ----------------------------------------------------
        meta = build_round_metadata(
            role=role,
            round_id=r,
            energy_record=energy_record,
            train_loss=train_loss,
            train_samples=train_samples,
            update_bytes=processed_bytes,
            upload_latency_sec=up_latency,
        )
        upload_metadata(r, role, meta)

    # Final energy summary for this client
    log_event(
        "client_energy_summary.log",
        {
            "role": role,
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "rounds": fl_rounds,
            "total_energy_j": total_energy_j,
        },
    )

    print(
        f"[{role}] Finished {fl_rounds} rounds. "
        f"Total estimated energy={total_energy_j:.2f} J."
    )


if __name__ == "__main__":
    main()
