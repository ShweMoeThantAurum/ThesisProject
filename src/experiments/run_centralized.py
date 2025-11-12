"""
Runs the centralized (non-FL) baseline.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import WindowedArray
from src.models.simple_gru import SimpleGRU
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.metrics import eval_loader_metrics
from src.utils.run_logger import init_csv, log_row
from src.core.energy import cpu_mem_snapshot, compute_round_energy_j


def run(config_path="configs/centralized_sz.yaml"):
    cfg = load_config(config_path)
    set_global_seed(cfg["experiment"]["seed"])

    exp = cfg["experiment"]
    train = cfg["training"]
    eva = cfg["evaluation"]

    PROC = exp["proc_dir"]
    OUT = exp["output_dir"]
    os.makedirs(OUT, exist_ok=True)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        and eva.get("device_preference") == "mps"
        else "cpu"
    )

    # datasets
    train_ds = WindowedArray(f"{PROC}/X_train.npy", f"{PROC}/y_train.npy")
    val_ds = WindowedArray(f"{PROC}/X_valid.npy", f"{PROC}/y_valid.npy")
    test_ds = WindowedArray(f"{PROC}/X_test.npy", f"{PROC}/y_test.npy")

    train_ld = DataLoader(train_ds, batch_size=train["batch_size"], shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=128)
    test_ld = DataLoader(test_ds, batch_size=128)

    # model
    num_nodes = train_ds[0][0].shape[-1]
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=train["hidden_size"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train["lr"])
    loss_fn = nn.MSELoss()

    csv_path = os.path.join(OUT, "round_log.csv")
    init_csv(
        csv_path,
        ["epoch", "val_mae", "val_rmse", "secs", "cpu_percent", "mem_mb", "energy_j"],
    )

    print(f"[Centralized] epochs={train['epochs']} | device={device}")

    total_energy_j = 0.0

    # training loop
    for ep in range(1, train["epochs"] + 1):
        t0 = time.time()
        model.train()

        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        # validation
        v_mae, v_rmse, *_ = eval_loader_metrics(model, val_ld, device)

        secs = time.time() - t0
        cpu_p, mem_mb = cpu_mem_snapshot()

        e_j = compute_round_energy_j(secs, cpu_p, 0, "edge")
        total_energy_j += e_j

        log_row(
            csv_path,
            [
                ep,
                f"{v_mae:.6f}",
                f"{v_rmse:.6f}",
                f"{secs:.3f}",
                f"{cpu_p:.2f}",
                f"{mem_mb:.2f}",
                f"{e_j:.6f}",
            ],
        )

        print(f"Epoch {ep:02d} | MAE={v_mae:.4f} | RMSE={v_rmse:.4f}")

    # final test
    t_mae, t_rmse, *_ = eval_loader_metrics(model, test_ld, device)

    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(f"TEST MAE: {t_mae:.6f}\n")
        f.write(f"TEST RMSE: {t_rmse:.6f}\n")
        f.write(f"TOTAL ENERGY_J: {total_energy_j:.6f}\n")

    print("Centralized baseline completed.")
