"""
Trains a centralized GRU baseline on SZ windowed data with CSV logging.
"""
import os, math, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.simple_gru import SimpleGRU
from src.data.dataset_sz import WindowedArray
from src.utils.run_logger import init_csv, log_row
from src.utils.profiler_utils import cpu_mem_snapshot, energy_proxy

PROC = "data/processed/sz/prepared"
OUT = "outputs/centralized_sz"
CSV_PATH = os.path.join(OUT, "round_log.csv")
CSV_HEADER = [
    "epoch", "chosen", "val_mae", "val_rmse",
    "avg_comm_ratio", "cpu_percent", "mem_mb",
    "round_secs", "energy_proxy"
]
os.makedirs(OUT, exist_ok=True)

def mae(yhat, y):
    """Computes Mean Absolute Error."""
    return (yhat - y).abs().mean().item()

def rmse(yhat, y):
    """Computes Root Mean Squared Error."""
    return math.sqrt(((yhat - y) ** 2).mean().item())

def train_one_epoch(model, loader, opt, loss_fn, device):
    """Trains for one epoch."""
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

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

def run():
    """Runs centralized training, logs per-epoch CSV, saves results."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_ds = WindowedArray(f"{PROC}/X_train.npy", f"{PROC}/y_train.npy")
    valid_ds = WindowedArray(f"{PROC}/X_valid.npy", f"{PROC}/y_valid.npy")
    test_ds  = WindowedArray(f"{PROC}/X_test.npy",  f"{PROC}/y_test.npy")

    num_nodes = train_ds[0][0].shape[-1]
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=64).to(device)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
    valid_ld = DataLoader(valid_ds, batch_size=128, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_mae, best_state = float("inf"), None

    init_csv(CSV_PATH, CSV_HEADER)

    epochs = 10
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_ld, opt, loss_fn, device)
        v_mae, v_rmse = evaluate(model, valid_ld, device)

        round_secs = time.time() - t0
        cpu_p, mem_mb = cpu_mem_snapshot()
        avg_comm = 1.0
        eproxy = energy_proxy(round_secs, cpu_p, avg_comm)

        print(
            f"Epoch {epoch+1:02d} | train_loss {train_loss:.4f} "
            f"| val_MAE {v_mae:.4f} | val_RMSE {v_rmse:.4f} "
            f"| cpu {cpu_p:.1f}% | mem {mem_mb:.1f}MB "
            f"| {round_secs:.1f}s | eproxy {eproxy:.4f}"
        )

        log_row(CSV_PATH, [
            epoch + 1, 1, f"{v_mae:.6f}", f"{v_rmse:.6f}",
            f"{avg_comm:.4f}", f"{cpu_p:.2f}", f"{mem_mb:.2f}",
            f"{round_secs:.3f}", f"{eproxy:.6f}"
        ])

        if v_mae < best_mae:
            best_mae = v_mae
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    te_mae, te_rmse = evaluate(model, test_ld, device)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(f"TEST MAE: {te_mae:.6f}\nTEST RMSE: {te_rmse:.6f}\n")
    torch.save(model.state_dict(), os.path.join(OUT, "simple_gru.pt"))
    print(f"Saved {OUT}/results.txt, simple_gru.pt and {CSV_PATH}")

if __name__ == "__main__":
    run()
