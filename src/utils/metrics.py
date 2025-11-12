"""
Computes evaluation metrics for traffic forecasting models.
"""

import torch
import torch.nn.functional as F


def mae(yhat, y): return torch.mean(torch.abs(yhat - y)).item()
def rmse(yhat, y): return torch.sqrt(F.mse_loss(yhat, y)).item()

def mape(yhat, y):
    eps = 1e-6
    return torch.mean(torch.abs((yhat - y) / (y + eps))).item()

def smape(yhat, y):
    eps = 1e-6
    return torch.mean(2.0 * torch.abs(yhat - y) /
                      (torch.abs(yhat) + torch.abs(y) + eps)).item()

def r2_score(yhat, y):
    y_mean = torch.mean(y)
    ss_tot = torch.sum((y - y_mean) ** 2)
    ss_res = torch.sum((y - yhat) ** 2)
    return (1 - ss_res / (ss_tot + 1e-8)).item()


@torch.no_grad()
def eval_loader_metrics(model, loader, device):
    """Computes MAE, RMSE, MAPE, sMAPE, and R² over a DataLoader."""
    model.eval()
    all_mae, all_rmse, all_mape, all_smape, all_r2 = [], [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        all_mae.append(mae(yhat, y))
        all_rmse.append(rmse(yhat, y))
        all_mape.append(mape(yhat, y))
        all_smape.append(smape(yhat, y))
        all_r2.append(r2_score(yhat, y))
    return (
        sum(all_mae) / len(all_mae),
        sum(all_rmse) / len(all_rmse),
        sum(all_mape) / len(all_mape),
        sum(all_smape) / len(all_smape),
        sum(all_r2) / len(all_r2),
    )
