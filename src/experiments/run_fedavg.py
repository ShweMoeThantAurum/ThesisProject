"""
Runs the standard FedAvg baseline using the unified federation engine.
"""

from src.core.federation import run_federated


def run(config_path="configs/fedavg_sz.yaml"):
    """Executes FedAvg baseline for a selected dataset."""
    run_federated(config_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run FedAvg baseline")
    p.add_argument(
        "--config",
        type=str,
        default="configs/fedavg_sz.yaml",
        help=(
            "Path to FedAvg config file. Examples:\n"
            "  configs/fedavg_sz.yaml\n"
            "  configs/fedavg_los.yaml\n"
            "  configs/fedavg_pems08.yaml"
        ),
    )
    args = p.parse_args()
    run(args.config)
