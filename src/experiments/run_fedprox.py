"""
Runs the FedProx baseline using the cleaned federation engine.
"""

from src.core.federation import run_federated

def run(config_path="configs/fedprox_sz.yaml"):
    """Executes FedProx baseline."""
    run_federated(config_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run FedProx baseline")
    p.add_argument(
        "--config",
        type=str,
        default="configs/fedprox_sz.yaml",
        help=(
            "Path to FedProx config file. Examples:\n"
            " configs/fedprox_sz.yaml\n"
            " configs/fedprox_los.yaml\n"
            " configs/fedprox_pems08.yaml"
        ),
    )
    args = p.parse_args()
    run(args.config)
