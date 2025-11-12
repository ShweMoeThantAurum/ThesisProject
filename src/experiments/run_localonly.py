"""
Runs the Local-Only baseline (no communication).
"""

from src.core.federation import run_federated

def run(config_path="configs/localonly_sz.yaml"):
    """Executes Local-Only experiment."""
    run_federated(config_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run Local-Only baseline")
    p.add_argument(
        "--config",
        type=str,
        default="configs/localonly_sz.yaml",
        help=(
            "Path to Local-Only config file. Examples:\n"
            " configs/localonly_sz.yaml\n"
            " configs/localonly_los.yaml\n"
            " configs/localonly_pems08.yaml"
        ),
    )
    args = p.parse_args()
    run(args.config)
