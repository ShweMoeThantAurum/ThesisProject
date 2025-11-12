"""
Runs AEFL experiments across multiple datasets.
"""

from src.core.federation import run_federated


def run(config_path="configs/aefl_sz.yaml"):
    """Executes AEFL (GRU) experiment for a selected dataset."""
    run_federated(config_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run AEFL (GRU) experiment")
    p.add_argument(
        "--config",
        type=str,
        default="configs/aefl_sz.yaml",
        help=(
            "Path to AEFL config file. Examples:\n"
            "  configs/aefl_sz.yaml\n"
            "  configs/aefl_los.yaml\n"
            "  configs/aefl_pems08.yaml"
        ),
    )
    args = p.parse_args()
    run(args.config)
