"""
Server-side energy aggregation utilities.

Reads all client energy logs and produces:
 - aggregated per-client totals
 - global totals
 - mean energy per client
 - dataset/mode/variant metadata

Output is saved as:
   outputs/<dataset>/<mode>/energy_summary.json
"""

import os
import json
from pathlib import Path
from src.fl.logger import LOG_DIR


def _load_energy_logs():
    """Load all per-round energy logs from run_logs (energy_<role>.jsonl)."""
    energy_entries = []
    if not os.path.exists(LOG_DIR):
        return energy_entries

    for fname in os.listdir(LOG_DIR):
        # Per-client per-round logs are stored as energy_<role>.jsonl
        if fname.startswith("energy_") and fname.endswith(".jsonl"):
            path = os.path.join(LOG_DIR, fname)
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            energy_entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            # Skip malformed lines but keep going
                            pass
            except FileNotFoundError:
                # File may have been removed between listdir and open
                continue

    return energy_entries


def aggregate_energy_logs():
    """
    Aggregate energy logs written by clients into a single summary.

    This function filters logs by the current dataset/mode/variant to avoid
    mixing results from different runs when logs are not cleared.
    """
    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = os.environ.get("FL_MODE", "aefl").strip().lower()
    variant = os.environ.get("VARIANT_ID", "").strip()

    out_dir = Path(f"outputs/{dataset}/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "energy_summary.json"

    entries = _load_energy_logs()

    # Filter by dataset/mode/variant in case multiple runs share the same log dir
    filtered = [
        e
        for e in entries
        if e.get("dataset", "").lower() == dataset
        and str(e.get("mode", "")).strip().lower() == mode
        and str(e.get("variant", "")).strip() == variant
    ]

    if not filtered:
        summary = {
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "error": "No energy logs found for this dataset/mode/variant.",
            "clients": [],
            "client_breakdown": {},
            "total_energy_j": 0.0,
            "mean_energy_per_client_j": 0.0,
        }
        out_path.write_text(json.dumps(summary, indent=4))
        print(f"[SERVER] No energy logs for dataset={dataset}, mode={mode}, variant={variant}")
        return summary

    # Aggregate per-client totals across rounds
    client_totals = {}
    for e in filtered:
        role = e.get("role", "unknown")
        client_totals.setdefault(role, 0.0)
        client_totals[role] += float(e.get("total_energy_j", 0.0))

    total_energy = sum(client_totals.values())
    mean_energy = total_energy / max(len(client_totals), 1)

    summary = {
        "dataset": dataset,
        "mode": mode,
        "variant": variant,
        "clients": list(client_totals.keys()),
        "client_breakdown": client_totals,
        "total_energy_j": total_energy,
        "mean_energy_per_client_j": mean_energy,
    }

    out_path.write_text(json.dumps(summary, indent=4))
    print(f"[SERVER] Saved energy summary → {out_path}")
    print(
        f"[SERVER] Energy totals | dataset={dataset}, mode={mode}, variant={variant}, "
        f"total={total_energy:.4f} J, mean_per_client={mean_energy:.4f} J"
    )

    return summary

