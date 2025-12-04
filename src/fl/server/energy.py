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
    """Load all energy logs from run_logs."""
    energy_entries = []
    for fname in os.listdir(LOG_DIR):
        if fname.startswith("energy_") and fname.endswith(".jsonl"):
            with open(os.path.join(LOG_DIR, fname), "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            energy_entries.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            pass
    return energy_entries


def aggregate_energy_logs():
    """Aggregate energy logs written by clients."""
    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = os.environ.get("FL_MODE", "aefl").strip().lower()
    variant = os.environ.get("VARIANT_ID", "").strip()

    out_dir = Path(f"outputs/{dataset}/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "energy_summary.json"

    entries = _load_energy_logs()

    if not entries:
        summary = {
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "error": "No energy logs found.",
        }
        out_path.write_text(json.dumps(summary, indent=4))
        return summary

    # Aggregate per-client totals
    client_totals = {}
    for e in entries:
        role = e["role"]
        client_totals.setdefault(role, 0.0)
        client_totals[role] += float(e["total_energy_j"])

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

    return summary
