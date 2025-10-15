"""
Appends per-round metrics to a CSV file.
"""
import os
import csv

def ensure_dir(path):
    """Creates directory if missing."""
    os.makedirs(path, exist_ok=True)

def init_csv(path, header):
    """Initializes CSV with a header if file is new."""
    ensure_dir(os.path.dirname(path))
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)

def log_row(path, row):
    """Appends a single row to CSV."""
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)
