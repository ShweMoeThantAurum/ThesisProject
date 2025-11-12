"""
Handles CSV-based experiment logging.
"""

import os
import csv


def init_csv(path, header):
    """Initializes a CSV file with header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def log_row(path, row):
    """Appends a row to an existing CSV log."""
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)
