"""
heat_common.py - Shared utilities for Parallel 2D Heat Diffusion

Implements:
- Grid initialization and Dirichlet boundary conditions with corner convention (columns override rows)
- Hotspot application
- Shared CLI argument parser
- CSV logging adhering strictly to the required schema
- Checksum computation
"""

import argparse
import csv
import datetime
import os
import numpy as np

# Dirichlet boundary temperatures
TOP_TEMP = 100.0
BOTTOM_TEMP = 0.0
LEFT_TEMP = 60.0
RIGHT_TEMP = 0.0
HOTSPOT_TEMP = 100.0

CSV_HEADER = [
    "timestamp", "target", "version", "kernel", "N", "p", "steps",
    "check_every", "rep", "t_total", "t_compute_max", "t_compute_mean",
    "t_halo_max", "t_halo_mean", "t_reduce_max", "t_reduce_mean",
    "t_plot", "iterations", "final_diff", "checksum",
    "oversubscribed", "threading_layer", "notes"
]

def apply_boundaries(grid):
    """
    Apply Dirichlet boundary conditions in-place on an N x N global grid.
    Corner convention: columns override rows.
    Top-left = 60, top-right = 0, bottom-left = 60, bottom-right = 0.
    """
    grid[0, :] = TOP_TEMP
    grid[-1, :] = BOTTOM_TEMP
    grid[:, 0] = LEFT_TEMP
    grid[:, -1] = RIGHT_TEMP
    return grid

def apply_hotspot_global(grid, N):
    """
    Apply circular hotspot held at 100 in the center of an N x N grid.
    """
    ci, cj = N // 2, N // 2
    radius = max(2, N // 8)
    r2 = radius * radius
    # Bounding box for efficiency
    i_min = max(1, ci - radius)
    i_max = min(N - 1, ci + radius + 1)
    j_min = max(1, cj - radius)
    j_max = min(N - 1, cj + radius + 1)
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            if (i - ci) ** 2 + (j - cj) ** 2 <= r2:
                grid[i, j] = HOTSPOT_TEMP
    return grid

def apply_hotspot_local(local_grid, N, rank, rows):
    """
    Apply circular hotspot to a local rank slice (rows + 2, N) where local row 1 is global row start_row.
    """
    ci, cj = N // 2, N // 2
    radius = max(2, N // 8)
    r2 = radius * radius
    start_row = rank * rows

    # Check if hotspot overlaps this rank's interior rows
    local_i_start = 1
    local_i_end = rows + 1
    for li in range(local_i_start, local_i_end):
        gi = start_row + (li - 1)
        if abs(gi - ci) <= radius:
            for j in range(max(1, cj - radius), min(N - 1, cj + radius + 1)):
                if (gi - ci) ** 2 + (j - cj) ** 2 <= r2:
                    local_grid[li, j] = HOTSPOT_TEMP
    return local_grid

def init_grid(N, hotspot=False):
    """
    Initialize an N x N global grid with zero interior, fixed boundaries, and optional hotspot.
    """
    grid = np.zeros((N, N), dtype=np.float64, order="C")
    apply_boundaries(grid)
    if hotspot:
        apply_hotspot_global(grid, N)
    return grid

def compute_checksum(grid):
    """
    Compute reproducible checksum of the entire grid (float64 sum).
    """
    return float(np.sum(grid))

def get_cpu_counts():
    """
    Return (physical_cores, logical_cores) used only to flag oversubscribed runs.
    Falls back to an os.cpu_count() estimate when psutil is not installed; the
    flag is a measurement hint, not a scheduling guarantee.
    """
    logical = os.cpu_count() or 1
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except ImportError:
        physical = max(1, logical // 2)
    return physical, logical


def get_base_parser():
    """
    Create standard argument parser matching instructions.md.
    """
    parser = argparse.ArgumentParser(description="Parallel 2D Heat Diffusion Solver")
    parser.add_argument("--N", type=int, default=512, help="Grid size (N x N)")
    parser.add_argument("--steps", type=int, default=2000, help="Number of time steps")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance on max absolute change")
    parser.add_argument("--check-every", type=int, default=50, help="Check convergence every k steps")
    parser.add_argument("--kernel", choices=["numpy", "numba"], default="numba", help="Stencil update kernel")
    parser.add_argument("--out", type=str, default="results/raw/scaling_local.csv", help="Output CSV path")
    parser.add_argument("--save-every", type=int, default=0, help="Save grid frame every S steps (0 to disable)")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions to run")
    parser.add_argument("--target", type=str, default="local", choices=["local", "docker", "vm", "lan"], help="Deployment target")
    parser.add_argument("--hotspot", action="store_true", help="Enable central hot spot held at 100")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test mode (small N, few steps, 1 rep)")
    return parser

def append_result_row(csv_path, row_dict):
    """
    Append a benchmark result row to the given CSV file. Creates header if file doesn't exist.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        # Fill missing fields with defaults
        complete_row = {k: row_dict.get(k, "") for k in CSV_HEADER}
        if not complete_row["timestamp"]:
            complete_row["timestamp"] = datetime.datetime.now().isoformat()
        writer.writerow(complete_row)

