"""
Common utilities for Matrix Multiplication Mini-Project.
High-Performance Computing (HPC) - Master / Student-Engineer Level.
"""

import os
import sys
import time
import json
import csv
from typing import Tuple, Dict, Any, Optional
import numpy as np


def generate_matrices(n: int, seed: int = 42, dtype=np.float64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two square n x n matrices A and B with deterministic pseudorandom numbers.
    Using double precision (float64) as specified in project requirements.
    """
    rng = np.random.default_rng(seed)
    A = rng.random((n, n), dtype=dtype)
    B = rng.random((n, n), dtype=dtype)
    return A, B


def compute_reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute reference result using NumPy BLAS (A @ B).
    """
    return A @ B


def verify_result(C_test: np.ndarray, C_ref: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[bool, float, float]:
    """
    Verify numerical tolerance using np.allclose and return max absolute / relative error.
    """
    is_close = bool(np.allclose(C_test, C_ref, rtol=rtol, atol=atol))
    abs_diff = np.abs(C_test - C_ref)
    max_abs_err = float(np.max(abs_diff))
    max_rel_err = float(np.max(abs_diff / (np.abs(C_ref) + 1e-12)))
    return is_close, max_abs_err, max_rel_err


def calculate_gflops(n: int, elapsed_time_seconds: float) -> float:
    """
    Calculate computational throughput in GFLOP/s.
    Dense square matrix multiplication requires 2 * n^3 operations (n^3 multiplications + n^3 additions).
    """
    if elapsed_time_seconds <= 0:
        return 0.0
    total_flops = 2.0 * (n ** 3)
    return (total_flops / 1e9) / elapsed_time_seconds


def get_matrix_memory_mb(n: int, dtype=np.float64) -> float:
    """
    Memory in megabytes for a single n x n matrix.
    """
    bytes_per_elem = np.dtype(dtype).itemsize
    return (n * n * bytes_per_elem) / (1024.0 * 1024.0)


def save_csv_results(filepath: str, fieldnames: list, rows: list) -> None:
    """
    Save experiment benchmark rows to CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[INFO] Saved {len(rows)} records to {filepath}")
