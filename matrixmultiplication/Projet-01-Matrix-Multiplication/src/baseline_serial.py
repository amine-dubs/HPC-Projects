"""
Version 0: Serial Baseline Matrix Multiplication
High-Performance Computing (HPC) Project

This module implements the serial baseline for dense matrix multiplication
using Python and Numba @njit compilation.

Memory Access Optimization Note:
Standard naive matmul uses loop order (i, j, k):
    for i: for j: for k: C[i, j] += A[i, k] * B[k, j]
In row-major (C-contiguous) arrays, B[k, j] changes row on every iteration of the inner loop k,
leading to a cache miss rate proportional to n.
Reordering loops to (i, k, j):
    for i: for k:
        a = A[i, k]
        for j: C[i, j] += a * B[k, j]
ensures that the inner loop traverses B[k, j] and C[i, j] contiguously in memory (stride-1 access),
drastically reducing cache misses and maximizing SIMD vectorization.
"""

import sys
import os
import time
import argparse
import numpy as np
from numba import njit

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import generate_matrices, compute_reference, verify_result, calculate_gflops, get_matrix_memory_mb


@njit(fastmath=False)
def matmul_serial_ikj(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    """
    Serial dense matrix multiplication with cache-friendly loop order (i, k, j).
    Computes C = A x B in-place.
    """
    n, m = A.shape
    p = B.shape[1]
    for i in range(n):
        for k in range(m):
            a = A[i, k]
            for j in range(p):
                C[i, j] += a * B[k, j]


@njit(fastmath=False)
def matmul_serial_ijk(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    """
    Naive serial matrix multiplication with (i, j, k) loop order.
    Used for comparative analysis to demonstrate the impact of cache misses.
    """
    n, m = A.shape
    p = B.shape[1]
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += A[i, k] * B[k, j]
            C[i, j] = s


def warmup_serial(n_warmup: int = 64) -> None:
    """
    Perform a warm-up call to compile the Numba functions before any timing is recorded.
    This eliminates JIT compilation overhead from experimental benchmarks.
    """
    A_w = np.ones((n_warmup, n_warmup), dtype=np.float64)
    B_w = np.ones((n_warmup, n_warmup), dtype=np.float64)
    C_w = np.zeros((n_warmup, n_warmup), dtype=np.float64)
    matmul_serial_ikj(A_w, B_w, C_w)
    C_w.fill(0.0)
    matmul_serial_ijk(A_w, B_w, C_w)


def benchmark_serial(n: int, repeats: int = 3, verify: bool = True) -> dict:
    """
    Run serial benchmark for matrix size n with specified repetitions.
    Returns median execution time and GFLOP/s.
    """
    print(f"\n[Version 0 - Serial] Benchmarking size n={n} (Matrix memory: {get_matrix_memory_mb(n):.2f} MB)...")
    
    # 1. Warm-up JIT compiler
    warmup_serial()

    # 2. Generate matrices outside timed section
    print(f"  Generating matrices A and B (n={n})...")
    A, B = generate_matrices(n, seed=42)
    C = np.zeros((n, n), dtype=np.float64)

    # 3. Optional correctness verification against NumPy BLAS
    if verify:
        print("  Computing reference C_ref = A @ B via NumPy BLAS...")
        C_ref = compute_reference(A, B)
        C.fill(0.0)
        matmul_serial_ikj(A, B, C)
        is_ok, max_abs, max_rel = verify_result(C, C_ref)
        print(f"  Verification result: {'PASSED' if is_ok else 'FAILED'} (Max abs err: {max_abs:.2e}, Max rel err: {max_rel:.2e})")
        if not is_ok:
            raise ValueError(f"Serial computation verification failed for n={n}!")

    # 4. Timed executions (repeats)
    times = []
    for r in range(repeats):
        C.fill(0.0)
        t_start = time.perf_counter()
        matmul_serial_ikj(A, B, C)
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        times.append(elapsed)
        gflops = calculate_gflops(n, elapsed)
        print(f"  Run {r+1}/{repeats}: {elapsed:.4f} s | {gflops:.3f} GFLOP/s")

    median_time = float(np.median(times))
    min_time = float(np.min(times))
    max_time = float(np.max(times))
    median_gflops = calculate_gflops(n, median_time)

    result = {
        "version": "Version 0 (Serial)",
        "n": n,
        "p": 1,
        "median_time_s": median_time,
        "min_time_s": min_time,
        "max_time_s": max_time,
        "gflops": median_gflops,
        "memory_mb": get_matrix_memory_mb(n) * 3,  # A, B, C
        "verified": True
    }
    print(f"  --> Median Runtime: {median_time:.4f} s | Median Throughput: {median_gflops:.3f} GFLOP/s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Version 0 Serial Matrix Multiplication Benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048], help="Matrix dimensions n (default: 1024 2048)")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions per size (default: 3)")
    args = parser.parse_args()

    for size in args.sizes:
        benchmark_serial(n=size, repeats=args.repeats, verify=True)
