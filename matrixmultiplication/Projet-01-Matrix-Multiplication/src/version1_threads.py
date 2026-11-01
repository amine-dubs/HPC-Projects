"""
Version 1: Shared-Memory Parallel Matrix Multiplication (OpenMP-style)
High-Performance Computing (HPC) Project

This module implements multi-threaded matrix multiplication using Numba's `prange`,
which maps directly to an OpenMP-style parallel loop across CPU cores.

Features:
1. Dynamic thread control via numba.set_num_threads(p).
2. Naive parallel row distribution: parallelizing outer loop over rows of C.
3. Cache-blocked (tiled) 2D/3D parallel variant to optimize L1/L2/L3 cache residency.
4. Numerical validation using np.allclose against serial baseline.
5. Speedup S = T_1 / T_p and Efficiency E = S / p computation.
"""

import sys
import os
import time
import argparse
import numpy as np
import numba
from numba import njit, prange

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import generate_matrices, compute_reference, verify_result, calculate_gflops, get_matrix_memory_mb


@njit(parallel=True, fastmath=False)
def matmul_threads_naive(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    """
    Standard parallel matrix multiplication over rows of C using Numba prange (OpenMP parallel-for).
    Loop order: i (parallel) -> k (serial) -> j (serial, stride-1 in B and C).
    """
    n, m = A.shape
    p = B.shape[1]
    for i in prange(n):
        for k in range(m):
            a = A[i, k]
            for j in range(p):
                C[i, j] += a * B[k, j]


@njit(parallel=True, fastmath=False)
def matmul_threads_blocked(A: np.ndarray, B: np.ndarray, C: np.ndarray, block_size: int = 64) -> None:
    """
    Cache-blocked (tiled) parallel matrix multiplication.
    Decomposes the matrices into sub-blocks of size (block_size x block_size) that fit
    completely inside the CPU L1/L2 cache, drastically reducing memory bandwidth contention.
    Iterates over tile indices with unit-stride prange to satisfy Numba parallel loop constraints.
    """
    n, m = A.shape
    p = B.shape[1]
    num_blocks_i = (n + block_size - 1) // block_size
    num_blocks_k = (m + block_size - 1) // block_size
    num_blocks_j = (p + block_size - 1) // block_size

    for bi in prange(num_blocks_i):
        ii = bi * block_size
        i_end = min(ii + block_size, n)
        for bk in range(num_blocks_k):
            kk = bk * block_size
            k_end = min(kk + block_size, m)
            for bj in range(num_blocks_j):
                jj = bj * block_size
                j_end = min(jj + block_size, p)
                for i in range(ii, i_end):
                    for k in range(kk, k_end):
                        a = A[i, k]
                        for j in range(jj, j_end):
                            C[i, j] += a * B[k, j]


def warmup_threads(n_warmup: int = 64) -> None:
    """
    Compile parallel Numba functions with small matrices before running benchmarks.
    """
    A_w = np.ones((n_warmup, n_warmup), dtype=np.float64)
    B_w = np.ones((n_warmup, n_warmup), dtype=np.float64)
    C_w = np.zeros((n_warmup, n_warmup), dtype=np.float64)
    matmul_threads_naive(A_w, B_w, C_w)
    C_w.fill(0.0)
    matmul_threads_blocked(A_w, B_w, C_w, block_size=16)


def run_single_thread_experiment(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                                num_threads: int, kernel_type: str = "naive",
                                block_size: int = 64) -> float:
    """
    Execute matrix multiplication with a specified thread count and return execution time in seconds.
    """
    numba.set_num_threads(num_threads)
    C.fill(0.0)
    t_start = time.perf_counter()
    if kernel_type == "blocked":
        matmul_threads_blocked(A, B, C, block_size=block_size)
    else:
        matmul_threads_naive(A, B, C)
    t_end = time.perf_counter()
    return t_end - t_start


def benchmark_threads(n: int, thread_counts: list = [1, 2, 4, 8], repeats: int = 3,
                      kernel_type: str = "naive", block_size: int = 64) -> list:
    """
    Run multi-threaded scaling benchmark for matrix size n across various thread counts.
    Returns list of result dictionaries with Speedup and Efficiency.
    """
    print(f"\n=================================================================")
    print(f"[Version 1 - Threads ({kernel_type.upper()})] Benchmarking size n={n}")
    print(f"Threading layer: {numba.threading_layer()} | Logical cores: {os.cpu_count()}")
    print(f"=================================================================")

    warmup_threads()

    print(f"  Generating matrices A and B (n={n})...")
    A, B = generate_matrices(n, seed=42)
    C = np.zeros((n, n), dtype=np.float64)

    # Reference validation
    print("  Validating against NumPy reference...")
    C_ref = compute_reference(A, B)
    numba.set_num_threads(1)
    C.fill(0.0)
    if kernel_type == "blocked":
        matmul_threads_blocked(A, B, C, block_size=block_size)
    else:
        matmul_threads_naive(A, B, C)
    is_ok, max_abs, max_rel = verify_result(C, C_ref)
    print(f"  Verification result: {'PASSED' if is_ok else 'FAILED'} (Max abs err: {max_abs:.2e})")
    if not is_ok:
        raise ValueError(f"Threaded computation verification failed for n={n}!")

    results = []
    t1_median = None

    for p in thread_counts:
        times = []
        print(f"\n  --- Thread Count p = {p} ---")
        for r in range(repeats):
            elapsed = run_single_thread_experiment(A, B, C, num_threads=p,
                                                  kernel_type=kernel_type,
                                                  block_size=block_size)
            times.append(elapsed)
            gflops = calculate_gflops(n, elapsed)
            print(f"    Run {r+1}/{repeats}: {elapsed:.4f} s | {gflops:.3f} GFLOP/s")

        median_time = float(np.median(times))
        min_time = float(np.min(times))
        max_time = float(np.max(times))
        gflops = calculate_gflops(n, median_time)

        if p == 1 or t1_median is None:
            t1_median = median_time

        speedup = t1_median / median_time
        efficiency = speedup / p

        res = {
            "version": f"Version 1 ({kernel_type})",
            "n": n,
            "p": p,
            "median_time_s": median_time,
            "min_time_s": min_time,
            "max_time_s": max_time,
            "gflops": gflops,
            "speedup": speedup,
            "efficiency": efficiency,
            "kernel": kernel_type,
            "block_size": block_size if kernel_type == "blocked" else 0
        }
        results.append(res)
        print(f"  --> p={p}: Median={median_time:.4f}s | Speedup={speedup:.2f}x | Efficiency={efficiency*100:.1f}% | {gflops:.2f} GFLOP/s")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Version 1 Threaded Matrix Multiplication Benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048], help="Matrix dimensions n")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8], help="Thread counts p")
    parser.add_argument("--repeats", type=int, default=3, help="Repetitions per test")
    parser.add_argument("--kernel", type=str, choices=["naive", "blocked"], default="naive", help="Kernel type")
    parser.add_argument("--block-size", type=int, default=64, help="Block size for tiled kernel")
    args = parser.parse_args()

    for size in args.sizes:
        benchmark_threads(n=size, thread_counts=args.threads, repeats=args.repeats,
                          kernel_type=args.kernel, block_size=args.block_size)
