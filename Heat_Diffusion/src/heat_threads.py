"""
heat_threads.py - Version 1: Shared-Memory Multithreading with Numba prange

Uses OpenMP threading layer (or fallback) via Numba parallel=True.
Runs with a configurable number of threads and records timing and efficiency.
"""

import os
import sys
import time
import numpy as np
import numba
from numba import njit, prange, set_num_threads, threading_layer
from heat_common import init_grid, apply_hotspot_global, compute_checksum, get_cpu_counts, get_base_parser, append_result_row

@njit(parallel=True, cache=True)
def jacobi_step_threads(u, v, N):
    """
    Perform one 5-point stencil Jacobi update using multithreaded row slices.
    """
    for i in prange(1, N - 1):
        for j in range(1, N - 1):
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

@njit(parallel=True, cache=True)
def compute_max_diff_threads(u, v, N):
    """
    Compute maximum absolute difference across interior cells safely using per-row buffer.
    """
    row_diffs = np.zeros(N, dtype=np.float64)
    for i in prange(1, N - 1):
        m_row = 0.0
        for j in range(1, N - 1):
            d = abs(v[i, j] - u[i, j])
            if d > m_row:
                m_row = d
        row_diffs[i] = m_row
    return np.max(row_diffs[1:-1])

def warmup_threads():
    """
    Warmup JIT compilation and parallel worker pool.
    """
    dummy_u = np.zeros((16, 16), dtype=np.float64, order="C")
    dummy_v = dummy_u.copy()
    jacobi_step_threads(dummy_u, dummy_v, 16)
    compute_max_diff_threads(dummy_u, dummy_v, 16)

def solve_numba_threads(N, steps=2000, tol=1e-4, check_every=50, hotspot=False):
    """
    Solve heat equation using Numba prange multithreading.
    Returns: (final_grid, elapsed_time, iterations, final_diff, threading_layer_name)
    """
    u = init_grid(N, hotspot=hotspot)
    v = u.copy()

    iterations = 0
    final_diff = 0.0
    current_grid = u

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        iterations = step
        jacobi_step_threads(u, v, N)
        if hotspot:
            apply_hotspot_global(v, N)

        current_grid = v

        if check_every > 0 and step % check_every == 0:
            final_diff = float(compute_max_diff_threads(u, v, N))
            if final_diff < tol:
                break

        u, v = v, u
        current_grid = u

    t1 = time.perf_counter()
    th_layer = threading_layer()
    return current_grid, (t1 - t0), iterations, final_diff, th_layer

def main():
    parser = get_base_parser()
    parser.add_argument("--threads", type=int, default=None, help="Number of threads (default: NUMBA_NUM_THREADS or all logical cores)")
    args = parser.parse_args()

    if args.quick:
        args.N = 64
        args.steps = 100
        args.reps = 1
        args.check_every = 10

    if args.threads is not None and args.threads > 0:
        set_num_threads(args.threads)
        num_threads = args.threads
    else:
        num_threads = numba.get_num_threads()

    warmup_threads()
    layer = threading_layer()

    _, max_cores = get_cpu_counts()
    oversubscribed = 1 if num_threads > max_cores else 0

    print(f"[Version 1 - Numba prange Threads] threads={num_threads} (layer={layer}), N={args.N}, steps={args.steps}, tol={args.tol}, reps={args.reps}")
    for rep in range(1, args.reps + 1):
        grid, elapsed, iters, diff, _ = solve_numba_threads(
            args.N, steps=args.steps, tol=args.tol,
            check_every=args.check_every, hotspot=args.hotspot
        )
        csum = compute_checksum(grid)
        print(f"  Rep {rep}: Time={elapsed:.4f}s, Iterations={iters}, FinalDiff={diff:.2e}, Checksum={csum:.4f}")

        row = {
            "target": args.target,
            "version": "threads_numba",
            "kernel": "numba",
            "N": args.N,
            "p": num_threads,
            "steps": args.steps,
            "check_every": args.check_every,
            "rep": rep,
            "t_total": elapsed,
            "t_compute_max": elapsed,
            "t_compute_mean": elapsed,
            "t_halo_max": 0.0,
            "t_halo_mean": 0.0,
            "t_reduce_max": 0.0,
            "t_reduce_mean": 0.0,
            "t_plot": 0.0,
            "iterations": iters,
            "final_diff": diff,
            "checksum": csum,
            "oversubscribed": oversubscribed,
            "threading_layer": layer,
            "notes": f"Version 1 Numba OpenMP prange with {num_threads} threads"
        }
        append_result_row(args.out, row)

if __name__ == "__main__":
    main()

