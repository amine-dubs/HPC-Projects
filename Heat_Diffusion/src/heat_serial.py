"""
heat_serial.py - Version 0b: Numba Serial Solver

The single-threaded baseline for timing and scaling comparison.
Uses @njit with float64 precision and explicit warmup.
"""

import time
import numpy as np
from numba import njit
from heat_common import init_grid, apply_hotspot_global, compute_checksum, get_base_parser, append_result_row

@njit(cache=True)
def jacobi_step_serial(u, v, N):
    """
    Perform one 5-point stencil Jacobi update on the interior of an N x N grid.
    Boundaries are not overwritten.
    """
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

@njit(cache=True)
def compute_max_diff_serial(u, v, N):
    """
    Compute maximum absolute difference on interior cells.
    """
    max_d = 0.0
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            d = abs(v[i, j] - u[i, j])
            if d > max_d:
                max_d = d
    return max_d

def warmup_numba():
    """
    Warmup JIT compilation with a dummy 16x16 grid.
    """
    dummy_u = np.zeros((16, 16), dtype=np.float64, order="C")
    dummy_v = dummy_u.copy()
    jacobi_step_serial(dummy_u, dummy_v, 16)
    compute_max_diff_serial(dummy_u, dummy_v, 16)

def solve_numba_serial(N, steps=2000, tol=1e-4, check_every=50, hotspot=False):
    """
    Solve heat equation using Numba @njit serial kernels.
    Returns: (final_grid, elapsed_time, iterations, final_diff)
    """
    u = init_grid(N, hotspot=hotspot)
    v = u.copy()

    iterations = 0
    final_diff = 0.0
    current_grid = u

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        iterations = step
        jacobi_step_serial(u, v, N)
        if hotspot:
            apply_hotspot_global(v, N)

        current_grid = v

        if check_every > 0 and step % check_every == 0:
            final_diff = compute_max_diff_serial(u, v, N)
            if final_diff < tol:
                break

        u, v = v, u
        current_grid = u

    t1 = time.perf_counter()
    return current_grid, (t1 - t0), iterations, final_diff

def main():
    parser = get_base_parser()
    args = parser.parse_args()
    if args.quick:
        args.N = 64
        args.steps = 100
        args.reps = 1
        args.check_every = 10

    print("[Warmup] Compiling Numba serial kernels...")
    warmup_numba()

    print(f"[Version 0b - Numba Serial Baseline] N={args.N}, steps={args.steps}, tol={args.tol}, reps={args.reps}")
    for rep in range(1, args.reps + 1):
        grid, elapsed, iters, diff = solve_numba_serial(
            args.N, steps=args.steps, tol=args.tol,
            check_every=args.check_every, hotspot=args.hotspot
        )
        csum = compute_checksum(grid)
        print(f"  Rep {rep}: Time={elapsed:.4f}s, Iterations={iters}, FinalDiff={diff:.2e}, Checksum={csum:.4f}")

        row = {
            "target": args.target,
            "version": "serial_numba",
            "kernel": "numba",
            "N": args.N,
            "p": 1,
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
            "oversubscribed": 0,
            "threading_layer": "none",
            "notes": "Version 0b Numba serial baseline"
        }
        append_result_row(args.out, row)

if __name__ == "__main__":
    main()

