"""
heat_reference.py - Version 0a: Pure NumPy Reference Solver

Used as an independent baseline for correctness verification.
Computes 2D heat diffusion using NumPy slicing.
"""

import time
import numpy as np
from heat_common import init_grid, apply_hotspot_global, compute_checksum, get_base_parser, append_result_row

def solve_numpy(N, steps=2000, tol=1e-4, check_every=50, hotspot=False):
    """
    Solve heat equation using NumPy array slicing.
    Returns: (final_grid, elapsed_time, iterations, final_diff)
    """
    u = init_grid(N, hotspot=hotspot)
    v = u.copy()

    t0 = time.perf_counter()
    iterations = 0
    final_diff = 0.0
    current_grid = u

    for step in range(1, steps + 1):
        iterations = step
        # 5-point stencil update on interior
        v[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
        if hotspot:
            apply_hotspot_global(v, N)

        current_grid = v

        if check_every > 0 and step % check_every == 0:
            final_diff = float(np.abs(v[1:-1, 1:-1] - u[1:-1, 1:-1]).max())
            if final_diff < tol:
                break

        # Swap buffers for next step
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

    print(f"[Version 0a - NumPy Reference] N={args.N}, steps={args.steps}, tol={args.tol}, reps={args.reps}")
    for rep in range(1, args.reps + 1):
        grid, elapsed, iters, diff = solve_numpy(
            args.N, steps=args.steps, tol=args.tol,
            check_every=args.check_every, hotspot=args.hotspot
        )
        csum = compute_checksum(grid)
        print(f"  Rep {rep}: Time={elapsed:.4f}s, Iterations={iters}, FinalDiff={diff:.2e}, Checksum={csum:.4f}")

        row = {
            "target": args.target,
            "version": "ref_numpy",
            "kernel": "numpy",
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
            "notes": "Version 0a pure NumPy serial reference"
        }
        append_result_row(args.out, row)

if __name__ == "__main__":
    main()

