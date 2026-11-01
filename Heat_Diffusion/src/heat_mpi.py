"""
heat_mpi.py - Distributed-Memory 2D Heat Diffusion Solver using MPI (mpi4py)

1D row-strip domain decomposition with halo exchange via Sendrecv.
Collectives for global convergence and output gathering.
Supports blocking halo exchange and optional debug-skip-halo mode.
"""

import os
import sys
import time
import argparse
import numpy as np

# Force 1 thread per rank to prevent thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from mpi4py import MPI
from numba import njit
from heat_common import (
    TOP_TEMP, BOTTOM_TEMP, LEFT_TEMP, RIGHT_TEMP,
    apply_hotspot_local, compute_checksum, get_cpu_counts, get_base_parser, append_result_row
)

@njit(cache=True)
def jacobi_step_mpi_numba(u, v, r_start, r_end, N):
    """
    Perform 5-point stencil update on local interior rows [r_start, r_end).
    Boundary columns 0 and N-1 are left untouched.
    """
    for i in range(r_start, r_end):
        for j in range(1, N - 1):
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

@njit(cache=True)
def compute_max_diff_mpi_numba(u, v, r_start, r_end, N):
    """
    Compute local maximum absolute difference strictly on updated interior cells.
    """
    max_d = 0.0
    for i in range(r_start, r_end):
        for j in range(1, N - 1):
            d = abs(v[i, j] - u[i, j])
            if d > max_d:
                max_d = d
    return max_d

def warmup_mpi(kernel_type):
    """
    Warmup JIT compilation on a 4x4 dummy slice so compilation overhead is not timed.
    """
    if kernel_type == "numba":
        dummy_u = np.zeros((4, 4), dtype=np.float64, order="C")
        dummy_v = dummy_u.copy()
        jacobi_step_mpi_numba(dummy_u, dummy_v, 1, 3, 4)
        compute_max_diff_mpi_numba(dummy_u, dummy_v, 1, 3, 4)

def run_mpi_solver(comm, N, steps, tol, check_every, kernel, hotspot, debug_skip_halo=False, save_every=0):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if N % size != 0:
        raise ValueError(f"Grid size N={N} must be evenly divisible by MPI rank count p={size}")

    rows = N // size
    # Allocate (rows + 2, N) for local slice with top/bottom halo rows
    u = np.zeros((rows + 2, N), dtype=np.float64, order="C")
    v = np.zeros((rows + 2, N), dtype=np.float64, order="C")

    # Set Dirichlet boundaries once in both arrays at initialization
    # Corner convention: columns override rows
    if rank == 0:
        u[1, :] = TOP_TEMP
        v[1, :] = TOP_TEMP
    if rank == size - 1:
        u[-2, :] = BOTTOM_TEMP
        v[-2, :] = BOTTOM_TEMP

    u[:, 0] = LEFT_TEMP
    v[:, 0] = LEFT_TEMP
    u[:, -1] = RIGHT_TEMP
    v[:, -1] = RIGHT_TEMP

    if hotspot:
        apply_hotspot_local(u, N, rank, rows)
        apply_hotspot_local(v, N, rank, rows)

    # Determine interior row range to update (excluding global boundaries)
    r_start = 2 if rank == 0 else 1
    r_end = rows if rank == size - 1 else rows + 1

    # Neighbors for halo exchange
    up = rank - 1 if rank > 0 else MPI.PROC_NULL
    down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    # Warmup kernel
    warmup_mpi(kernel)

    # Synchronization before timing
    comm.Barrier()
    t_start = MPI.Wtime()

    local_t_halo = 0.0
    local_t_compute = 0.0
    local_t_reduce = 0.0
    local_t_plot = 0.0

    iterations = 0
    final_diff = 0.0
    current_grid = u

    saved_frames = []

    for step in range(1, steps + 1):
        iterations = step

        # Halo Exchange
        th0 = MPI.Wtime()
        if not debug_skip_halo:
            comm.Sendrecv(sendbuf=u[1, :], dest=up, recvbuf=u[-1, :], source=down)
            comm.Sendrecv(sendbuf=u[-2, :], dest=down, recvbuf=u[0, :], source=up)
        th1 = MPI.Wtime()
        local_t_halo += (th1 - th0)

        # Computation
        tc0 = MPI.Wtime()
        if kernel == "numba":
            jacobi_step_mpi_numba(u, v, r_start, r_end, N)
        else:
            v[r_start:r_end, 1:-1] = 0.25 * (
                u[r_start - 1 : r_end - 1, 1:-1] +
                u[r_start + 1 : r_end + 1, 1:-1] +
                u[r_start:r_end, :-2] +
                u[r_start:r_end, 2:]
            )

        if hotspot:
            apply_hotspot_local(v, N, rank, rows)
        tc1 = MPI.Wtime()
        local_t_compute += (tc1 - tc0)

        current_grid = v

        # Periodic Convergence Check
        if check_every > 0 and step % check_every == 0:
            tr0 = MPI.Wtime()
            if kernel == "numba":
                local_diff = compute_max_diff_mpi_numba(u, v, r_start, r_end, N)
            else:
                local_diff = float(np.abs(v[r_start:r_end, 1:-1] - u[r_start:r_end, 1:-1]).max()) if (r_end > r_start) else 0.0

            final_diff = comm.allreduce(local_diff, op=MPI.MAX)
            tr1 = MPI.Wtime()
            local_t_reduce += (tr1 - tr0)

            if final_diff < tol:
                break

        # Save frame if requested (plotting excluded from simulation time)
        if save_every > 0 and step % save_every == 0:
            tp0 = MPI.Wtime()
            # Gather interior rows to rank 0
            sendbuf = np.ascontiguousarray(current_grid[1:rows+1, :])
            full_frame = np.empty((N, N), dtype=np.float64, order="C") if rank == 0 else None
            comm.Gather(sendbuf, full_frame, root=0)
            if rank == 0:
                saved_frames.append((step, full_frame.copy()))
            tp1 = MPI.Wtime()
            local_t_plot += (tp1 - tp0)

        # Pointer swap
        u, v = v, u
        current_grid = u

    comm.Barrier()
    t_end = MPI.Wtime()
    local_t_total = t_end - t_start

    # Gather full final grid to rank 0 for verification & checksum
    sendbuf = np.ascontiguousarray(current_grid[1:rows+1, :])
    full_grid = np.empty((N, N), dtype=np.float64, order="C") if rank == 0 else None
    comm.Gather(sendbuf, full_grid, root=0)

    # Reduction of timing statistics across ranks
    t_comp_max = comm.reduce(local_t_compute, op=MPI.MAX, root=0)
    t_comp_sum = comm.reduce(local_t_compute, op=MPI.SUM, root=0)
    t_halo_max = comm.reduce(local_t_halo, op=MPI.MAX, root=0)
    t_halo_sum = comm.reduce(local_t_halo, op=MPI.SUM, root=0)
    t_reduce_max = comm.reduce(local_t_reduce, op=MPI.MAX, root=0)
    t_reduce_sum = comm.reduce(local_t_reduce, op=MPI.SUM, root=0)
    t_total_max = comm.reduce(local_t_total, op=MPI.MAX, root=0)
    t_plot_max = comm.reduce(local_t_plot, op=MPI.MAX, root=0)

    results = None
    if rank == 0:
        csum = compute_checksum(full_grid)
        results = {
            "full_grid": full_grid,
            "iterations": iterations,
            "final_diff": final_diff,
            "checksum": csum,
            "t_total": t_total_max,
            "t_compute_max": t_comp_max,
            "t_compute_mean": t_comp_sum / size,
            "t_halo_max": t_halo_max,
            "t_halo_mean": t_halo_sum / size,
            "t_reduce_max": t_reduce_max,
            "t_reduce_mean": t_reduce_sum / size,
            "t_plot": t_plot_max,
            "saved_frames": saved_frames
        }
    return results

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = get_base_parser()
    parser.add_argument("--debug-skip-halo", action="store_true", help="Deliberately omit halo exchange to visualize seam")
    args = parser.parse_args()

    if args.quick:
        args.N = 64
        args.steps = 100
        args.reps = 1
        args.check_every = 10

    physical_cores, _ = get_cpu_counts()
    oversubscribed = 1 if size > physical_cores else 0

    if rank == 0:
        print(f"[Version 2-4 MPI] size={size}, N={args.N}, steps={args.steps}, tol={args.tol}, kernel={args.kernel}, reps={args.reps}, target={args.target}")

    for rep in range(1, args.reps + 1):
        res = run_mpi_solver(
            comm, args.N, args.steps, args.tol,
            args.check_every, args.kernel, args.hotspot,
            debug_skip_halo=args.debug_skip_halo, save_every=args.save_every
        )

        if rank == 0:
            print(f"  Rep {rep}: Time={res['t_total']:.4f}s (comp={res['t_compute_max']:.4f}s, halo={res['t_halo_max']:.4f}s, red={res['t_reduce_max']:.4f}s), "
                  f"Iters={res['iterations']}, Diff={res['final_diff']:.2e}, Checksum={res['checksum']:.4f}")

            row = {
                "target": args.target,
                "version": "mpi",
                "kernel": args.kernel,
                "N": args.N,
                "p": size,
                "steps": args.steps,
                "check_every": args.check_every,
                "rep": rep,
                "t_total": res["t_total"],
                "t_compute_max": res["t_compute_max"],
                "t_compute_mean": res["t_compute_mean"],
                "t_halo_max": res["t_halo_max"],
                "t_halo_mean": res["t_halo_mean"],
                "t_reduce_max": res["t_reduce_max"],
                "t_reduce_mean": res["t_reduce_mean"],
                "t_plot": res["t_plot"],
                "iterations": res["iterations"],
                "final_diff": res["final_diff"],
                "checksum": res["checksum"],
                "oversubscribed": oversubscribed,
                "threading_layer": "none",
                "notes": f"Version 2-4 MPI blocking Sendrecv (kernel={args.kernel})"
            }
            append_result_row(args.out, row)

if __name__ == "__main__":
    main()

