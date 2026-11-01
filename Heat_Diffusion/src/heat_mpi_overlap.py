"""
heat_mpi_overlap.py - Non-blocking MPI 2D Heat Diffusion Solver

Overlaps halo communication with computation of interior core rows using
non-blocking MPI_Isend and MPI_Irecv.
"""

import os
import sys
import time
import argparse
import numpy as np

# Force 1 thread per rank
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
def jacobi_row_update(u, v, row_idx, N):
    """Update a single row index."""
    for j in range(1, N - 1):
        v[row_idx, j] = 0.25 * (u[row_idx - 1, j] + u[row_idx + 1, j] + u[row_idx, j - 1] + u[row_idx, j + 1])

@njit(cache=True)
def jacobi_rows_range(u, v, r_start, r_end, N):
    """Update contiguous rows [r_start, r_end)."""
    for i in range(r_start, r_end):
        for j in range(1, N - 1):
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

@njit(cache=True)
def compute_max_diff_mpi_numba(u, v, r_start, r_end, N):
    max_d = 0.0
    for i in range(r_start, r_end):
        for j in range(1, N - 1):
            d = abs(v[i, j] - u[i, j])
            if d > max_d:
                max_d = d
    return max_d

def warmup_overlap():
    dummy_u = np.zeros((6, 6), dtype=np.float64, order="C")
    dummy_v = dummy_u.copy()
    jacobi_row_update(dummy_u, dummy_v, 1, 6)
    jacobi_rows_range(dummy_u, dummy_v, 2, 4, 6)
    compute_max_diff_mpi_numba(dummy_u, dummy_v, 1, 5, 6)

def run_mpi_overlap_solver(comm, N, steps, tol, check_every, hotspot):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if N % size != 0:
        raise ValueError(f"Grid size N={N} must be evenly divisible by MPI rank count p={size}")

    rows = N // size
    u = np.zeros((rows + 2, N), dtype=np.float64, order="C")
    v = np.zeros((rows + 2, N), dtype=np.float64, order="C")

    # Set Dirichlet boundaries
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

    r_start = 2 if rank == 0 else 1
    r_end = rows if rank == size - 1 else rows + 1

    up = rank - 1 if rank > 0 else MPI.PROC_NULL
    down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    # Inner rows that DO NOT depend on halos: local rows [2, rows)
    inner_start = max(r_start, 2)
    inner_end = min(r_end, rows)

    warmup_overlap()

    comm.Barrier()
    t_start = MPI.Wtime()

    local_t_halo = 0.0
    local_t_compute = 0.0
    local_t_reduce = 0.0

    iterations = 0
    final_diff = 0.0
    current_grid = u

    for step in range(1, steps + 1):
        iterations = step

        # 1. Post non-blocking receives and sends
        th0 = MPI.Wtime()
        reqs = []
        if down != MPI.PROC_NULL:
            reqs.append(comm.Irecv(u[-1, :], source=down))
            reqs.append(comm.Isend(u[-2, :], dest=down))
        if up != MPI.PROC_NULL:
            reqs.append(comm.Irecv(u[0, :], source=up))
            reqs.append(comm.Isend(u[1, :], dest=up))
        th1 = MPI.Wtime()

        # 2. Compute inner rows while communication is in-flight
        tc0 = MPI.Wtime()
        if inner_end > inner_start:
            jacobi_rows_range(u, v, inner_start, inner_end, N)
        tc1 = MPI.Wtime()

        # 3. Wait for halo exchanges to complete
        th2 = MPI.Wtime()
        if reqs:
            MPI.Request.Waitall(reqs)
        th3 = MPI.Wtime()

        local_t_halo += (th1 - th0) + (th3 - th2)

        # 4. Compute boundary-adjacent rows with freshly received halos
        tc2 = MPI.Wtime()
        # Top-most interior row (row 1) if this rank is not global top boundary
        if r_start <= 1:
            jacobi_row_update(u, v, 1, N)
        # Bottom-most interior row (row rows) if this rank is not global bottom boundary
        if r_end > rows:
            jacobi_row_update(u, v, rows, N)

        if hotspot:
            apply_hotspot_local(v, N, rank, rows)
        tc3 = MPI.Wtime()
        local_t_compute += (tc1 - tc0) + (tc3 - tc2)

        current_grid = v

        # 5. Convergence check
        if check_every > 0 and step % check_every == 0:
            tr0 = MPI.Wtime()
            local_diff = compute_max_diff_mpi_numba(u, v, r_start, r_end, N)
            final_diff = comm.allreduce(local_diff, op=MPI.MAX)
            tr1 = MPI.Wtime()
            local_t_reduce += (tr1 - tr0)
            if final_diff < tol:
                break

        u, v = v, u
        current_grid = u

    comm.Barrier()
    t_end = MPI.Wtime()
    local_t_total = t_end - t_start

    # Gather full final grid to rank 0
    sendbuf = np.ascontiguousarray(current_grid[1:rows+1, :])
    full_grid = np.empty((N, N), dtype=np.float64, order="C") if rank == 0 else None
    comm.Gather(sendbuf, full_grid, root=0)

    t_comp_max = comm.reduce(local_t_compute, op=MPI.MAX, root=0)
    t_comp_sum = comm.reduce(local_t_compute, op=MPI.SUM, root=0)
    t_halo_max = comm.reduce(local_t_halo, op=MPI.MAX, root=0)
    t_halo_sum = comm.reduce(local_t_halo, op=MPI.SUM, root=0)
    t_reduce_max = comm.reduce(local_t_reduce, op=MPI.MAX, root=0)
    t_reduce_sum = comm.reduce(local_t_reduce, op=MPI.SUM, root=0)
    t_total_max = comm.reduce(local_t_total, op=MPI.MAX, root=0)

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
            "t_reduce_mean": t_reduce_sum / size
        }
    return results

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = get_base_parser()
    args = parser.parse_args()

    if args.quick:
        args.N = 64
        args.steps = 100
        args.reps = 1
        args.check_every = 10

    physical_cores, _ = get_cpu_counts()
    oversubscribed = 1 if size > physical_cores else 0

    if rank == 0:
        print(f"[Version Extra - MPI Non-blocking Overlap] size={size}, N={args.N}, steps={args.steps}, tol={args.tol}, reps={args.reps}")

    for rep in range(1, args.reps + 1):
        res = run_mpi_overlap_solver(
            comm, args.N, args.steps, args.tol,
            args.check_every, args.hotspot
        )

        if rank == 0:
            print(f"  Rep {rep}: Time={res['t_total']:.4f}s (comp={res['t_compute_max']:.4f}s, halo={res['t_halo_max']:.4f}s, red={res['t_reduce_max']:.4f}s), "
                  f"Iters={res['iterations']}, Diff={res['final_diff']:.2e}, Checksum={res['checksum']:.4f}")

            row = {
                "target": args.target,
                "version": "mpi_overlap",
                "kernel": "numba",
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
                "t_plot": 0.0,
                "iterations": res["iterations"],
                "final_diff": res["final_diff"],
                "checksum": res["checksum"],
                "oversubscribed": oversubscribed,
                "threading_layer": "none",
                "notes": "Version Extra MPI non-blocking Isend/Irecv overlap"
            }
            append_result_row(args.out, row)

if __name__ == "__main__":
    main()

