"""
Local MPI Matrix Multiplication
High-Performance Computing (HPC) Project

Run all MPI ranks on the same machine:
    mpiexec -n <p> python src/mpi_matmul.py --n <size> --kernel <numba|numpy>

Algorithm:
Row-Block Matrix Decomposition:
1. Rank 0 allocates matrices A and B (double precision float64).
2. Rank 0 broadcasts matrix B to all ranks: comm.Bcast(B, root=0).
3. Rank 0 scatters row blocks of A to all ranks: comm.Scatter(A, A_loc, root=0).
4. Each rank computes its local chunk C_loc = A_loc x B.
   - Using either single-thread Numba kernel or NumPy BLAS with OMP_NUM_THREADS=1.
5. Rank 0 gathers C_loc from all ranks: comm.Gather(C_loc, C, root=0).
6. Rank 0 verifies correctness against serial reference using np.allclose.
7. Separate timers record:
   - t_bcast (Matrix B broadcast)
   - t_scatter (Matrix A row blocks distribution)
   - t_compute (Local arithmetic)
   - t_gather (Matrix C row blocks aggregation)
   - t_total = t_bcast + t_scatter + t_compute + t_gather
"""

import os
import sys

# Force single-threaded BLAS to avoid core oversubscription across ranks.
# These variables MUST be set BEFORE NumPy/BLAS is imported (BLAS reads them
# at import time), as required by the proposal (Section 4.2).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import argparse
import json
import numpy as np
from numba import njit

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import generate_matrices, compute_reference, verify_result, calculate_gflops, get_matrix_memory_mb


@njit(fastmath=False)
def local_kernel_numba(A_loc: np.ndarray, B: np.ndarray, C_loc: np.ndarray) -> None:
    """
    Local matrix multiplication kernel using Numba (single-threaded).
    Computes C_loc = A_loc x B.
    A_loc: (rows x m), B: (m x p), C_loc: (rows x p).
    """
    rows, m = A_loc.shape
    p = B.shape[1]
    for i in range(rows):
        for k in range(m):
            a = A_loc[i, k]
            for j in range(p):
                C_loc[i, j] += a * B[k, j]


def warmup_mpi_kernel() -> None:
    """
    Warm-up local JIT compilation on each rank before recording benchmark timers.
    """
    a = np.ones((16, 16), dtype=np.float64)
    b = np.ones((16, 16), dtype=np.float64)
    c = np.zeros((16, 16), dtype=np.float64)
    local_kernel_numba(a, b, c)


def run_mpi_matmul(n: int, kernel_choice: str = "numba", verify: bool = True, output_json: str = None):
    """
    Executes local MPI matrix multiplication via mpi4py.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        print("[ERROR] mpi4py is not installed in this environment.")
        print("Install mpi4py and a local MPI implementation, then run with mpiexec.")
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    processor_name = MPI.Get_processor_name()

    if n % size != 0:
        if rank == 0:
            print(f"[ERROR] Matrix dimension n={n} must be divisible by process count p={size}.")
        sys.exit(1)

    rows = n // size

    # Pre-compile Numba kernel on all ranks
    if kernel_choice == "numba":
        warmup_mpi_kernel()
    comm.Barrier()

    # Step 1: Memory Allocation
    if rank == 0:
        print(f"\n========================================================")
        print(f"[MPI Matrix Multiplication] Size n={n} | Processes p={size} | Kernel={kernel_choice.upper()}")
        print(f"Memory per n x n matrix: {get_matrix_memory_mb(n):.2f} MB | Rows per rank: {rows}")
        print(f"========================================================")
        A, B = generate_matrices(n, seed=42)
    else:
        A = None
        B = np.empty((n, n), dtype=np.float64)

    A_loc = np.empty((rows, n), dtype=np.float64)
    C_loc = np.zeros((rows, n), dtype=np.float64)
    C = np.empty((n, n), dtype=np.float64) if rank == 0 else None

    comm.Barrier()

    # Step 2: Timed Broadcast of Matrix B
    t0_bcast = MPI.Wtime()
    comm.Bcast([B, MPI.DOUBLE], root=0)
    t1_bcast = MPI.Wtime()
    t_bcast = t1_bcast - t0_bcast

    # Step 3: Timed Scatter of Matrix A (Row Blocks)
    t0_scatter = MPI.Wtime()
    comm.Scatter([A, MPI.DOUBLE], [A_loc, MPI.DOUBLE], root=0)
    t1_scatter = MPI.Wtime()
    t_scatter = t1_scatter - t0_scatter

    # Step 4: Timed Local Computation
    t0_compute = MPI.Wtime()
    if kernel_choice == "numba":
        local_kernel_numba(A_loc, B, C_loc)
    else:
        # NumPy BLAS multiplication with OMP_NUM_THREADS=1
        C_loc = A_loc @ B
    t1_compute = MPI.Wtime()
    t_compute = t1_compute - t0_compute

    # Step 5: Timed Gather of Matrix C
    t0_gather = MPI.Wtime()
    comm.Gather([C_loc, MPI.DOUBLE], [C, MPI.DOUBLE], root=0)
    t1_gather = MPI.Wtime()
    t_gather = t1_gather - t0_gather

    t_total = t_bcast + t_scatter + t_compute + t_gather

    # Collect timing statistics across all ranks using MPI Reductions
    max_bcast = comm.reduce(t_bcast, op=MPI.MAX, root=0)
    max_scatter = comm.reduce(t_scatter, op=MPI.MAX, root=0)
    max_compute = comm.reduce(t_compute, op=MPI.MAX, root=0)
    max_gather = comm.reduce(t_gather, op=MPI.MAX, root=0)
    max_total = comm.reduce(t_total, op=MPI.MAX, root=0)

    avg_compute = comm.reduce(t_compute, op=MPI.SUM, root=0)
    if rank == 0:
        avg_compute /= size

    # Step 6: Verification and Reporting on Rank 0
    if rank == 0:
        print("\n--- Detailed Timing Breakdown (Critical Path / Max Across Ranks) ---")
        print(f"  1. Broadcast (Matrix B, {get_matrix_memory_mb(n):.2f} MB): {max_bcast:.5f} s")
        print(f"  2. Scatter   (Matrix A, {get_matrix_memory_mb(n):.2f} MB): {max_scatter:.5f} s")
        print(f"  3. Compute   (Local Chunk {rows}x{n}):           {max_compute:.5f} s (Avg: {avg_compute:.5f} s)")
        print(f"  4. Gather    (Matrix C, {get_matrix_memory_mb(n):.2f} MB): {max_gather:.5f} s")
        print(f"  -------------------------------------------------------------")
        print(f"  Total Wall Time (Comm + Comp):               {max_total:.5f} s")
        
        # Communication time on the critical-path (slowest) rank. We derive it
        # as total minus compute so that the two ratios below always sum to
        # 100%: the per-phase MAXIMA can come from different ranks, so summing
        # them (max_bcast + max_scatter + max_gather) can exceed max_total and
        # would produce ratios above 100%.
        comm_time = max_total - max_compute
        comp_ratio = (max_compute / max_total) * 100.0 if max_total > 0 else 0
        comm_ratio = (comm_time / max_total) * 100.0 if max_total > 0 else 0
        gflops = calculate_gflops(n, max_total)
        compute_only_gflops = calculate_gflops(n, max_compute)

        print(f"  Computation Time Ratio:                      {comp_ratio:.1f}%")
        print(f"  Communication Time Ratio:                    {comm_ratio:.1f}%")
        print(f"  Overall Throughput:                          {gflops:.2f} GFLOP/s")
        print(f"  Raw Compute Throughput:                      {compute_only_gflops:.2f} GFLOP/s")

        is_ok = True
        max_abs = 0.0
        if verify:
            print("\n  Verifying result against reference (np.allclose)...")
            C_ref = compute_reference(A, B)
            is_ok, max_abs, max_rel = verify_result(C, C_ref)
            print(f"  Numerical Verification: {'PASSED [OK]' if is_ok else 'FAILED [ERROR]'} (Max abs err: {max_abs:.2e})")

        metrics = {
            "n": n,
            "p": size,
            "kernel": kernel_choice,
            "t_bcast": max_bcast,
            "t_scatter": max_scatter,
            "t_compute": max_compute,
            "t_gather": max_gather,
            "t_comm": comm_time,
            "t_total": max_total,
            "comp_ratio_pct": comp_ratio,
            "comm_ratio_pct": comm_ratio,
            "gflops": gflops,
            "verified": is_ok,
            "max_abs_err": max_abs
        }

        if output_json:
            with open(output_json, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"[INFO] Metrics written to {output_json}")

        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local MPI Row-Block Matrix Multiplication using mpi4py")
    parser.add_argument("--n", type=int, default=2048, help="Matrix dimension n (default: 2048)")
    parser.add_argument("--kernel", type=str, choices=["numba", "numpy"], default="numba", help="Local computation kernel")
    parser.add_argument("--no-verify", action="store_true", help="Skip numerical verification to save time")
    parser.add_argument("--output-json", type=str, default=None, help="Save metrics to JSON file")
    args = parser.parse_args()

    run_mpi_matmul(n=args.n, kernel_choice=args.kernel, verify=not args.no_verify, output_json=args.output_json)
