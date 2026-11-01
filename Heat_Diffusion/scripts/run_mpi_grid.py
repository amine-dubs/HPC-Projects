"""
scripts/run_mpi_grid.py - Automated Runner for MPI Scaling & Overlap (Experiments E3 & E8)

Runs:
- E3: Local MPI scaling with Numba kernel: N in {512, 1024, 2048}, p in {1, 2, 4, 8}, 2000 steps, 3 reps
- E3 (Kernel Comparison): MPI with NumPy kernel for N=1024, p in {1, 2, 4, 8}, 2000 steps, 3 reps
- E8: Non-blocking MPI overlap: N in {1024, 2048}, p in {2, 4, 8}, 2000 steps, 3 reps
Appends directly to results/raw/scaling_local.csv.
"""

import os
import sys
import shutil
import subprocess
import argparse

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_DIR, "src")

def get_mpirun_cmd():
    msmpi = os.path.join(os.environ.get("MSMPI_BIN", r"C:\Program Files\Microsoft MPI\Bin"), "mpiexec.exe")
    if os.path.exists(msmpi):
        return msmpi
    which = shutil.which("mpirun") or shutil.which("mpiexec")
    return which or "mpirun"

def run_mpi_grid(out_csv="results/raw/scaling_local.csv", quick=False):
    if quick and out_csv == "results/raw/scaling_local.csv":
        out_csv = "results/raw/scaling_quick.csv"
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(PROJECT_DIR, out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    mpi_cmd = get_mpirun_cmd()

    if quick:
        grid_sizes = [64, 128]
        p_ranks = [1, 2, 4]
        steps = 100
        reps = 1
        check_every = 10
        overlap_sizes = [128]
    else:
        grid_sizes = [512, 1024, 2048]
        p_ranks = [1, 2, 4, 8]
        steps = 2000
        reps = 3
        check_every = 50
        overlap_sizes = [1024, 2048]

    print("=========================================================")
    print("EXPERIMENT E3: LOCAL MPI SCALING (NUMBA KERNEL)")
    print(f"Grid sizes: {grid_sizes}, Ranks: {p_ranks}, Steps: {steps}, Reps: {reps}")
    print("=========================================================\n")

    # 1. Main MPI Scaling Grid (Numba Kernel)
    for N in grid_sizes:
        for p in p_ranks:
            print(f"\n>>> Running MPI (Numba): N={N}, p={p}, steps={steps}, reps={reps} <<<")
            cmd = [
                mpi_cmd, "-n", str(p),
                sys.executable, "-u", os.path.join(SRC_DIR, "heat_mpi.py"),
                "--N", str(N),
                "--steps", str(steps),
                "--reps", str(reps),
                "--check-every", str(check_every),
                "--kernel", "numba",
                "--out", out_csv
            ]
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                print(f"Error in MPI Numba N={N}, p={p}", file=sys.stderr)

    # 2. NumPy Kernel Comparison (for N=1024)
    if not quick:
        print("\n=========================================================")
        print("EXPERIMENT E3 COMPARISON: MPI WITH NUMPY KERNEL (N=1024)")
        print("=========================================================\n")
        N_comp = 1024
        for p in p_ranks:
            print(f"\n>>> Running MPI (NumPy kernel): N={N_comp}, p={p}, steps={steps}, reps={reps} <<<")
            cmd = [
                mpi_cmd, "-n", str(p),
                sys.executable, "-u", os.path.join(SRC_DIR, "heat_mpi.py"),
                "--N", str(N_comp),
                "--steps", str(steps),
                "--reps", str(reps),
                "--check-every", str(check_every),
                "--kernel", "numpy",
                "--out", out_csv
            ]
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                print(f"Error in MPI NumPy N={N_comp}, p={p}", file=sys.stderr)

    # 3. Experiment E8: Non-blocking MPI Overlap
    print("\n=========================================================")
    print("EXPERIMENT E8: NON-BLOCKING OVERLAP (MPI_ISEND / MPI_IRECV)")
    print("=========================================================\n")
    overlap_ranks = [2, 4, 8] if not quick else [2, 4]
    for N in overlap_sizes:
        for p in overlap_ranks:
            print(f"\n>>> Running MPI Overlap: N={N}, p={p}, steps={steps}, reps={reps} <<<")
            cmd = [
                mpi_cmd, "-n", str(p),
                sys.executable, "-u", os.path.join(SRC_DIR, "heat_mpi_overlap.py"),
                "--N", str(N),
                "--steps", str(steps),
                "--reps", str(reps),
                "--check-every", str(check_every),
                "--out", out_csv
            ]
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                print(f"Error in MPI Overlap N={N}, p={p}", file=sys.stderr)

    print("\n>>> All MPI scaling and overlap experiments completed. Output in", out_csv, "<<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/raw/scaling_local.csv")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_mpi_grid(out_csv=args.out, quick=args.quick)

