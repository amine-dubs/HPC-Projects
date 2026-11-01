"""
scripts/run_threads.py - Automated Runner for Version 1 Thread Scaling (Experiment E2)

Runs Numba serial baseline and Numba prange threads across thread counts {1, 2, 4, 8, 12}
for grid sizes N in {512, 1024, 2048} with 2000 steps and 3 repetitions.
Each thread count is executed in a fresh Python process to isolate threading runtimes.
Appends directly to results/raw/scaling_local.csv.
"""

import os
import sys
import subprocess
import argparse

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_DIR, "src")

def run_threads_grid(out_csv="results/raw/scaling_local.csv", quick=False):
    if quick and "scaling_local.csv" in out_csv:
        out_csv = "results/raw/scaling_quick.csv"
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(PROJECT_DIR, out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if quick:
        grid_sizes = [64, 128]
        thread_counts = [1, 2, 4]
        steps = 100
        reps = 1
        check_every = 10
    else:
        grid_sizes = [512, 1024, 2048]
        # Logical cores on this machine = 12, physical = 6
        thread_counts = [1, 2, 4, 8, 12]
        steps = 2000
        reps = 3
        check_every = 50

    print("=========================================================")
    print("EXPERIMENT E2: SHARED-MEMORY THREAD SCALING (NUMBA PRANGE)")
    print(f"Grid sizes: {grid_sizes}, Threads: {thread_counts}, Steps: {steps}, Reps: {reps}")
    print("=========================================================\n")

    # 1. First run Version 0b Numba serial baseline for each N
    for N in grid_sizes:
        print(f"\n>>> Running Version 0b Serial Baseline: N={N}, steps={steps}, reps={reps} <<<")
        cmd = [
            sys.executable,
            os.path.join(SRC_DIR, "heat_serial.py"),
            "--N", str(N),
            "--steps", str(steps),
            "--reps", str(reps),
            "--check-every", str(check_every),
            "--out", out_csv
        ]
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            print(f"Error in serial baseline N={N}", file=sys.stderr)

    # 2. Run Version 1 Threads in fresh process for each (N, threads)
    for N in grid_sizes:
        for t in thread_counts:
            print(f"\n>>> Running Version 1 Threads: N={N}, threads={t}, steps={steps}, reps={reps} <<<")
            cmd = [
                sys.executable,
                os.path.join(SRC_DIR, "heat_threads.py"),
                "--N", str(N),
                "--threads", str(t),
                "--steps", str(steps),
                "--reps", str(reps),
                "--check-every", str(check_every),
                "--out", out_csv
            ]
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                print(f"Error in threads run N={N}, threads={t}", file=sys.stderr)

    print("\n>>> Thread scaling experiments completed. Results logged to", out_csv, "<<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/raw/scaling_local.csv")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_threads_grid(out_csv=args.out, quick=args.quick)

