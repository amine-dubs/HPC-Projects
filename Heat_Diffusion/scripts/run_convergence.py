"""
scripts/run_convergence.py - Experiment E4 Convergence & k-sweep Study

1. Convergence tracking:
   For N in {128, 256, 512}, logs max change every 10 iterations until tol < 1e-4 (or cap 10,000).
   Saves to results/raw/convergence_N<N>.csv.
2. k-sweep:
   For N=256, p=4, sweeps check_every k in {1, 10, 50, 200} with 3 reps.
   Measures runtime, t_reduce share, and stopping iteration.
   Saves to results/raw/ksweep.csv.
"""

import os
import sys
import csv
import argparse
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from heat_common import init_grid
from heat_serial import jacobi_step_serial, compute_max_diff_serial, warmup_numba
from run_mpi_grid import get_mpirun_cmd

def run_convergence_curves(grid_sizes=[128, 256, 512], tol=1e-4, max_steps=10000, quick=False):
    if quick:
        grid_sizes = [64, 128]
        max_steps = 1000

    warmup_numba()
    print("=========================================================")
    print("EXPERIMENT E4 (PART 1): CONVERGENCE HISTORY (TOL = 1e-4)")
    print("=========================================================")

    for N in grid_sizes:
        fname = f"quick_convergence_N{N}.csv" if quick else f"convergence_N{N}.csv"
        csv_path = os.path.join(PROJECT_DIR, "results", "raw", fname)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        print(f"\n>>> Tracking convergence for N={N} (max steps: {max_steps}) <<<")

        u = init_grid(N, hotspot=False)
        v = u.copy()

        history = []
        stopped_step = max_steps
        final_d = 0.0

        for step in range(1, max_steps + 1):
            jacobi_step_serial(u, v, N)

            if step % 10 == 0:
                d = compute_max_diff_serial(u, v, N)
                history.append((step, d))
                final_d = d
                if d < tol:
                    stopped_step = step
                    break

            u, v = v, u

        print(f"  N={N} stopped at step {stopped_step} with diff={final_d:.4e} (logged {len(history)} points)")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "max_change"])
            w.writerows(history)
        print(f"  Saved to {csv_path}")

def run_ksweep(N=256, p=4, k_values=[1, 10, 50, 200], tol=1e-4, max_steps=5000, reps=3, quick=False):
    import subprocess
    import tempfile

    if quick:
        k_values = [1, 10, 50]
        reps = 1
        max_steps = 1000

    print("\n=========================================================")
    print(f"EXPERIMENT E4 (PART 2): k-SWEEP STUDY (N={N}, p={p})")
    print(f"k in {k_values}, tol={tol}, reps={reps}")
    print("=========================================================")

    fname = "quick_ksweep.csv" if quick else "ksweep.csv"
    out_csv = os.path.join(PROJECT_DIR, "results", "raw", fname)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    mpi_cmd = get_mpirun_cmd()
    ksweep_rows = []

    for k in k_values:
        print(f"\n>>> Running k-sweep with k={k} ({reps} reps) <<<")
        for rep in range(1, reps + 1):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_json = tmp.name

            code = f"""
import sys, json
sys.path.insert(0, r"{SRC_DIR}")
from mpi4py import MPI
from heat_mpi import run_mpi_solver

comm = MPI.COMM_WORLD
res = run_mpi_solver(comm, {N}, {max_steps}, {tol}, {k}, "numba", False)
if comm.Get_rank() == 0:
    out = {{
        "iterations": res["iterations"],
        "final_diff": res["final_diff"],
        "t_total": res["t_total"],
        "t_compute_max": res["t_compute_max"],
        "t_compute_mean": res["t_compute_mean"],
        "t_halo_max": res["t_halo_max"],
        "t_halo_mean": res["t_halo_mean"],
        "t_reduce_max": res["t_reduce_max"],
        "t_reduce_mean": res["t_reduce_mean"]
    }}
    with open(r"{tmp_json}", "w") as f:
        json.dump(out, f)
"""
            cmd = [mpi_cmd, "-n", str(p), sys.executable, "-u", "-c", code]
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                print(f"Error in k-sweep k={k}, rep={rep}", file=sys.stderr)
                continue

            import json
            with open(tmp_json, "r") as f:
                data = json.load(f)
            try:
                os.remove(tmp_json)
            except OSError:
                pass

            t_total = data["t_total"]
            t_red = data["t_reduce_mean"]
            red_pct = (t_red / t_total * 100.0) if t_total > 0 else 0.0

            print(f"  k={k:3d} (Rep {rep}): Time={t_total:.4f}s | Reduce={t_red:.4f}s ({red_pct:5.1f}%) | Iters={data['iterations']}")

            ksweep_rows.append({
                "N": N,
                "p": p,
                "k": k,
                "rep": rep,
                "iterations": data["iterations"],
                "final_diff": data["final_diff"],
                "t_total": t_total,
                "t_compute_max": data["t_compute_max"],
                "t_compute_mean": data["t_compute_mean"],
                "t_halo_max": data["t_halo_max"],
                "t_halo_mean": data["t_halo_mean"],
                "t_reduce_max": data["t_reduce_max"],
                "t_reduce_mean": data["t_reduce_mean"],
                "reduce_share_pct": red_pct
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["N", "p", "k", "rep", "iterations", "final_diff", "t_total", "t_compute_max", "t_compute_mean", "t_halo_max", "t_halo_mean", "t_reduce_max", "t_reduce_mean", "reduce_share_pct"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ksweep_rows)

    print(f"\nk-sweep complete. Results saved to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    run_convergence_curves(quick=args.quick)
    run_ksweep(quick=args.quick)

if __name__ == "__main__":
    main()

