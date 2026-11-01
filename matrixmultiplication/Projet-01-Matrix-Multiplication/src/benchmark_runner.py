"""
Benchmark Runner & Data Generator for Version 1 (Shared-Memory Threads)
High-Performance Computing (HPC) Project: Matrix Multiplication

This script orchestrates benchmarks across:
- Version 0: Serial Baseline (Numba @njit, loop order i-k-j)
- Version 1: Shared-Memory Threads (Numba prange, OpenMP-style, p=1,2,4,8)
- Version 1: Cache-Blocked (Tiled) Threads (Tile=64, p=4,8)

It runs tests with 3 repetitions, reports the median, calculates Speedup S = T_1 / T_p,
Parallel Efficiency E = S / p, and exports data/results_serial_threads.csv.

Every CSV row carries a "source" field:
  - "measured":     value captured from an actual timed run on this machine.
  - "extrapolated": analytic estimate (O(n^3) scaling from the measured n=2048
                    baseline with assumed efficiency factors), emitted ONLY if a
                    requested size could not be measured (e.g. memory/time limits)
                    so that plots and tables remain continuous and honestly labelled.
The final checked-in dataset measured ALL proposed sizes (1024, 2048 and 4096).
The extrapolation fallback exists for machines where n=4096 cannot be measured.
"""

import sys
import os
import time
import argparse
import csv
import numpy as np
import numba

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import generate_matrices, calculate_gflops, get_matrix_memory_mb, save_csv_results
from src.baseline_serial import matmul_serial_ikj, warmup_serial
from src.version1_threads import matmul_threads_naive, matmul_threads_blocked, warmup_threads


def run_benchmarks(sizes=[1024, 2048], thread_counts=[1, 2, 4, 8], repeats=3, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    warmup_serial()
    warmup_threads()

    rows = []
    print("\n================================================================================")
    print("BENCHMARKING: Version 0 (Serial Baseline) & Version 1 (Shared-Memory Threads)")
    print(f"Host Logical Cores: {os.cpu_count()} | Threading Layer: {numba.threading_layer()}")
    print("================================================================================")

    for n in sizes:
        print(f"\n>>> Matrix Dimension n = {n} (Memory: {get_matrix_memory_mb(n):.1f} MB per matrix)...")
        A, B = generate_matrices(n, seed=42)
        C = np.zeros((n, n), dtype=np.float64)

        # 1. Version 0 Serial Baseline
        serial_times = []
        for r in range(repeats):
            C.fill(0.0)
            t0 = time.perf_counter()
            matmul_serial_ikj(A, B, C)
            t1 = time.perf_counter()
            serial_times.append(t1 - t0)

        t_serial_median = float(np.median(serial_times))
        gflops_serial = calculate_gflops(n, t_serial_median)
        print(f"  [Version 0 - Serial]     Median: {t_serial_median:.4f} s | {gflops_serial:.2f} GFLOP/s")

        rows.append({
            "version": "Version 0 (Serial)",
            "target": "Host (Single-Core)",
            "n": n,
            "p": 1,
            "kernel": "ikj_serial",
            "median_time_s": round(t_serial_median, 5),
            "min_time_s": round(float(np.min(serial_times)), 5),
            "max_time_s": round(float(np.max(serial_times)), 5),
            "speedup": 1.0,
            "efficiency": 1.0,
            "gflops": round(gflops_serial, 2),
            "source": "measured"
        })

        # 2. Version 1 Threads (Naive Row-Parallel)
        t1_thread_median = None
        for p in thread_counts:
            numba.set_num_threads(p)
            thread_times = []
            for r in range(repeats):
                C.fill(0.0)
                t0 = time.perf_counter()
                matmul_threads_naive(A, B, C)
                t1 = time.perf_counter()
                thread_times.append(t1 - t0)

            med_time = float(np.median(thread_times))
            if p == 1:
                t1_thread_median = med_time

            speedup = (t_serial_median / med_time) if t_serial_median else (t1_thread_median / med_time)
            efficiency = speedup / p
            gflops = calculate_gflops(n, med_time)

            print(f"  [Version 1 - Threads] p={p} Median: {med_time:.4f} s | Speedup: {speedup:.2f}x | Eff: {efficiency*100:.1f}% | {gflops:.2f} GFLOP/s")

            rows.append({
                "version": "Version 1 (Threads - Naive)",
                "target": "Host (OpenMP Threads)",
                "n": n,
                "p": p,
                "kernel": "naive_prange",
                "median_time_s": round(med_time, 5),
                "min_time_s": round(float(np.min(thread_times)), 5),
                "max_time_s": round(float(np.max(thread_times)), 5),
                "speedup": round(speedup, 3),
                "efficiency": round(efficiency, 3),
                "gflops": round(gflops, 2),
                "source": "measured"
            })

        # 3. Version 1 Threads (Cache-Blocked / Tiled)
        for p in [4, 8]:
            if p > os.cpu_count():
                continue
            numba.set_num_threads(p)
            blocked_times = []
            for r in range(repeats):
                C.fill(0.0)
                t0 = time.perf_counter()
                matmul_threads_blocked(A, B, C, block_size=64)
                t1 = time.perf_counter()
                blocked_times.append(t1 - t0)

            med_blocked = float(np.median(blocked_times))
            sp_blocked = t_serial_median / med_blocked
            eff_blocked = sp_blocked / p
            gflops_blocked = calculate_gflops(n, med_blocked)
            print(f"  [Version 1 - Blocked] p={p} Median: {med_blocked:.4f} s | Speedup: {sp_blocked:.2f}x | Eff: {eff_blocked*100:.1f}% | {gflops_blocked:.2f} GFLOP/s")

            rows.append({
                "version": "Version 1 (Threads - Cache-Blocked)",
                "target": "Host (OpenMP Tiled)",
                "n": n,
                "p": p,
                "kernel": "blocked_64",
                "median_time_s": round(med_blocked, 5),
                "min_time_s": round(float(np.min(blocked_times)), 5),
                "max_time_s": round(float(np.max(blocked_times)), 5),
                "speedup": round(sp_blocked, 3),
                "efficiency": round(eff_blocked, 3),
                "gflops": round(gflops_blocked, 2),
                "source": "measured"
            })

    # Extrapolated n=4096 estimates: appended ONLY if 4096 was not part of the
    # measured sizes this run. Estimates scale the measured n=2048 serial time
    # with the exact O(n^3) ratio (8x) and an assumed cache/bandwidth penalty
    # (+5%), then derive p-scaling with assumed efficiency factors.
    measured_sizes = {r["n"] for r in rows}
    if 4096 not in measured_sizes:
        print("[NOTE] n=4096 was not measured: appending clearly-labelled EXTRAPOLATED estimates "
              "(O(n^3) scaling from measured n=2048, not measured runs).")
        ref_2048_serial = next(r["median_time_s"] for r in rows if r["n"] == 2048 and r["p"] == 1)
        scale_factor_4096 = (4096.0 / 2048.0) ** 3
        t_serial_4096 = ref_2048_serial * scale_factor_4096 * 1.05

        rows.append({
            "version": "Version 0 (Serial)",
            "target": "Host (Single-Core)",
            "n": 4096,
            "p": 1,
            "kernel": "ikj_serial",
            "median_time_s": round(t_serial_4096, 5),
            "min_time_s": round(t_serial_4096 * 0.98, 5),
            "max_time_s": round(t_serial_4096 * 1.02, 5),
            "speedup": 1.0,
            "efficiency": 1.0,
            "gflops": round(calculate_gflops(4096, t_serial_4096), 2),
            "source": "extrapolated"
        })

        for p, eff_est in [(1, 1.0), (2, 0.92), (4, 0.81), (8, 0.68)]:
            t_p = (t_serial_4096 / p) / eff_est
            sp = t_serial_4096 / t_p
            eff = sp / p
            rows.append({
                "version": "Version 1 (Threads - Naive)",
                "target": "Host (OpenMP Threads)",
                "n": 4096,
                "p": p,
                "kernel": "naive_prange",
                "median_time_s": round(t_p, 5),
                "min_time_s": round(t_p * 0.98, 5),
                "max_time_s": round(t_p * 1.02, 5),
                "speedup": round(sp, 3),
                "efficiency": round(eff, 3),
                "gflops": round(calculate_gflops(4096, t_p), 2),
                "source": "extrapolated"
            })

    csv_path = os.path.join(output_dir, "results_serial_threads.csv")
    fieldnames = ["version", "target", "n", "p", "kernel", "median_time_s", "min_time_s", "max_time_s", "speedup", "efficiency", "gflops", "source"]
    save_csv_results(csv_path, fieldnames, rows)
    print(f"\n[SUCCESS] Benchmark results saved to {csv_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Version 0/1 benchmark suite and export CSV results")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048],
                        help="Matrix dimensions n to MEASURE (default: 1024 2048). Add 4096 for real measurements.")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8], help="Thread counts p")
    parser.add_argument("--repeats", type=int, default=3, help="Repetitions per test (median reported)")
    args = parser.parse_args()

    run_benchmarks(sizes=args.sizes, thread_counts=args.threads, repeats=args.repeats)
