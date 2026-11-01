"""
Benchmark Driver for the Local MPI Implementation
High-Performance Computing (HPC) Project: Matrix Multiplication

Runs src/mpi_matmul.py under mpiexec for a grid of (n, p) configurations and
repeats, collects the per-phase timing metrics written by rank 0 (JSON), and
exports a single consolidated CSV: data/results_mpi_local.csv.

For every (n, p) the run whose total wall time is the MEDIAN of the repeats is
reported (median-run record), as required by the project methodology
(repeat each run at least three times and report the median).

Usage:
    python src/benchmark_mpi.py --sizes 1024 2048 --procs 1 2 4 8 --repeats 3
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

FIELDNAMES = [
    "n", "p", "kernel", "repeats",
    "t_bcast_s", "t_scatter_s", "t_compute_s", "t_gather_s",
    "median_total_s", "min_total_s", "max_total_s",
    "comp_ratio_pct", "comm_ratio_pct", "gflops", "verified", "max_abs_err"
]


def run_single_mpi(n: int, p: int, kernel: str, verify: bool) -> dict:
    """Run mpiexec -n <p> for size <n> once and return the rank-0 metrics dict."""
    handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    handle.close()

    cmd = [
        "mpiexec", "-n", str(p),
        sys.executable, os.path.join("src", "mpi_matmul.py"),
        "--n", str(n),
        "--kernel", kernel,
        "--output-json", handle.name,
    ]
    if not verify:
        cmd.append("--no-verify")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"mpiexec returned code {proc.returncode} for n={n}, p={p}.\n"
                f"--- stdout (tail) ---\n{proc.stdout[-2000:]}\n"
                f"--- stderr (tail) ---\n{proc.stderr[-2000:]}"
            )
        with open(handle.name, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    finally:
        try:
            os.remove(handle.name)
        except OSError:
            pass
    return metrics


def benchmark_mpi(sizes, proc_counts, repeats, kernel="numba", verify=True,
                  output_csv="data/results_mpi_local.csv"):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    rows = []

    print("\n=================================================================")
    print("BENCHMARKING: Local MPI Matrix Multiplication (mpi4py)")
    print(f"Sizes: {sizes} | Processes: {proc_counts} | Repeats: {repeats} | Kernel: {kernel}")
    print("=================================================================")

    for n in sizes:
        if n % max(proc_counts) != 0:
            print(f"[SKIP] n={n} is not divisible by the largest process count {max(proc_counts)}.")
            continue
        for p in proc_counts:
            if n % p != 0:
                print(f"[SKIP] n={n} not divisible by p={p}.")
                continue
            print(f"\n>>> n={n} | p={p} | {repeats} repeats ...")
            records = []
            for r in range(repeats):
                metrics = run_single_mpi(n=n, p=p, kernel=kernel, verify=verify)
                records.append(metrics)
                print(f"    Run {r+1}/{repeats}: total={metrics['t_total']:.4f} s "
                      f"(comp={metrics['t_compute']:.4f} s, comm_ratio={metrics['comm_ratio_pct']:.1f}%) "
                      f"| {metrics['gflops']:.2f} GFLOP/s")

            # Median-run record: the repetition whose total time is the median.
            records.sort(key=lambda m: m["t_total"])
            med = records[len(records) // 2] if len(records) % 2 == 1 else records[len(records) // 2 - 1]
            totals = [m["t_total"] for m in records]

            rows.append({
                "n": n,
                "p": p,
                "kernel": kernel,
                "repeats": repeats,
                "t_bcast_s": round(med["t_bcast"], 5),
                "t_scatter_s": round(med["t_scatter"], 5),
                "t_compute_s": round(med["t_compute"], 5),
                "t_gather_s": round(med["t_gather"], 5),
                "median_total_s": round(med["t_total"], 5),
                "min_total_s": round(min(totals), 5),
                "max_total_s": round(max(totals), 5),
                "comp_ratio_pct": round(med["comp_ratio_pct"], 1),
                "comm_ratio_pct": round(med["comm_ratio_pct"], 1),
                "gflops": round(med["gflops"], 2),
                "verified": med["verified"],
                "max_abs_err": med.get("max_abs_err", 0.0),
            })
            print(f"  --> n={n} p={p}: median total={med['t_total']:.4f} s | "
                  f"compute {med['comp_ratio_pct']:.1f}% | communication {med['comm_ratio_pct']:.1f}%")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[SUCCESS] Saved {len(rows)} records to {output_csv}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark local MPI matmul across (n, p) and export CSV")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048], help="Matrix dimensions n")
    parser.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8], help="MPI process counts p")
    parser.add_argument("--repeats", type=int, default=3, help="Repetitions per (n, p) pair")
    parser.add_argument("--kernel", type=str, choices=["numba", "numpy"], default="numba", help="Local compute kernel")
    parser.add_argument("--no-verify", action="store_true", help="Skip numerical verification")
    parser.add_argument("--output-csv", type=str, default="data/results_mpi_local.csv", help="Output CSV path")
    args = parser.parse_args()

    benchmark_mpi(sizes=args.sizes, proc_counts=args.procs, repeats=args.repeats,
                  kernel=args.kernel, verify=not args.no_verify,
                  output_csv=args.output_csv)
