"""
MPI Ping-Pong Benchmark for Local Communication Characterization
High-Performance Computing (HPC) Project

This program measures point-to-point MPI communication between two ranks on one machine.
It models transfer time using the classical Hockney model:
    T(m) = alpha + m / beta
where:
    alpha : Latency / startup overhead (seconds or microseconds)
    m     : Message size in bytes
    beta  : Asymptotic communication bandwidth (Bytes/sec or MB/s)

Run Command:
    mpiexec -n 2 python src/pingpong.py --output-csv data/pingpong_local.csv
"""

import sys
import os
import time
import argparse
import csv
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_pingpong_mpi(output_csv: str = "data/pingpong_local.csv", num_trials: int = 50):
    try:
        from mpi4py import MPI
    except ImportError:
        print("[ERROR] mpi4py is required to run the live MPI Ping-Pong test.")
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print(f"[ERROR] Ping-pong benchmark must be run with exactly 2 processes (got size={size}).")
        sys.exit(1)

    # Message sizes from 8 bytes to 32 MB (powers of 2)
    message_sizes = [
        8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304,
        8388608, 16777216, 33554432
    ]

    results = []

    if rank == 0:
        print("\n========================================================")
        print("Starting Local MPI Ping-Pong Latency & Bandwidth Benchmark")
        print(f"Ranks: 2 | Trials per message: {num_trials}")
        print("========================================================")
        print(f"{'Size (Bytes)':<14} {'Size (Readable)':<16} {'One-Way Time (us)':<18} {'Bandwidth (MB/s)':<18}")
        print("-" * 68)

    for sz in message_sizes:
        send_buf = np.ones(sz, dtype=np.uint8)
        recv_buf = np.empty(sz, dtype=np.uint8)

        # Warm-up transfers
        for _ in range(5):
            if rank == 0:
                comm.Send([send_buf, MPI.BYTE], dest=1, tag=99)
                comm.Recv([recv_buf, MPI.BYTE], source=1, tag=99)
            elif rank == 1:
                comm.Recv([recv_buf, MPI.BYTE], source=0, tag=99)
                comm.Send([send_buf, MPI.BYTE], dest=0, tag=99)

        comm.Barrier()

        # Timed ping-pong iterations
        t0 = MPI.Wtime()
        for _ in range(num_trials):
            if rank == 0:
                comm.Send([send_buf, MPI.BYTE], dest=1, tag=10)
                comm.Recv([recv_buf, MPI.BYTE], source=1, tag=10)
            elif rank == 1:
                comm.Recv([recv_buf, MPI.BYTE], source=0, tag=10)
                comm.Send([send_buf, MPI.BYTE], dest=0, tag=10)
        t1 = MPI.Wtime()

        if rank == 0:
            total_rtt = t1 - t0
            one_way_time = total_rtt / (2.0 * num_trials)  # in seconds
            one_way_us = one_way_time * 1e6
            bandwidth_mb_s = (sz / (1024.0 * 1024.0)) / one_way_time if one_way_time > 0 else 0.0

            # Format readable size
            if sz < 1024:
                size_str = f"{sz} B"
            elif sz < 1048576:
                size_str = f"{sz / 1024.0:.1f} KB"
            else:
                size_str = f"{sz / (1024.0 * 1024.0):.1f} MB"

            print(f"{sz:<14} {size_str:<16} {one_way_us:<18.2f} {bandwidth_mb_s:<18.2f}")

            results.append({
                "message_bytes": sz,
                "size_readable": size_str,
                "one_way_time_s": one_way_time,
                "one_way_us": one_way_us,
                "bandwidth_mb_s": bandwidth_mb_s
            })

    if rank == 0:
        # Fit Hockney Model: Time = alpha + size * (1 / beta)
        # Using linear regression on large sizes for beta and smallest size for alpha
        alpha_lat_us = results[0]["one_way_us"]
        max_bw = max(r["bandwidth_mb_s"] for r in results)

        print("\n--- Local MPI Communication Summary ---")
        print(f"  Latency (alpha):            {alpha_lat_us:.2f} us")
        print(f"  Max Bandwidth (beta):       {max_bw:.2f} MB/s ({max_bw * 8 / 1000.0:.2f} Gbps)")

        if output_csv:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["message_bytes", "size_readable", "one_way_time_s", "one_way_us", "bandwidth_mb_s"])
                writer.writeheader()
                writer.writerows(results)
            print(f"[INFO] Ping-pong benchmark data written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local MPI Ping-Pong Latency & Bandwidth Benchmark")
    parser.add_argument("--output-csv", type=str, default="data/pingpong_local.csv", help="Output CSV path")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials per message size")
    args = parser.parse_args()

    run_pingpong_mpi(output_csv=args.output_csv, num_trials=args.trials)
