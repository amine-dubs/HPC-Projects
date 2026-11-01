"""MPI Ping-Pong benchmark to measure communication latency and bandwidth.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB

Decision: On a single machine, this measures local shared-memory transfer
speeds between processes, not physical Ethernet networking.
Supports an optional --hostfile for multi-node clusters in subsequent phases.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from mpi4py import MPI
import numpy as np


def run_pingpong(hostfile=None, out_csv=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print("ERROR: Ping-pong requires exactly 2 ranks.", file=sys.stderr)
        sys.exit(1)

    # Message sizes: 1 B to 8 MiB
    sizes = [
        1,
        8,
        64,
        512,
        4 * 1024,
        32 * 1024,
        256 * 1024,
        1024 * 1024,
        2 * 1024 * 1024,
        8 * 1024 * 1024,
    ]

    # Repetitions schedule: many for small, fewer for large
    def reps_for_size(s):
        if s <= 64:
            return 500
        elif s <= 4096:
            return 200
        elif s <= 256 * 1024:
            return 50
        elif s <= 2 * 1024 * 1024:
            return 20
        else:
            return 10

    results = []

    # Warm-up exchange
    warm_buf = np.zeros(1024, dtype=np.uint8)
    for _ in range(10):
        if rank == 0:
            comm.Send(warm_buf, dest=1, tag=99)
            comm.Recv(warm_buf, source=1, tag=99)
        else:
            comm.Recv(warm_buf, source=0, tag=99)
            comm.Send(warm_buf, dest=0, tag=99)
    comm.Barrier()

    for s in sizes:
        reps = reps_for_size(s)
        buf = np.ones(s, dtype=np.uint8)

        comm.Barrier()
        t0 = time.perf_counter()

        if rank == 0:
            for _ in range(reps):
                comm.Send(buf, dest=1, tag=s % 1000)
                comm.Recv(buf, source=1, tag=s % 1000)
            t1 = time.perf_counter()
            total_time = t1 - t0
            round_trip = total_time / reps
            latency = round_trip / 2.0  # one-way time
            bandwidth_mbps = (s / latency) / (1024.0 * 1024.0) if latency > 0 else 0.0

            results.append({
                "message_size_bytes": s,
                "repetitions": reps,
                "round_trip_sec": round_trip,
                "latency_sec": latency,
                "bandwidth_mbytes_per_sec": bandwidth_mbps,
            })
            print(
                f"[Ping-Pong] Size: {s:>8} B | Latency: {latency*1e6:>8.2f} us | "
                f"Bandwidth: {bandwidth_mbps:>8.2f} MB/s ({reps} reps)"
            )
        else:
            for _ in range(reps):
                comm.Recv(buf, source=0, tag=s % 1000)
                comm.Send(buf, dest=0, tag=s % 1000)

    if rank == 0:
        if out_csv is None:
            out_csv = Path("results/pingpong.csv")
        else:
            out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "message_size_bytes",
            "repetitions",
            "round_trip_sec",
            "latency_sec",
            "bandwidth_mbytes_per_sec",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Ping-pong results saved to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="MPI Ping-Pong Benchmark")
    parser.add_argument("--hostfile", type=str, default=None, help="Hostfile for multi-node execution")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    run_pingpong(args.hostfile, args.out)


if __name__ == "__main__":
    main()

