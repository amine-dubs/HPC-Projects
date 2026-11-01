"""
pingpong.py - MPI Ping-Pong Benchmark for Latency and Bandwidth (Experiment E5)

Measures point-to-point communication time between 2 ranks across message sizes
from 8 bytes to 8 MB. Saves results to results/raw/pingpong_<target>.csv.
"""

import argparse
import csv
import os
import sys
import numpy as np
from mpi4py import MPI

def run_pingpong(target="local", out_path=None, max_bytes=8*1024*1024, quick=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print("Error: pingpong requires exactly 2 ranks.", file=sys.stderr)
        sys.exit(1)

    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if out_path is None:
        if quick:
            out_path = os.path.join(PROJECT_DIR, "results", "raw", "test_pingpong.csv")
        else:
            out_path = os.path.join(PROJECT_DIR, "results", "raw", f"pingpong_{target}.csv")
    elif not os.path.isabs(out_path):
        out_path = os.path.join(PROJECT_DIR, out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Powers of two from 8 bytes to max_bytes
    byte_sizes = []
    b = 8
    while b <= max_bytes:
        byte_sizes.append(b)
        b *= 2

    if quick:
        byte_sizes = [8, 64, 512, 4096, 65536, 1048576]

    if rank == 0:
        print(f"[E5 Ping-Pong Benchmark] Target: {target}, sizes: {len(byte_sizes)}, max_size: {max(byte_sizes)/(1024*1024):.1f} MB")
        file_exists = os.path.exists(out_path)
        csv_file = open(out_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["target", "bytes", "doubles", "reps", "round_trip_sec", "one_way_sec", "bandwidth_MBs"])

    for num_bytes in byte_sizes:
        num_doubles = max(1, num_bytes // 8)
        # Adapt reps based on message size
        if quick:
            reps = 10
        elif num_bytes <= 1024:
            reps = 1000
        elif num_bytes <= 65536:
            reps = 300
        elif num_bytes <= 1048576:
            reps = 100
        else:
            reps = 30

        send_buf = np.ones(num_doubles, dtype=np.float64)
        recv_buf = np.zeros(num_doubles, dtype=np.float64)

        # Warmup
        for _ in range(5):
            if rank == 0:
                comm.Send(send_buf, dest=1, tag=99)
                comm.Recv(recv_buf, source=1, tag=99)
            elif rank == 1:
                comm.Recv(recv_buf, source=0, tag=99)
                comm.Send(send_buf, dest=0, tag=99)

        comm.Barrier()
        t0 = MPI.Wtime()

        for _ in range(reps):
            if rank == 0:
                comm.Send(send_buf, dest=1, tag=10)
                comm.Recv(recv_buf, source=1, tag=11)
            elif rank == 1:
                comm.Recv(recv_buf, source=0, tag=10)
                comm.Send(send_buf, dest=0, tag=11)

        t1 = MPI.Wtime()
        elapsed = t1 - t0

        if rank == 0:
            round_trip = elapsed / reps
            one_way = round_trip / 2.0
            bandwidth_MBs = (num_bytes / (1024 * 1024)) / one_way if one_way > 0 else 0.0
            print(f"  {num_bytes:10d} bytes | Reps: {reps:5d} | Round-trip: {round_trip*1e6:8.2f} us | One-way: {one_way*1e6:8.2f} us | BW: {bandwidth_MBs:9.2f} MB/s")
            writer.writerow([target, num_bytes, num_doubles, reps, round_trip, one_way, bandwidth_MBs])
            csv_file.flush()

    if rank == 0:
        csv_file.close()
        print(f"Ping-pong benchmark results saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="MPI Ping-Pong Benchmark")
    parser.add_argument("--target", type=str, default="local", help="Deployment target (local, docker, vm, lan)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--max-mb", type=int, default=8, help="Maximum message size in MB")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer sizes and reps")
    args = parser.parse_args()

    max_bytes = args.max_mb * 1024 * 1024
    run_pingpong(target=args.target, out_path=args.out, max_bytes=max_bytes, quick=args.quick)

if __name__ == "__main__":
    main()

