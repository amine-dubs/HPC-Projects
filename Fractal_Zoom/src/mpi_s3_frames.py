"""MPI Strategy 3: Whole frames per rank (S3, block and cyclic).

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB
"""

import argparse
import csv
import datetime
import os
import sys
import time
from pathlib import Path

# Constrain Numba / OpenMP to 1 thread per rank before import
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI
import numpy as np

# Ensure src is in import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fractal_core import (
    load_config,
    frame_params,
    render_block,
    row_block,
    hash_frame,
)


def load_serial_hashes():
    hash_file = Path("results/serial_hashes.csv")
    if not hash_file.is_file():
        return {}
    hashes = {}
    with open(hash_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            hashes[int(r["frame"])] = r["sha256"]
    return hashes


def main():
    parser = argparse.ArgumentParser(description="MPI S3: Frames per Rank")
    parser.add_argument("--assignment", choices=["block", "cyclic"], default="cyclic", help="Frame assignment")
    parser.add_argument("--rep", type=int, default=1, help="Repetition index")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--busy-out", type=str, default=None, help="Busy time CSV path")
    parser.add_argument("--notes", type=str, default="", help="Run notes")
    parser.add_argument("--hostfile", type=str, default=None, help="Hostfile (optional)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    config = load_config()
    w = config["width"]
    h = config["height"]
    total_frames = config["total_frames"]
    bench_indices = config.get("bench_frame_indices")
    if bench_indices is None:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = bench_indices
    n_frames = len(frame_indices)

    # Determine assigned frames for this rank
    if args.assignment == "block":
        k0, k1 = row_block(rank, p, n_frames)
        my_frame_indices = frame_indices[k0:k1]
    else:  # cyclic
        my_frame_indices = [frame_indices[i] for i in range(rank, n_frames, p)]

    # Warm-up call on tiny block
    _ = render_block(0, 4, 0, 4, -2.0, 1.0, 0.01, 0.01, 50)
    comm.Barrier()

    t_render_start = time.perf_counter()

    t_compute = 0.0
    iter_sum = 0
    my_rendered_frames = {}

    # Rendering phase (pure compute, NO communication)
    for k in my_frame_indices:
        xmin, ymax, dx, dy, maxit = frame_params(k, config)
        tc0 = time.perf_counter()
        counts = render_block(0, h, 0, w, xmin, ymax, dx, dy, maxit)
        tc1 = time.perf_counter()
        t_compute += (tc1 - tc0)
        iter_sum += int(np.sum(counts, dtype=np.int64))
        my_rendered_frames[k] = counts

    # Communication phase: collect all frames on rank 0
    t_comm_start = time.perf_counter()
    assembled_frames = {}
    if rank == 0:
        assembled_frames.update(my_rendered_frames)
        for sender_rank in range(1, p):
            frames_from_sender = comm.recv(source=sender_rank, tag=300)
            assembled_frames.update(frames_from_sender)
    else:
        comm.send(my_rendered_frames, dest=0, tag=300)

    t_render_end = time.perf_counter()
    t_wait = t_render_end - t_comm_start
    t_render = t_render_end - t_render_start
    t_total = t_compute + t_wait

    # Gather per-rank statistics
    stats = comm.gather({
        "rank": rank,
        "role": "worker",
        "n_tasks": len(my_frame_indices),
        "t_compute": t_compute,
        "t_wait": t_wait,
        "t_total": t_total,
        "iterations_sum": iter_sum,
        "messages": 1 if rank != 0 else (p - 1),
    }, root=0)

    if rank == 0:
        # Correctness check outside timed region
        ref_hashes = load_serial_hashes()
        checksum_ok = True
        if ref_hashes:
            for k in frame_indices:
                h_val = hash_frame(assembled_frames[k])
                if ref_hashes.get(k) != h_val:
                    checksum_ok = False
                    print(f"ERROR: S3 checksum mismatch on frame {k}!", file=sys.stderr)

        fps = n_frames / t_render if t_render > 0 else 0.0
        oversubscribed = p > 6

        # Append to mpi_scaling.csv
        out_csv = Path(args.out) if args.out else Path("results/mpi_scaling.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "experiment", "target", "version", "strategy", "assignment",
            "p", "workers", "tile", "frames", "width", "height", "rep",
            "t_render", "t_comm", "t_io", "fps", "checksum_ok",
            "oversubscribed", "threading_layer", "launcher_cmd", "timestamp", "notes"
        ]
        file_exists = out_csv.is_file()
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([
                "E4", "local", "MPI", "S3", args.assignment,
                p, p, "", n_frames, w, h, args.rep,
                f"{t_render:.6f}", f"{t_wait:.6f}", "", f"{fps:.4f}",
                checksum_ok, oversubscribed, "none",
                f"mpiexec -n {p} python src/mpi_s3_frames.py --assignment {args.assignment}",
                datetime.datetime.now().isoformat(), args.notes
            ])

        # Append to busy_time.csv
        busy_csv = Path(args.busy_out) if args.busy_out else Path("results/busy_time.csv")
        busy_header = [
            "experiment", "target", "strategy", "assignment", "p", "tile",
            "rep", "rank", "role", "n_tasks", "t_compute", "t_wait", "t_total",
            "iterations_sum", "messages"
        ]
        busy_exists = busy_csv.is_file()
        with open(busy_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not busy_exists:
                writer.writerow(busy_header)
            for s in stats:
                writer.writerow([
                    "E4", "local", "S3", args.assignment, p, "",
                    args.rep, s["rank"], s["role"], s["n_tasks"],
                    f"{s['t_compute']:.6f}", f"{s['t_wait']:.6f}", f"{s['t_total']:.6f}",
                    s["iterations_sum"], s["messages"]
                ])

        print(
            f"[MPI S3-{args.assignment} p={p}] Rep {args.rep}: {n_frames} frames in {t_render:.3f} s "
            f"({fps:.2f} fps) - checksum_ok={checksum_ok}"
        )


if __name__ == "__main__":
    main()

