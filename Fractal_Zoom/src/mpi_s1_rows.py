"""MPI Strategy 1: Static row block decomposition (S1).

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
    total_iterations,
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
    parser = argparse.ArgumentParser(description="MPI S1: Static Rows")
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

    # Pre-calculate row ranges for all ranks
    all_row_bounds = [row_block(r, p, h) for r in range(p)]
    r0, r1 = all_row_bounds[rank]
    my_rows = r1 - r0

    # Warm-up call on tiny block
    _ = render_block(0, 4, 0, 4, -2.0, 1.0, 0.01, 0.01, 50)
    comm.Barrier()

    t_render_start = time.perf_counter()

    t_compute = 0.0
    t_wait = 0.0
    iter_sum = 0
    assembled_frames = {} if rank == 0 else None

    # Render each frame
    for k in frame_indices:
        xmin, ymax, dx, dy, maxit = frame_params(k, config)

        tc0 = time.perf_counter()
        block = render_block(r0, r1, 0, w, xmin, ymax, dx, dy, maxit)
        tc1 = time.perf_counter()
        t_compute += (tc1 - tc0)
        iter_sum += int(np.sum(block, dtype=np.int64))

        tw0 = time.perf_counter()
        # Gather row blocks to rank 0
        blocks = comm.gather(block, root=0)
        tw1 = time.perf_counter()
        t_wait += (tw1 - tw0)

        if rank == 0:
            full_frame = np.empty((h, w), dtype=np.int32)
            for r_idx, blk in enumerate(blocks):
                br0, br1 = all_row_bounds[r_idx]
                full_frame[br0:br1, :] = blk
            assembled_frames[k] = full_frame

    t_render_end = time.perf_counter()
    t_render = t_render_end - t_render_start
    t_total = t_compute + t_wait

    # Gather per-rank statistics
    stats = comm.gather({
        "rank": rank,
        "role": "worker",
        "n_tasks": n_frames,
        "t_compute": t_compute,
        "t_wait": t_wait,
        "t_total": t_total,
        "iterations_sum": iter_sum,
        "messages": n_frames,
    }, root=0)

    if rank == 0:
        # Check correctness outside timed region
        ref_hashes = load_serial_hashes()
        checksum_ok = True
        if ref_hashes:
            for k in frame_indices:
                h_val = hash_frame(assembled_frames[k])
                if ref_hashes.get(k) != h_val:
                    checksum_ok = False
                    print(f"ERROR: S1 checksum mismatch on frame {k}!", file=sys.stderr)

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
                "E4", "local", "MPI", "S1", "static_rows",
                p, p, "", n_frames, w, h, args.rep,
                f"{t_render:.6f}", f"{t_wait:.6f}", "", f"{fps:.4f}",
                checksum_ok, oversubscribed, "none",
                f"mpiexec -n {p} python src/mpi_s1_rows.py",
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
                    "E4", "local", "S1", "static_rows", p, "",
                    args.rep, s["rank"], s["role"], s["n_tasks"],
                    f"{s['t_compute']:.6f}", f"{s['t_wait']:.6f}", f"{s['t_total']:.6f}",
                    s["iterations_sum"], s["messages"]
                ])

        print(
            f"[MPI S1 p={p}] Rep {args.rep}: {n_frames} frames in {t_render:.3f} s ({fps:.2f} fps) "
            f"- checksum_ok={checksum_ok}"
        )


if __name__ == "__main__":
    main()

