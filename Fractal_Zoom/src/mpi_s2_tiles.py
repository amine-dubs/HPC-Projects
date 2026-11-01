"""MPI Strategy 2: Master-Worker dynamic tile load balancing (S2).

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
    tile_list,
    hash_frame,
)

TAG_WORK = 1
TAG_STOP = 2


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
    parser = argparse.ArgumentParser(description="MPI S2: Master-Worker Tiles")
    parser.add_argument("--tile", type=int, default=20, help="Tile size in pixels")
    parser.add_argument("--experiment", type=str, default="E4", choices=["E4", "E5"], help="Experiment ID")
    parser.add_argument("--rep", type=int, default=1, help="Repetition index")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--busy-out", type=str, default=None, help="Busy time CSV path")
    parser.add_argument("--notes", type=str, default="", help="Run notes")
    parser.add_argument("--hostfile", type=str, default=None, help="Hostfile (optional)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    if p < 2:
        if rank == 0:
            print("ERROR: S2 requires at least 2 ranks (1 master + >=1 worker).", file=sys.stderr)
        sys.exit(1)

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
    tile_size = args.tile

    # Precompute frame params for all benchmark frames
    params_cache = {k: frame_params(k, config) for k in frame_indices}

    # Warm-up call on tiny block
    _ = render_block(0, 4, 0, 4, -2.0, 1.0, 0.01, 0.01, 50)
    comm.Barrier()

    if rank == 0:
        # Master rank
        t_render_start = time.perf_counter()
        t_master_wait = 0.0
        master_messages = 0

        # Build task list covering all frames
        tiles_per_frame = tile_list(w, h, tile_size)
        tasks = []
        for k in frame_indices:
            for (r0, r1, c0, c1) in tiles_per_frame:
                tasks.append((k, r0, r1, c0, c1))

        n_tasks_total = len(tasks)
        task_idx = 0
        active_workers = 0

        # Initial distribution
        for worker_rank in range(1, p):
            if task_idx < n_tasks_total:
                comm.send(tasks[task_idx], dest=worker_rank, tag=TAG_WORK)
                task_idx += 1
                active_workers += 1
                master_messages += 1
            else:
                comm.send(None, dest=worker_rank, tag=TAG_STOP)
                master_messages += 1

        assembled_frames = {k: np.empty((h, w), dtype=np.int32) for k in frame_indices}
        status = MPI.Status()

        # Dynamic distribution loop
        while active_workers > 0:
            tw0 = time.perf_counter()
            data = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_WORK, status=status)
            tw1 = time.perf_counter()
            t_master_wait += (tw1 - tw0)
            master_messages += 1

            source_rank = status.Get_source()
            (k, r0, r1, c0, c1), tile_counts = data
            assembled_frames[k][r0:r1, c0:c1] = tile_counts

            if task_idx < n_tasks_total:
                comm.send(tasks[task_idx], dest=source_rank, tag=TAG_WORK)
                task_idx += 1
                master_messages += 1
            else:
                comm.send(None, dest=source_rank, tag=TAG_STOP)
                master_messages += 1
                active_workers -= 1

        t_render_end = time.perf_counter()
        t_render = t_render_end - t_render_start

        # Receive stats from each worker
        worker_stats = []
        for worker_rank in range(1, p):
            w_stat = comm.recv(source=worker_rank, tag=100)
            worker_stats.append(w_stat)

        master_stat = {
            "rank": 0,
            "role": "master",
            "n_tasks": 0,
            "t_compute": 0.0,
            "t_wait": t_master_wait,
            "t_total": t_render,
            "iterations_sum": 0,
            "messages": master_messages,
        }
        all_stats = [master_stat] + worker_stats

        # Correctness check outside timed region
        ref_hashes = load_serial_hashes()
        checksum_ok = True
        if ref_hashes:
            for k in frame_indices:
                h_val = hash_frame(assembled_frames[k])
                if ref_hashes.get(k) != h_val:
                    checksum_ok = False
                    print(f"ERROR: S2 checksum mismatch on frame {k}!", file=sys.stderr)

        fps = n_frames / t_render if t_render > 0 else 0.0
        oversubscribed = p > 6
        workers_count = p - 1

        # Determine output CSV
        if args.out:
            out_csv = Path(args.out)
        elif args.experiment == "E5":
            out_csv = Path("results/tile_size.csv")
        else:
            out_csv = Path("results/mpi_scaling.csv")
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
                args.experiment, "local", "MPI", "S2", "master_worker",
                p, workers_count, tile_size, n_frames, w, h, args.rep,
                f"{t_render:.6f}", f"{t_master_wait:.6f}", "", f"{fps:.4f}",
                checksum_ok, oversubscribed, "none",
                f"mpiexec -n {p} python src/mpi_s2_tiles.py --tile {tile_size}",
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
            for s in all_stats:
                writer.writerow([
                    args.experiment, "local", "S2", "master_worker", p, tile_size,
                    args.rep, s["rank"], s["role"], s["n_tasks"],
                    f"{s['t_compute']:.6f}", f"{s['t_wait']:.6f}", f"{s['t_total']:.6f}",
                    s["iterations_sum"], s["messages"]
                ])

        print(
            f"[MPI S2 p={p} tile={tile_size}] Rep {args.rep}: {n_frames} frames in {t_render:.3f} s "
            f"({fps:.2f} fps, workers={workers_count}) - checksum_ok={checksum_ok}"
        )

    else:
        # Worker rank
        status = MPI.Status()
        t_compute = 0.0
        t_wait = 0.0
        tasks_done = 0
        iter_sum = 0
        worker_messages = 0

        while True:
            tw0 = time.perf_counter()
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tw1 = time.perf_counter()
            t_wait += (tw1 - tw0)
            worker_messages += 1

            tag = status.Get_tag()
            if tag == TAG_STOP or task is None:
                break

            k, r0, r1, c0, c1 = task
            xmin, ymax, dx, dy, maxit = params_cache[k]

            tc0 = time.perf_counter()
            tile_counts = render_block(r0, r1, c0, c1, xmin, ymax, dx, dy, maxit)
            tc1 = time.perf_counter()
            t_compute += (tc1 - tc0)
            iter_sum += int(np.sum(tile_counts, dtype=np.int64))
            tasks_done += 1

            comm.send((task, tile_counts), dest=0, tag=TAG_WORK)
            worker_messages += 1

        t_total = t_compute + t_wait
        # Send stats back to master
        comm.send({
            "rank": rank,
            "role": "worker",
            "n_tasks": tasks_done,
            "t_compute": t_compute,
            "t_wait": t_wait,
            "t_total": t_total,
            "iterations_sum": iter_sum,
            "messages": worker_messages,
        }, dest=0, tag=100)


if __name__ == "__main__":
    main()
