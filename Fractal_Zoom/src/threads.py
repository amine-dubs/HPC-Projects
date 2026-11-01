"""Shared-memory threads implementation using Numba prange (V1).

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB
"""

import argparse
import csv
import datetime
import os
import random
import sys
import time
from pathlib import Path

# Ensure OpenMP threading layer is prioritized if possible
if "NUMBA_THREADING_LAYER" not in os.environ:
    os.environ["NUMBA_THREADING_LAYER"] = "omp"

import numba
from numba import njit, prange
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


@njit(parallel=True)
def render_frame_rows(H, W, xmin, ymax, dx, dy, maxit):
    """Variant A: parallel over image rows using prange.

    F-08: pixels are computed and written directly into the preallocated
    output array instead of calling render_block per row (which allocated a
    fresh 1-row array on every loop iteration, 120 x 360 = 43,200 times per
    run). The escape loop mirrors fractal_core.render_block exactly (same
    <= 4.0 bound, same float operation order) so the output stays bitwise
    identical to the serial V0 reference.
    """
    out = np.empty((H, W), dtype=np.int32)
    for j in prange(H):
        ci = ymax - j * dy
        for i in range(W):
            cr = xmin + i * dx
            zr = 0.0
            zi = 0.0
            k = 0
            while k < maxit and (zr * zr + zi * zi) <= 4.0:
                zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                k += 1
            out[j, i] = k
    return out


@njit(parallel=True)
def render_frame_tiles(H, W, tiles, xmin, ymax, dx, dy, maxit):
    """Variant B: parallel over shuffled tiles using prange."""
    out = np.empty((H, W), dtype=np.int32)
    num_tiles = tiles.shape[0]
    for t in prange(num_tiles):
        r0 = tiles[t, 0]
        r1 = tiles[t, 1]
        c0 = tiles[t, 2]
        c1 = tiles[t, 3]
        tile = render_block(r0, r1, c0, c1, xmin, ymax, dx, dy, maxit)
        for jr in range(r1 - r0):
            for ic in range(c1 - c0):
                out[r0 + jr, c0 + ic] = tile[jr, ic]
    return out


def load_serial_hashes():
    """Load reference hashes from results/serial_hashes.csv."""
    hash_file = Path("results/serial_hashes.csv")
    if not hash_file.is_file():
        return {}
    hashes = {}
    with open(hash_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            hashes[int(r["frame"])] = r["sha256"]
    return hashes


def run_threads(variant, p, tile_size=20, rep=1, out_csv=None, notes=""):
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

    # Configure threads
    numba.set_num_threads(p)
    actual_threads = numba.get_num_threads()
    assert actual_threads == p, f"Thread count mismatch: requested {p}, got {actual_threads}"

    # Prepare tiles if Variant B
    tiles_arr = None
    if variant == "B":
        tiles = tile_list(w, h, tile_size)
        rng = random.Random(config.get("shuffle_seed", 42))
        rng.shuffle(tiles)
        tiles_arr = np.array(tiles, dtype=np.int32)

    # Warm-up call
    xmin_w, ymax_w, dx_w, dy_w, maxit_w = frame_params(0, config)
    if variant == "A":
        _ = render_frame_rows(h, w, xmin_w, ymax_w, dx_w, dy_w, 50)
    else:
        _ = render_frame_tiles(h, w, tiles_arr, xmin_w, ymax_w, dx_w, dy_w, 50)

    threading_layer = numba.threading_layer()
    ref_hashes = load_serial_hashes()
    checksum_all_ok = True

    # Timed loop
    t0 = time.perf_counter()
    for k in frame_indices:
        xmin, ymax, dx, dy, maxit = frame_params(k, config)
        if variant == "A":
            counts = render_frame_rows(h, w, xmin, ymax, dx, dy, maxit)
        else:
            counts = render_frame_tiles(h, w, tiles_arr, xmin, ymax, dx, dy, maxit)

        if ref_hashes:
            h_val = hash_frame(counts)
            if ref_hashes.get(k) != h_val:
                checksum_all_ok = False
                print(f"ERROR: Checksum mismatch on frame {k}!", file=sys.stderr)

    t1 = time.perf_counter()
    t_render = t1 - t0
    fps = n_frames / t_render if t_render > 0 else 0.0

    # Oversubscribed check (host has 6 physical cores, 12 logical cores)
    # Marked oversubscribed if p > physical cores (or > logical cores)
    oversubscribed = p > 6

    if out_csv is None:
        out_csv = Path("results/threads.csv")
    else:
        out_csv = Path(out_csv)
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
            "E3", "local", "V1", f"threads_{variant}",
            "rows" if variant == "A" else "shuffled_tiles",
            p, p, "" if variant == "A" else tile_size,
            n_frames, w, h, rep,
            f"{t_render:.6f}", "", "", f"{fps:.4f}",
            checksum_all_ok, oversubscribed, threading_layer,
            f"python src/threads.py --variant {variant} -p {p}",
            datetime.datetime.now().isoformat(), notes
        ])

    print(
        f"[V1-{variant} Threads p={p} tile={tile_size if variant == 'B' else 'N/A'}] "
        f"Rep {rep}: {n_frames} frames in {t_render:.3f} s ({fps:.2f} fps) - checksum_ok={checksum_all_ok}"
    )
    return t_render, checksum_all_ok


def main():
    parser = argparse.ArgumentParser(description="V1 Threads Mandelbrot Renderer")
    parser.add_argument("--variant", choices=["A", "B"], default="A", help="Variant A (rows) or B (shuffled tiles)")
    parser.add_argument("-p", type=int, default=4, help="Thread count")
    parser.add_argument("--tile", type=int, default=20, help="Tile size for Variant B")
    parser.add_argument("--rep", type=int, default=1, help="Repetition index")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--notes", type=str, default="", help="Run notes")
    args = parser.parse_args()

    run_threads(args.variant, args.p, args.tile, args.rep, args.out, args.notes)


if __name__ == "__main__":
    main()

