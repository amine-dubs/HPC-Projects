"""Production run and movie generator (E10).

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio

# Ensure src is in import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fractal_core import (
    load_config,
    frame_params,
    render_block,
    hash_frame,
)
from threads import render_frame_tiles
import random
from fractal_core import tile_list


def color_map_counts(counts, maxit):
    """Convert iteration counts to RGB image without flickering.
    
    Decision: Cyclic colormap based on modulo arithmetic ensures that colours
    do not shift or flicker as the iteration limit increases from 200 to 1000.
    Interior points (count == maxit) are rendered pure black.
    """
    cmap = plt.get_cmap("magma")
    norm_val = (counts % 64) / 63.0
    rgb = (cmap(norm_val)[:, :, :3] * 255).astype(np.uint8)
    rgb[counts == maxit] = [0, 0, 0]
    return rgb


def generate_production_movie(p=6, tile_size=20, rep=1, out_csv=None, notes=""):
    config = load_config()
    w = config["width"]
    h = config["height"]
    total_frames = config["total_frames"]
    fps_target = config.get("fps", 24)

    print(f"Generating full {total_frames}-frame fractal zoom movie ({w}x{h})...")

    # 1. Timed rendering phase
    # Prepare tiles for multi-threaded rendering (fastest local execution)
    tiles = tile_list(w, h, tile_size)
    rng = random.Random(config.get("shuffle_seed", 42))
    rng.shuffle(tiles)
    tiles_arr = np.array(tiles, dtype=np.int32)

    import numba
    numba.set_num_threads(p)

    # Warm-up call
    _ = render_frame_tiles(h, w, tiles_arr, -2.0, 1.0, 0.01, 0.01, 50)

    t0_render = time.perf_counter()
    all_counts = []
    for k in range(total_frames):
        xmin, ymax, dx, dy, maxit = frame_params(k, config)
        counts = render_frame_tiles(h, w, tiles_arr, xmin, ymax, dx, dy, maxit)
        all_counts.append((k, counts, maxit))
    t1_render = time.perf_counter()
    t_render = t1_render - t0_render
    fps_render = total_frames / t_render if t_render > 0 else 0.0

    print(f"Rendering complete: {t_render:.3f} s ({fps_render:.2f} render fps)")

    # 2. Timed I/O & Movie Assembly phase
    t0_io = time.perf_counter()
    frames_dir = Path("output/frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    rgb_frames = []
    for k, counts, maxit in all_counts:
        rgb = color_map_counts(counts, maxit)
        rgb_frames.append(rgb)
        frame_path = frames_dir / f"frame_{k:03d}.png"
        imageio.imwrite(frame_path, rgb)

    # Assemble GIF
    gif_path = Path("output/zoom.gif")
    # GIF delays in imageio are in seconds; 24 fps is ~0.04167s (GIF rounds to centiseconds: 40ms = 25fps)
    imageio.mimsave(gif_path, rgb_frames, fps=fps_target, loop=0)

    # Assemble MP4 with H.264, macro_block_size=1 to preserve exact 640x360
    mp4_path = Path("output/zoom.mp4")
    try:
        writer = imageio.get_writer(
            mp4_path,
            fps=fps_target,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1
        )
        for rgb in rgb_frames:
            writer.append_data(rgb)
        writer.close()
    except Exception as e:
        print(f"Warning: ffmpeg direct writer error: {e}. Falling back to mimsave.", file=sys.stderr)
        imageio.mimsave(mp4_path, rgb_frames, fps=fps_target, macro_block_size=1)

    t1_io = time.perf_counter()
    t_io = t1_io - t0_io
    total_time = t_render + t_io
    fps_total = total_frames / total_time if total_time > 0 else 0.0

    print(f"I/O and video encoding complete: {t_io:.3f} s")
    print(f"Total production time: {total_time:.3f} s (overall {fps_total:.2f} fps)")

    # Verify movie readability and frame count
    reader = imageio.get_reader(mp4_path)
    read_frames = reader.count_frames()
    reader.close()
    print(f"Verification: {mp4_path} exists and contains {read_frames} frames.")

    # Record in production.csv
    if out_csv is None:
        out_csv = Path("results/production.csv")
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
            "E10", "local", "V1", "threads_B", "shuffled_tiles",
            p, p, tile_size, total_frames, w, h, rep,
            f"{t_render:.6f}", "", f"{t_io:.6f}", f"{fps_render:.4f}",
            True, False, numba.threading_layer(),
            f"python src/make_movie.py -p {p} --tile {tile_size}",
            datetime.datetime.now().isoformat(), notes
        ])

    return t_render, t_io


def main():
    parser = argparse.ArgumentParser(description="Generate Fractal Zoom Movie and production timings")
    parser.add_argument("-p", type=int, default=6, help="Process or thread count")
    parser.add_argument("--tile", type=int, default=20, help="Tile size")
    parser.add_argument("--rep", type=int, default=1, help="Repetition index")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--notes", type=str, default="", help="Notes")
    args = parser.parse_args()

    generate_production_movie(args.p, args.tile, args.rep, args.out, args.notes)


if __name__ == "__main__":
    main()

