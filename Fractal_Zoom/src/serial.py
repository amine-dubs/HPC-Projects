"""Serial Mandelbrot baseline (V0).

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB
"""

import argparse
import csv
import datetime
import sys
import time
from pathlib import Path
import numpy as np

# Ensure src is in import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fractal_core import (
    load_config,
    frame_params,
    render_block,
    hash_frame,
    total_iterations,
)


def run_serial(config, save_reference=False, rep=1, out_csv=None, notes=""):
    # 1. Warm-up call outside measurement
    _ = render_block(0, 10, 0, 10, -2.0, 1.0, 0.01, 0.01, 50)

    w = config["width"]
    h = config["height"]
    total_frames = config["total_frames"]
    bench_indices = config.get("bench_frame_indices")
    if bench_indices is None:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = bench_indices

    n_frames = len(frame_indices)

    # Reference storage
    hashes_records = []
    ref_dir = Path("results/reference")
    ref_frames = [
        frame_indices[0],
        frame_indices[len(frame_indices) // 3],
        frame_indices[2 * len(frame_indices) // 3],
        frame_indices[-1],
    ]

    # 2. Timed rendering
    t0 = time.perf_counter()
    for k in frame_indices:
        xmin, ymax, dx, dy, maxit = frame_params(k, config)
        counts = render_block(0, h, 0, w, xmin, ymax, dx, dy, maxit)

        if save_reference:
            h_val = hash_frame(counts)
            tot_iter = total_iterations(counts)
            hashes_records.append({
                "frame": k,
                "maxit": maxit,
                "sha256": h_val,
                "total_iterations": tot_iter,
            })
            if k in ref_frames:
                ref_dir.mkdir(parents=True, exist_ok=True)
                np.save(ref_dir / f"frame_{k:03d}.npy", counts)

    t1 = time.perf_counter()
    t_render = t1 - t0
    fps = n_frames / t_render if t_render > 0 else 0.0

    # Save hashes if requested
    if save_reference:
        hash_file = Path("results/serial_hashes.csv")
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(hash_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "maxit", "sha256", "total_iterations"])
            writer.writeheader()
            writer.writerows(hashes_records)
        print(f"Saved {len(hashes_records)} frame hashes to {hash_file}")

    # Record timing in CSV
    if out_csv is None:
        out_csv = Path("results/serial_times.csv")
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
            "E2", "local", "V0", "serial", "none",
            1, 1, "", n_frames, w, h, rep,
            f"{t_render:.6f}", "", "", f"{fps:.4f}", True,
            False, "none", "serial", datetime.datetime.now().isoformat(), notes
        ])

    print(f"[V0 Serial] Rep {rep}: {n_frames} frames in {t_render:.3f} s ({fps:.2f} fps)")
    return t_render


def main():
    parser = argparse.ArgumentParser(description="V0 Serial Mandelbrot Renderer")
    parser.add_argument("--save-reference", action="store_true", help="Save hashes and reference arrays")
    parser.add_argument("--rep", type=int, default=1, help="Repetition index")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--notes", type=str, default="", help="Run notes")
    args = parser.parse_args()

    config = load_config()
    run_serial(config, save_reference=args.save_reference, rep=args.rep, out_csv=args.out, notes=args.notes)


if __name__ == "__main__":
    main()

