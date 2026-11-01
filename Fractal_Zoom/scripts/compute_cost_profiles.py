"""Experiment E6: Compute cost profiles from iteration counts.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB

Decision: Cost is defined purely by iteration counts (hardware-independent).
This isolates theoretical computational imbalance from system noise.
"""

import csv
import sys
from pathlib import Path
import numpy as np

# Ensure src is in import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fractal_core import (
    load_config,
    frame_params,
    render_block,
    row_block,
    tile_list,
)


def compute_cost_profiles():
    config = load_config()
    w = config["width"]
    h = config["height"]
    total_frames = config["total_frames"]

    sample_frames = [0, total_frames // 2, total_frames - 1]
    p_values = config["process_counts"]

    rowblock_csv = Path("results/cost_per_rowblock.csv")
    rowblock_csv.parent.mkdir(parents=True, exist_ok=True)

    rowblock_records = []
    tile_cost_data = {}

    for k in sample_frames:
        xmin, ymax, dx, dy, maxit = frame_params(k, config)
        counts = render_block(0, h, 0, w, xmin, ymax, dx, dy, maxit)
        frame_total_iters = int(np.sum(counts, dtype=np.int64))

        # 1. Row block costs for S1
        for p in p_values:
            for r in range(p):
                r0, r1 = row_block(r, p, h)
                block_iters = int(np.sum(counts[r0:r1, :], dtype=np.int64))
                share = block_iters / frame_total_iters if frame_total_iters > 0 else 0.0
                rowblock_records.append({
                    "frame": k,
                    "maxit": maxit,
                    "p": p,
                    "rank": r,
                    "row_start": r0,
                    "row_stop": r1,
                    "n_rows": r1 - r0,
                    "iterations_sum": block_iters,
                    "iterations_share": f"{share:.6f}",
                })

        # 2. Tile cost grids for 20px and 40px
        for t_sz in [20, 40]:
            ny = int(np.ceil(h / t_sz))
            nx = int(np.ceil(w / t_sz))
            grid = np.zeros((ny, nx), dtype=np.int64)
            for j, r0 in enumerate(range(0, h, t_sz)):
                r1 = min(r0 + t_sz, h)
                for i, c0 in enumerate(range(0, w, t_sz)):
                    c1 = min(c0 + t_sz, w)
                    grid[j, i] = np.sum(counts[r0:r1, c0:c1], dtype=np.int64)
            tile_cost_data[f"frame_{k}_tile_{t_sz}"] = grid

    # Save row block CSV
    fieldnames = [
        "frame", "maxit", "p", "rank", "row_start", "row_stop",
        "n_rows", "iterations_sum", "iterations_share"
    ]
    with open(rowblock_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rowblock_records)
    print(f"Cost per row block saved to {rowblock_csv}")

    # Save tile cost NPZ
    tile_npz = Path("results/cost_per_tile.npz")
    np.savez_compressed(tile_npz, **tile_cost_data)
    print(f"Cost per tile grids saved to {tile_npz}")


if __name__ == "__main__":
    compute_cost_profiles()

