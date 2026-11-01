"""Core Mandelbrot computation, coordinates, configuration, and helpers.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB
"""

import json
import math
import hashlib
from pathlib import Path
import numpy as np
from numba import njit


@njit
def render_block(row0, row1, col0, col1, xmin, ymax, dx, dy, maxit):
    """Compute Mandelbrot iteration counts for a rectangular pixel block.
    
    Decision: No fastmath is enabled to ensure deterministic IEEE 754 float
    computations across all ranks, threads, and tile sizes.
    Coordinates use global indices (i, j) rather than relative tile bounds to
    eliminate floating-point rounding discrepancies across tile edges.
    """
    out = np.empty((row1 - row0, col1 - col0), dtype=np.int32)
    for j in range(row0, row1):
        ci = ymax - j * dy
        for i in range(col0, col1):
            cr = xmin + i * dx
            zr = 0.0
            zi = 0.0
            k = 0
            while k < maxit and (zr * zr + zi * zi) <= 4.0:
                zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                k += 1
            out[j - row0, i - col0] = k
    return out


def load_config(config_path=None):
    """Load configuration from config.json."""
    if config_path is None:
        # Search relative to this file or current working directory
        candidates = [
            Path(__file__).resolve().parent.parent / "config.json",
            Path("config.json"),
            Path("Fractal_Zoom/config.json"),
        ]
        for c in candidates:
            if c.is_file():
                config_path = c
                break
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def maxit_for_frame(k, config=None):
    """Linear iteration limit schedule from maxit_start to maxit_end."""
    if config is None:
        config = load_config()
    total_frames = config["total_frames"]
    m_start = config["maxit_start"]
    m_end = config["maxit_end"]
    if total_frames <= 1:
        return m_start
    return int(round(m_start + (m_end - m_start) * k / (total_frames - 1)))


def frame_params(k, config=None):
    """Compute (xmin, ymax, dx, dy, maxit) for frame k.
    
    Decision: Window width shrinks by shrink_factor per frame.
    Row 0 maps to ymax (top of image), and col 0 maps to xmin (left of image).
    Pixel aspect ratio is maintained at 1:1 (dx == dy).
    """
    if config is None:
        config = load_config()

    w = config["width"]
    h = config["height"]
    center_x, center_y = config["zoom_center"]
    shrink = config["shrink_factor"]
    init_width = config["initial_width"]

    frame_width = init_width * (shrink ** k)
    frame_height = frame_width * (h / w)

    dx = frame_width / w
    dy = frame_height / h

    # Center coordinates on (center_x, center_y)
    xmin = center_x - (w - 1) / 2.0 * dx
    ymax = center_y + (h - 1) / 2.0 * dy
    maxit = maxit_for_frame(k, config)

    return xmin, ymax, dx, dy, maxit


def row_block(rank, p, H):
    """Partition H rows into p blocks, giving one extra row to the first H % p ranks."""
    rows_per = H // p
    remainder = H % p
    if rank < remainder:
        r0 = rank * (rows_per + 1)
        r1 = r0 + rows_per + 1
    else:
        r0 = remainder * (rows_per + 1) + (rank - remainder) * rows_per
        r1 = r0 + rows_per
    return r0, r1


def tile_list(W, H, tile_size):
    """Generate list of (r0, r1, c0, c1) tiles covering W x H, handling edge tiles."""
    tiles = []
    for r0 in range(0, H, tile_size):
        r1 = min(r0 + tile_size, H)
        for c0 in range(0, W, tile_size):
            c1 = min(c0 + tile_size, W)
            tiles.append((r0, r1, c0, c1))
    return tiles


def hash_frame(counts):
    """Compute deterministic SHA-256 hash of int32 iteration count array."""
    arr = np.ascontiguousarray(counts, dtype=np.int32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def total_iterations(counts):
    """Compute total iteration count as hardware-independent cost proxy."""
    return int(np.sum(counts, dtype=np.int64))

