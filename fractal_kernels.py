"""Numba Mandelbrot kernels corresponding to the serial reference renderer."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit
def _mandelbrot_serial_kernel(
    x_values: np.ndarray,
    y_values: np.ndarray,
    max_iterations: int,
) -> np.ndarray:
    """Compute iteration counts serially inside compiled Numba code."""
    height = y_values.size
    width = x_values.size
    counts = np.full((height, width), max_iterations, dtype=np.int32)

    for row in range(height):
        ci = y_values[row]
        for column in range(width):
            cr = x_values[column]
            zr = 0.0
            zi = 0.0
            for iteration in range(1, max_iterations + 1):
                zr_next = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = zr_next
                if zr * zr + zi * zi > 4.0:
                    counts[row, column] = iteration
                    break
    return counts


@njit(parallel=True)
def _mandelbrot_parallel_kernel(
    x_values: np.ndarray,
    y_values: np.ndarray,
    max_iterations: int,
) -> np.ndarray:
    """Compute iteration counts in parallel over independent image rows."""
    height = y_values.size
    width = x_values.size
    counts = np.full((height, width), max_iterations, dtype=np.int32)

    for row in prange(height):
        ci = y_values[row]
        for column in range(width):
            cr = x_values[column]
            zr = 0.0
            zi = 0.0
            for iteration in range(1, max_iterations + 1):
                zr_next = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = zr_next
                if zr * zr + zi * zi > 4.0:
                    counts[row, column] = iteration
                    break
    return counts


@njit(parallel=True)
def _mandelbrot_tiled_parallel_kernel(
    x_values: np.ndarray,
    y_values: np.ndarray,
    max_iterations: int,
    tile_rows: np.ndarray,
    tile_columns: np.ndarray,
    tile_size: int,
) -> np.ndarray:
    """Compute independently shuffled tiles in parallel."""
    height = y_values.size
    width = x_values.size
    counts = np.full((height, width), max_iterations, dtype=np.int32)

    for task in prange(tile_rows.size):
        row_start = tile_rows[task]
        column_start = tile_columns[task]
        row_stop = min(row_start + tile_size, height)
        column_stop = min(column_start + tile_size, width)
        for row in range(row_start, row_stop):
            ci = y_values[row]
            for column in range(column_start, column_stop):
                cr = x_values[column]
                zr = 0.0
                zi = 0.0
                for iteration in range(1, max_iterations + 1):
                    zr_next = zr * zr - zi * zi + cr
                    zi = 2.0 * zr * zi + ci
                    zr = zr_next
                    if zr * zr + zi * zi > 4.0:
                        counts[row, column] = iteration
                        break
    return counts


def _coordinates(
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build coordinates using the same mapping as ``fractal.py``."""
    x_values = np.linspace(x_min, x_max, width, dtype=np.float64)
    y_values = np.linspace(y_max, y_min, height, dtype=np.float64)
    return x_values, y_values


def render_numba_serial(
    width: int,
    height: int,
    max_iterations: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Return Mandelbrot counts using a compiled, non-parallel kernel."""
    x_values, y_values = _coordinates(
        width, height, x_min, x_max, y_min, y_max
    )
    return _mandelbrot_serial_kernel(x_values, y_values, max_iterations)


def render_numba_parallel(
    width: int,
    height: int,
    max_iterations: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Return Mandelbrot counts using a compiled kernel parallelized by rows."""
    x_values, y_values = _coordinates(
        width, height, x_min, x_max, y_min, y_max
    )
    return _mandelbrot_parallel_kernel(x_values, y_values, max_iterations)


def _shuffled_tile_schedule(
    width: int, height: int, tile_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a reproducibly shuffled list of tile origins."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    row_starts = range(0, height, tile_size)
    column_starts = range(0, width, tile_size)
    origins = [(row, column) for row in row_starts for column in column_starts]
    generator = np.random.default_rng(seed)
    generator.shuffle(origins)
    tile_rows = np.array([origin[0] for origin in origins], dtype=np.int64)
    tile_columns = np.array([origin[1] for origin in origins], dtype=np.int64)
    return tile_rows, tile_columns


def render_numba_tiled_parallel(
    width: int,
    height: int,
    max_iterations: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    tile_size: int = 16,
    shuffle_seed: int = 0,
) -> np.ndarray:
    """Return counts using parallel, reproducibly shuffled square tiles."""
    x_values, y_values = _coordinates(
        width, height, x_min, x_max, y_min, y_max
    )
    tile_rows, tile_columns = _shuffled_tile_schedule(
        width, height, tile_size, shuffle_seed
    )
    return _mandelbrot_tiled_parallel_kernel(
        x_values,
        y_values,
        max_iterations,
        tile_rows,
        tile_columns,
        tile_size,
    )
