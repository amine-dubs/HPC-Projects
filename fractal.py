"""Serial Mandelbrot renderer and command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
DEFAULT_MAX_ITERATIONS = 200
DEFAULT_BOUNDS = (-2.2, 0.8, -1.0, 1.0)
DEFAULT_OUTPUT = Path("results/fractal.png")


def render_mandelbrot(
    width: int,
    height: int,
    max_iterations: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Return an image of Mandelbrot iteration counts.

    Pixels are mapped with column 0 at ``x_min`` and the last column at
    ``x_max``. Row 0 maps to ``y_max`` and the last row to ``y_min`` so the
    returned array can be displayed with Matplotlib's ``origin="upper"``.
    A count of zero is not used: an escaping point receives the one-based
    iteration on which ``abs(z) > 2`` first occurs, while a point that does
    not escape receives ``max_iterations``.
    """
    _validate_dimensions_and_iterations(width, height, max_iterations)
    _validate_bounds(x_min, x_max, y_min, y_max)

    x_values = np.linspace(x_min, x_max, width, dtype=np.float64)
    y_values = np.linspace(y_max, y_min, height, dtype=np.float64)
    real, imaginary = np.meshgrid(x_values, y_values)
    constants = real + 1j * imaginary
    z = np.zeros_like(constants)
    counts = np.full(constants.shape, max_iterations, dtype=np.int32)
    active = np.ones(constants.shape, dtype=bool)

    for iteration in range(1, max_iterations + 1):
        z[active] = z[active] * z[active] + constants[active]
        escaped_now = active & (z.real * z.real + z.imag * z.imag > 4.0)
        counts[escaped_now] = iteration
        active[escaped_now] = False
        if not np.any(active):
            break

    return counts


def save_fractal_image(iteration_counts: np.ndarray, output_path: Path) -> None:
    """Render iteration counts as an image and save them to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = iteration_counts.shape
    figure, axis = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    try:
        axis.imshow(iteration_counts, cmap="magma", origin="upper")
        axis.set_axis_off()
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        figure.savefig(output_path, dpi=100)
    finally:
        plt.close(figure)


def run_self_check() -> None:
    """Check known interior and exterior points using the iteration convention."""
    counts = render_mandelbrot(
        width=3,
        height=3,
        max_iterations=10,
        x_min=-2.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
    )
    assert counts[1, 0] == 10, "c=-2 should remain bounded for this escape rule"
    assert counts[1, 1] == 10, "c=-0.5 should be interior"
    assert counts[1, 2] == 3, "c=1 should escape on iteration 3"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _validate_dimensions_and_iterations(
    width: int, height: int, max_iterations: int
) -> None:
    if width <= 0 or height <= 0 or max_iterations <= 0:
        raise ValueError("width, height, and max_iterations must be positive")


def _validate_bounds(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> None:
    if not x_min < x_max:
        raise ValueError("x_min must be less than x_max")
    if not y_min < y_max:
        raise ValueError("y_min must be less than y_max")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Render a serial Mandelbrot image.")
    parser.add_argument("--width", type=_positive_int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=_positive_int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--max-iterations",
        type=_positive_int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum iterations per complex point (default: 200).",
    )
    parser.add_argument("--x-min", type=float, default=DEFAULT_BOUNDS[0])
    parser.add_argument("--x-max", type=float, default=DEFAULT_BOUNDS[1])
    parser.add_argument("--y-min", type=float, default=DEFAULT_BOUNDS[2])
    parser.add_argument("--y-max", type=float, default=DEFAULT_BOUNDS[3])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path (default: results/fractal.png).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate arguments, run the self-check, compute, and save the image."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        _validate_bounds(
            arguments.x_min,
            arguments.x_max,
            arguments.y_min,
            arguments.y_max,
        )
        run_self_check()
        iteration_counts = render_mandelbrot(
            arguments.width,
            arguments.height,
            arguments.max_iterations,
            arguments.x_min,
            arguments.x_max,
            arguments.y_min,
            arguments.y_max,
        )
        save_fractal_image(iteration_counts, arguments.output)
    except (OSError, ValueError, AssertionError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
