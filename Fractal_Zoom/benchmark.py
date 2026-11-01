"""Correctness checks, CSV benchmarks, and plots for Mandelbrot kernels."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numba
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fractal import (
    DEFAULT_BOUNDS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_WIDTH,
    render_mandelbrot,
    run_self_check,
)
from fractal_kernels import (
    render_numba_parallel,
    render_numba_serial,
    render_numba_tiled_parallel,
)


Bounds = tuple[float, float, float, float]
Arguments = tuple[int, int, int, float, float, float, float]
Renderer = Callable[..., np.ndarray]
CSV_COLUMNS = (
    "timestamp",
    "implementation",
    "width",
    "height",
    "max_iterations",
    "repetition",
    "elapsed_seconds",
    "numba_threads",
    "tile_size",
    "shuffle_seed",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parameters(
    width: int, height: int, max_iterations: int, bounds: Bounds
) -> Arguments:
    return (width, height, max_iterations, *bounds)


def check_correctness(
    width: int,
    height: int,
    max_iterations: int,
    bounds: Bounds,
    tile_size: int,
    shuffle_seed: int,
) -> None:
    """Compare every implementation and report the first mismatch."""
    arguments = _parameters(width, height, max_iterations, bounds)
    reference = render_mandelbrot(*arguments)
    implementations = [
        ("Numba serial", render_numba_serial(*arguments)),
        ("Numba parallel rows", render_numba_parallel(*arguments)),
        (
            "Numba parallel shuffled tiles",
            render_numba_tiled_parallel(
                *arguments, tile_size=tile_size, shuffle_seed=shuffle_seed
            ),
        ),
    ]
    for name, result in implementations:
        if not np.array_equal(reference, result):
            difference = np.argwhere(reference != result)[0]
            raise RuntimeError(
                f"{name} mismatch at row {difference[0]}, "
                f"column {difference[1]}"
            )


def _measure(
    renderer: Renderer,
    arguments: Arguments,
    repetitions: int,
    **renderer_options: int,
) -> list[float]:
    """Measure renderer calls only; Numba compilation happens before this."""
    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        renderer(*arguments, **renderer_options)
        samples.append(time.perf_counter() - start)
    return samples


def _append_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    """Append observed rows without deleting an existing results file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def benchmark(
    width: int,
    height: int,
    max_iterations: int,
    bounds: Bounds,
    repetitions: int,
    tile_size: int,
    shuffle_seed: int,
    csv_path: Path,
) -> None:
    """Warm up kernels, record samples, and print summary statistics."""
    arguments = _parameters(width, height, max_iterations, bounds)
    implementations: list[tuple[str, Renderer, dict[str, int]]] = [
        ("Python/NumPy reference", render_mandelbrot, {}),
        ("Numba serial", render_numba_serial, {}),
        ("Numba parallel rows", render_numba_parallel, {}),
        (
            "Numba parallel shuffled tiles",
            render_numba_tiled_parallel,
            {"tile_size": tile_size, "shuffle_seed": shuffle_seed},
        ),
    ]

    # Warm-up calls compile every Numba signature before timing starts.
    for _, renderer, options in implementations[1:]:
        renderer(*arguments, **options)

    
    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    print(f"dimensions: {width} x {height}")
    print(f"max iterations: {max_iterations}")
    print(f"repetitions: {repetitions}")
    print(f"Numba threads: {numba.get_num_threads()}")
    print("timings (computation only, seconds):")
    for name, renderer, options in implementations:
        samples = _measure(renderer, arguments, repetitions, **options)
        print(
            f"  {name}: min={min(samples):.6f}, "
            f"median={statistics.median(samples):.6f}"
        )
        for repetition, elapsed in enumerate(samples, start=1):
            rows.append(
                {
                    "timestamp": timestamp,
                    "implementation": name,
                    "width": width,
                    "height": height,
                    "max_iterations": max_iterations,
                    "repetition": repetition,
                    "elapsed_seconds": f"{elapsed:.12f}",
                    "numba_threads": numba.get_num_threads(),
                    "tile_size": options.get("tile_size", ""),
                    "shuffle_seed": options.get("shuffle_seed", ""),
                }
            )
    _append_csv(rows, csv_path)
    print(f"CSV appended: {csv_path}")


def plot_csv(csv_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    """Plot median time, speedup, and efficiency from the recorded CSV."""
    if not csv_path.exists():
        raise ValueError(f"CSV file does not exist: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV file has no benchmark rows: {csv_path}")

    latest_timestamp = max(row["timestamp"] for row in rows)
    rows = [row for row in rows if row["timestamp"] == latest_timestamp]
    required = {"implementation", "elapsed_seconds", "numba_threads"}
    if not required.issubset(rows[0]):
        raise ValueError("CSV is missing required benchmark columns")
    grouped: dict[str, list[float]] = {}
    threads: dict[str, int] = {}
    for row in rows:
        try:
            grouped.setdefault(row["implementation"], []).append(
                float(row["elapsed_seconds"])
            )
            threads[row["implementation"]] = int(row["numba_threads"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("CSV contains an invalid benchmark row") from error
    baseline_name = "Python/NumPy reference"
    if baseline_name not in grouped:
        raise ValueError("latest CSV run has no Python/NumPy reference data")
    names = list(grouped)
    medians = [statistics.median(grouped[name]) for name in names]
    baseline = statistics.median(grouped[baseline_name])
    speedups = [baseline / median for median in medians]
    efficiencies = [
        speedup / threads[name] if threads[name] > 1 else float("nan")
        for name, speedup in zip(names, speedups)
    ]

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    time_path = output_prefix.with_name(output_prefix.name + "_times.png")
    metrics_path = output_prefix.with_name(
        output_prefix.name + "_speedup_efficiency.png"
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(names, medians)
    axis.set_ylabel("Median computation time (seconds)")
    axis.set_title("Mandelbrot computation time")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(time_path)
    plt.close(figure)

    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions - 0.2, speedups, width=0.4, label="Speedup")
    axis.bar(positions + 0.2, efficiencies, width=0.4, label="Efficiency")
    axis.set_xticks(positions, names, rotation=25, ha="right")
    axis.set_ylabel("Ratio")
    axis.set_title(
        "Speedup and parallel efficiency "
        f"(baseline: {baseline_name})"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(metrics_path)
    plt.close(figure)
    print(f"Plots written: {time_path}, {metrics_path}")
    return time_path, metrics_path


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare and benchmark Mandelbrot implementations."
    )
    parser.add_argument("--width", type=_positive_int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=_positive_int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--max-iterations",
        type=_positive_int,
        default=DEFAULT_MAX_ITERATIONS,
    )
    parser.add_argument("--x-min", type=float, default=DEFAULT_BOUNDS[0])
    parser.add_argument("--x-max", type=float, default=DEFAULT_BOUNDS[1])
    parser.add_argument("--y-min", type=float, default=DEFAULT_BOUNDS[2])
    parser.add_argument("--y-max", type=float, default=DEFAULT_BOUNDS[3])
    parser.add_argument("--repetitions", type=_positive_int, default=5)
    parser.add_argument("--threads", type=_positive_int, default=None)
    parser.add_argument("--tile-size", type=_positive_int, default=16)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/benchmark.csv"),
        help="CSV path to append benchmark samples.",
    )
    parser.add_argument(
        "--plot-prefix",
        type=Path,
        default=Path("results/benchmark"),
        help="Prefix for generated plot files.",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate options, check correctness, benchmark, and optionally plot."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.plot_only and (arguments.check_only or arguments.plot):
        parser.error("--plot-only cannot be combined with --check-only or --plot")
    bounds: Bounds = (
        arguments.x_min,
        arguments.x_max,
        arguments.y_min,
        arguments.y_max,
    )
    if not arguments.x_min < arguments.x_max:
        parser.error("x_min must be less than x_max")
    if not arguments.y_min < arguments.y_max:
        parser.error("y_min must be less than y_max")
    if arguments.threads is not None:
        try:
            numba.set_num_threads(arguments.threads)
        except ValueError as error:
            parser.error(str(error))

    try:
        if arguments.plot_only:
            plot_csv(arguments.csv, arguments.plot_prefix)
            return 0
        run_self_check()
        check_correctness(
            arguments.width,
            arguments.height,
            arguments.max_iterations,
            bounds,
            arguments.tile_size,
            arguments.shuffle_seed,
        )
        print("correctness: all four implementations match exactly")
        if not arguments.check_only:
            benchmark(
                arguments.width,
                arguments.height,
                arguments.max_iterations,
                bounds,
                arguments.repetitions,
                arguments.tile_size,
                arguments.shuffle_seed,
                arguments.csv,
            )
            if arguments.plot:
                plot_csv(arguments.csv, arguments.plot_prefix)
    except (RuntimeError, ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
