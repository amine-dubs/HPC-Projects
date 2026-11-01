# Parallel Fractal Zoom in Python

This project compares three implementations of the same Mandelbrot
computation:

1. the readable Python/NumPy serial reference from Milestone 1;
2. a Numba-compiled serial kernel;
3. a Numba kernel parallelized across independent image rows with `prange`.

MPI, multiprocessing, distributed execution, and zoom animation are outside
this milestone.

## Proposal alignment and delivery roadmap

The proposal calls the complete project **Parallel Fractal Zoom Movie in
Python**. This repository is being built incrementally over four weeks:

| Proposal deliverable | Status |
|---|---|
| Serial Mandelbrot reference and single-frame image | Complete |
| Numba serial and shared-memory `prange` implementations | Complete |
| Correctness comparison against the serial reference | Complete |
| Thread/process scaling measurements and plots | Started by `benchmark.py`; broader runs pending |
| MPI static-row strategy (S1) | Not started |
| MPI master-worker tile strategy (S2) | Not started |
| MPI whole-frame strategy (S3) | Not started |
| VM deployment and documented bridged-LAN setup | Not started |
| Docker deployment and Compose/network scripts | Not started |
| Physical-LAN deployment and connectivity checks | Not started |
| Zoom frames and final GIF/MP4 movie | Not started |
| Raw CSV data, speedup/efficiency/load-balance plots | Not started |
| Four-to-six-page report and live demonstration | Not started |

The proposal's suggested final parameters are 640 x 360 frames, about 120
frames, a zoom centre near `-0.7436 + 0.1318i`, a per-frame zoom factor near
`0.95`, and an iteration limit that grows from roughly 200 toward 1000. These
are future movie settings, not changes to the current single-frame baseline.
The current milestone deliberately does not add MPI, Docker, VM, LAN, or
animation code.

The next implementation step is to extend the shared-memory experiment with
the proposal's alternative small-tile scheduling variant, then record
repeatable scaling data before beginning local MPI strategy S1. Deployment
work should begin only after local MPI correctness is established.

## Setup

The project uses Python 3, NumPy, Numba, and Matplotlib. Numba accelerates the
two experimental kernels but does not replace the original reference.

From the project directory, create or activate a virtual environment and
install the required runtime packages:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy matplotlib numba
```

This workspace also includes an existing environment at `3.13\Scripts`. Its
exact installation command is:

```powershell
.\3.13\Scripts\python.exe -m pip install numba
```

## Run

The default command creates `results/fractal.png`:

```powershell
python fractal.py
```

For the existing workspace environment, use:

```powershell
.\3.13\Scripts\python.exe fractal.py
```

All main rendering parameters are configurable:

```powershell
python fractal.py --width 800 --height 500 --max-iterations 300 `
  --x-min -2.2 --x-max 0.8 --y-min -1.0 --y-max 1.0 `
  --output results\custom.png
```

The program runs a small correctness self-check before rendering. Use
`python fractal.py --help` to see all arguments.

## Correctness comparison

Run the three implementations on identical inputs and require their complete
iteration-count arrays to match:

```powershell
.\3.13\Scripts\python.exe benchmark.py --width 32 --height 18 `
  --max-iterations 40 --threads 2 --check-only
```

The command first runs the existing serial self-check, then compares the
Python/NumPy reference, Numba serial, and Numba parallel results with exact
array equality.

## Algorithm

For each complex pixel value `c`, the renderer starts at `z = 0` and repeats:

```text
z <- z² + c
```

The loop stops when `|z| > 2` or when the maximum iteration count is reached.
The output is a NumPy array of integer iteration counts. An escaping point is
assigned the one-based iteration on which it first exceeds 2. A point that
does not escape receives the maximum iteration value. The escape test uses
`z.real² + z.imag² > 4`, avoiding an unnecessary square root.

Pixel mapping is explicit:

* column 0 maps to `x_min`, and the last column maps to `x_max`;
* row 0 maps to `y_max`, and the last row maps to `y_min`;
* Matplotlib displays the array with `origin="upper"`.

This makes the top of the saved image correspond to positive imaginary
coordinates and preserves the expected vertical orientation.

Computation is separated from plotting and file output in
`render_mandelbrot` and `save_fractal_image`. The Numba kernels are in
`fractal_kernels.py`; they contain only numerical array and scalar operations,
with no plotting or file I/O.

## Benchmark

Run a small benchmark first:

```powershell
.\3.13\Scripts\python.exe benchmark.py --width 320 --height 180 `
  --max-iterations 100 --repetitions 3 --threads 2
```

For the initial-size comparison:

```powershell
.\3.13\Scripts\python.exe benchmark.py --repetitions 5 --threads 2
```

The harness checks correctness before benchmarking. It calls both Numba
functions once as a warm-up so JIT compilation is completed before timing
begins. Each reported sample measures only the renderer call; coordinate
construction is included consistently in each wrapper, while plotting and
disk output are excluded. It reports the minimum and median of the requested
repetitions.

`--threads` configures Numba's thread pool. If omitted, Numba's configured
default is used. The output reports the actual `numba.get_num_threads()` value.
The Numba serial version does not use parallel execution even when multiple
threads are available.

JIT compilation translates the numerical Python function into machine code on
its first call. Warm-up prevents that one-time compilation cost from
dominating the measured samples. A speedup is the reference runtime divided by
an experimental runtime; parallel efficiency is speedup divided by the number
of threads. These values should be calculated from observed benchmark output,
not assumed in advance. Small images can be dominated by overhead, so a
parallel result is not automatically faster.

## Initial configuration and expected output

The defaults are:

* 640 x 360 pixels;
* `x = [-2.2, 0.8]`;
* `y = [-1.0, 1.0]`;
* maximum of 200 iterations;
* output path `results/fractal.png`.

After a successful run, the `results` directory is created if needed and
contains a PNG image of the Mandelbrot set. The benchmark has been validated
with a small local run, but no final scaling or deployment conclusion is
claimed yet.

## Troubleshooting

* **`ModuleNotFoundError` for NumPy, Matplotlib, or Numba:** activate the
  environment and run `python -m pip install numpy matplotlib numba`.
* **PowerShell blocks activation:** run the script with the environment's
  Python executable directly, for example
  `.\3.13\Scripts\python.exe fractal.py`.
* **Invalid dimensions or iteration limit:** width, height, and
  `--max-iterations` must be positive integers.
* **Invalid coordinate bounds:** each minimum must be strictly less than its
  corresponding maximum.
* **No display is available:** this program selects Matplotlib's non-interactive
  `Agg` backend and writes the image without opening a window.
* **Invalid thread count:** `--threads` must be a positive integer accepted by
  Numba. Try omitting it to use Numba's default, or choose a supported count.
