# Learning log

## 2026-10-14 — Serial Mandelbrot renderer

### Objective

Implement the first milestone of the four-week HPC mini-project: a correct,
readable serial Mandelbrot renderer without parallelism or Numba JIT
compilation.

### Setup

* Python 3.12.10 virtual environment in `3.13\`
* NumPy and Matplotlib
* Non-interactive Matplotlib `Agg` backend

### Commands

Planned setup and run commands:

```powershell
.\3.13\Scripts\python.exe -m pip install numpy matplotlib
.\3.13\Scripts\python.exe fractal.py
```

### Results

* The built-in interior/exterior self-check passed.
* The default run created `results/fractal.png` with dimensions 640 x 360.
* Invalid input `--width 0` was rejected with a clear command-line error.
* No performance timings were collected for this milestone.

## 2026-10-14 — Numba serial and parallel comparison

### Objective

Add a Numba-compiled serial implementation and a row-parallel Numba
implementation while preserving the original Python/NumPy reference.

### Setup

* Existing Python 3.12.10 virtual environment in `3.13\`
* Installed Numba 0.68.0 and its required `llvmlite` dependency
* Existing NumPy 2.5.4 and Matplotlib 3.11.2

### Commands executed

```powershell
.\3.13\Scripts\python.exe -c "import numpy, numba; ..."
.\3.13\Scripts\python.exe -m py_compile fractal.py fractal_kernels.py benchmark.py
.\3.13\Scripts\python.exe benchmark.py --width 32 --height 18 `
  --max-iterations 40 --repetitions 2 --threads 2 --check-only
```

### Results

* NumPy 2.5.4 and Numba 0.68.0 imported successfully.
* Python syntax compilation passed for all three Python files.
* All three iteration-count arrays matched exactly for the small comparison.
* A 64 x 36 benchmark with 50 iterations, 2 repetitions, and 2 Numba
  threads completed after warm-up. The observed minimum/median times in
  seconds were: Python/NumPy `0.004626/0.004664`, Numba serial
  `0.000237/0.000268`, and Numba parallel `0.000152/0.000165`.
* These are one small run on this machine only; no general performance
  conclusion is recorded.

### Proposal deliverable status

The proposal was reviewed against the repository on 2026-10-14:

* Complete: serial reference, single-frame PNG, Numba serial kernel, shared
  memory `prange` kernel, correctness comparison, and warm-up benchmark.
* Started: repeatable local timing harness for minimum and median runtimes.
* Not started: the alternative shuffled-tile thread variant, MPI strategies
  S1/S2/S3, VM deployment, Docker deployment, physical-LAN deployment, zoom
  movie, CSV/plot collection, final report, and live demonstration.

The proposal's suggested 120-frame zoom, boundary-point centre, growing
iteration limit, and final GIF/MP4 remain future work. They were not added to
the current single-frame implementation.

### Next actions

1. Add and validate the proposal's alternative small-tile shared-memory
   scheduling experiment.
2. Extend benchmark output to durable CSV data and scaling plots.
3. Implement and validate MPI static rows locally before attempting VM,
   Docker, or physical-LAN deployment.
