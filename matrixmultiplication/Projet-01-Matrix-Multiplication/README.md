# Parallel Matrix Multiplication in Python: Threads and Local MPI

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/Accelerated-Numba%20%28OpenMP%29-blueviolet.svg)](https://numba.pydata.org/)
[![MPI](https://img.shields.io/badge/MPI-Local%20mpi4py-orange.svg)](https://mpi4py.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-success.svg)](src/verify_all.py)

This project covers serial matrix multiplication, Version 1 shared-memory threading, and local MPI execution on one machine. It does not include multi-node, Docker, or virtual-machine deployment.

---

## 1. Project Overview

- **Algorithm:** Dense double-precision square matrix multiplication: $C = A \times B$ where $A, B \in \mathbb{R}^{n \times n}$.
- **Execution Model:** Shared-memory multi-threading with Numba's `@njit(parallel=True)` and `prange(n)`.
- **Threading Runtime:** OpenMP (`NUMBA_THREADING_LAYER=omp`).
- **Memory Optimization:** $(i, k, j)$ loop ordering for unit-stride cache locality, plus Cache-Blocked (tiled) sub-matrices ($64 \times 64$).
- **Local MPI:** `mpi4py` row-block decomposition across ranks launched on the same host; no hostfile or network cluster is required.
- **Numerical Validation:** Verified against NumPy BLAS (`A @ B`) using `np.allclose(atol=1e-8, rtol=1e-5)`.

---

## 2. Directory Layout

```text
Projet 01 - Matrix Multiplication/
├── Matrix_Multiplication_Mini_Project_Proposal.docx # Original specification
├── REPORT.docx                                       # Academic Word Report (threads and local MPI)
├── REPORT.md                                         # Markdown Technical Report
├── README.md                                         # Project Guide (this file)
├── requirements.txt                                  # Python dependencies
├── generate_report_docx.py                           # Automated DOCX report generator
│
├── src/                                              # Source Code Implementations
│   ├── common.py                                     # Utilities, verification, FLOP calculation
│   ├── baseline_serial.py                            # Version 0: Serial Baseline (Numba @njit)
│   ├── version1_threads.py                           # Version 1: Multi-Threaded & Cache-Blocked Kernels
│   ├── mpi_matmul.py                                 # Local MPI row-block matrix multiplication
│   ├── pingpong.py                                   # Local two-rank MPI communication benchmark
│   ├── benchmark_runner.py                           # Automated benchmark runner (3 reps median)
│   ├── benchmark_mpi.py                              # MPI benchmark driver -> data/results_mpi_local.csv
│   ├── plot_results.py                               # 300 DPI publication plotting suite
│   └── verify_all.py                                 # Automated unit test suite (100% passing)
│
├── data/                                             # Empirical Benchmark Datasets
│   ├── results_serial_threads.csv                    # Measured runtimes, speedup, efficiency, GFLOP/s
│   ├── results_mpi_local.csv                         # Measured local MPI timings (per-phase breakdown)
│   └── pingpong_local.csv                            # Measured MPI latency/bandwidth (Hockney model)
│
└── plots/                                            # Generated High-Resolution Figures (300 DPI)
    ├── speedup_threads.png                           # Thread speedup vs Ideal linear scaling
    ├── efficiency_threads.png                        # Parallel efficiency curve
    └── cache_blocking_comparison.png                 # Cache-Blocked vs Naive Row-Parallel
```

---

## 3. Quickstart & Usage

### 3.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.2 Run Automated Verification Tests
```bash
python src/verify_all.py
```
*Validates numerical correctness of Serial, Naive Threaded, and Cache-Blocked kernels against NumPy BLAS reference.*

### 3.3 Execute Version 0 (Serial Baseline)
```bash
python src/baseline_serial.py --sizes 1024 2048 --repeats 3
```

### 3.4 Execute Version 1 (Multi-Threaded Scaling)
```bash
# Standard row-parallel scaling across threads:
python src/version1_threads.py --sizes 1024 2048 --threads 1 2 4 8 --kernel naive

# Cache-blocked (tiled) variant:
python src/version1_threads.py --sizes 2048 --threads 4 8 --kernel blocked --block-size 64
```

### 3.5 Execute Local MPI Matrix Multiplication
Install a local MPI implementation and `mpi4py` for your platform, then launch ranks on this machine:
```bash
mpiexec -n 4 python src/mpi_matmul.py --n 2048 --kernel numba
```
The matrix dimension must be divisible by the number of ranks. Rank 0 verifies the result against the NumPy reference. No `--hostfile` is used.

To run the two-rank local MPI communication benchmark:
```bash
mpiexec -n 2 python src/pingpong.py --output-csv data/pingpong_local.csv
```

### 3.6 Run Thread Benchmark Suite & Regenerate Plots
```bash
# Execute benchmarks across all configurations (add 4096 for real measurements):
python src/benchmark_runner.py

# Record local MPI timings across (n, p) into data/results_mpi_local.csv:
python src/benchmark_mpi.py

# Generate publication-quality plots in plots/:
python src/plot_results.py

# Recompile the Word academic report:
python generate_report_docx.py
```

---

## 4. Key Results Summary

The checked-in benchmark dataset covers serial, threaded, and (measured on this machine) local-MPI execution — all fully measured:

- **Serial baseline (n=2048):** 4.44 s (3.87 GFLOP/s); measured n=4096 serial: 39.0 s.
- **Threads (n=1024):** 3.29x at p=4 (82.1% efficiency) and 6.00x at p=8 (75.0%) — B (8 MB) fits in shared L3.
- **Threads (n=2048):** 4.48x at p=8 (56.0%); at the measured n=4096, scaling saturates: 2.59x at p=4 vs 2.63x at p=8 (32.8% efficiency) — memory-bandwidth-bound.
- **Blocked variant:** consistently slower than the naive kernel (0.78x–1.63x of serial) — documented as an honest negative result with analysis.
- **Local MPI (n=2048):** compute-bound with 97-99.7% of time computing; best speedup 2.05x at p=4; communication 0.3-3.3% locally.
- **Ping-pong:** latency 2.25 microseconds, peak bandwidth ~8.2 GB/s (one host, shared-memory transport).
- Every row in `data/results_serial_threads.csv` is measured (`source=measured`).

No VM, Docker, or multi-node performance results are included (out of scope for this submission).
