# Parallel Matrix Multiplication in Python: Threads and Local MPI

**Course:** High-Performance Computing (HPC) / Calcul Intensif (S9 - Student-Engineer Project)
**Date:** October 2026  
**Language & Runtime:** Python 3.11, NumPy, Numba (`@njit`, `prange`), OpenMP, mpi4py
**Project Scope:** Serial baseline, Version 1 shared-memory threads, and local MPI on one machine; no VM, Docker, or multi-node deployment.

---

## Executive Summary & Abstract

This report investigates serial, shared-memory, and local message-passing approaches to dense matrix multiplication:
$$C = A \times B, \quad A, B, C \in \mathbb{R}^{n \times n}$$
using double-precision floating-point arithmetic (`float64`).

Standard Python multi-threading is constrained by the **Global Interpreter Lock (GIL)**, preventing compute-bound loops from utilizing multiple CPU cores. We resolve this by compiling matrix multiplication kernels using **Numba's JIT compiler** (`@njit(parallel=True)`) and parallelizing the outer loop over matrix rows with `prange(n)`. Under the hood, Numba interfaces directly with the native **OpenMP runtime**.

We implement and evaluate:
1. **Version 0 (Baseline Serial):** Serial Numba `@njit` with cache-friendly loop ordering $(i, k, j)$.
2. **Version 1 (Shared-Memory Threads):** OpenMP-style row-parallel matrix multiplication, with a cache-blocked variant.
3. **Local MPI:** Row-block decomposition across MPI ranks launched on the same host. Rank 0 broadcasts $B$, scatters rows of $A$, gathers the output, and verifies the result. No multi-node execution is part of this project.

Threaded and MPI results are numerically verified against NumPy BLAS (`A @ B`) using `np.allclose(atol=1e-8, rtol=1e-5)`. The checked-in benchmark dataset reports serial and threaded measurements; local MPI can be run to measure the current machine, but no MPI performance results are claimed here.

---

## 1. Computational Foundations & Mathematical Complexity

### 1.1 Arithmetic Intensity and FLOP Count
Dense matrix multiplication computes each entry $C_{i,j}$ via a vector dot product:
$$C_{i,j} = \sum_{k=0}^{n-1} A_{i,k} \cdot B_{k,j}$$
For an $n \times n$ matrix, computing $n^2$ entries requires $n$ multiplications and $n-1$ additions per entry. The total floating-point operation count is:
$$\text{Total FLOPs} = 2n^3 - n^2 \approx 2n^3$$

| Dimension ($n$) | Total FLOPs | Single Matrix Size | Working Set ($A, B, C$) | Computational Workload |
| :--- | :--- | :--- | :--- | :--- |
| **$n = 1024$** | $2.147 \times 10^9$ (2.15 GFLOPs) | 8.00 MB | 24.00 MB | Fast (~0.09s to 0.55s) |
| **$n = 2048$** | $17.180 \times 10^9$ (17.18 GFLOPs) | 32.00 MB | 96.00 MB | Medium (~0.99s to 4.44s) |
| **$n = 4096$** | $137.439 \times 10^9$ (137.44 GFLOPs) | 128.00 MB | 384.00 MB | Heavy (~14.9s to 39.0s) |

Because operation count scales cubically ($O(n^3)$) while data volume scales quadratically ($O(n^2)$), the computational intensity increases linearly with matrix dimension ($O(n)$), providing ideal conditions for multi-core scaling.

---

## 2. Algorithmic Design & Optimization

```
+-----------------------------------------------------------------------------------+
|               Shared-Memory Thread Parallelism (Version 1)                        |
|                                                                                   |
|           Matrix A (n x n)                       Matrix B (n x n)                 |
|    +-----------------------------+       +-----------------------------+          |
|    | Thread 0: Rows 0 .. (n/p)-1 |       |                             |          |
|    +-----------------------------+       |                             |          |
|    | Thread 1: Rows (n/p) ..     |       |   Shared in Main Memory     |          |
|    +-----------------------------+       |     Read Concurrently       |          |
|    | ...                         |       |       by All Threads        |          |
|    +-----------------------------+       |                             |          |
|    | Thread p-1: Rows ... (n-1)  |       +-----------------------------+          |
|    +-----------------------------+                                                |
|                                                                                   |
|                   Parallel Loop: for i in prange(n)                               |
|                                                                                   |
|           Matrix C (n x n)                                                        |
|    +-----------------------------+                                                |
|    | Thread 0 computes Rows      | ===> Written directly into shared C array     |
|    +-----------------------------+                                                |
|    | Thread 1 computes Rows      | ===> Written directly into shared C array     |
|    +-----------------------------+                                                |
|    | Thread p-1 computes Rows    | ===> Written directly into shared C array     |
|    +-----------------------------+                                                |
+-----------------------------------------------------------------------------------+
```

### 2.1 Memory Locality: Why $(i, k, j)$ Beats $(i, j, k)$
In C-contiguous (row-major) order, array elements in the same row are adjacent in memory ($B[k, j]$ and $B[k, j+1]$ are consecutive 8-byte doubles).

If the textbook loop order $(i, j, k)$ is executed:
```python
# Naive (i, j, k) - Cache Unfriendly
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]  # B[k, j] strides by n*8 bytes every iteration!
```
The innermost loop $k$ jumps across rows of $B$, striding by $n \times 8$ bytes (16 KB for $n=2048$). This causes continuous L1/L2 cache misses.

By reordering loops to $(i, k, j)$:
```python
# Optimized (i, k, j) - Cache Friendly & SIMD Vectorizable
for i in range(n):
    for k in range(n):
        a = A[i, k]
        for j in range(n):
            C[i, j] += a * B[k, j]  # Stride-1 unit access in both B and C
```
The innermost loop $j$ reads $B[k, j]$ and writes $C[i, j]$ contiguously with unit stride (8 bytes), fully leveraging 64-byte cache lines, prefetch buffers, and SIMD registers (AVX2).

### 2.2 Version 1: Multi-Threading with OpenMP (`prange`)
Using Numba's `@njit(parallel=True)` with `prange(n)`, Numba chunks the outer loop across available worker threads:
- Each thread is assigned a distinct slice of rows of $A$ and computes the corresponding rows of $C$.
- No thread synchronization or locks are required during computation since threads write to disjoint rows of $C$.
- The OpenMP threading runtime is controlled dynamically via `numba.set_num_threads(p)` for $p \in \{1, 2, 4, 8\}$.

### 2.3 Cache-Blocked (Tiled) Variant
When matrix dimensions exceed the CPU cache hierarchy, sub-blocks of size $64 \times 64$ are used:
- A $64 \times 64$ double-precision tile occupies $32\ \text{KB}$, which fits entirely inside the L1 data cache of each CPU core.
- Active sub-tiles are reused extensively before eviction, mitigating memory bus contention.

### 2.4 Local MPI Row-Block Decomposition
The MPI implementation runs multiple ranks on a single machine using `mpi4py`:
1. Rank 0 creates matrices $A$ and $B$.
2. Rank 0 broadcasts $B$ and scatters equal row blocks of $A$.
3. Each rank computes its local output block using a single-threaded Numba or NumPy kernel.
4. Rank 0 gathers the output blocks and checks the result against the NumPy reference.

Run it with `mpiexec -n 4 python src/mpi_matmul.py --n 2048 --kernel numba`. The matrix dimension must be divisible by the number of ranks. This local run does not require hostfiles or network configuration.

---

## 3. Empirical Results & Performance Analysis

Every value in the table below was measured on the team workstation in a single clean benchmark session (laptop plugged in, background idle). Each configuration was repeated 3 times and the median is reported; JIT compilation was pre-warmed so compile time is excluded. Speedup is computed against the Version 0 serial median measured **in the same session**. Every row is flagged `source=measured` in `data/results_serial_threads.csv` (the benchmark runner can emit `source=extrapolated` scale estimates only if a size is skipped, so no row is ever silently synthetic). MPI timings are in Section 3.2; no multi-node performance data is claimed.

### 3.1 Benchmark Data Table

| Matrix Dimension ($n$) | Kernel Version | Threads ($p$) | Median Time (s) | Speedup ($S = T_1/T_p$) | Efficiency ($E = S/p$) | Throughput (GFLOP/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$n = 1024$** | Version 0 (Serial) | 1 | 0.5476 s | 1.00x | 100.0% | 3.92 GFLOP/s |
| **$n = 1024$** | Version 1 (Naive Threads) | 1 | 0.5384 s | 1.02x | 101.7% | 3.99 GFLOP/s |
| **$n = 1024$** | Version 1 (Naive Threads) | 2 | 0.3271 s | 1.67x | 83.7% | 6.57 GFLOP/s |
| **$n = 1024$** | Version 1 (Naive Threads) | 4 | 0.1667 s | 3.29x | 82.1% | 12.88 GFLOP/s |
| **$n = 1024$** | Version 1 (Naive Threads) | 8 | 0.0913 s | 6.00x | 75.0% | 23.53 GFLOP/s |
| **$n = 1024$** | Version 1 (Cache-Blocked) | 4 | 0.6628 s | 0.83x | 20.7% | 3.24 GFLOP/s |
| **$n = 1024$** | Version 1 (Cache-Blocked) | 8 | 0.3497 s | 1.57x | 19.6% | 6.14 GFLOP/s |
| **$n = 2048$** | Version 0 (Serial) | 1 | 4.4360 s | 1.00x | 100.0% | 3.87 GFLOP/s |
| **$n = 2048$** | Version 1 (Naive Threads) | 1 | 4.3703 s | 1.02x | 101.5% | 3.93 GFLOP/s |
| **$n = 2048$** | Version 1 (Naive Threads) | 2 | 2.9014 s | 1.53x | 76.4% | 5.92 GFLOP/s |
| **$n = 2048$** | Version 1 (Naive Threads) | 4 | 1.8103 s | 2.45x | 61.3% | 9.49 GFLOP/s |
| **$n = 2048$** | Version 1 (Naive Threads) | 8 | 0.9906 s | 4.48x | 56.0% | 17.34 GFLOP/s |
| **$n = 2048$** | Version 1 (Cache-Blocked) | 4 | 5.7264 s | 0.78x | 19.4% | 3.00 GFLOP/s |
| **$n = 2048$** | Version 1 (Cache-Blocked) | 8 | 2.9869 s | 1.49x | 18.6% | 5.75 GFLOP/s |
| **$n = 4096$** | Version 0 (Serial) | 1 | 39.0061 s | 1.00x | 100.0% | 3.52 GFLOP/s |
| **$n = 4096$** | Version 1 (Naive Threads) | 1 | 38.6823 s | 1.01x | 100.8% | 3.55 GFLOP/s |
| **$n = 4096$** | Version 1 (Naive Threads) | 2 | 25.5108 s | 1.53x | 76.5% | 5.39 GFLOP/s |
| **$n = 4096$** | Version 1 (Naive Threads) | 4 | 15.0727 s | 2.59x | 64.7% | 9.12 GFLOP/s |
| **$n = 4096$** | Version 1 (Naive Threads) | 8 | 14.8497 s | 2.63x | 32.8% | 9.26 GFLOP/s |
| **$n = 4096$** | Version 1 (Cache-Blocked) | 4 | 43.4696 s | 0.90x | 22.4% | 3.16 GFLOP/s |
| **$n = 4096$** | Version 1 (Cache-Blocked) | 8 | 23.9665 s | 1.63x | 20.3% | 5.73 GFLOP/s |

### 3.2 Local MPI Benchmark Data

The local MPI implementation was benchmarked on a single host with the `numba` kernel, 3 repetitions per configuration, median-run records (script: `src/benchmark_mpi.py`, data: `data/results_mpi_local.csv`). Timings are the MAX across ranks for each phase (critical path):

| Matrix Dimension ($n$) | Processes ($p$) | Broadcast (s) | Scatter (s) | Compute (s) | Gather (s) | Total (s) | Compute Ratio | Communication Ratio | Throughput (GFLOP/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$n = 1024$** | 1 | 0.00005 | 0.00191 | 0.53345 | 0.00183 | 0.53725 | 99.3% | 0.7% | 4.00 |
| **$n = 1024$** | 2 | 0.00338 | 0.00285 | 0.30008 | 0.00336 | 0.30780 | 97.5% | 2.5% | 6.98 |
| **$n = 1024$** | 4 | 0.00718 | 0.00554 | 0.24778 | 0.01389 | 0.26128 | 94.8% | 5.2% | 8.22 |
| **$n = 1024$** | 8 | 0.01534 | 0.00581 | 0.30228 | 0.01520 | 0.32346 | 93.5% | 6.5% | 6.64 |
| **$n = 2048$** | 1 | 0.00005 | 0.00734 | 4.36796 | 0.00740 | 4.38275 | 99.7% | 0.3% | 3.92 |
| **$n = 2048$** | 2 | 0.01636 | 0.01051 | 2.72803 | 0.01835 | 2.76485 | 98.7% | 1.3% | 6.21 |
| **$n = 2048$** | 4 | 0.02943 | 0.02145 | 2.08731 | 0.05662 | 2.13587 | 97.7% | 2.3% | 8.04 |
| **$n = 2048$** | 8 | 0.05162 | 0.02061 | 2.31421 | 0.04818 | 2.39265 | 96.7% | 3.3% | 7.18 |

Observations (single host, shared-memory transport under MS-MPI):
1. **Communication is negligible locally:** MPI collectives are implemented over shared-memory copies inside one machine (measured copy bandwidth up to ~8 GB/s, Section 3.3), so compute dominates with 93–99.7% of the total time. On a real LAN (1–10 Gb/s), moving the same 8–128 MB matrices would make broadcast/scatter/gather dominant, which is precisely what deployment targets 2–4 (excluded from this scope) would have quantified.
2. **Best speedup at $p = 4$** (2.05x at $n = 2048$): $p = 4$ matches the number of physical cores available to the ranks; $p = 8$ oversubscribes the 6-core CPU.
3. **Kernel note:** each MPI rank uses a plain single-threaded `@njit` kernel (no `prange`). In the clean benchmark session the two kernels agree almost exactly at $p = 1$ (Version 1: $0.538\ \text{s}$ vs MPI: $0.537\ \text{s}$ at $n = 1024$; $4.37\ \text{s}$ vs $4.38\ \text{s}$ at $n = 2048$), so timings are directly comparable across versions within the same session. Absolute times measured in different sessions on this laptop must not be cross-compared: thermal state and background load shift each session's absolute scale (an earlier, heavier-loaded session produced different absolute times for the identical code).

### 3.3 Local Ping-Pong Communication Results

Two-rank ping-pong benchmark on one host (`src/pingpong.py`, data: `data/pingpong_local.csv`, 30 trials per message size):

| Metric | Measured Value |
| :--- | :--- |
| Latency $\alpha$ (8 B message) | $2.25\ \mu s$ |
| Peak bandwidth (1 MB messages) | $8157\ \text{MB/s}$ (65.3 Gbps) |
| Bandwidth plateau ($\geq 2$ MB) | $3.6 - 3.8\ \text{GB/s}$ (shared-memory copy limit) |

The Hockney model $T(m) = \alpha + m/\beta$ fits well: small messages are latency-dominated ($\alpha \approx 2\ \mu s$), large messages saturate at the memcpy bandwidth of the machine. Both numbers are far better than any real network target (typical LAN: $\alpha \approx 50-100\ \mu s$, $\beta \approx 110\ \text{MB/s}$ on gigabit Ethernet), which explains why the local MPI runs are compute-bound while a LAN deployment would become communication-bound.

---

## 4. Key Performance Observations & Bottlenecks

### 4.1 Thread Scaling ($p = 1$ to $4$)

Across all sizes, scaling is respectable but sub-linear once $p > 2$:
- For $n = 1024$ (16 MB working set, fits largely in caches): speedup is near-linear in practice — $1.67\text{x}$ at $p = 2$, $3.29\text{x}$ at $p = 4$ ($82.1\%$ efficiency, $12.88\ \text{GFLOP/s}$) and $6.00\text{x}$ at $p = 8$ ($75.0\%$).
- For $n = 2048$ ($96\ \text{MB}$ working set): $2.90\ \text{s}$ at $p = 2$ ($1.53\text{x}$), $1.81\ \text{s}$ at $p = 4$ ($2.45\text{x}$, $61.3\%$).
- For $n = 4096$ ($384\ \text{MB}$ working set, measured): $25.5\ \text{s}$ at $p = 2$ ($1.53\text{x}$), $15.1\ \text{s}$ at $p = 4$ ($2.59\text{x}$, $64.7\%$).

Efficiency below $100\%$ reflects the two classic shared-memory taxes that our scalar (deliberately non-BLAS) kernel exposes: every thread streams the whole matrix $B$ from memory (bandwidth sharing), and threads compete for shared L2/L3 capacity once the working set exceeds the caches. There is no load imbalance: rows are split evenly and no locks are taken (threads write disjoint rows of $C$).

### 4.2 Bandwidth Saturation with Growing Problem Size ($p = 8$)

The relative benefit of $p = 8$ (hyper-threaded logical threads beyond the 6 physical cores) degrades as $n$ grows: $6.00\text{x}$ at $n = 1024$, $4.48\text{x}$ at $n = 2048$, but only $2.63\text{x}$ at $n = 4096$ and essentially no gain over $p = 4$ there ($14.85\ \text{s}$ vs $15.07\ \text{s}$):
- **Arithmetic intensity in space:** for each row of $C$, every thread re-reads the entire $32$-$128\ \text{MB}$ matrix $B$ from RAM; aggregate demand for memory bandwidth grows with $p$ and the working set, so at $n = 4096$ the system becomes bandwidth-bound and extra logical threads add mostly contention.
- **Why $p=8$ still helps at small $n$:** at $n = 1024$, $B$ ($8\ \text{MB}$) fits in the shared L3, so threads largely serve $B$ from cache rather than RAM; that memory-stall overlap is exactly the scenario where hyper-threading adds throughput.
- **Hyper-threading contention at large $n$:** logical siblings share ALUs/L1/L2; with the memory bus already saturated by 6 physical cores, sibling threads only multiply the bandwidth demand (efficiency falls to $32.8\%$ at $n = 4096$, $p = 8$).

### 4.3 Amdahl's Law Implications

Max speedup is capped by the serial fraction $s$: $S_{max} = 1/(s + (1-s)/p)$. Thread management and scheduling overhead constitute a minimal serial fraction here; the observed ceiling is therefore set by **memory bandwidth**, not by serial overhead.

### 4.4 Cache-Blocked (Tiled) Variant: An Honest Negative Result

The $64 \times 64$ tiled kernel measured consistently slower than the naive row-parallel kernel ($0.78\text{x}$–$1.63\text{x}$ of the serial baseline — i.e., slower than serial at $p = 4$ for every size). Its tiling does remove DRAM traffic for matrix $B$, but the generated inner loops are not register-blocked (each `C[i,j]` element is still touched once per `k`, although that traffic stays in L1/L2), and the extra per-tile index arithmetic defeats some codegen optimizations. BLAS-style register blocking, panel packing, or `fastmath=True` vectorization are identified as future work.

---

## 5. Live Demonstration Guide

To present Version 1 live to the professor:

```bash
# 1. Run Automated Numerical Verification Suite (6 unit tests, np.allclose):
python src/verify_all.py

# 2. Run Version 0 Serial Baseline:
python src/baseline_serial.py --sizes 1024 2048 --repeats 3

# 3. Run Version 1 Multi-Threaded Scaling Benchmark:
python src/version1_threads.py --sizes 2048 --threads 1 2 4 8

# 4. Run Cache-Blocked (Tiled) Comparison:
python src/version1_threads.py --sizes 2048 --threads 4 8 --kernel blocked --block-size 64

# 5. Run the full benchmark suite and regenerate all data and plots:
python src/benchmark_runner.py
python src/plot_results.py

# 6. Run four local MPI ranks (requires mpi4py and a local MPI runtime):
mpiexec -n 4 python src/mpi_matmul.py --n 2048 --kernel numba

# 7. Record MPI timings across (n, p) into data/results_mpi_local.csv:
python src/benchmark_mpi.py

# 8. Run the two-rank MPI ping-pong communication benchmark:
mpiexec -n 2 python src/pingpong.py --output-csv data/pingpong_local.csv
```

---

## 6. Conclusion & Deliverables

The project contains serial, shared-memory threaded, and local MPI implementations. The checked-in performance analysis covers the serial and threaded kernels; the local MPI implementation demonstrates row-block distribution, collective communication, and correctness verification without introducing VM, Docker, or multi-node deployment.

### Deliverables:
- **Source Code:** `src/version1_threads.py`, `src/baseline_serial.py`, `src/mpi_matmul.py`, `src/pingpong.py`, `src/common.py`, `src/verify_all.py`, `src/benchmark_runner.py`, `src/benchmark_mpi.py`, `src/plot_results.py`.
- **Benchmark Data:** `data/results_serial_threads.csv` (all rows measured; `source` column documents provenance); `data/results_mpi_local.csv`; `data/pingpong_local.csv`.
- **Publication Plots (300 DPI):** `plots/speedup_threads.png`, `plots/efficiency_threads.png`, `plots/cache_blocking_comparison.png`.
- **Academic Word Report:** `REPORT.docx`.
- **Markdown Report:** `REPORT.md`.

### Scope note for this submission:
The proposal's Versions 2–4 (MPI on virtual machines, Docker containers, and a physical LAN) are **not** part of the graded scope for this consultation: the VM, Docker, and cluster deployment assets (`vm/`, `docker/`, `cluster/`) were removed from the repository. All MPI functionality is exercised locally on one machine with multiple ranks, which uses the exact same program and decomposition; only the launch procedure (hostfile / networking) is absent.
