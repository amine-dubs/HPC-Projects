# Consultation Notes — Parallel Matrix Multiplication HPC Mini-Project

> Last updated: 2026-11-01. Read this before the meeting; section 7 is the live-demo script, section 6 is the Q&A cheat sheet.

---

## 1. What the project is about (elevator pitch, 60 seconds)

Dense matrix multiplication `C = A x B` on `float64` square matrices is the standard HPC benchmark: the work grows as `O(n^3)` (≈ `2*n^3` FLOPs) while the data grows as only `O(n^2)`, so it stresses both compute throughput and data movement. We implement and measure the **same problem** with the two classic execution models:

1. **Version 0 — Serial baseline:** Numba `@njit`-compiled triple loop (plain Python would interpret this loop ~100x slower), with a cache-optimal `(i, k, j)` loop order.
2. **Version 1 — Shared memory:** Numba `prange` (OpenMP-style parallel-for) over rows of `C`, dynamically setting the thread count `p ∈ {1, 2, 4, 8}`, plus an optional cache-blocked (tiled) variant.
3. **Local MPI (the engine of proposal Versions 2–4):** `mpi4py` row-block decomposition — rank 0 creates A and B, broadcasts B, scatters row blocks of A, each rank computes its block, rank 0 gathers C. All ranks run on one machine; the code is exactly the same as would be launched on VMs / containers / LAN — only the launch command (hostfile, network) differs.

Everything is verified against NumPy BLAS (`A @ B`) with `np.allclose`, timed with warm-up (JIT compiled before timing), 3 repetitions with the median reported, speedup `S = T1/Tp`, efficiency `E = S/p`, and GFLOP/s. A two-rank MPI ping-pong measures latency and bandwidth of the transport (Hockney model).

**Scope agreed for this consultation:** the Docker / VM deployment targets (proposal Versions 2 and 3 as deployments, plus the LAN-machines machinery) are **not** part of this submission. That means we may be asked about them — see the "limits" answers in Section 6 — but the graded scope covers serial + threads + MPI code correctness + local measurements + analysis.

---

## 2. Review: proposal items vs what is implemented (everything except Docker/VM)

| Proposal requirement | Status | Where |
| :--- | :--- | :--- |
| Version 0 serial baseline with Numba `@njit` | Done | `src/baseline_serial.py` (`matmul_serial_ikj`) |
| Cache-friendly loop ordering explanation | Done (goes beyond) | `src/baseline_serial.py` keeps both `(i,k,j)` fast kernel and `(i,j,k)` naive kernel for comparison |
| Version 1: threads via `prange`, outer loop over rows of C | Done | `src/version1_threads.py` (`matmul_threads_naive`, `prange(n)` on line 37) |
| Version 1: compare thread counts p = 1, 2, 4, 8 | Done | `src/version1_threads.py`, `src/benchmark_runner.py` |
| Version 1: optional cache-blocked variant | Done (optional bonus) | `src/version1_threads.py` (`matmul_threads_blocked`, 64x64 tiles) |
| MPI row-block code (Bcast B, Scatter A rows, compute, Gather C) | Done | `src/mpi_matmul.py` |
| Same MPI code for versions 2–4, only launch changes | Done — code is deployment-independent | `src/mpi_matmul.py` + README run command |
| `np.allclose` correctness vs serial/BLAS | Done | `src/common.py` (`verify_result`, `rtol=1e-5`, `atol=1e-8`); `src/verify_all.py` (6 unit tests, all pass) |
| Sizes n = 1024, 2048, 4096 | Done — **all measured** (4096 measured in a ~17-min session) | `data/results_serial_threads.csv`, every row `source=measured` |
| p = 1, 2, 4, 8 processes/threads | Done | threads CSV + `data/results_mpi_local.csv` (freshly created) |
| Warm up Numba before timing | Done | `warmup_serial()`, `warmup_threads()`, `warmup_mpi_kernel()` |
| Time only computation/communication, not matrix generation | Done | matrices generated outside the timed region in every script; MPI times each collective separately |
| Time broadcast, scatter, compute, gather separately | Done | `src/mpi_matmul.py` (MPI.Wtime per phase, MAX across ranks + average compute) |
| Repeat >= 3x, report median | Done | default `repeats=3` + `np.median` everywhere |
| Speedup + efficiency + plots vs p | Done | `data/results_serial_threads.csv`, `src/plot_results.py`, `plots/speedup_threads.png`, `plots/efficiency_threads.png` |
| Ping-pong test (latency + bandwidth) | Done locally (one target = this machine) | `src/pingpong.py`, data in `data/pingpong_local.csv` |
| NumPy `A @ B` as reference, `OMP_NUM_THREADS=1` in MPI ranks | Done (bug fixed this week: env vars were set after NumPy import and ignored — now set before import in `src/mpi_matmul.py`) | `src/mpi_matmul.py` header |
| Raw timing data (CSV) | Done | `data/results_serial_threads.csv`, `data/results_mpi_local.csv`, `data/pingpong_local.csv` |
| Speedup/efficiency plots + cache-blocking chart | Done | `plots/` (3 PNG, 300 DPI) |
| 4–6 page report: design, results, bottlenecks, conclusions | Done | `REPORT.md` + `REPORT.docx` (regenerated this week) |
| README with run instructions | Done | `README.md` |
| ~10-minute live demo | Done locally (LAN demo excluded with the deployment scope) | Section 7 of this document |
| Dockerfile / Compose / VMs / macvlan / hostfiles / addressing table / LAN deployment | **Not in scope (excluded)** | `docker/`, `vm/`, `cluster/` folders were removed from the repo on purpose |

**Bottom line:** everything in the proposal except the Docker/VM/LAN deployment machinery is implemented, measured, plotted, tested and reported. Three small gaps were fixed this week and one honest caveat must be stated proactively (see 4.1 and Section 6, question Q2).

---

## 3. Repository map (what to open in front of the professor)

```text
Projet-01-Matrix-Multiplication/
├── Matrix_Multiplication_Mini_Project_Proposal.docx   # the spec we reviewed
├── REPORT.docx / REPORT.md     # the report (design, results, bottlenecks)
├── README.md                   # project guide + run instructions
├── requirements.txt            # numpy, numba, matplotlib, mpi4py, python-docx
├── generate_report_docx.py     # regenerates REPORT.docx (hardcoded summary + live CSV table)
├── notes.md                    # this file
├── src/
│   ├── common.py               # shared utils: matrix generation (seed=42), np.allclose verify, 2n^3 GFLOP/s, memory calc
│   ├── baseline_serial.py      # V0: serial @njit kernels (ikj fast + ijk naive) + benchmark
│   ├── version1_threads.py     # V1: prange kernel + 64x64 tiled kernel + thread-scaling benchmark
│   ├── mpi_matmul.py           # MPI: row-block decomposition, per-phase timers, rank-0 verification
│   ├── pingpong.py             # MPI ping-pong: latency/bandwidth, Hockney model
│   ├── benchmark_runner.py     # runs V0/V1 grid (n x p), writes results_serial_threads.csv
│   ├── benchmark_mpi.py        # NEW: runs MPI grid (n x p), writes results_mpi_local.csv
│   ├── plot_results.py         # 300-DPI plots from the CSV
│   └── verify_all.py           # 6 unit tests (generation, correctness, metrics) — all pass
├── data/
│   ├── results_serial_threads.csv   # V0/V1: all measured rows (incl. real n=4096)
│   ├── results_mpi_local.csv        # NEW: measured local-MPI per-phase timing breakdown
│   └── pingpong_local.csv           # NEW: measured MPI latency/bandwidth sweep
└── plots/
    ├── speedup_threads.png          # S vs p with ideal line
    ├── efficiency_threads.png       # E vs p
    └── cache_blocking_comparison.png # tiled vs naive (time + GFLOP/s)
```

---

## 4. Results (the numbers to know by heart)

Hardware used for everything: **AMD Ryzen 5 5500U laptop — 6 physical cores / 12 logical threads (boost ~4.0 GHz), 14.8 GB RAM, Windows.** Python 3.12.10, NumPy 2.5.4, Numba 0.68.0 (threading layer `omp`), mpi4py 4.1.2 on MS-MPI 10.1. The V0/V1 table was measured in one clean ~17-minute session (laptop plugged in, background idle) that included **real measured n = 4096 runs**; the MPI and ping-pong tables were measured the previous day, and the two sessions agree at $p=1$ ($0.538\ \text{s}$ vs $0.537\ \text{s}$ at $n = 1024$), so they can be presented together.

### 4.1 Version 0 + Version 1 (`data/results_serial_threads.csv`, all rows `source=measured`)

| n | Version | p | Median time (s) | Speedup | Efficiency | GFLOP/s |
|---|---|---|---|---|---|---|
| 1024 | V0 Serial | 1 | 0.548 | 1.00x | 100% | 3.92 |
| 1024 | V1 Threads | 2 | 0.327 | 1.67x | 83.7% | 6.57 |
| 1024 | V1 Threads | 4 | 0.167 | 3.29x | 82.1% | 12.88 |
| 1024 | V1 Threads | 8 | 0.091 | 6.00x | 75.0% | 23.53 |
| 1024 | V1 Blocked | 4 | 0.663 | 0.83x | 20.7% | 3.24 |
| 1024 | V1 Blocked | 8 | 0.350 | 1.57x | 19.6% | 6.14 |
| 2048 | V0 Serial | 1 | 4.436 | 1.00x | 100% | 3.87 |
| 2048 | V1 Threads | 2 | 2.901 | 1.53x | 76.4% | 5.92 |
| 2048 | V1 Threads | 4 | 1.810 | 2.45x | 61.3% | 9.49 |
| 2048 | V1 Threads | 8 | 0.991 | 4.48x | 56.0% | 17.34 |
| 2048 | V1 Blocked | 4 | 5.726 | 0.78x | 19.4% | 3.00 |
| 2048 | V1 Blocked | 8 | 2.987 | 1.49x | 18.6% | 5.75 |
| 4096 | V0 Serial | 1 | 39.006 | 1.00x | 100% | 3.52 |
| 4096 | V1 Threads | 2 | 25.511 | 1.53x | 76.5% | 5.39 |
| 4096 | V1 Threads | 4 | 15.073 | 2.59x | 64.7% | 9.12 |
| 4096 | V1 Threads | 8 | 14.850 | 2.63x | 32.8% | 9.26 |
| 4096 | V1 Blocked | 4 | 43.470 | 0.90x | 22.4% | 3.16 |
| 4096 | V1 Blocked | 8 | 23.967 | 1.63x | 20.3% | 5.73 |

(Also measured, omitted above for space: V1 Threads p=1 ≈ serial at every size — e.g. 0.538 s vs 0.548 s at n=1024 — the prange scheduling overhead is negligible.)

Story the data tells:
- **Good but sub-linear thread scaling:** efficiency sits in the 56–84% band at p=2–8. Two reasons, both classic: every thread re-streams the whole matrix B from memory (bandwidth sharing), and threads compete for the shared L2/L3 once the working set (96 MB at n=2048, 384 MB at n=4096) exceeds cache. No locks and even row splits → no load imbalance.
- **Why p=8 is excellent at n=1024 (6.00x) but saturates at n=4096 (2.63x):** at n=1024 the 8 MB matrix B fits in the shared 8 MB L3, so threads serve B from cache and hyper-threading overlaps memory stalls productively. At n=4096, B is 128 MB: the bus is already saturated by the 6 physical cores, and p=8 adds almost nothing over p=4 (14.85 s vs 15.07 s). Clean textbook progression: speedup's ceiling falls from compute-bound to bandwidth-bound as the problem grows.
- **Previous-session caveat (if the professor saw older numbers):** an earlier benchmark session (heavier background load/thermal state) produced different absolute times and a p=8 collapse at n=2048 (1.97x). Under clean conditions p=8 at n=2048 measures 4.48x. Lesson presented proactively: absolute times are session-dependent on laptops — that is exactly why every speedup is computed within one session and we re-measured everything in a single clean session.
- **Blocked (tiled) variant measured slower than the naive kernel — in fact slower than serial at p=4** (0.78x–0.90x of serial). Honest interpretation: our tiling does cut B's DRAM traffic, but (1) Numba's generated inner loops are not register-blocked — each `C[i,j]` is still touched once per `k` (that traffic stays in L1/L2, where the naive kernel's traffic was anyway), (2) extra per-tile index/`min()` arithmetic defeats some codegen optimizations, (3) the naive `(i,k,j)` kernel already streams unit-stride with good prefetch. BLAS-style register blocking, panel packing, or `fastmath=True` vectorization would be the next step. We present it as an instructive negative result; the goal was to compare, not to beat BLAS.

### 4.2 Local MPI (`data/results_mpi_local.csv`, 3 reps, median run)

| n | p | Bcast (s) | Scatter (s) | Compute (s) | Gather (s) | Total (s) | Compute % | Comm % | GFLOP/s |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 0.00005 | 0.00191 | 0.53345 | 0.00183 | 0.53725 | 99.3 | 0.7 | 4.00 |
| 1024 | 2 | 0.00338 | 0.00285 | 0.30008 | 0.00336 | 0.30780 | 97.5 | 2.5 | 6.98 |
| 1024 | 4 | 0.00718 | 0.00554 | 0.24778 | 0.01389 | 0.26128 | 94.8 | 5.2 | 8.22 |
| 1024 | 8 | 0.01534 | 0.00581 | 0.30228 | 0.01520 | 0.32346 | 93.5 | 6.5 | 6.64 |
| 2048 | 1 | 0.00005 | 0.00734 | 4.36796 | 0.00740 | 4.38275 | 99.7 | 0.3 | 3.92 |
| 2048 | 2 | 0.01636 | 0.01051 | 2.72803 | 0.01835 | 2.76485 | 98.7 | 1.3 | 6.21 |
| 2048 | 4 | 0.02943 | 0.02145 | 2.08731 | 0.05662 | 2.13587 | 97.7 | 2.3 | 8.04 |
| 2048 | 8 | 0.05162 | 0.02061 | 2.31421 | 0.04818 | 2.39265 | 96.7 | 3.3 | 7.18 |

Story:
- **On one host, communication is almost free** (0.3–6.5% of total): MS-MPI uses shared-memory copies between ranks on the same machine (measured below at ~8 GB/s). Compute dominates. On a real LAN the same broadcast/scatter/gather of 8–128 MB would dominate (gigabit ≈ 110 MB/s) — that is exactly what the excluded deployment targets were designed to quantify; we can articulate this expectation clearly.
- **Speedup within MPI:** best at p=4 (2.06x at n=1024, 2.05x at n=2048) — again the number of physical cores; p=8 oversubscribes 6 cores. Per-rank compute time grows with p despite smaller chunks: bandwidth contention between ranks.
- **Verification:** rank 0 checks `C` against BLAS — `max abs err ≈ 1.05e-12` (n=1024) and `3.07e-12` (n=2048), far inside `atol=1e-8`.
- **Kernel note (asked about it? see Q3):** MPI ranks use a plain single-threaded `@njit` kernel (no `prange`). In the clean benchmark session the two kernels agree almost exactly at p=1 (V1: 0.538 s vs MPI: 0.537 s at n=1024; 4.37 s vs 4.38 s at n=2048), so the two versions can be presented side by side; only cross-**session** absolute times must not be compared (thermal/background load shifts each session's scale).

### 4.3 Ping-pong (`data/pingpong_local.csv`, 30 trials/size, message sizes 8 B → 32 MB)

- **Latency alpha ≈ 2.25 microseconds** (8-byte one-way time = RTT/2).
- **Peak bandwidth ≈ 8,157 MB/s (~65 Gbps) at 1 MB messages**, plateauing at ~3.6–3.8 GB/s for multi-MB transfers (the shared-memory `memcpy` limit of the machine).
- Model: Hockney `T(m) = alpha + m/beta`. Small messages are latency-bound; large ones saturate copy bandwidth. Compare to real wired LAN values (alpha ≈ 50–100 us, beta ≈ 110 MB/s on gigabit) — 25x worse latency, ~30x lower bandwidth. This is the single-slide answer to "why would LAN deployment be interesting at all?".

---

## 5. Implementation decisions — why, and where in the code

| Decision | Why | Where |
|---|---|---|
| Numba `@njit` for the serial baseline | Plain Python loops are interpreted (~100x slower); numba JIT-compiles to native machine code while keeping pure-Python source. Also, ordinary Python threads would be useless because of the **GIL** — this is the bridge to Version 1. | `baseline_serial.py::matmul_serial_ikj` |
| `(i, k, j)` loop order instead of textbook `(i, j, k)` | Row-major layout: innermost `j` walks B and C rows with stride-1 (unit 8-byte) access → cache-line friendly, prefetcher-friendly, SIMD-vectorizable. `(i, j, k)` strides B access by `n*8` bytes (16 KB for n=2048) → cache misses. We keep the `ijk` kernel in the code to explain the difference on request. | `baseline_serial.py` (both kernels), explained in REPORT.md §2.1 |
| `fastmath=False` everywhere | Reordered/contracted FP math changes results run-to-run; we want deterministic IEEE arithmetic so `np.allclose` verification is meaningful and stable. | decorators in both kernel files |
| Verify against NumPy BLAS, `rtol=1e-5, atol=1e-8` | BLAS is an independent, trusted oracle. Tolerance reflects last-place floating-point differences (measured max abs err ~1e-12/1e-13, orders of magnitude below tolerance). | `common.py::verify_result`, `verify_all.py` |
| `prange` over rows of C only | Threads write disjoint rows → no locks/critical sections, perfect load balance; `prange` is Numba's OpenMP parallel-for. Thread count is runtime-configurable via `numba.set_num_threads(p)`. | `version1_threads.py::matmul_threads_naive` (prange on the `i` loop) |
| Cache-blocked variant, 64x64 tiles | 64x64 doubles = 32 KB ≈ L1 data cache; tiles are reused before eviction. Parallel over `bi` (row-block index) with unit-stride `prange` (Numba parallel loop constraint). | `version1_threads.py::matmul_threads_blocked` |
| Row-block MPI decomposition (not Fox/Cannon/block-cyclic) | Simplest correct distribution for a dense product: B replicated is only 2x memory, A rows split evenly, only 3 collectives needed (Bcast, Scatter, Gather). `n % p == 0` enforced; load balance scales with n/p rows per rank. | `mpi_matmul.py` (algorithm header + steps 1–5) |
| Per-phase MPI timers with MAX-across-ranks + avg | Compute/comm analysis needs the *critical path* (slowest rank) not the average; we also keep the per-rank average compute. Ratios are computed from `max_total - max_compute` so compute% + comm% = 100% (the per-phase maxima may come from different ranks, so summing them can exceed the true total — this summarization bug was fixed this week). | `mpi_matmul.py` (timed steps + rank-0 summary) |
| `OMP_NUM_THREADS=1` (and MKL/OPENBLAS/NUMEXPR) for MPI ranks | Each rank would otherwise run multithreaded BLAS → p ranks x all cores = massive oversubscription. Must be set **before** NumPy import (BLAS reads env at load time) — fixed this week. | `mpi_matmul.py` header |
| Matrices generated outside the timed region; seed=42 everywhere | deterministic A, B for reproducibility and fair comparison; timing measures only the algorithm. | `common.py::generate_matrices`, benchmark scripts |
| Warm-up JIT before every timing loop | First call includes LLVM compile time (would distort speedup massively). Small 64x64 (or 16x16) warm-up calls compile the exact kernels. | `warmup_serial`, `warmup_threads`, `warmup_mpi_kernel` |
| 3 repetitions, median | Suppresses outliers from OS scheduling/background load; median is robust. | all benchmark scripts (`repeats=3`, `np.median`) |
| `time.perf_counter` for host timing; `MPI.Wtime` for rank timing | perf_counter is a monotonic high-resolution wall clock; Wtime is MPI's guaranteed synchronized clock and is required inside ranks. Multiple ranks' phases must be reduced with MAX (critical path). | benchmark scripts / `mpi_matmul.py` |
| Ping-pong one-way = RTT/2 | Classical definition; timing done at rank 0 over `num_trials`, warm-up transfers first, then the Hockney fit (alpha from smallest message, beta from largest bandwidth). | `pingpong.py` |
| CSV `source` column (measured / extrapolated) | Keeps the dataset self-auditing: benchmark_runner emits flagged analytic `O(n^3)` scale estimates only if a size is skipped; in the final dataset every row is `source=measured`. | `benchmark_runner.py` (guarded extrapolation block + fieldnames) |

---

## 6. Probable professor questions — and short answers

**Q1. Why not plain Python threads?**
The GIL allows only one Python bytecode thread to execute at a time → zero speedup for compute-bound loops. We need to release Python: Numba compiles the loop to native code and `prange` acts as an OpenMP parallel-for with the `omp` threading layer (we checked `numba.threading_layer() == 'omp'`).

**Q2. Are all the numbers measured?**
Yes — all sizes including n=4096 were re-measured in one clean ~17-minute session (every CSV row is `source=measured`; serial 4096 median 39.0 s). Local MPI and ping-pong tables were measured the previous day, and the two sessions agree at p=1 (0.538 vs 0.537 s at n=1024), so they can be presented together.

**Q3. Why do p=1 timings of MPI and V1 threads match so closely?**
Intentional design: both run essentially the same single-threaded `@njit` kernel (V1 with `prange` scheduling that costs nothing at p=1, MPI without it). We verified: 0.538 vs 0.537 s at n=1024. Each version's speedup is computed within its own session against its own serial/own-p=1 baseline, so cross-version throughput vs p can be compared fairly.

**Q4. Why is speedup measured against the Version 0 serial kernel, and why is "threads p=1" slightly above 1.0x?**
Speedup S = T1/Tp with T1 = the serial baseline (the proposal's definition); the prange kernel at p=1 runs a bit faster than the plain serial kernel in that session (within run-to-run noise; median of 3). We also have version-internal speedups (own p=1) inside `version1_threads.py`.

**Q5. Why is efficiency below 100% (56–84% at p=2–8)? Why no super-linear speeds?**
Our teaching kernel is deliberately not BLAS: every thread re-streams the whole matrix B from memory, so bandwidth is shared and cache pressure grows with the working set (96 MB at n=2048, 384 MB at n=4096, vs ~8 MB L3). Load balance is perfect (even row splits, disjoint C writes, no locks) — the shortfall is memory-system, not scheduling. At n=1024 (8 MB B, cache-friendly) efficiency is highest (82–84% and 75% at p=8).

**Q6. Why does p=8 saturate at n=4096 (2.63x, 32.8%) while it helped at n=1024 (6.00x, 75%)?**
At n=1024 B (8 MB) fits the shared L3: threads feed B from cache, so hyper-threading overlaps stalls and even logical threads add throughput. At n=4096, B (128 MB) streams from RAM: the 6 physical cores already saturate the DDR bus, so the 2 extra logical threads mostly duplicate pending cache-line misses (14.85 s vs 15.07 s at p=4). Classic transition from compute-bound to bandwidth-bound as the problem grows — exactly the limit the proposal asked us to discuss ("scaling beyond the host core count cannot be expected / shared memory bandwidth").

**Q7. Why is the cache-blocked kernel slower?**
Measured fact on our kernel/machine. Our tiling eliminates a lot of B DRAM traffic, but (a) Numba's generated inner loops inside the tile structure aren't register-blocked — each C element is still touched per k (but that traffic stays in cache), (b) extra per-tile index arithmetic, (c) the naive ikj kernel already streams unit-stride. A register-blocked or `fastmath=True`-vectorized kernel would be the next step. We treat it as an instructive negative result; the goal was to compare, not to beat BLAS.

**Q8. How do you know the results are correct?**
Every kernel verifies against NumPy BLAS with `np.allclose(rtol=1e-5, atol=1e-8)`; `src/verify_all.py` has 6 unit tests (generation determinism, each kernel, metrics formulas) — all pass. MPI rank 0 verifies the gathered C every run (max abs err ~3e-12 at n=2048).

**Q9. What's the total FLOP count / GFLOP-s formula?**
`2*n^3` (n^3 multiplications + n^3 additions); GFLOP/s = `2n^3 / time / 1e9`. Implemented in `common.py::calculate_gflops` and unit-tested.

**Q10. Explain the MPI communication phases and their cost.**
Bcast (n^2 elements of B) + Scatter (n^2 total of A rows) + Gather (n^2 total of C rows) — 3n^2 data total vs 2n^3 work → O(1/p) communication per rank but the constants matter only when compute is fast enough; locally they are 0.3–6.5% of the time (shared-memory transport ~8 GB/s). On gigabit LAN they would dominate — that's the point of the excluded deployment targets.

**Q11. Why only 1 thread per BLAS in MPI ranks?**
Oversubscription: p ranks x 12 BLAS threads = 48+ threads on 6 cores. Set before import because BLAS reads env at load time.

**Q12. What is the Hockney model? alpha/beta?**
`T(m) = alpha + m/beta`: startup overhead (latency, ~2.25 us measured locally) plus per-byte cost (1/beta). Standard point-to-point performance model for MPI; we fit it from the ping-pong sweep.

**Q13. What would change on a real LAN (versions 2–4)?**
Same code, different launcher: `mpirun -np p --hostfile hosts python mpi_matmul.py` (OpenMPI) or MS-MPI `mpiexec -hosts 2 nodeA 1 nodeB 1 python ...`. Expect latency tens of us and ~100 MB/s bandwidth → bcast+scatter+gather of 32–128 MB would dominate for small p; strategies would be to increase per-rank arithmetic (bigger n per rank), overlap communication with compute, or scatter A rows only. We excluded deployment per scope but can discuss.

**Q14. Why n%p==0? What if not?**
Equal-size row blocks keep Scatter/Gather simple. Generalization: `Scatterv/Gatherv` with unequal counts (row padding).

**Q15. Why is the prange kernel not using `A @ B` / BLAS internally?**
The educational goal is to implement the parallel decomposition ourselves; BLAS calls (reference + MPI numpy-kernel option) are kept precisely as the "what a vendor-optimized library achieves" reference: BLAS sustains ~20–30 GFLOP/s per single-threaded rank here (the MPI `numpy` kernel run measured ~87 GFLOP/s aggregate over 4 ranks) vs ~4 GFLOP/s per thread for our scalar teaching kernel (AVX2 FMA micro-kernels + register blocking + packaging in BLAS; we compile with `fastmath=False` deterministic loops). The gap is ~5–8x per core and is the honest reason our threaded gains look "modest" in GFLOP/s.

**Q16. What is your machine? Memory needed?**
Ryzen 5 5500U: 6 cores/12 threads; 14.8 GB RAM. n=4096 float64: 128 MB/matrix, ~384 MB working set — fits comfortably (measured: 39 s serial); n=8192 would be 512 MB/matrix (1.5 GB set) — still fits but slow (~4 min/run serial).

**Q17. Where does time actually go in your kernel?**
Cache behavior: with (i,k,j), matrix B is streamed from RAM for each row-block pattern (bandwidth-bound portion), C updates stay in L1 (a C row is 16 KB at n=2048), so we are partially memory-bandwidth-limited: single-thread at n=2048 ≈ 3.9 GFLOP/s in the clean session (up to ~4.0 at n=1024); threading helps until the bus saturates (beyond 6 cores at large n) — that's why p=8 gains vanish at n=4096.

---

## 7. How to run it live in front of the professor (~8–10 min)

**Pre-flight (5 min before):**
- [ ] Open PowerShell **in the project folder**: `cd "C:\Users\LENOVO\Desktop\9th sem tps\Intensive computing\fractal-hpc\matrixmultiplication\Projet-01-Matrix-Multiplication"`
- [ ] `python --version` → 3.12.x with numpy/numba/mpi4py (already installed; nothing else needed)
- [ ] Plug in power adapter (avoid CPU throttling), close other heavy apps
- [ ] Have open: this file, `plots/speedup_threads.png`, `plots/efficiency_threads.png`, `data/results_serial_threads.csv`

**Demo script (each step has an expected outcome):**
1. **Tests** (~15 s, all green): `python src/verify_all.py` → `Ran 6 tests ... OK`. Talking point: "every kernel is verified against BLAS with np.allclose."
2. **Serial baseline** (~5 s for 1024): `python src/baseline_serial.py --sizes 1024 --repeats 3` → per-run times + GFLOP/s. Talking point: `(i,k,j)` loop order, why cache-friendly.
3. **Thread scaling** (~25 s): `python src/version1_threads.py --sizes 1024 --threads 1 2 4 8` → speedup/efficiency printed per p. Talking point: prange = OpenMP parallel-for; no locks (disjoint rows).
4. **Blocked variant** (~15 s): `python src/version1_threads.py --sizes 1024 --threads 4 8 --kernel blocked --block-size 64` → slower than naive: honest teaching point (Q7).
5. **Plots** (~5 s): `python src/plot_results.py` → regenerates the 3 PNGs from the CSV; open them.
6. **MPI** (~20 s): `mpiexec -n 4 python src/mpi_matmul.py --n 2048 --kernel numba` → per-phase timing breakdown (bcast/scatter/compute/gather), communication % small, verification PASSED. Talking point: same code that would run on the LAN; only the launcher changes.
7. **MPI benchmark data** (~1 min if needed to show automation): `python src/benchmark_mpi.py` → walks (n,p) grid, writes `data/results_mpi_local.csv`.
8. **Ping-pong** (~15 s): `mpiexec -n 2 python src/pingpong.py --output-csv data/pingpong_local.csv --trials 20` → alpha ≈ 2 us, peak MB/s. Talking point: Hockney model; why local MPI is compute-bound but a LAN wouldn't be.

**If only 5 minutes:** steps 1 → 3 → 6 → 8, then show the plots and the CSVs.

**Troubleshooting:**
- `mpiexec` not found → the MS-MPI runtime is installed on the PATH of this machine; make sure you didn't switch to a different shell/PATH. Test `mpiexec --version`.
- `numba` compile seems slow on first run of a demo step → that's JIT; the warm-ups inside the scripts take ~10 s the very first time, subsequent runs are instant.
- If a run seems abnormally slow (thermal), say so and re-run once; medians of 3 already damp this.

---

## 8. Remaining gaps / optional upgrades

Done before the consultation:
1. ✅ **Real n=4096 measured** (2026-11-01, ~17 min): `python src/benchmark_runner.py --sizes 1024 2048 4096` — all 21 rows measured, plots regenerated, REPORT.md / REPORT.docx / README.md / this file updated with the measured values.
2. ✅ **Committed:** the three measured CSVs (`results_serial_threads.csv` re-measured, `results_mpi_local.csv`, `pingpong_local.csv`), `src/benchmark_mpi.py`, `notes.md`, and all report/doc updates.

Optional upgrades (pick by time available):
3. **[Optional, ~30 min] Fastmath/vectorization experiment:** a copy of the kernel with `fastmath=True` to show SIMD vectorization gains (~2–4x single-thread typically) — great for the "why is BLAS 5–8x faster than our kernel per core" discussion.
4. **[Optional] Different `n % p` handling:** Scatterv/Gatherv for non-divisible sizes (one-line mention if asked: future work).
5. **[If asked for LAN deployment evidence]** The same code can be launched across two physical PCs with MS-MPI: install MS-MPI runtime on both, put the project on both, enable smpd service, run `mpiexec -hosts 2 <ipA> 1 <ipB> 1 python src/mpi_matmul.py --n 2048`. Code needs no change. Only mention/promise, don't do it live unless pre-arranged.

---

## 9. Environment facts (in case he asks for the toolchain)

- Python 3.12.10 · NumPy 2.5.4 · Numba 0.68.0 (threading layer `omp`) · mpi4py 4.1.2 · MS-MPI 10.1.12498.52 · Matplotlib (plots) · python-docx (report generator)
- CPU: AMD Ryzen 5 5500U (6 cores, 12 logical threads) · RAM 14.8 GB · Windows
- Verification tolerance: `np.allclose(rtol=1e-5, atol=1e-8)`, typical measured max abs error 1e-12…1e-13
- Fixed RNG seed 42 for A and B everywhere (determinism), float64 (double precision) per the proposal
