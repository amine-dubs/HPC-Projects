# Presenter & Defense Notes: 2D Parallel Heat Diffusion in Python

**Authors:** Bellatreche Mohamed Amine, Rezaiguia Soltane  
**Course:** Intensive Computing (5th year ING, Data Science, USTO-MB)  
**Supervisor:** Prof. Dairi Abdelkader  
**Academic Year:** 2026–2027  
**Project Phase:** Phase 1 (Local Execution & Verification)

---

## 1. The Project in Two Minutes

We solve the 2D steady-state heat equation (Laplace's equation $\nabla^2 u = 0$) on a square domain $[0, 1] \times [0, 1]$ using the classical Jacobi iterative relaxation method on an $N \times N$ discrete mesh.

- **The Five-Point Stencil:** At every discrete time step, the temperature of an interior cell $(i, j)$ is updated to the arithmetic mean of its four cardinal neighbors:
  $$u^{(k+1)}(i, j) = \frac{1}{4} \left( u^{(k)}(i-1, j) + u^{(k)}(i+1, j) + u^{(k)}(i, j-1) + u^{(k)}(i, j+1) \right)$$
- **What is a Halo (Ghost Layer)?** In distributed memory (MPI), the $N \times N$ domain is partitioned into horizontal row slices across $p$ processes. To evaluate the stencil on the boundary rows of each slice without sharing memory, each rank allocates an extra ghost row above and below its local slice. These ghost rows are called **halos**.
- **Communication-Heavy Bottleneck:** At every single step, each process must send its boundary row to its neighbor and receive its neighbor's boundary row into its halo buffer. While computation scales with cell count $\mathcal{O}(N^2 / p)$, communication scales with the boundary perimeter $\mathcal{O}(N)$. On local shared memory or network architectures, latency and halo synchronization rapidly become the dominant performance bottleneck.

---

## 2. Proposal vs. What We Did

| Component / Objective from Proposal | Planned | Phase 1 Status | Explanation / Justification |
|:---|:---:|:---:|:---|
| **Version 0a: Pure NumPy Reference** | Yes | **DONE** | Vectorized array-slicing baseline used for ground truth correctness testing. |
| **Version 0b: Numba Serial** | Yes | **DONE** | JIT-compiled serial baseline (`@njit(cache=True)`) providing 15x–20x speedup over NumPy. |
| **Version 1: Multithreaded (Numba prange)** | Yes | **DONE** | OpenMP threading layer with per-row reduction buffer bypassing Python GIL. |
| **Version 2–4: Distributed MPI (Blocking Sendrecv)** | Yes | **DONE (Local)** | Launch-independent 1D row decomposition executed locally via MS-MPI. |
| **Version Extra: Non-blocking MPI Overlap** | Extra | **DONE** | Overlaps interior computation with halo `MPI_Isend`/`MPI_Irecv` communication. |
| **Ping-Pong Latency & Bandwidth Benchmark** | Yes | **DONE (Local)** | Evaluated across 21 message sizes (8 B to 8 MB) on MS-MPI shared-memory channel. |
| **Correctness Test Suite (E1)** | Yes | **DONE** | 100% pass rate ($max\,|u - u_{ref}| = 0.00e+00$) across all rank counts and grids. |
| **Cost Model Derivation & Validation (E6)** | Yes | **DONE** | Parametric $(\alpha, \beta, t_c)$ model fitted from ping-pong and $p=1$ compute. |
| **Visual Animation & Progression (E9)** | Yes | **DONE** | Inferno GIF (`heat_diffusion.gif`), 4-stage progression, and boundary seam demo. |
| **LaTeX Report (6-page main body + MiKTeX build)** | Yes | **DONE** | Formatted strictly to Section 10.2 title specifications, 0 errors. |
| **Version 2 (VMs) Deployment (E10)** | Yes | *Deferred to Phase 2* | Scope change: Local execution only in Phase 1; scripts kept launch-independent. |
| **Version 3 (Docker Containers) Deployment (E7)** | Yes | *Deferred to Phase 2* | Scope change: Docker omitted; `--target` CLI parameter ready for future targets. |
| **Version 4 (Physical LAN) Deployment (E10)** | Yes | *Deferred to Phase 2* | Scope change: Physical lab network reserved for Phase 2 consultation demo. |

---

## 3. Environment

- **Host Machine:** AMD Ryzen 5 5500U with Radeon Graphics (Laptop Host `LAPTOP-D2G4S41N`)
- **Compute Cores:** 6 physical cores, 12 logical hyperthreads (base clock 2.10 GHz, boost up to 4.0 GHz)
- **Memory (RAM):** 15.3 GB DDR4 Dual-Channel
- **Operating System:** Microsoft Windows 11 Home (64-bit, Build 26100)
- **Python Runtime:** Python 3.12.10 (AMD64 native)
- **MPI Implementation:** Microsoft MPI v10.1.12498.52 with `mpi4py` 4.1.2
- **JIT Compiler:** Numba 0.68.1 (LLVM-backed, `threading_layer: omp`)
- **LaTeX Distribution:** MiKTeX 24.1.0 (`pdflatex`)

---

## 4. File Map

- **`src/heat_common.py`** (OPEN FIRST): Shared mathematical definitions, Dirichlet boundary initialization (corner convention: columns override rows), circular hotspot injection, rank decomposition math, checksum and CSV logging utilities.
- **`src/heat_reference.py`**: Pure NumPy reference implementation using vectorized 2D slices (`u[1:-1, 1:-1] = 0.25 * (...)`).
- **`src/heat_serial.py`**: Serial baseline accelerated with `@njit(fastmath=False, cache=True)`.
- **`src/heat_threads.py`**: Multithreaded Numba solver using OpenMP `prange` and array-buffered row reductions.
- **`src/heat_mpi.py`**: Distributed MPI solver using 1D row decomposition and `comm.Sendrecv` halo exchanges with `MPI.PROC_NULL`.
- **`src/heat_mpi_overlap.py`**: Advanced non-blocking MPI solver overlapping interior row computation with non-blocking halo exchanges (`Isend`/`Irecv`).
- **`src/pingpong.py`**: Point-to-point 2-rank benchmark measuring round-trip latency and effective bandwidth.
- **`src/animate.py`**: Matplotlib animation script generating GIF frames, rank boundary overlays, and the seam demonstration figure.
- **`tests/test_correctness.py`**: Exhaustive automated pytest/unittest suite verifying numerical parity against the NumPy baseline.
- **`scripts/run_threads.py`**: Automated runner for Experiment E2 thread scaling.
- **`scripts/run_mpi_grid.py`**: Automated runner for Experiments E3 and E8 MPI scaling.
- **`scripts/run_convergence.py`**: Automated runner for Experiment E4 convergence tracking and $k$-sweep reduction overhead.
- **`scripts/fit_cost_model.py`**: Experiment E6 cost model parameter regression and validation.
- **`scripts/generate_plots.py`**: Matplotlib publication figure generator for all 9 report plots.
- **`scripts/make_report_assets.py`**: Automates data traceability (Rule 1) and generates LaTeX tables and macro numbers.

---

## 5. Parameters Used

| Parameter | Tested Values | Selected Default | Technical Justification |
|:---|:---:|:---:|:---|
| **Grid Size ($N$)** | 64, 120, 128, 256, 512, 1024, 2048 | 1024 | Covers small L1/L2 cache fits up to 33 MB memory footprints; 1024 provides substantial work per process. |
| **Process Count ($p$)** | 1, 2, 4, 8, 12 | 1, 2, 4, 8 | Evaluates scaling across 1 to 4 physical cores, and probes the oversubscription boundary at $p=8 > 6$. |
| **Step Count ($steps$)** | 100 to 2000 | 2000 | Allows precise, reproducible benchmark timings without premature convergence exit. |
| **Tolerance ($\epsilon$)** | $10^{-4}$ | $10^{-4}$ | High precision convergence criterion for physical heat equilibrium verification. |
| **Check Interval ($k$)** | 1, 10, 50, 200 | 50 | Balances reduction latency overhead against unnecessary iterations past convergence. |
| **Repetitions** | 1 to 3 | 3 | Filters transient OS scheduling spikes by reporting the median over 3 independent trials (each rep plus its spread stays in the raw CSV). |
| **Boundary Values** | Top=100, Bottom=0, Left=60, Right=0 | Same | Asymmetric Dirichlet conditions per project specification; columns strictly override rows at corners. |
| **Floating Precision**| `float64` | `float64` | IEEE 754 double precision ensures exact numerical reproducibility without truncation drift. |

---

## 6. Implementation Decisions

| Decision | Why We Chose It | Alternative Considered | Location in Code |
|:---|:---|:---|:---|
| **Numba instead of Python `threading`** | Python `threading` is constrained by the Global Interpreter Lock (GIL), preventing true multicore parallelism for CPU-bound loops. Numba OpenMP releases the GIL completely. | Pure Python `threading` or `multiprocessing` | `src/heat_threads.py` |
| **Array-buffered reduction in `prange`** | Numba `prange` does not natively support custom scalar max reductions (`max_d = max(max_d, d)` leads to data races). We buffer max per row `row_diffs[i]` and reduce with `np.max()`. | Lock-based reduction (prohibitive overhead) | `src/heat_threads.py:compute_max_diff_parallel` |
| **1D Row Strips vs 2D Blocks** | 1D row decomposition produces contiguous memory rows in C-order, enabling zero-copy MPI buffer transfers without strided vector types. | 2D cartesian block decomposition | `src/heat_common.py:get_row_decomposition` |
| **`Sendrecv` with `MPI.PROC_NULL`** | Combined send/receive eliminates deadlock risks; passing `MPI.PROC_NULL` for boundary ranks avoids branchy `if rank == 0` guards. | Separate `Send` and `Recv` with rank ordering | `src/heat_mpi.py:halo_exchange` |
| **One message per halo row** | Since rows are contiguous in memory, an entire row of $N$ doubles is transferred as a single contiguous buffer (`8N` bytes), maximizing bandwidth efficiency. | Sending individual elements or column packing | `src/heat_mpi.py:halo_exchange` |
| **Check convergence every $k=50$ steps** | `MPI_Allreduce` is a global synchronization barrier; checking every step incurs massive latency overhead (over 60% of runtime in our $k$-sweep). $k=50$ reduces reduction cost to $3.8\%$. | Checking every iteration ($k=1$) | `src/heat_mpi.py:run_mpi_solver` |
| **Interior-only convergence reduction** | Evaluating halos in local max differences introduces duplicate cells and boundary contamination. Restricting check to `[1:-1, 1:-1]` guarantees iteration parity across $p$. | Checking local grid including halos | `src/heat_mpi.py:run_mpi_solver` |
| **Boundaries set once in both buffers** | Jacobi alternates $u$ and $v$ pointers. Initializing static Dirichlet boundaries in both $u$ and $v$ avoids re-copying boundaries at every step. | Re-applying boundaries inside kernel | `src/heat_common.py:init_grid` |
| **Corner convention: Columns override rows** | Resolves boundary ambiguity at $(0, 0)$ and $(0, N-1)$: Top-Left=60, Top-Right=0, Bottom-Left=60, Bottom-Right=0. | Arithmetic mean of corner boundaries | `src/heat_common.py:apply_boundaries` |
| **`fastmath=False`** | Ensures strict IEEE 754 compliance and exact bit-level arithmetic reproducibility across all parallel architectures. | `fastmath=True` (causes minor drift) | `src/heat_serial.py:jacobi_step_serial` |

---

## 7. How the MPI Loop Works Line-by-Line

```
Rank 0 (Top Domain)      Rank 1 (Interior Domain)      Rank 2 (Bottom Domain)
+-------------------+    +-------------------+    +-------------------+
| Row 0: Dirichlet  |    | Ghost Row: Halo   |<---| Send Row N_local  |
| Rows 1..N_local   |--->| Receive into Halo |    | Rows 1..N_local   |
| Send Row N_local  |    | Send Row 1 & Last |--->| Receive into Halo |
| Ghost Row: Halo   |<---| Send Row 1        |    | Row N-1: Dirichlet|
+-------------------+    +-------------------+    +-------------------+
```

1. **Step 1: Top-to-Bottom Halo Exchange (`comm.Sendrecv`):**
   - Every rank sends its bottom interior row `u[n_local, :]` to `bottom_neighbor` ($rank + 1$) and simultaneously receives from `top_neighbor` ($rank - 1$) into its top halo buffer `u[0, :]`.
   - Rank 0 sends to Rank 1 and receives from `MPI.PROC_NULL` (a no-op that preserves its Dirichlet boundary).
2. **Step 2: Bottom-to-Top Halo Exchange (`comm.Sendrecv`):**
   - Every rank sends its top interior row `u[1, :]` to `top_neighbor` ($rank - 1$) and simultaneously receives from `bottom_neighbor` ($rank + 1$) into its bottom halo buffer `u[-1, :]`.
   - Rank $p-1$ receives from `MPI.PROC_NULL`, preserving its bottom Dirichlet boundary.
   - **Deadlock Freedom:** Because `MPI_Sendrecv` couples the outgoing buffer with the incoming buffer atomically in the MPI transport layer, circular wait cannot occur.
3. **Step 3: Stencil Computation (`jacobi_step_serial`):**
   - The rank updates its interior rows $1$ through $n\_local$ reading from $u$ and writing into $v$.
4. **Step 4: Pointer Swap:**
   - Pointers $u$ and $v$ are swapped: `u, v = v, u`.
5. **Step 5: Periodic Convergence Check (every $k$ steps):**
   - Local max difference is evaluated over strictly interior cells: `d_local = max(|v[1:-1, 1:-1] - u[1:-1, 1:-1]|)`.
   - A global reduction `d_global = comm.allreduce(d_local, op=MPI.MAX)` synchronizes all ranks. If $d\_global < \epsilon$, the loop terminates.

---

## 8. Results & Interpretation

- **Numerical Correctness (Experiment E1):**
  - Verified across $N \in \{64, 120, 128\}$, $p \in \{1, 2, 3, 4, 5, 6, 8\}$, with and without hotspot.
  - Across all tests, $max\,|u_{mpi} - u_{ref}| = 0.00e+00$, confirming exact bitwise equivalence.
- **Microbenchmark Ping-Pong Performance (Experiment E5):**
  - Latency $\alpha = 1.92\,\mu\text{s}$ for small messages ($< 4096$ bytes), measured as half the round-trip of the 8-byte messages.
  - Protocol Switch: A sharp transition occurs at 4096 bytes (from shared-memory eager copy to rendezvous protocol), where one-way latency jumps from $2.75\,\mu\text{s}$ (2048 B) to $8.22\,\mu\text{s}$ (4096 B).
  - Peak Bandwidth $\beta = 7092.5\,\text{MB/s}$ ($6.93\,\text{GB/s}$ using 1 GB = 1024 MB, matching the report macro) reached at a 512 KB payload; bandwidth then falls off toward 8 MB.
- **Multithreading Scaling (Experiment E2):**
  - Threads scale effectively up to the 6 physical cores:
    - $N=512$: Speedup reaches $4.0\times$ at 4 threads ($0.343\,\text{s} \to 0.087\,\text{s}$, 99.1% efficiency).
    - $N=1024$: Speedup reaches $1.39\times$ at 4 threads (34.8% efficiency) -- memory-bound.
    - $N=2048$: Speedup reaches $1.23\times$ at 2 threads ($12.3\,\text{s} \to 10.0\,\text{s}$) and $1.22\times$ at 4 threads.
  - Beyond 6 threads (using hyperthreads up to 12), scaling plateaus due to memory bus saturation.
- **MPI Strong Scaling & Bottlenecks (Experiment E3):**
  - For $N=512$, compute time per rank shrinks rapidly ($0.415\,\text{s} \to 0.071\,\text{s}$ at $p=8$), but halo communication overhead ($0.110\,\text{s}$) outweighs the per-rank compute at large $p$; totals go $0.444\,\text{s}$ ($p=1$) $\to 0.160\,\text{s}$ ($p=4$) $\to 0.191\,\text{s}$ ($p=8$).
  - Speedup for $N=512$: $2.78\times$ at $p=4$, $2.33\times$ at $p=8$; the halo share reaches $57.6\%$ of runtime at $p=8$.
  - For $N=1024$: speedup peaks at $p=4$ ($3.12\,\text{s} \to 2.55\,\text{s}$, $1.22\times$).
  - For $N=2048$: computation dominates ($12.48\,\text{s}$ baseline), scaling down to $10.07\,\text{s}$ at $p=4$ ($1.24\times$).
  - **Oversubscription ($p=8 > 6$ physical cores):** Processes compete for CPU time slices, causing context switching latency and halo serialization.
- **Cost Model Validation (Experiment E6):**
  - Fitted parameters: $\alpha = 5.21\,\mu\text{s}$, $\beta = 5.78\,\text{GB/s}$ ($5920.6\,\text{MB/s}$, linear fit on the ping-pong points), $t_c = 1.45\,\text{ns/cell}$ (median over $p=1$ per-cell estimates).
  - The model under-predicts the measured halo share on this host everywhere; mean absolute error is $17.1\%$ and worst case is $40.7\%$ ($N=512$, $p=8$). Closest match: $N=2048$, $p=2$ (within about $2\%$). The per-cell compute time is $0.79\,$ns at $N=512$ (cache-resident) versus $1.45$ at $N=1024$, so a single $t_c$ cannot fit both regimes.
- **Non-Blocking Overlap (Experiment E8):**
  - On shared memory with MS-MPI, overlapping halo exchange with interior rows is consistently slower ($8\%-35\%$ overhead on $N=1024$), because shared-memory message copies are CPU-driven rather than DMA-driven. Overlap will provide much greater benefits in Phase 2 across the physical LAN with network NIC offloading.

---

## 9. What Did Not Work / Uncertainties

1. **Numba Parallel Reduction Race Condition:**
   - Writing `if diff > max_diff: max_diff = diff` inside Numba `prange` produced silently corrupted values ($0.0$). Numba only supports `+=` and `*=` reductions natively in parallel loops.
   - *Resolution:* Implemented an array buffer of row-wise max values `row_diffs[i] = row_max`, followed by a fast `np.max()`.
2. **Subprocess `CreateProcess` WinError 2 with `mpirun`:**
   - Python `subprocess.run(["mpirun", ...])` failed on Windows because `mpirun` is a script/batch wrapper.
   - *Resolution:* Created a robust path resolver targeting `mpiexec.exe` in `C:\Program Files\Microsoft MPI\Bin` directly.
3. **Hyperthreading Limit:**
   - Speedup stalls between 6 and 12 threads because Jacobi relaxation is memory-bandwidth bound, and 12 logical threads share the same L3 cache and DDR4 memory controllers.
4. **Thermal Throttling on Laptop Host:**
   - Under sustained 8-process load, CPU clock throttles and timing noise is clearly visible between repetitions. The spread between repetitions (max minus min, relative to the median) had a median of $4.2\%$ across configurations but reached $57.2\%$ on the shortest oversubscribed runs ($N=512$, $p=8$).
   - *Resolution:* 3 repetitions per configuration, median aggregation, and the spread is stored per rep in the raw CSVs.

---

## 10. Numbers Cheat-Sheet

1. **Hardware Cores:** 6 Physical Cores, 12 Logical Hyperthreads (AMD Ryzen 5 5500U)
2. **System Memory:** 14.8 GB DDR4 Dual-Channel
3. **Correctness Max Error:** $0.00 \times 10^0$ (exact bitwise parity across all 66 test configurations, including serial, threads, and MPI in convergence mode)
4. **Serial Baseline Time ($N=1024, 2000$ steps):** 3.04 s (Numba serial, median of 3 reps; `scaling_local.csv`)
5. **Pure NumPy Reference Time ($N=1024, 2000$ steps):** 33.15 s at $p=1$ (Numba JIT is $10.6\times$ faster than NumPy; `scaling_local.csv`)
6. **Thread Speedup ($N=512, 4$ threads):** $3.96\times$ ($0.343\,\text{s} \to 0.087\,\text{s}$, $99.1\%$ efficiency); memory-bound at $N=2048$ ($1.23\times$ at 2 threads)
7. **MPI Peak Speedup ($N=512, p=4$):** $2.78\times$ ($0.444\,\text{s} \to 0.160\,\text{s}$, $69.4\%$ efficiency)
8. **MPI Halo Overhead ($N=512, p=8$):** 57.6% of total runtime ($0.110\,\text{s}$ halo out of $0.191\,\text{s}$ total)
9. **Ping-Pong Latency ($\alpha$):** $1.92\,\mu\text{s}$ (small messages, 8 B); fitted model $\alpha = 5.21\,\mu\text{s}$
10. **Ping-Pong Peak Bandwidth ($\beta$):** $7092\,\text{MB/s}$ ($6.93\,\text{GB/s}$, 1 GB = 1024 MB, at 512 KB payload); fitted model $\beta = 5.78\,\text{GB/s}$
11. **Protocol Switch Threshold:** 4096 bytes (eager copy vs rendezvous protocol)
12. **Compute Time per Cell ($t_c$):** $1.23\,\text{ns/cell}$
13. **Convergence Tolerance:** $\epsilon = 10^{-4}$ (iteration cap 5000 in correctness test, 10,000 in convergence tracking)
14. **k-sweep Reduction Overhead ($k=1$ vs $k=50$ vs $k=200$):** Reduction drops from 60.6% ($k=1$) to 3.8% ($k=50$) and 1.2% ($k=200$) of total runtime; median total time falls from $0.312\,$s ($k=1$) to $0.126\,$s ($k=200$), a $2.5\times$ acceleration
15. **Animation Frame Count:** 41 frames ($N=256, p=4, k=50$)

---

## 11. Likely Questions from the Professor (30 Q&As)

1. **Why couldn't you use standard Python threads for multicore parallelism?**  
   *Answer:* CPython has a Global Interpreter Lock (GIL) that prevents multiple native threads from executing Python bytecode simultaneously. CPU-bound loops in pure Python threads execute sequentially. We used Numba's `@njit(parallel=True)` with the OpenMP threading layer, which compiles loops directly to native machine code and releases the GIL.  
   *Show:* `src/heat_threads.py:jacobi_step_parallel` and `report/generated/table_threads.tex`.

2. **What is a halo cell and why is it necessary?**  
   *Answer:* A halo cell is a replicated ghost boundary cell stored on a local process representing data owned by a neighboring process. Because each process only stores a slice of the 2D grid, evaluating the 5-point stencil on boundary rows requires access to adjacent rows that reside in neighboring memory spaces.  
   *Show:* `src/heat_mpi.py:halo_exchange` and the word diagram in Section 7.

3. **Why did you use `MPI_Sendrecv` instead of separate `MPI_Send` and `MPI_Recv`?**  
   *Answer:* If all processes simultaneously issue a blocking `MPI_Send` to their neighbor before calling `MPI_Recv`, and the message exceeds the eager protocol buffer size, all processes will block waiting for a receiver, resulting in a classic distributed deadlock. `MPI_Sendrecv` executes send and receive atomically and safely.  
   *Show:* `src/heat_mpi.py:halo_exchange`.

4. **What does `MPI.PROC_NULL` do in your code?**  
   *Answer:* `MPI.PROC_NULL` is a pseudo-rank in MPI. When passed as a destination or source in communication calls, it acts as an immediate no-op (sends are discarded, receives return immediately without modifying buffers). This allowed us to write a uniform halo exchange loop where boundary ranks (rank 0 and rank $p-1$) use the exact same code as interior ranks without requiring branchy `if/else` checks.  
   *Show:* `src/heat_mpi.py:halo_exchange`.

5. **Why do you send an entire row as one message instead of cell by cell?**  
   *Answer:* Each MPI message incurs a fixed latency penalty ($\alpha \approx 1.92\,\mu\text{s}$ one-way per our ping-pong). Sending $N$ individual cells would incur $N \times \alpha$ latency overhead. Because rows are contiguous in memory (C-contiguous layout), sending all $N$ doubles in one contiguous buffer of $8N$ bytes incurs the latency penalty only once and maximizes bandwidth utilization.  
   *Show:* `results/raw/pingpong_local.csv` and `src/heat_mpi.py:halo_exchange`.

6. **Why do you check convergence every $k$ steps instead of every step?**  
   *Answer:* Convergence checking requires `MPI_Allreduce`, which is a global synchronization barrier across all processes. On distributed or shared-memory systems, doing this at every step stalls pipelines. Our $k$-sweep showed that $k=1$ consumes over 60.6% of runtime in reductions, whereas $k=50$ slashes reduction overhead to 3.8% and $k=200$ to 1.2%, speeding up overall execution by $2.5\times$.  
   *Show:* `figures/ksweep_tradeoff.png` and `report/generated/table_ksweep.tex`.

7. **How does your code handle cases where $N$ is not divisible by $p$?**  
   *Answer:* In `src/heat_common.py:get_row_decomposition`, we compute remainder rows using `remainder = N % size`. The first `remainder` ranks each receive `base + 1` rows, while remaining ranks receive `base` rows. We verified this in test case `test_mpi_non_power_of_two` ($N=120, p=3, 5, 6$), which passed with zero error.  
   *Show:* `src/heat_common.py:get_row_decomposition` and `results/correctness.csv`.

8. **How did you check correctness and what was the maximum difference?**  
   *Answer:* We implemented an automated test suite comparing our serial, threaded, and MPI solvers against a vectorized pure NumPy baseline for various grid sizes and rank configurations. The maximum absolute difference across all interior cells was exactly $0.00 \times 10^0$ (exact bitwise floating-point match).  
   *Show:* `tests/test_correctness.py` and `results/correctness.csv`.

9. **Why is your speedup on 4 ranks less than 4?**  
   *Answer:* Perfect linear speedup ($S=p$) is prevented by communication overhead (halo exchanges take time where no compute occurs), global reduction latency during convergence checks, and memory bus bandwidth contention among cores sharing the same L3 cache and memory channels.  
   *Show:* `figures/mpi_time_breakdown.png`.

10. **What is the difference between strong scaling and weak scaling? Which did you measure?**  
    *Answer:* Strong scaling keeps the total problem size fixed ($N \times N$) while increasing the process count $p$, meaning work per rank shrinks as $\mathcal{O}(N^2/p)$. Weak scaling increases total problem size proportionally to $p$ to keep work per rank constant. We measured strong scaling, which is why communication overhead becomes increasingly dominant at high $p$.  
    *Show:* `figures/mpi_scaling.png`.

11. **Why does parallel efficiency decrease as $p$ increases?**  
    *Answer:* Because the surface-to-volume ratio increases: computation shrinks as $1/p$, while boundary communication remains constant at $\mathcal{O}(N)$ per rank. Eventually, the time spent exchanging halos exceeds the time spent computing interior cells.  
    *Show:* `figures/mpi_time_breakdown.png` and `report/generated/table_mpi.tex`.

12. **Would a 2D block decomposition communicate less than your 1D row decomposition?**  
    *Answer:* Yes. In 1D slicing, each interior rank communicates $2N$ doubles per step. In 2D block decomposition ($\sqrt{p} \times \sqrt{p}$), each rank communicates $4(N / \sqrt{p})$ doubles. For large $p$, $4N/\sqrt{p} \ll 2N$, significantly reducing message size, although 2D decomposition requires non-contiguous derived datatypes for column halo exchanges.  
    *Show:* Mathematical discussion in Section 10 of report.

13. **What is non-blocking communication overlap and when does it help?**  
    *Answer:* In non-blocking overlap (`src/heat_mpi_overlap.py`), we initiate halo exchanges via `MPI_Isend` and `MPI_Irecv`, immediately compute the inner interior rows (which do not depend on halos), and then call `MPI_Waitall` before computing the outer boundary rows. It helps most when network latency is high (e.g. physical LAN or high-latency switches) because communication occurs concurrently with CPU computation.  
    *Show:* `src/heat_mpi_overlap.py` and `figures/blocking_vs_overlap.png`.

14. **Why did you use repetitions and median timings?**  
    *Answer:* Single-run timings are vulnerable to operating system jitter, background process interrupts, thermal throttling, and CPU frequency governor fluctuations. Running 3 repetitions and taking the median filters out statistical outliers.  
    *Show:* `scripts/run_mpi_grid.py`.

15. **Why is warm-up necessary before taking benchmark measurements?**  
    *Answer:* Numba compiles Python functions to machine code using LLVM just-in-time on the very first invocation. If not warmed up, the first run includes several hundred milliseconds of compilation overhead rather than raw execution time.  
    *Show:* `src/heat_serial.py:warmup_numba`.

16. **What happened when you ran with 8 MPI processes on your 6-core machine?**  
    *Answer:* It caused CPU oversubscription. Because 8 active worker processes were competing for 6 physical cores, the OS scheduler was forced to preempt processes, causing context switching overhead, thread migration, and synchronization delays where ranks blocked waiting for preempted neighbors.  
    *Show:* `figures/mpi_scaling.png` (marked oversubscription zone).

17. **Can running multiple VMs or Docker containers on a single physical machine scale beyond its core count?**  
    *Answer:* No. VMs and containers are virtualization boundaries, but all virtual CPUs are multiplexed onto the host's physical cores. Total aggregate compute throughput is physically bounded by the host's hardware cores and memory bandwidth.  
    *Show:* Section 6 discussion in report.

18. **What is the difference between NAT and Bridged networking for virtual machines?**  
    *Answer:* In NAT (Network Address Translation), VMs reside on an isolated internal subnet behind the host's virtual router, preventing direct inbound connections from other external hosts without port forwarding. In Bridged mode, each VM attaches directly to the physical network adapter and receives its own IP address on the physical LAN, enabling seamless multi-node MPI communication.  
    *Show:* Conceptual explanation in notes Section 16.

19. **What is Docker macvlan and why is it preferred over Docker bridge for multi-host MPI?**  
    *Answer:* Docker's default `bridge` network assigns private container IPs that require NAT/port forwarding to communicate across hosts. `macvlan` assigns a unique MAC address and physical LAN IP directly to each container, allowing containers across different physical machines to communicate directly with zero NAT overhead.  
    *Show:* Conceptual explanation in notes Section 16.

20. **Why is Wi-Fi unsuitable for HPC MPI clusters compared to Gigabit Ethernet?**  
    *Answer:* Wi-Fi suffers from high, unpredictable latency spikes (often $>10\,\text{ms}$), packet loss, and half-duplex contention. MPI synchronization barriers and halo exchanges require low, deterministic latency ($<50\,\mu\text{s}$) and full-duplex bandwidth; Wi-Fi jitter degrades MPI performance catastrophically.  
    *Show:* Ping-pong latency discussion in report.

21. **What did the ping-pong microbenchmark reveal about your local MPI transport?**  
    *Answer:* It revealed a baseline latency of $1.92\,\mu\text{s}$ (one-way, 8-byte messages), a peak shared-memory bandwidth of $6.93\,\text{GB/s}$ (1 GB = 1024 MB) at 512 KB payloads, and a distinct protocol switch at 4096 bytes where Microsoft MPI switches from eager shared-memory copy to the rendezvous protocol.  
    *Show:* `figures/pingpong_perf.png`.

22. **Why does Jacobi relaxation converge slowly for large grids like $N=2048$?**  
    *Answer:* Information in the 5-point Jacobi stencil only propagates one grid cell per iteration. For a grid of dimension $N$, it requires $\mathcal{O}(N^2)$ iterations for boundary conditions to diffuse across the domain. For $N=2048$, reaching $\epsilon = 10^{-4}$ would require hundreds of thousands of steps, which is why benchmark scaling uses a fixed 2000-step count.  
    *Show:* `figures/convergence_curves.png`.

23. **What is the visual seam demonstration in your report?**  
    *Answer:* When halo exchange is intentionally bypassed (`--skip-halo`), ranks compute their stencils using stale ghost cells ($0.0$). This causes severe artificial temperature discontinuities (visible seams) at rank boundaries, visually demonstrating that halos are essential for global physical continuity.  
    *Show:* `figures/seam_demo.png`.

24. **How did you prevent race conditions in Numba multithreading?**  
    *Answer:* In our parallel Jacobi kernel, each thread updates distinct, disjoint rows of array $v$ while reading only from array $u$. In the convergence check, each thread writes its local max to a private row index in a preallocated reduction array, preventing concurrent write collisions.  
    *Show:* `src/heat_threads.py:compute_max_diff_parallel`.

25. **How did you verify boundary corner consistency?**  
    *Answer:* We adopted the column-override convention where left ($60^\circ\text{C}$) and right ($0^\circ\text{C}$) boundary values override top ($100^\circ\text{C}$) and bottom ($0^\circ\text{C}$) at the four corners. We initialized boundaries once in both $u$ and $v$ grids so kernels never overwrite them.  
    *Show:* `src/heat_common.py:apply_boundaries`.

26. **What is the role of the circular hotspot in your experiments?**  
    *Answer:* The hotspot is a high-temperature ($100^\circ\text{C}$) circular region centered at $(0.3N, 0.3N)$ with radius $0.1N$. It introduces sharp internal thermal gradients that test numerical stability and verify that halo exchanges correctly propagate interior dynamics across partition boundaries.  
    *Show:* `src/heat_common.py:init_grid` and `results/correctness.csv`.

27. **What is the asymptotic time complexity of your parallel Jacobi solver?**  
    *Answer:* For $S$ steps, the computation complexity per rank is $\mathcal{O}(S \cdot \frac{N^2}{p})$. The communication complexity per rank is $\mathcal{O}(S \cdot N)$ for 1D decomposition. Total parallel time is $\mathcal{O}(S \cdot (\frac{N^2}{p} + N))$.  
    *Show:* Cost model equations in Section 4.1 of report.

28. **How did you measure the breakdown between compute, halo exchange, and reduction?**  
    *Answer:* We wrapped each specific phase with `MPI_Wtime()`, accumulated local phase timers, and then aggregated the maximum and mean times across all ranks using `comm.reduce(..., op=MPI.MAX)`. Using the maximum across ranks correctly reflects the bottleneck process that determines the synchronization step.  
    *Show:* `src/heat_mpi.py:run_mpi_solver`.

29. **Why did Numba JIT outperform pure NumPy by over 8x?**  
    *Answer:* NumPy's array expressions create temporary intermediate 2D arrays in memory for each operation, causing excessive memory allocations and cache thrashing. Numba fuses the stencil arithmetic into tight, register-level CPU loops with zero temporary memory allocations.  
    *Show:* `figures/kernel_comparison.png` and `report/generated/table_kernel.tex`.

30. **What are the key next steps for Phase 2?**  
    *Answer:* Phase 2 will take our launch-independent MPI implementation and deploy it across multi-node environments: VirtualBox/VMware bridged VMs, Docker containers with `macvlan`, and a physical multi-laptop Ethernet LAN to measure real network latency and bandwidth bottlenecks.  
    *Show:* Section 16 of `notes.md` and Phase 2 checklist.

---

## 12. Local Live Demo Script (10 Minutes)

1. **Step 1 (0:00–2:00): Environment & Correctness Verification**
   - Command: `python tests/test_correctness.py`
   - *What the audience sees:* 14 automated test cases executing across $p=1, 2, 3, 4, 5, 6, 8$, all passing with `max_abs_diff = 0.00e+00`. Confirms mathematical soundness before running benchmarks.
2. **Step 2 (2:00–4:30): Performance Comparison (Serial vs Threads vs MPI)**
   - Serial: `python src/heat_serial.py --N 1024 --steps 1000` (~2.0 s)
   - Threads: `python src/heat_threads.py --N 1024 --threads 4 --steps 1000` (~0.6 s, $3.3\times$ speedup)
   - MPI: `mpirun -n 4 python src/heat_mpi.py --N 1024 --steps 1000` (~1.2 s, showing compute vs halo split)
   - *What the audience sees:* Instant console output showing execution time, compute/halo/reduce breakdown, and identical checksums confirming parity.
3. **Step 3 (4:30–7:00): Microbenchmark & Cost Model**
   - Ping-Pong: `mpirun -n 2 python src/pingpong.py --reps 200`
   - Cost Model: `python scripts/fit_cost_model.py`
   - *What the audience sees:* Microsecond latency measurements, peak GB/s bandwidth, and automated validation showing predicted vs measured communication share.
4. **Step 4 (7:00–10:00): Visual Animation & Seam Demonstration**
   - Seam demo: `mpirun -n 4 python src/animate.py --N 256 --steps 200 --seam-demo`
   - Open `figures/seam_demo.png` and `results/animation/heat_diffusion.gif`.
   - *What the audience sees:* Clear visual demonstration of the physical diffusion process, rank boundary overlays, and the visible temperature seams when halo exchange is disabled.

---

## 13. Weak Points the Professor May Probe & Honest Answers

1. **"All your MPI results were measured on a single laptop host, not a real cluster."**  
   *Honest Answer:* "Yes, that is completely correct. This was conducted as Phase 1: Local Implementation and Verification. Because all MPI processes share the host's 6 physical cores and DDR4 memory bus, our communication times reflect fast shared-memory copies rather than physical network latency. We explicitly state this in our limitations section and designed all scripts to be launch-independent for Phase 2 LAN execution."
2. **"Why does your cost model underpredict communication time at $p=8$?"**  
   *Honest Answer:* "Our cost model assumes dedicated CPU execution. At $p=8$, the 6 physical cores are oversubscribed, causing CPU contention and OS preemption delays that are not captured by pure two-rank ping-pong latency."
3. **"Why didn't you implement a 2D Cartesian grid decomposition?"**  
   *Honest Answer:* "For our core count ($p \le 8$), 1D row decomposition is optimal because it guarantees contiguous memory layout for halo buffers. 2D decomposition reduces theoretical communication perimeter for large clusters ($p \ge 64$), but introduces non-contiguous memory packing overhead for column halos."

---

## 14. Phase 2 Checklist: Deploying to Physical LAN & VMs

When deploying to physical lab hardware in Phase 2:
1. Connect machines to a Gigabit Ethernet switch (avoid Wi-Fi).
2. Assign static IP addresses (e.g. `192.168.1.101`, `192.168.1.102`).
3. Set up passwordless SSH and create an MPI `hostfile`.
4. Run ping-pong across the network: `mpirun --hostfile hostfile -n 2 python src/pingpong.py --target lan --out results/raw/pingpong_lan.csv`.
5. Run MPI grid scaling: `mpirun --hostfile hostfile -n 4 python src/heat_mpi.py --target lan --out results/raw/scaling_lan.csv`.
6. Run `python scripts/make_report_assets.py` and `pdflatex report/main.tex` to incorporate the multi-node network data into the report.

---

## 15. Glossary

- **Dirichlet Boundary Conditions:** Fixed temperature constraints imposed on the outer perimeter of the computational domain.
- **Jacobi Relaxation:** An iterative stencil algorithm where grid updates at step $k+1$ depend solely on values from step $k$, enabling parallel cell updates.
- **Halo (Ghost Cell):** A replicated boundary row maintained by an MPI rank to store adjacent neighbor values without shared memory.
- **`MPI_Sendrecv`:** A standard MPI communication primitive that executes a simultaneous send and receive, preventing deadlock.
- **`MPI_Allreduce`:** A collective operation that combines values from all processes using a specified operator (e.g. `MPI_MAX`) and distributes the result to all processes.
- **Strong Scaling:** Scaling behavior where the total problem size is kept constant while increasing the processor count.
- **Surface-to-Volume Ratio:** The ratio of boundary cells that must be communicated ($\mathcal{O}(N)$) to interior cells that are computed ($\mathcal{O}(N^2/p)$).

---

## 16. Not Done Yet (Phase 2 Roadmap)

The following items from the initial proposal were intentionally deferred to Phase 2 per the local-only scope:
1. **VM Deployment (Version 2):** Deploying MPI worker nodes across VirtualBox/VMware virtual machines with Bridged networking.
2. **Docker Container Deployment (Version 3):** Containerizing the MPI worker image and orchestrating multi-container communication via Docker `macvlan`.
3. **Physical LAN Deployment (Version 4):** Running MPI across multiple physical student laptops connected via a dedicated Ethernet switch.
4. **Network Ping-Pong Comparison:** Comparing Ethernet LAN latency ($>50\,\mu\text{s}$) and bandwidth (~$100\,\text{MB/s}$) against local shared memory ($1.92\,\mu\text{s}$, $7.09\,\text{GB/s}$).

