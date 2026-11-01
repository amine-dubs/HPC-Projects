# Student Consultation & Defense Notes: Parallel Fractal Zoom Movie

**Project:** Parallel Fractal Zoom Movie in Python: Threads, Virtual Machines, Docker Containers and a Physical LAN  
**Course:** Intensive Computing, USTO-MB  
**Authors:** Bellatreche Mohamed Amine, Rezaiguia Soltane  
**Supervisor:** Prof. Dairi Abdelkader  
**Date:** November 2026  

---

## 1. The Project in Five Lines
- We built a high-performance Python fractal renderer computing a 120-frame Mandelbrot zoom movie ($640 \times 360$).
- Every pixel is mathematically independent, making the workload embarrassingly parallel yet plagued by severe spatial and temporal load imbalance.
- We implemented and verified five parallel variants: Numba multi-threading (rows vs. shuffled tiles) and three MPI strategies (static rows, dynamic master-worker tiles, and whole frames block/cyclic).
- Under this local project phase, we ran complete benchmarks on an AMD Ryzen 5 5500U (6 cores / 12 threads) using Microsoft MPI and OpenMP.
- Distributed executions across bridged VMs, Docker overlays, and physical LAN nodes were deferred to the subsequent phase.

---

## 2. Mathematical Background and the Load Imbalance Problem
1. **The Mandelbrot Sequence:** For each pixel representing complex coordinate $c = c_r + i c_i$, compute $z_{n+1} = z_n^2 + c$ starting from $z_0 = 0$.
2. **Escape Criterion:** If $|z_n| > 2$ (equivalently $z_r^2 + z_i^2 > 4.0$), the sequence is guaranteed to diverge. The pixel colour is determined by the escape iteration $n$.
3. **Interior Points:** If $|z_n| \le 2$ across all iterations up to the threshold $M$, the point is considered inside the Mandelbrot set.
4. **Why Work is Uneven (Spatial Imbalance):** Exterior background pixels escape in 1 to 5 iterations. Filament boundary pixels require hundreds of iterations. Interior cardioid pixels must iterate all the way to $M$. Dividing an image into equal-sized spatial blocks assigns massive work to interior workers while exterior workers idle immediately.
5. **Why Zoom Increases Imbalance (Temporal Imbalance):** To maintain visual sharpness as the camera zooms into boundary structures, the iteration limit $M_k$ must increase dynamically (from $M_0 = 200$ to $M_{119} = 1000$). Deep frames are up to 5 times more computationally expensive than initial frames.

---

## 3. Architecture and Implementation Files

| File | Purpose and Role |
| :--- | :--- |
| `src/fractal_core.py` | Shared Numba `@njit` kernel, coordinate formulas, config loader, partition helpers, SHA-256 hash. |
| `src/serial.py` | V0 serial reference implementation; generates reference hashes and baseline benchmarks. |
| `src/threads.py` | V1 shared-memory parallel engine using Numba `prange` (Variant A: rows, Variant B: shuffled tiles). |
| `src/mpi_s1_rows.py` | MPI Strategy 1: Static horizontal row block decomposition gathered per frame. |
| `src/mpi_s2_tiles.py` | MPI Strategy 2: Dynamic master-worker tile queue across all frames without inter-frame barriers. |
| `src/mpi_s3_frames.py` | MPI Strategy 3: Embarrassingly parallel whole frames per rank (Block vs. Cyclic). |
| `src/make_movie.py` | Production renderer (E10): computes 120 frames, applies cyclic color palette, outputs PNGs, GIF, and MP4. |
| `src/pingpong.py` | MPI Ping-Pong benchmark measuring local shared-memory latency and bandwidth across buffer sizes. |
| `tests/run_correctness.py` | Validation test suite checking Cartesian orientation, coverage, and bitwise identity against V0. |
| `scripts/env_info.py` | Collects CPU, RAM, OS, MPI, Numba, and compiler metadata into `results/environment.json`. |
| `scripts/mpi_launch.py` | Cross-platform MPI launcher wrapper supporting MS-MPI and Open MPI with oversubscription control. |
| `scripts/run_benchmarks.py` | Automated experimental harness for E2–E5 with random shuffling and resume capability. |
| `scripts/make_plots.py` | Generates publication-ready figures (Figures 1–8) into `figures/`. |
| `scripts/make_tables.py` | Formats LaTeX tables and exports alphabetic number macros into `report/generated/`. |

---

## 4. Parameters Used

- **Resolution:** $640 \times 360$ (16:9 widescreen aspect ratio).
- **Frames:** 120 frames (approx. 5.0 seconds at 24 fps).
- **Zoom Centre:** $c = -0.7436 + 0.1318i$ (Seahorse Valley).
- **Window Width:** $W_0 = 3.0$; shrink factor $= 0.95$ per frame ($W_k = 3.0 \times 0.95^k$).
- **Iteration Limits:** $M_0 = 200 \rightarrow M_{119} = 1000$ (linear ramp: $200 + 800 \times k / 119$).
- **Escape Radius:** $|z|^2 \le 4.0$.
- **Data Types:** Iteration counts as `int32`, coordinates as `float64`.
- **Tile Sizes:** $10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$ pixels.
- **Process / Thread Counts:** $p \in \{1, 2, 4, 8, 12\}$.
- **Repetitions:** 5 repetitions for V0 serial, 3 repetitions for all parallel configurations.

---

## 5. Key Implementation Decisions

| Decision | Justification | Code Location |
| :--- | :--- | :--- |
| **Global-Index Coordinates** | Computing $c_r = x_{min} + i \cdot dx$ from global $(i, j)$ prevents boundary rounding errors across tiles. | `src/fractal_core.py:render_block` |
| **No `fastmath`** | Strict IEEE 754 compliance ensures identical floating-point operations across all ranks and threads. | `src/fractal_core.py:render_block` |
| **Cartesian Orientation** | Row 0 maps to $y_{max}$ so $Im > 0$ is at the top of the image (verified by unit test). | `src/fractal_core.py:frame_params` |
| **One Shared Kernel** | V0, V1, S1, S2, and S3 all execute the exact same JIT function, isolating scheduling effects. | `src/fractal_core.py:render_block` |
| **Linear Iteration Schedule** | Detail increases smoothly as magnification deepens without sudden jumps in computational cost. | `src/fractal_core.py:maxit_for_frame` |
| **Shuffled Tiles in V1-B** | Randomizing tile execution order breaks spatial clustering, balancing OpenMP worker threads. | `src/threads.py:run_threads` |
| **OpenMP Threading Layer** | Set `NUMBA_THREADING_LAYER=omp` to use native OpenMP runtime instead of Workqueue or TBB. | `src/threads.py:line 16` |
| **1 Thread per MPI Rank** | Set `NUMBA_NUM_THREADS=1` and `OMP_NUM_THREADS=1` to ensure fair MPI comparisons without thread contention. | `src/mpi_s*.py:lines 14-15` |
| **Global S2 Task Queue** | One task list covers all frames so workers never idle at inter-frame barriers. | `src/mpi_s2_tiles.py:main` |
| **Edge Tile Clipping** | Clamps $r_1 = \min(r_0 + T, H)$ and $c_1 = \min(c_0 + T, W)$ so arbitrary tile sizes divide cleanly. | `src/fractal_core.py:tile_list` |
| **S3 Cyclic Partitioning** | Interleaving frames ($k \pmod p$) balances the time-varying cost across all ranks. | `src/mpi_s3_frames.py:main` |
| **Decoupled I/O Timings** | File writing, colour conversion, and MP4 encoding are timed separately from core rendering. | `src/make_movie.py:main` |
| **Modulo Colormapping** | Formula $(k \pmod{64})/63.0$ avoids palette shifting or flickering as $M_k$ increases. | `src/make_movie.py:color_map_counts` |
| **SHA-256 Bitwise Audit** | Every frame array is hashed and matched against V0 serial reference arrays. | `src/fractal_core.py:hash_frame` |

---

## 6. What We Ran

1. **E0 (Environment):** `python scripts/env_info.py` $\rightarrow$ `results/environment.json`.
2. **E1 (Correctness):** `python tests/run_correctness.py` $\rightarrow$ `results/correctness.csv`.
3. **E2 (Serial Baseline):** `python src/serial.py --rep 1..5` $\rightarrow$ `results/serial_times.csv`, `results/serial_hashes.csv`.
4. **E3 (Threads Scaling):** `python src/threads.py --variant {A,B} -p {1..12}` $\rightarrow$ `results/threads.csv`.
5. **E4 (MPI Scaling):** `mpiexec -n {1..12} python src/mpi_s*.py` $\rightarrow$ `results/mpi_scaling.csv`, `results/busy_time.csv`.
6. **E5 (Tile Study):** `mpiexec -n 4 python src/mpi_s2_tiles.py --tile {10,20,40,80}` $\rightarrow$ `results/tile_size.csv`.
7. **E6 (Cost Profile):** `python scripts/compute_cost_profiles.py` $\rightarrow$ `results/cost_per_tile.npz`, `results/cost_per_rowblock.csv`.
8. **E7 (Ping-Pong):** `mpiexec -n 2 python src/pingpong.py` $\rightarrow$ `results/pingpong.csv`.
9. **E10 (Production Movie):** `python src/make_movie.py -p 6 --tile 20` $\rightarrow$ `output/zoom.mp4`, `results/production.csv`.

---

## 7. Results Summary by Figure
- **Figure 1 (Scaling & Efficiency):** Shows that V1-B (shuffled tiles) scales better than V1-A (rows). In MPI, S3-cyclic achieves the highest speedup. Beyond $p = 6$, hyperthreading yields diminishing returns.
- **Figure 2 (Busy Time Breakdown at $p=4$):** Exposes severe imbalance in S1 (middle ranks compute 2–3$\times$ longer). S2 shows balanced workers but an idle master. S3-cyclic exhibits near-perfect compute balance.
- **Figure 3 (Tile Size Study for S2):** Demonstrates the communication-granularity trade-off. $10 \times 10$ tiles saturate the master with over 550,000 messages ($18.38$~s). Because our unified task queue covers all 120 frames without inter-frame barriers, $80 \times 80$ tiles provide 4,800 tasks—sufficient to balance 3 workers without tail latency—while minimizing message traffic to 9,603 messages, achieving the optimal execution time of $13.83$~s (with $40 \times 40$ close at $14.13$~s).
- **Figure 4 (Cost Heatmaps & S1 Rows):** Visualizes the concentration of work along boundary filaments and cardioid interior. S1 row blocks 1 and 2 absorb over 70% of total iterations.
- **Figure 5 (Busy Time vs. Iterations):** Scatter points align tightly on $y = x$, proving that iteration count accurately dictates compute time.
- **Figure 6 (S3 Block vs. Cyclic):** Compares S3 block (rank 3 computes expensive deep frames, rank 0 finishes early) against S3 cyclic (uniform compute across all ranks).
- **Figure 7 (Ping-Pong Latency & Bandwidth):** Displays minimum latency of $\sim 2.5~\mu$s and peak bandwidth of $\sim 3.88$~GB/s on local shared memory.
- **Figure 8 (Filmstrip):** Four clean sample frames showing the zoom path into Seahorse Valley.

---

## 8. What Worked, What Did Not, and Uncertainties
- **What Worked:**
  - JIT compilation with Numba `@njit` yielded serial rendering of 120 frames in $\sim 36.5$~s.
  - Bitwise correctness was verified across all implementations with zero differing pixels.
  - Shuffled tiles effectively eliminated OpenMP spatial imbalance.
  - S3 cyclic frame assignment provided linear scaling up to the physical core limit.
- **What Did Not Work / Bugs Encountered:**
  - **S2 Tag Collision Deadlock:** In `mpi_s2_tiles.py`, master polled with `ANY_TAG`, intercepting worker terminal stats (`tag=100`) as tile data. Resolved by restricting the work loop to `TAG_WORK`.
  - **Numba Thread Pool Constraint:** Setting `NUMBA_NUM_THREADS=1` globally prevented `numba.set_num_threads(p)` from creating more than 1 thread. Resolved by keeping thread limits scoped to MPI ranks only.
  - **MS-MPI PATH Missing:** Windows PowerShell did not locate `mpiexec` by default. Resolved by auto-detecting `C:\Program Files\Microsoft MPI\Bin` in `mpi_launch.py`.
- **What Was Not Run (Deferred):**
  - Bridged Virtual Machines (V2), Docker macvlan containers (V3), and physical LAN clustering (V4) were deferred per scope.
- **Uncertainties:**
  - Single-socket memory bus contention when 12 ranks simultaneously access shared L3 cache.
  - Exact OpenMP thread scheduling mechanics inside Numba's runtime (unobservable without intrusive C-level hooks).

---

## 9. Likely Questions from the Professor (with Prepared Answers)

### Mandelbrot Algorithm & Maths
1. **Q: Why does the Mandelbrot calculation cause load imbalance?**  
   *A:* Pixels outside the set escape after 1 to 5 iterations. Pixels on boundary filaments or inside the set iterate up to the maximum limit (200 to 1000). A uniform spatial partition therefore assigns drastically unequal workloads to workers.
2. **Q: Why does iteration limit grow with frame index?**  
   *A:* As the view magnifies by $0.95^k$ (up to $500\times$ zoom at frame 119), boundary filaments become infinitesimally thin. Lower iteration limits cause deep boundaries to blur into black blobs. Higher limits resolve fine details.
3. **Q: Why does $|z|^2 > 4.0$ guarantee divergence?**  
   *A:* By triangle inequality, if $|z| > 2$ and $|c| \le 2$, then $|z^2 + c| \ge |z|^2 - |c| > 2|z| - 2 = |z| + (|z| - 2) > |z|$. Thus $|z_n|$ grows strictly monotonically toward infinity.
4. **Q: Why are pixel coordinates derived from global indices instead of tile bounds?**  
   *A:* In floating-point arithmetic, $(x_1 - x_0) \cdot i / w$ evaluated on sub-tiles can differ by 1 unit in the last place (ULP) compared to full-frame coordinates. Using global indices $(i, j)$ ensures deterministic, bitwise identical values across all partition strategies.

### Python, Numba, and Shared Memory
5. **Q: How does Numba bypass Python's Global Interpreter Lock (GIL)?**  
   *A:* When Numba compiles a function with `@njit(parallel=True)`, the compiled native machine code runs outside the CPython runtime, releasing the GIL during `prange` loop execution.
6. **Q: Why did Variant B (shuffled tiles) outperform Variant A (rows)?**  
   *A:* Variant A splits horizontal rows. Because the Mandelbrot set is centered near the horizontal axis, middle threads receive all expensive interior rows while outer threads finish almost immediately. Shuffling tiles scatters expensive and cheap blocks across all threads.
7. **Q: What threading layer was used by Numba on this machine?**  
   *A:* OpenMP (`omp`), verified dynamically by querying `numba.threading_layer()`.
8. **Q: Can we measure per-thread idle time inside `prange`?**  
   *A:* No. Numba manages thread dispatch internally through OpenMP work-sharing constructs without exposing per-thread execution timestamps to Python. We quantify imbalance using iteration sums per tile.
9. **Q: Why did we set `cache=False` when testing inline scripts?**  
   *A:* Numba's function cache requires a tangible source code file on disk to locate functions. Passing scripts via `python -c` causes `enable_caching()` to raise a runtime error.

### MPI Strategies and Architecture
10. **Q: Why is S1 (static rows) the slowest MPI strategy?**  
    *A:* S1 assigns fixed row blocks to ranks. Ranks computing outer rows sit idle waiting at the per-frame `gather` barrier while middle ranks compute the dense cardioid interior.
11. **Q: Why does S2 have no $p = 1$ measurement?**  
    *A:* S2 requires a dedicated master rank to distribute tiles and assemble frames, plus at least one worker rank to perform computation ($p \ge 2$). At $p = 2$, only 1 worker computes.
12. **Q: Why is S2 efficiency lower at $p = 2$?**  
    *A:* At $p = 2$, half the allocated processes (rank 0) act purely as a dispatcher. The computational resource is effectively 1 core, giving a theoretical maximum speedup of $1.0\times$ and efficiency of 50\%.
13. **Q: At what tile size does the S2 master become a bottleneck?**  
    *A:* At $10 \times 10$ pixels, the master must process tens of thousands of messages. The master's CPU becomes saturated receiving and replying to requests, causing worker starvation.
14. **Q: Why does S3 cyclic frame assignment beat block assignment?**  
    *A:* Because iteration limits grow from 200 to 1000, frame costs increase with frame index. Block assignment gives the cheapest frames to rank 0 and the most expensive frames to the last rank. Cyclic assignment interleaves cheap and expensive frames across all ranks.
15. **Q: Does S3 have any communication overhead during rendering?**  
    *A:* None. Each rank renders its assigned frames entirely in local memory. Communication occurs only once at the end when assembled arrays are sent to rank 0.
16. **Q: Why did we enforce 1 thread per MPI rank?**  
    *A:* If each MPI rank spawned 6 or 12 threads, $p = 4$ would instantiate up to 48 threads, severely overloading CPU cores and invalidating scaling comparisons against single-threaded V0.

### Performance, Amdahl, and Hardware
17. **Q: What is the Amdahl speedup limit in this problem?**  
    *A:* In pure rendering, the parallel fraction is near 99.9\%. However, when including serial I/O, colour mapping, and video compression (approx. 4.5 seconds), Amdahl's law dictates that speedup asymptotically caps around $T_{total} / T_{io}$.
18. **Q: Is speedup beyond $p = 6$ real?**  
    *A:* It is partially real, but limited. The AMD Ryzen 5 5500U has 6 physical cores and 12 logical threads via Simultaneous Multithreading (SMT). SMT shares integer and floating-point execution pipelines, so speedup between 6 and 12 scales by 20–30\% rather than doubling.
19. **Q: Why do we report median instead of mean?**  
    *A:* Median is robust to transient operating system background interruptions (antivirus scans, context switches, thermal throttling) that distort the arithmetic mean.
20. **Q: Why did ping-pong bandwidth plateau around 3.88 GB/s?**  
    *A:* Because it ran on a single machine, transfers pass through host DRAM and CPU L3 cache via shared memory pipes, rather than network interface cards.

### Networking, Virtualization & Cluster Theory (Conceptual / Untested)
21. **Q: Why can't NAT-mode VMs communicate across a physical cluster?**  
    *A:* NAT isolates each VM behind an internal host NAT router with private RFC 1918 addresses not routable from external physical hosts. Bridged networking attaches VMs directly to the physical layer-2 broadcast domain.
22. **Q: What is Docker macvlan networking?**  
    *A:* Macvlan assigns a distinct physical MAC address to each container interface, allowing containers to appear as independent physical nodes directly on the LAN subnet.
23. **Q: Why can't a Linux host communicate with its own macvlan container by default?**  
    *A:* The Linux kernel drops packets routed between a parent physical interface and its child macvlan sub-interfaces for security. A dedicated macvlan shim bridge on the host is required to bypass this restriction.
24. **Q: What switch requirement is critical for macvlan and bridged VMs?**  
    *A:* The physical switch port must support multiple MAC addresses per port (promiscuous mode / MAC learning). If port security is enabled, it will drop traffic or disable the port.
25. **Q: Why does Open MPI require `--oversubscribe`?**  
    *A:* Open MPI enforces strict core-binding slots. When launching more processes than detected physical slots, Open MPI aborts unless explicitly permitted via `--oversubscribe`.

### Methodology, Limitations & Consultation Defense
26. **Q: Did you run benchmarks in Docker or VMs?**  
    *A:* No. Per project scope, Docker, VMs, and physical LAN clusters were marked as deferred to the subsequent phase. All data in our report is measured natively on the local machine.
27. **Q: Did you invent any benchmark numbers?**  
    *A:* Absolutely not. Every table cell, macro, and plot point is parsed directly from saved CSV and JSON files in `results/`.
28. **Q: What would you change if you started over?**  
    *A:* Implement non-blocking MPI communication (`Isend`/`Irecv`) in S2 and S3, and test a dynamic frame-level work queue for S3 to eliminate any residual frame-cost imbalance.
29. **Q: Why did you decouple rendering time from I/O time?**  
    *A:* Rendering represents algorithmic CPU parallelization, whereas I/O depends on disk bandwidth and video compression codecs. Mixing them obscures parallel scaling analysis.
30. **Q: How do you prove that your movie zoom is mathematically correct?**  
    *A:* Every frame's pixel array matches the serial reference array bit-for-bit, verified by identical SHA-256 digests in `results/serial_hashes.csv`.

---

## 10. Live Demo Plan (Local Execution)
*Total Time: 10 minutes*

1. **Minute 1: Environment & Sanity Check**
   - Command: `python scripts/env_info.py`
   - Show detected CPU (6 physical / 12 logical cores) and OpenMP / MS-MPI status.
2. **Minute 2–3: Correctness Suite**
   - Command: `python tests/run_correctness.py`
   - Show unit tests and 100% bitwise identity match across all parallel versions.
3. **Minute 4–6: Live Parallel Speedup Demonstration**
   - Run V0 Serial baseline on 10 frames: `python src/serial.py`
   - Run V1-B Threads on 10 frames: `python src/threads.py --variant B -p 6`
   - Run MPI S3 Cyclic: `mpiexec -n 6 python src/mpi_s3_frames.py --assignment cyclic`
   - Point out the speedup in console output.
4. **Minute 7–8: Master-Worker Tile Demonstration (S2)**
   - Run S2 with tile size 20: `mpiexec -n 4 python src/mpi_s2_tiles.py --tile 20`
   - Show dynamic load balancing output.
5. **Minute 9–10: Final Zoom Video Display**
   - Open and play `output/zoom.mp4` and `output/zoom.gif`.
   - *Note on Next Phase:* State clearly that multi-node execution across bridged VMs, Docker overlays, and physical lab switches will be added in the next phase.
6. **Fallback Plan:** If any live command fails, display pre-computed figures in `figures/` and raw CSV logs in `results/`.

---

## 11. Live Interactive Modifications (What the Professor May Ask to Change)

1. **Change Zoom Centre:**
   - Modify `"zoom_center": [-0.7436, 0.1318]` in `config.json` to Seahorse tail `[-0.75, 0.1]`.
   - Run: `python src/make_movie.py -p 6`
2. **Change Tile Size in S2 Live:**
   - Run: `mpiexec -n 4 python src/mpi_s2_tiles.py --tile 80`
   - Observe the difference in rendering time and worker idle wait.
3. **Change Process Count Live:**
   - Run with 2 cores: `mpiexec -n 2 python src/mpi_s3_frames.py --assignment cyclic`
   - Run with 6 cores: `mpiexec -n 6 python src/mpi_s3_frames.py --assignment cyclic`
4. **Compare S3 Block vs. Cyclic Live:**
   - Block: `mpiexec -n 4 python src/mpi_s3_frames.py --assignment block`
   - Cyclic: `mpiexec -n 4 python src/mpi_s3_frames.py --assignment cyclic`
   - Observe the immediate execution time difference on the console.

---

## 12. Numbers Cheat-Sheet (Key Metrics to Memorize)

- **V0 Serial Baseline Time:** $40.41$~seconds ($2.97$~FPS) (`results/serial_times.csv`).
- **Best Threads Speedup (V1-B):** $8.37\times$ at $p = 12$ ($T_p = 4.83$~s vs. $5.39$~s for Variant A) (`results/threads.csv`).
- **Best MPI Strategy:** S3 Cyclic (`results/mpi_scaling.csv`).
- **Best MPI Speedup:** $10.20\times$ at $p = 12$ ($T_p = 3.96$~s, $30.30$~FPS) (`results/mpi_scaling.csv`).
- **S3 Block vs Cyclic Imbalance:** Block $\lambda \approx 1.48$ vs. Cyclic $\lambda \approx 1.01$ (`results/busy_time.csv`).
- **S2 Master-Worker Optimal Tile Size:** $80 \times 80$ pixels ($T_p = 13.83$~s at $p = 4$, with $40 \times 40$ close at $14.13$~s) (`results/tile_size.csv`).
- **S2 Master Worker Imbalance:** Workers achieve near-perfect balance ($\lambda \approx 1.03$), but $p=2$ has only 1 compute worker ($T_p \approx 44.4$~s) (`results/busy_time.csv`).
- **Ping-Pong Shared-Memory Latency:** $2.24~\mu$s minimal one-way latency (`results/pingpong.csv`).
- **Ping-Pong Peak Bandwidth:** $3882.47$~MB/s ($\approx 3.88$~GB/s) for 8 MiB buffers (`results/pingpong.csv`).
- **Video Production Timings (E10):** Render time $8.11$~s, I/O + video encoding $11.41$~s, total production $19.52$~s (`results/production.csv`; every frame hash-verified against the serial reference).
- **Hardware Architecture:** AMD Ryzen 5 5500U, 6 physical cores, 12 logical threads, 15.54 GB RAM (`results/environment.json`).

---

## 13. Glossary of Terms

- **Speedup ($S$):** Ratio of single-process execution time to parallel execution time ($S = T_{serial} / T_p$).
- **Parallel Efficiency ($E$):** Speedup normalized by allocated processing units ($E = S / p$).
- **Load Imbalance ($\lambda$):** Ratio of maximum worker compute time to mean worker compute time ($\lambda = \max(t_{comp}) / \text{mean}(t_{comp})$).
- **GIL (Global Interpreter Lock):** Mutex in CPython preventing multiple native threads from executing Python bytecode simultaneously.
- **`prange`:** Numba parallel-for loop construct lowering to OpenMP runtime work-sharing.
- **Master-Worker:** Dynamic task-scheduling pattern where a coordinator process dispenses tasks to worker processes.
- **Rank:** Unique integer identifier assigned to each process in an MPI communicator.
- **Communicator:** Opaque handle representing a group of communicating MPI processes (`MPI.COMM_WORLD`).
- **Bridged Networking:** Virtual machine network mode attaching VM virtual NIC directly to physical layer-2 network.
- **NAT (Network Address Translation):** Virtual network mode hiding guests behind a private subnet managed by the host.
- **Macvlan:** Linux network driver assigning unique MAC addresses to container virtual interfaces.
- **Overlay Network:** Encapsulated software-defined network (VXLAN) routing container packets across multiple hosts.
- **Hostfile:** Text file listing available cluster node IP addresses and slot capacities for MPI launchers.
- **Slot:** Number of available processor cores/threads assigned to an MPI node in a hostfile.

---

## 14. Explicit List of What We Did NOT Test (Be Honest!)
When asked about any of the following items, state clearly: *"We did not test that in this phase."*
1. We did not run benchmarks inside Docker containers.
2. We did not deploy on bridged Virtual Machines.
3. We did not test over a physical multi-machine LAN switch.
4. We did not configure or test macvlan interfaces or host shims.
5. We did not measure physical Ethernet wire latency or switch MTU constraints.
6. We did not test dynamic frame assignment in S3 (only static Block and Cyclic).
7. We did not test non-blocking MPI calls (`Isend`/`Irecv`).
8. We did not test CUDA or GPU acceleration.

