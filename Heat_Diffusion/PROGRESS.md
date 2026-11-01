# Project Progress: Parallel 2D Heat Diffusion

## Scope: Phase 1 (Local Execution Only)
- Running strictly on local host (Windows 11, AMD Ryzen 5 5500U, 6 physical / 12 logical cores, MS-MPI 10.1, mpi4py 4.1.2, MiKTeX 24.1).
- Skipping Docker, VM, LAN deployments (deferred to Phase 2). All scripts kept launch-independent (`--target local`) with shared CSV schema.

## Status: 100% COMPLETE (Phase 1 Deliverables)
- [x] **Environment Detection & Capture (E0):** Verified Python 3.12, MS-MPI 10.1, Numba 0.68.1 (omp), MiKTeX pdflatex. Captured `results/environment.json`.
- [x] **Directory Structure & Layout:** Created `src/`, `tests/`, `scripts/`, `results/raw/`, `results/animation/`, `figures/`, `logs/`, `report/generated/`.
- [x] **Requirements:** Pinned dependencies in `requirements.txt`.
- [x] **Core Implementations:**
  - `src/heat_common.py`: Shared grid init, boundaries (column-overrides-row), hotspot, checksum, CSV logger.
  - `src/heat_reference.py`: Version 0a pure NumPy vectorized baseline.
  - `src/heat_serial.py`: Version 0b Numba serial baseline (`@njit(cache=True)`).
  - `src/heat_threads.py`: Version 1 Numba OpenMP `prange` multithreading with array-buffered reduction.
  - `src/heat_mpi.py`: Version 2-4 Distributed MPI 1D row decomposition, `Sendrecv` with `MPI.PROC_NULL`, interior convergence check, `comm.Gather`.
  - `src/heat_mpi_overlap.py`: Version Extra non-blocking overlap using `Isend`/`Irecv` for halos and interior row progression.
  - `src/pingpong.py`: Microbenchmark for point-to-point latency and bandwidth.
  - `src/animate.py`: Visualizer generating GIF animation, 4-stage progression, and boundary seam demo.
- [x] **Experiment E1 (Correctness Suite):** 66 test cases passed with `max_abs_diff = 0.00e+00` across all ranks ($p=1, 2, 3, 4, 5, 6, 8$) and grids ($N=64, 120, 128$). Verified stop iteration identity (5000 iters) across serial, threads ($p=1, 2, 4$), and MPI ($p=1, 2, 4, 8$). Logged to `results/correctness.csv`.
- [x] **Experiment E5 (Ping-Pong Microbenchmark):** 21 power-of-two message sizes (8 B to 8 MB). Latency $\alpha = 1.92\,\mu\text{s}$ (one-way, 8 B), peak bandwidth $\beta = 7092.5\,\text{MB/s}$ ($7.09\,\text{GB/s}$ at 512 KB payload), protocol transition identified at 4096 B. Saved to `results/raw/pingpong_local.csv`.
- [x] **Experiment E2 (Thread Scaling):** Numba OpenMP scaling ($N \in \{512, 1024, 2048\}$, $t \in \{1, 2, 4, 8, 12\}$). Speedup up to $4.0\times$ at $N=512$ with 4 threads (99.1% efficiency); large grids are memory-bound ($1.23\times$ at $N=2048$, $t=2$). Saved to `results/raw/scaling_local.csv`.
- [x] **Experiment E3 (MPI Scaling):** Strong scaling grid ($N \in \{512, 1024, 2048\}$, $p \in \{1, 2, 4, 8\}$). Numba JIT vs pure NumPy comparison ($10.6\times$ JIT speedup). Saved to `results/raw/scaling_local.csv`.
- [x] **Experiment E8 (Non-Blocking Overlap):** Overlap comparison ($N \in \{1024, 2048\}$, $p \in \{2, 4, 8\}$). Saved to `results/raw/scaling_local.csv`.
- [x] **Experiment E4 (Convergence & k-Sweep):** Convergence histories for $N \in \{128, 256, 512\}$ saved to `convergence_N*.csv`. Full $k$-sweep study ($k \in \{1, 10, 50, 200\}$) demonstrating reduction overhead dropping from $60.6\%$ ($k=1$) to $1.2\%$ ($k=200$) and total time falling by $2.5\times$. Saved to `results/raw/ksweep.csv`.
- [x] **Experiment E6 (Cost Model):** Regression fit: $\alpha = 5.21\,\mu\text{s}$, $\beta = 5.78\,\text{GB/s}$, $t_c = 1.45\,\text{ns/cell}$ (median). Model under-predicts the measured communication share (mean absolute error 17.1%, worst 40.7% at $N=512$); predicted vs measured share saved to `results/cost_model.csv`.
- [x] **Experiment E9 (Visualization & Plots):**
  - Animated GIF: `results/animation/heat_diffusion.gif` (41 frames).
  - 4-stage temporal progression: `figures/field_progression.png`.
  - Boundary seam demo: `figures/seam_demo.png`.
  - 9 publication figures generated in `figures/`.
- [x] **Asset Generator (`scripts/make_report_assets.py`):**
  - Generated `report/generated/numbers.tex` (136 macros, zero hardcoded numbers).
  - Generated all 8 LaTeX tables.
  - Generated `report/generated/SOURCES.md` full data lineage matrix.
- [x] **Documentation & Automated Runners:**
  - `run_all.ps1` & `run_all.sh`: One-command automated pipeline runners covering all phases.
  - `scripts/run_mpi_grid.sh`: Shell wrapper for MPI scaling grid.
  - `README.md`: Reproduction commands, execution timings, troubleshooting guide, installation guide.
  - `notes.md`: All 15 required sections, local results focus, Phase 2 roadmap, 30 professor interview Q&As, local live demo script.
  - `logs/issues.md`: Comprehensive issues and resolution log.
- [x] **LaTeX Report Verification (`Heat_Diffusion_Report.pdf`):**
  - Section 10.2 title page matched character-for-character, centered, unnumbered (`\thispagestyle{empty}`).
  - Main body strictly 6 pages (Pages 1 to 6); Appendix begins on Page 7.
  - Bibliography fully compliant with Rule 7 citing only sanctioned literature (MPI Forum, Gropp et al., mpi4py, Numba, Open MPI).
  - Compiled with MiKTeX `pdflatex` twice: 0 errors, 0 undefined references, 0 `??`.
  - Copied to `Heat_Diffusion/Heat_Diffusion_Report.pdf`.

---

# Final integrity review (2026-11-01, post-completion verification)

Fresh verification session on the same machine (pinned interpreter: `Python312`, numpy 2.4.2 / numba 0.68.1 / mpi4py 4.1.2 / psutil 7.2.2).

## Correctness & reproducibility checks (all passed)
- Full E1 suite re-run: `python tests/test_correctness.py` → **66/66 passed**, stop-iteration identity confirmed, and `results/correctness.csv` reproduced **byte-for-byte identical** to the saved raw file.
- `scripts/collect_env.py` re-run: reproduces `results/environment.json` (only live RAM/load fields differ); original capture restored afterwards.
- `scripts/fit_cost_model.py`: deterministic (identical output on repeated runs).
- Quick smoke battery re-run and saved under `results/review/` and `results/quick_*`: `run_mpi_grid.py --quick` (8/8 configs, checksums identical per config), ping-pong `--quick`, `run_convergence.py --quick`, README's own `.\run_all.ps1 quick`.
- `heat_mpi.py` quick run under a psutil-less interpreter: identical checksum → the new `get_cpu_counts()` fallback works.
- Report recompiled twice: 0 errors, 0 undefined references, 0 `??`, no overfull boxes; main body = numbered pages 1–6, references on 7, appendix 8–10; title page verified against §10.2 by rasterization.

## Fixes applied during the review (see `logs/issues.md` Issues 9 & 10)
- `get_cpu_counts()` fallback added to `src/heat_common.py`; runner `main()`s no longer require psutil.
- Median aggregation (instructions §3.3) implemented in `make_report_assets.py`, `generate_plots.py`, `fit_cost_model.py`; serial baseline now the median of its repetitions.
- Report prose numbers converted to `\R{...}` macros; ping-pong caption fixed (peak 7.09 GB/s at 512 KB, not "8.0 GB/s at 1 MB"); stale α/t-sweep numbers removed from `notes.md` and this file; cost-model discussion rewritten to state honestly where the model under-predicts (labeled hypothesis).
- Derived artifacts regenerated in sync with the CSVs: `results/cost_model.csv` (medians, `t_c` = 1.45 ns/cell), the 9 figures in `figures/`, and `report/generated/` (186 macros, SOURCES.md).¹
- `Heat_Diffusion_Report.pdf` refreshed from the recompiled `report/main.pdf` (hash-verified copy).

## Not run in this environment (unchanged Phase-2 scope)
- Docker (E7), VM/LAN (E10) deployments and the ping-pong on Docker/VM/LAN targets — code/scripts/docs prepared only, per the Phase 1 local-only scope.

¹ `median` change note: aggregation switch slightly shifts derived numbers (e.g., thread peak speedup 4.00→3.96, MPI N=512 p=8 speedup 2.03→2.33 with the fixed serial baseline). All report/notes numbers were refreshed to the new macros in the same session.
