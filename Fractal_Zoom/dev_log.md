# Development Log: Fractal Zoom Mini-Project

## 2026-11-01 - Project Initialization & Scope Setup
- Project root set up at `Fractal_Zoom/`.
- Proposal docx moved to `docs/Fractal_Zoom_Mini_Project_Proposal.docx`.
- Scope defined: Local execution on host machine. V0, V1, and local MPI runs are targeted for execution. V2 (VMs), V3 (Docker), and V4 (physical LAN) deferred to subsequent phase.
- Environment check:
  - Python 3.12.10 in `3.13` venv.
  - CPU: AMD Ryzen 5 5500U with Radeon Graphics (6 physical cores, 12 logical processors).
  - RAM: 15.54 GB.
  - OS: Windows 11 Pro (build 10.0.26200).
  - Power: AC plugged in.
  - Compiler / TeX: MiKTeX 24.1 (`pdflatex`).
  - MPI launcher: Microsoft MPI Startup Program v10.1 (`C:\Program Files\Microsoft MPI\Bin\mpiexec.exe`).
  - OpenMP threading layer available in Numba (`omp`).
- Discovered issue: `mpiexec` was not present in the default PowerShell PATH. Resolved by configuring `mpi_launch.py` to auto-detect `C:\Program Files\Microsoft MPI\Bin\mpiexec.exe` and add it to `PATH` dynamically.
- Discovered issue: `imageio-ffmpeg` was initially missing for MP4 encoding. Resolved by installing `imageio-ffmpeg==0.7.0`.
- Benchmark feasibility test:
  - Frame 0 (maxit 200): ~73 ms.
  - Frame 119 (maxit 1000): ~94 ms.
  - Full 120-frame serial run estimated at ~10-11 seconds. Fits within the 10-60s recommended benchmark window. Decided to benchmark all 120 frames without subsampling.

## 2026-11-01 - Serial Baseline (V0) & Reference Hashes
- Executed `src/serial.py --save-reference` for all 120 frames.
- Measured runtime: 36.483 s (3.29 fps).
- Hashes and iteration counts for all 120 frames saved to `results/serial_hashes.csv`.
- Saved reference `.npy` arrays for frames 0, 40, 80, 119 in `results/reference/`.
- Full 120-frame workload confirmed feasible without frame subsampling (36.5 s falls squarely within the 10-60 s window prescribed in Section 3).

## 2026-11-01 - MPI Strategy S2 Fix
- Discovered issue: In `src/mpi_s2_tiles.py`, master was polling with `tag=MPI.ANY_TAG` while active workers > 0. When an early-finishing worker received `TAG_STOP`, it sent its final statistics dictionary with `tag=100`, which was erroneously received by the master's work loop, triggering `ValueError: too many values to unpack (expected 2)`.
- Solution: Master work loop now strictly matches `tag=TAG_WORK`, ensuring all worker tile returns are processed before receiving terminal `tag=100` summary packets.

## 2026-11-01 - Ping-Pong Benchmark (E7)
- Executed `src/pingpong.py` locally on 2 ranks.
- Message sizes: 1 B to 8 MiB.
- Measured minimum latency: 2.24 - 2.79 microseconds (characteristic of inter-process shared-memory transfers on Windows MS-MPI).
- Maximum bandwidth: 3882.47 MB/s (~3.88 GB/s) for 8 MiB transfers.
- Results saved to `results/pingpong.csv`.

## 2026-11-01 - Test Suite, Movie Generator, and Harness Implementation
- Implemented `tests/run_correctness.py`:
  - Unit tests: Verified Cartesian orientation (Im=0 at row > H/2), row block complete coverage across $p \in [1, 12]$, tile list complete coverage including edge clipping for non-divisors.
  - Bitwise reference validation: Verified frames 0, 40, 80, 119 against saved serial `.npy` arrays (0 differing pixels).
  - Parallel identity tests: Verified V1-A, V1-B, S1, S2, S3 against V0 reference.
- Discovered issue: Setting `NUMBA_NUM_THREADS=1` globally at import time restricted Numba's thread pool capacity, causing `ValueError: The number of threads must be between 1 and 1` when invoking `numba.set_num_threads(p)` for $p > 1$. Resolved by scoping thread constraints strictly to MPI rank worker scripts.
- Executed `scripts/compute_cost_profiles.py` (E6): Generated `results/cost_per_tile.npz` and `results/cost_per_rowblock.csv`.
- Implemented `src/make_movie.py` (E10) with cyclic modulo palette to avoid brightness flicker across dynamic iteration thresholds.
- Configured automated benchmark harness `scripts/run_benchmarks.py` to evaluate E2–E5 with random shuffling within repetitions.
- Enhanced `scripts/audit_numbers.py` to audit forbidden academic/style buzzwords (delve, crucial, pivotal, robust, leverage, seamless, etc.) per Section 9.3.
- Replaced buzzwords in `report/main.tex` to maintain direct student tone.

## 2026-11-01 - Benchmark Completion, Production Run (E10), and PDF Compilation
- All 104 scheduled benchmark executions across repetitions 1, 2, and 3 completed successfully with zero failures and zero checksum mismatches (`checksum_ok = True` on 100% of runs).
- Executed production run E10 (`python src/make_movie.py -p 6 --tile 20`):
  - Render time: 7.70 s (15.59 render fps).
  - I/O and video encoding time: 10.62 s.
  - Total production time: 18.32 s.
  - Artifacts generated: 120 PNG frames in `output/frames/`, animated GIF `output/zoom.gif` (6.22 MB), and H.264 MP4 `output/zoom.mp4` (2.21 MB).
- Regenerated all 8 figures (`figures/fig1_scaling.png` through `fig8_filmstrip.png`) using `scripts/make_plots.py`.
- Regenerated all LaTeX tables and 39 macros in `report/generated/` using `scripts/make_tables.py`.
- Conducted number audit with `scripts/audit_numbers.py`: PASS on zero forbidden placeholders and zero buzzwords.
- Tuned float constraints and list spacing in `report/main.tex` to satisfy Section 9.1 page budget:
  - Page 1: Mandatory exact cover page.
  - Pages 2–7: 6-page body covering Introduction, Implementation, Experimental Setup, Results, Limitations, and Conclusion.
  - Pages 8+: References and Appendices A–E.
- Successfully compiled `report/main.pdf` using `pdflatex` twice with zero undefined references.
- Verified cover page with `pdftotext -enc UTF-8` confirming exact title, authors, supervisor, and institution strings.

## 2026-11-01 - Audit Review and Complete Fix Application
- Comprehensive audit conducted against `instructions.md` rules and specifications (`fixes.md`).
- Applied F-04: Ran 15 benchmark runs for Variant B Tile 40 ($p \in \{1, 2, 4, 8, 12\}$, reps 1--3). All passed checksum verification (`results/threads.csv` now complete with 45 rows).
- Applied F-03: Filtered rogue single-repetition $p=3$ row from `results/mpi_scaling.csv` and `results/busy_time.csv`. Enforced process count filter in `scripts/make_tables.py` for Table 7.
- Applied F-05: Redesigned Figure 1 in `scripts/make_plots.py` into a 3-panel figure (Speedup, Efficiency, FPS) with min-max error bars, hollow markers for oversubscribed core counts, and shaded SMT region ($p > 6$).
- Applied F-06: Added live MPI pipeline verification (`run_mpi_script` on S1, S2, S3) to `tests/run_correctness.py`. Complete test suite passed (unit tests, reference frames, parallel slices, and live MPI runs).
- Applied F-07: In `src/serial.py`, moved SHA-256 computation and `.npy` disk writes outside timed section.
- Applied F-09: Created `figures/README.md` providing an explicit mapping of disk files to PDF report figure numbers.
- Applied F-11: Whitelisted algorithmic parameters and dynamic macros in `scripts/audit_numbers.py`. Audit confirmed 0 flagged numbers, 0 buzzwords, and 0 placeholders.
- Applied F-02: Harmonized narrative on S2 optimal tile size between `report/main.tex` and `notes.md` line 103 ($80 \times 80$ tile is fastest at 13.80 s due to balancing 3 workers without master message flooding).
- Applied F-01: Re-balanced `report/main.tex` layout. Moved 34-row Table 2 to Appendix B, adjusted figure dimensions and bullet spacing, pulling the Conclusion cleanly onto Page 7 (page numbered 6). Verified with `pdftotext` that the 6-section body fits strictly on pages 2--7 (exactly 6 pages), followed by References on Page 8 (numbered 7) and Appendices on Pages 9--12.
- Compiled `report/main.pdf` cleanly with two passes of `pdflatex` with 0 errors and 0 undefined references.
