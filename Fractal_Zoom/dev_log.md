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

## 2026-11-01 - Independent Review and Verification Session
- Full re-verification of claimed fixes in `fixes.md` against code, saved results, and fresh runs. Backups of derived artifacts taken before verification runs (temp folder); no file under `results/` or `output/` was hand-edited in this session.
- Environment confirmed: all benchmarks and the movie were produced by Python 3.12.10 in the `3.13` venv with numpy 2.5.4 / numba 0.68.0 / pandas 3.0.6 / imageio 2.38.1 / mpi4py 4.1.2 (matches `requirements.txt` and `results/environment.json`).
- Re-ran `tests/run_correctness.py`: ALL PASS, including live MPI (S1 p=2, S2 p=3, S3 p=2). Regenerated `results/correctness.csv` byte-identical (MD5) to the committed copy.
- Re-ran `scripts/make_tables.py` and `scripts/audit_numbers.py`: all 9 files in `report/generated/` and `results/number_audit.md` reproduce byte-identically from the saved CSVs. No hand-edited derived artifacts.
- Recompiled report; final `report/main.pdf`: 12 pages, 0 errors, 0 undefined references, 0 overfull hboxes. Body = pages 2-7 (6 pages), References page 8, Appendices 9-12. Cover page 1 matches the required format.
- Report edits this session (wording/layout only, no numbers changed): added `\emergencystretch`; Figure 1 paragraph and caption updated to the 3-panel description (speedup / efficiency / FPS, min-max bars); removed two unbreakable em-dash phrases; made the lambda ratio math breakable; reworded the `src/fractal_core.py` reference; shrank float heights slightly so the Conclusion stays on page 7.
- Verified `output/`: 120 PNG frames (`frame_000.png`..`frame_119.png`), `zoom.mp4` = 120 frames, 24 fps, 640x360, 5.0 s; `zoom.gif` = 120 frames with 40 ms real delay (4.8 s) as documented.
- Verified `figures/README.md` mapping matches the PDF's actual figure numbers (filmstrip = Fig 5 in appendix, etc.). Added the corresponding pointer in `README.md` section 5, added `scripts/audit_numbers.py` to the README folder layout, and corrected the "Quick Test" heading to a realistic ~2 minutes (the suite includes full live MPI runs).
- `notes.md` Figure 3 text corrected: 13.80 s -> 13.83 s, the actual median of the tile-80 rows in `results/tile_size.csv`.
- Result audit: `serial_times.csv` 5 reps (median 40.41 s; rep1 36.48 s is a ~9.9% outlier, kept per Rule 6, just under the ~10% spread caveat); `threads.csv` 45 rows incl. 15 tile-40 runs (F-04); `mpi_scaling.csv` 57 rows fresh today, `checksum_ok=True` everywhere; `tile_size.csv` 12 rows (tile 80 median 13.83 s < tile 40 14.13 s < tile 10 18.38 s); `busy_time.csv` 370 lines exactly; 114 run logs with zero errors/Tracebacks.
- Open item 1 (F-08 NOT applied, also not claimed in this log): `src/threads.py` Variant A still calls `render_block` per row inside `prange`, allocating a 1-row array each call. Applying it requires re-running E3 Variant A (~15 runs) to refresh `threads.csv`.
- Open item 2 (F-10 NOT applied, not claimed): `src/make_movie.py` has no `--engine {threads,mpi}` option. Its production row in `results/production.csv` also writes `checksum_ok=True` unconditionally (no real hash comparison happened). Fixing this properly means re-running E10, which would rewrite `output/`.
- Open item 3 (E8/E9, V2/V3/V4): deferred phase confirmed - no `docker/`, `lan/`, or `scripts/run_cluster_bench.py` exist. `results/deployment_status.json` uses the status word "DEFERRED", which is not one of the spec's words (EXECUTED / PREPARED, NOT EXECUTED / FAILED); README/report share this wording.

## 2026-11-02 - Fixes F-08 and F-10 Applied, Benchmarks Re-Run
- Applied F-08 in `src/threads.py`: Variant A (`render_frame_rows`) now computes pixels and writes them directly into the preallocated output array instead of calling `render_block` per row (which allocated a fresh 1-row array 43,200 times per run). The escape loop mirrors `fractal_core.render_block` exactly (`<= 4.0` bound, same float operation order). Correctness suite re-run: ALL PASS, V1-A rows bitwise identical to V0 at p = 2, 3, 4.
- Applied F-10 in `src/make_movie.py`: added `--engine {threads,mpi}`. Default = threads (documented rationale in the code and below); mpi engine mirrors S3-cyclic (whole frames per rank, one thread per rank, gather on rank 0 which writes PNGs/GIF/MP4). Also made `checksum_ok` real: every rendered frame's SHA-256 is compared against `results/serial_hashes.csv` instead of writing a hardcoded True. Threads V1-B stays the E10 default because on a single workstation it fully occupies the 6 physical cores at p=6, avoids gathering 110 MB of image data through MPI, and bbox-free assembly stays on one process.
- Fixed a resume-matching bug in `scripts/run_benchmarks.py` (found while refreshing E3 Variant A): historical S2 rows in `results/mpi_scaling.csv` store tile as `20.0` while the harness matched `20`, so re-runs duplicated completed configurations. `count_existing_reps` now compares matched fields numerically with `--dry-run` confirming 0 pending. The 12 duplicated E4-S2 mpi_scaling rows and 78 duplicated busy_time rows appended during this discovery run were removed; `mpi_scaling.csv` (57 rows) and `busy_time.csv` (370 data lines) are back to their verified clean session state.
- Re-ran E3 Variant A (15 runs, p in {1,2,4,8,12} x 3 reps, fresh `results/threads.csv` rows): all checksum_ok=True. New medians: p=1 41.27 s, p=2 23.81 s, p=4 13.13 s, p=8 7.10 s, p=12 5.39 s (7.50x vs serial median 40.41 s, eff 62.6%). F-08 made Variant A slightly faster and no result changed qualitatively - Variant B (4.83 s, 8.37x) remains the threaded winner.
- Re-ran E10 production (`python src/make_movie.py -p 6 --tile 20 --engine threads`): render 8.11 s, I/O + encoding 11.41 s, total 19.52 s; verification printed "all 120 frames match the serial V0 reference" (real checksum). `output/` regenerated (120 PNGs, zoom.mp4 120 frames @ 24 fps, zoom.gif).
- Regenerated `report/generated/` tables/macros, `results/number_audit.md` (0 flagged), and all 8 figures from the updated CSVs. Recompiled `report/main.pdf`: 12 pages, 0 errors, 0 undefined references, 0 overfull, conclusion still ends on page 7 (numbered 6). `notes.md` updated: Variant A p=12 5.42 -> 5.39 s and production 7.70/10.62/18.32 -> 8.11/11.41/19.52 s.
- Pre-run backups of the replaced artifacts: `threads.csv`, `production.csv`, `output/` (temp backup folder); no other data touched.


