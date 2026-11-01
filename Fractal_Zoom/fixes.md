# Comprehensive Code & Specification Audit: Fractal Zoom Mini-Project
**Course:** Intensive Computing (USTO-MB)  
**Authors:** Bellatreche Mohamed Amine, Rezaiguia Soltane  
**Supervisor:** Prof. Dairi Abdelkader  
**Audit Date:** November 2026  
**Scope:** Local Single-Machine Execution Phase (V0, V1, MPI Local S1–S3; V2–V4 Deferred)

---

## Executive Summary

Gemini completed an impressive amount of work on the Fractal Zoom project:
- A fully unified, deterministic JIT kernel (`src/fractal_core.py`) with zero `fastmath`, correct Cartesian orientation, and global-coordinate index mapping.
- All core execution engines: Serial baseline (V0), shared-memory multi-threading with OpenMP (V1-A rows and V1-B shuffled tiles), and three MPI strategies (S1 static rows, S2 master-worker dynamic tiles, S3 block and cyclic whole frames).
- Local Ping-Pong latency/bandwidth benchmark (`src/pingpong.py`).
- 120-frame production movie generated at 24 fps (`output/zoom.mp4`, `output/zoom.gif`, and 120 PNG frames).
- 100% passing correctness test suite (`results/correctness.csv`) with zero differing pixels against serial reference arrays.
- All 8 figures plotted (`figures/fig1_scaling.png` to `fig8_filmstrip.png`), automated LaTeX tables and number macros generated, and `report/main.pdf` compiled with exact cover page text.

However, a strict line-by-line audit against `instructions.md` reveals **11 issues** ranging from critical report page-budget compliance and factual self-contradictions down to minor script enhancements.

Below is the structured catalog of all necessary fixes, categorized by severity, with exact file locations, requirement citations from `instructions.md`, observed behavior, and drop-in correction instructions.

---

## Table of Audit Findings

| ID | Category | Severity | Description | Target File(s) |
| :--- | :--- | :--- | :--- | :--- |
| **F-01** | Report | **CRITICAL** | Body exceeds 6-page maximum budget (spills onto Page 8) | `report/main.tex` |
| **F-02** | Report / Notes | **HIGH** | Self-contradictory prose on S2 optimal tile size ($80\times 80$) | `report/main.tex`, `notes.md` |
| **F-03** | Data / Table | **MEDIUM** | MPI S2 $p=3$ single-repetition rogue entry in Table 7 | `results/mpi_scaling.csv`, `scripts/make_tables.py` |
| **F-04** | Experiment | **MEDIUM** | Variant B skipped tile size 40 (only tile 20 was benchmarked) | `scripts/run_benchmarks.py`, `results/threads.csv` |
| **F-05** | Plots | **MEDIUM** | Figure 1 missing error bars, FPS panel, and oversubscription markers | `scripts/make_plots.py` |
| **F-06** | Tests | **MEDIUM** | `run_correctness.py` simulates MPI locally instead of invoking `mpiexec` | `tests/run_correctness.py` |
| **F-07** | Code / Timing | **LOW** | `src/serial.py` times SHA-256 and `.npy` disk writes when `--save-reference` is active | `src/serial.py` |
| **F-08** | Code | **LOW** | `src/threads.py` Variant A allocates 43,200 redundant 1-row NumPy arrays | `src/threads.py` |
| **F-09** | Report | **LOW** | Figure numbers in PDF captions disconnect from disk filenames | `report/main.tex`, `scripts/make_plots.py` |
| **F-10** | Production | **LOW** | `make_movie.py` uses Threads V1-B instead of the fastest E4 MPI strategy | `src/make_movie.py` |
| **F-11** | Audit Tool | **COSMETIC** | `audit_numbers.py` flags valid algorithmic constants as unknown numbers | `scripts/audit_numbers.py` |

---

## Detailed Audit Findings and Necessary Fixes

---

### Fix 1 (CRITICAL): Reduce LaTeX Report Body to Within 6 Pages
* **Reference:** `instructions.md` Section 9.1:
  > *"Page 1 is the cover page (section 1). After it: the body, at most 6 pages (the proposal asks 4 to 6 pages). Then references, then appendices (extra figures, full tables, environment, commands), which do not count in the 6 pages."*
* **Observed Defect:**
  In `report/main.pdf`:
  - Page 1 is the unnumbered title page.
  - Body pages span Pages 2 through 8 (7 pages total).
  - On Page 8, Section 6 (Conclusion) has its final 6 lines, followed by the bibliography (`\begin{thebibliography}`).
  - Appendices start on Page 9.
  - Even though `dev_log.md` claims "Pages 2–7: 6-page body", Section 6 actually spilled onto Page 8.
* **Root Cause:**
  Figures 1, 2, 3, 4, and 5 in the body have vertical heights that push Section 5 (Limitations) to the bottom of Page 7 and force Section 6 (Conclusion) onto Page 8.
* **Required Fix in `report/main.tex`:**
  1. Slightly reduce the heights of body figures:
     - Figure 1 (`fig1_scaling.png`): change `height=4.4cm` to `height=3.8cm`.
     - Figure 2 (`fig2_busy_time.png`): change `height=4.0cm` to `height=3.4cm`.
     - Figure 3 (`fig3_tile_size.png`): change `height=3.8cm` to `height=3.2cm`.
     - Figure 4 (`fig4_cost_map.png`): change `height=4.0cm` to `height=3.4cm`.
     - Figure 5 (`fig8_filmstrip.png`): change `width=0.72\linewidth` to `width=0.64\linewidth`.
  2. In Table 5 (`deployment_status.tex`), reduce row spacing slightly (`\renewcommand{\arraystretch}{0.9}`).
  3. Ensure Section 6 (Conclusion) ends completely on Page 7.
  4. Place `\clearpage` and `\begin{thebibliography}` on Page 8 immediately before `\appendix`. This makes the body exactly 6 pages (Pages 2–7).

---

### Fix 2 (HIGH): Resolve Self-Contradictory Prose on S2 Optimal Tile Size
* **Reference:** `instructions.md` Section 5.6 & Section 9.2:
  > *"The tile-size study (10, 20, 40, 80) must show the trade-off between balance and number of messages, using measured numbers."*
  > *"Never present expected behaviour from the proposal as a measured result: test it, and if the data disagree with the proposal, say so and look for the reason."*
* **Observed Defect:**
  - In `report/main.tex` (Section 4.4, lines 217–218):
    > *"Conversely, large tiles ($80 \times 80$) reduce message count but induce tail latency because worker idle times spike when finishing large boundary tiles. The optimal trade-off on our architecture occurs at \sTwoOptimalTileSize$\times$\sTwoOptimalTileSize~pixels ($T_p = \sTwoOptimalTileTime$~s)."*
    Because `\sTwoOptimalTileSize` dynamically expands to `80`, the sentence literally asserts that $80 \times 80$ causes idle tail latency and then immediately declares $80 \times 80$ optimal!
  - In `notes.md` Section 7:
    > *"Optimal tile size is $20 \times 20$ or $40 \times 40$."*
    This directly contradicts Section 12 (Cheat-Sheet) in the same file, which states:
    > *"S2 Master-Worker Optimal Tile Size: $80 \times 80$ pixels ($T_p = 13.83$~s at $p = 4$, with $40 \times 40$ close at $14.13$~s)."*
* **Root Cause:**
  Gemini copied a textbook explanation ("large tiles hurt balance, small tiles hurt communication, 20 or 40 is optimal") without noticing that the real benchmark data on this system measured $80 \times 80$ as the fastest (13.80s vs 14.13s for 40, 15.34s for 20, and 18.38s for 10). Because the global task queue spans all 120 frames continuously without barriers, $80 \times 80$ generates 4,800 tasks across 3 workers—more than enough for load balancing while reducing message handling by 98% compared to $10 \times 10$.
* **Required Fix in `report/main.tex` and `notes.md`:**
  Replace the contradictory sentence in `report/main.tex` with:
  ```latex
  Small tiles ($10 \times 10$) provide fine load granularity but flood the master with 552,963 messages, creating a communication bottleneck ($T_p = 18.38$~s). Because our unified task queue spans all 120 frames without inter-frame barriers, $80 \times 80$ tiles provide 4,800 total tasks—sufficient to balance 3 workers without tail latency—while slashing message traffic to 9,603 messages, achieving the fastest rendering time ($T_p = \sTwoOptimalTileTime$~s).
  ```
  In `notes.md` Section 7, update the description to state $80 \times 80$ as optimal on this workload, matching Section 12.

---

### Fix 3 (MEDIUM): Clean Up MPI S2 $p=3$ Rogue Row in Table 7
* **Reference:** `instructions.md` Section 3 & Section 6.1:
  > *"Process / thread counts: 1, 2, 4, 8 (and the logical core count if it is different) [= 12]"*  
  > *"Repetitions: at least 3 per configuration (5 if cheap), report the median and the min–max"*
* **Observed Defect:**
  In `results/mpi_scaling.csv`, row 1 contains a test execution of S2 with $p=3$ (`workers=2`).
  In Table 7 (`report/generated/scaling_summary.tex`), this appears as:
  ```latex
  MPI S2 (tiles 20) & 3 & 21.29 & 21.29--21.29 & 1.90\times & 63.3\% & 5.6 \\
  ```
  - It only has **1 repetition** (min and max are identical).
  - $p=3$ is not in `config.json` `process_counts` (`[1, 2, 4, 8, 12]`).
  - Every other row in Table 7 has 3 repetitions.
* **Required Fix:**
  In `scripts/make_tables.py`, filter scaling table rows to only include process counts defined in `config["process_counts"]`:
  ```python
  # In make_scaling_summary_table():
  allowed_p = set(cfg["process_counts"])
  for p, grp in df_mpi[df_mpi["strategy"] == "S2"].groupby("p"):
      if p not in allowed_p:
          continue
      ...
  ```
  Re-run `python scripts/make_tables.py` to regenerate `scaling_summary.tex`.

---

### Fix 4 (MEDIUM): Benchmark Variant B with Tile Size 40
* **Reference:** `instructions.md` Section 5.3:
  > *"Variant B: prange over tiles, the order of the tiles shuffled with the fixed seed so that cheap and expensive tiles are mixed (use 20 and 40 pixel tiles at least, all four sizes if time allows)."*
* **Observed Defect:**
  In `scripts/run_benchmarks.py` line 126:
  ```python
  for t_sz in [20]:
  ```
  Only tile size 20 was evaluated for Threads Variant B. Tile size 40 was omitted from `results/threads.csv`.
* **Required Fix:**
  1. In `scripts/run_benchmarks.py`, update line 126 to `for t_sz in [20, 40]:`.
  2. Execute the missing runs for tile 40:
     ```powershell
     python scripts/run_benchmarks.py
     ```
  3. Re-run `python scripts/make_plots.py` and `python scripts/make_tables.py`.

---

### Fix 5 (MEDIUM): Enhance Figure 1 (Scaling) per Section 7 Requirements
* **Reference:** `instructions.md` Section 7:
  > *"1. Speedup versus p with the ideal line, threads (A and B) and MPI (S1, S2, S3) on two panels. Efficiency versus p in the same way, and frames per second versus p."*  
  > *"min–max as error bars or a light band, oversubscribed points drawn differently"*
* **Observed Defect:**
  In `scripts/make_plots.py` (`plot_fig1_scaling`):
  - Only 2 panels are plotted (Speedup and Efficiency). FPS vs $p$ is missing.
  - Measured points do not have min-max error bars or light bands.
  - Oversubscribed process counts ($p=8$ and $p=12$) use the exact same solid line and filled markers as non-oversubscribed points ($p \le 6$).
* **Required Fix in `scripts/make_plots.py`:**
  1. Add error bars showing min and max values (`ax.errorbar(..., yerr=[med - min_val, max_val - med], capsize=3)`).
  2. Style oversubscribed points ($p > 6$) with hollow markers (e.g., `markerfacecolor='white'`) or dashed line segments beyond $p=6$.
  3. Add a third subplot for Frames Per Second (FPS) vs $p$, or document why a 2-panel layout was chosen to preserve LaTeX figure budget.

---

### Fix 6 (MEDIUM): Upgrade `tests/run_correctness.py` to Run Actual MPI Subprocesses
* **Reference:** `instructions.md` Section 5.10:
  > *"Compare V1-A, V1-B (several tile sizes...), S1, S2 (tile sizes...), S3 block and cyclic, with p in 2, 3, 4 (odd p catches uneven row blocks)."*
* **Observed Defect:**
  In `tests/run_correctness.py` lines 226–289:
  ```python
  # Using local simulation of MPI partitioning to ensure logic bitwise equality
  ```
  The script loops through ranks locally and calls `render_block()` sequentially. It does not launch `mpiexec` or test MPI communicators, buffer gathers, or master-worker queues.
* **Required Fix in `tests/run_correctness.py`:**
  Add a real MPI verification block that launches:
  - `mpiexec -n 2 python src/mpi_s1_rows.py --rep 1`
  - `mpiexec -n 3 python src/mpi_s2_tiles.py --tile 20 --rep 1`
  - `mpiexec -n 3 python src/mpi_s3_frames.py --assignment cyclic --rep 1`
  Verify that each process exits with code 0 and records `checksum_ok = True`.

---

### Fix 7 (LOW): Decouple Timing from Hashing in `src/serial.py`
* **Reference:** `instructions.md` Section 5.2:
  > *"Renders all frames... returns the counts, prints and saves the timing. --save-reference: writes results/serial_hashes.csv... and saves full count arrays... in results/reference/"*
* **Observed Defect:**
  In `src/serial.py` lines 53–71:
  ```python
  t0 = time.perf_counter()
  for k in frame_indices:
      ...
      counts = render_block(...)
      if save_reference:
          h_val = hash_frame(counts)
          tot_iter = total_iterations(counts)
          if k in ref_frames:
              np.save(...)
  t1 = time.perf_counter()
  ```
  When `--save-reference` is passed, `t_render` includes SHA-256 calculation, array iteration summing, and disk writes (`np.save`). This artificially inflated rep 1 timing (36.48s) compared to pure render reps.
* **Required Fix in `src/serial.py`:**
  Store the computed `counts` in a list during the timed loop, and perform `hash_frame()`, `total_iterations()`, and `np.save()` *after* `t1 = time.perf_counter()`.

---

### Fix 8 (LOW): Eliminate Row Allocation Overhead in `src/threads.py` (Variant A)
* **Reference:** `instructions.md` Section 5.1 & Section 5.3:
  > *"One kernel, shared by every version... prange over rows"*
* **Observed Defect:**
  In `src/threads.py` lines 40–44:
  ```python
  for j in prange(H):
      row = render_block(j, j + 1, 0, W, xmin, ymax, dx, dy, maxit)
      for i in range(W):
          out[j, i] = row[0, i]
  ```
  Calling `render_block(j, j + 1, ...)` allocates a new 1-row NumPy array (`out = np.empty((1, W))`) on every loop iteration. Across 120 frames with 360 rows, this performs **43,200 dynamic array allocations**, adding overhead to Variant A that is unrelated to computation.
* **Required Fix in `src/threads.py`:**
  Provide an in-place rendering function or slice directly:
  ```python
  @njit(parallel=True)
  def render_frame_rows_fast(H, W, xmin, ymax, dx, dy, maxit):
      out = np.empty((H, W), dtype=np.int32)
      for j in prange(H):
          ci = ymax - j * dy
          for i in range(W):
              cr = xmin + i * dx
              zr = zi = 0.0
              k = 0
              while k < maxit and (zr * zr + zi * zi) <= 4.0:
                  zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                  k += 1
              out[j, i] = k
      return out
  ```

---

### Fix 9 (LOW/COSMETIC): Synchronize Figure Filenames with PDF Figure Numbers
* **Reference:** `instructions.md` Section 7:
  > *"In the report, put the most important figures in the body (1, 2, 3, 4, 8), the others in the appendix."*
* **Observed Defect:**
  Because Figure 8 (`fig8_filmstrip.png`) is placed in Section 4.8 of the body, LaTeX assigns it the sequential caption **Figure 5**.
  Then in the appendices:
  - `fig6_s3_block_vs_cyclic.png` is captioned **Figure 6**
  - `fig5_busy_vs_iterations.png` is captioned **Figure 7**
  - `fig7_pingpong.png` is captioned **Figure 8**
  While all `\ref{}` links resolve properly in LaTeX, the discrepancy between the file names on disk (`fig5_...`, `fig8_...`) and the numbers in the captions can confuse readers inspecting the repository.
* **Required Fix:**
  Add a clear note in `figures/README.md` and `README.md` Section 5 explaining this mapping:
  - Disk `fig8_filmstrip.png` $\rightarrow$ PDF Report Figure 5 (Filmstrip)
  - Disk `fig6_s3_block_vs_cyclic.png` $\rightarrow$ PDF Report Figure 6 (S3 Comparison)
  - Disk `fig5_busy_vs_iterations.png` $\rightarrow$ PDF Report Figure 7 (Cost Correlation)
  - Disk `fig7_pingpong.png` $\rightarrow$ PDF Report Figure 8 (Ping-Pong Curves)

---

### Fix 10 (LOW): Document Strategy Choice in `src/make_movie.py` (E10)
* **Reference:** `instructions.md` Section 6.1:
  > *"E10 Production run: all 120 frames, 640 x 360, the best strategy and p found in E4, writes the PNG frames..."*
* **Observed Defect:**
  In E4, the fastest strategy was MPI S3 cyclic ($T_p = 3.96$~s at $p=12$). `src/make_movie.py` uses Threads V1-B at $p=6$ ($T_p = 7.70$~s). While using threads avoids MPI gather serialization, it strictly deviates from the "best strategy found in E4" instruction.
* **Required Fix in `src/make_movie.py`:**
  Add a CLI argument `--engine {threads, mpi}` allowing the user to render the production movie using either Threads V1-B or MPI S3 cyclic. Document in `dev_log.md` why Threads V1-B was set as default for single-workstation generation.

---

### Fix 11 (COSMETIC): Whitelist Algorithmic Constants in `scripts/audit_numbers.py`
* **Reference:** `instructions.md` Section 9.4:
  > *"Extract every number from the prose of main.tex. Each one must be (a) a generated macro or table cell, (b) a parameter from config.json, or (c) structural..."*
* **Observed Defect:**
  `results/number_audit.md` currently reports 6 numbers as `FLAGGED`:
  `0.5` (rounding formula), `100` (MPI tag), `256` (SHA-256), `3` (rank 3 example), `63.0` and `64` (color mapping formula). These are legitimate algorithmic constants, not unanchored benchmark numbers.
* **Required Fix in `scripts/audit_numbers.py`:**
  Add `"0.5"`, `"100"`, `"256"`, `"3"`, `"63.0"`, `"64"` to `known_config_numbers`. Re-running the script will yield a 100% clean traceability audit report.

---

## Action Plan to Apply All Fixes

To bring the project to 100% compliance:

```powershell
# 1. Edit report/main.tex (Fix 1, Fix 2)
# Reduce figure heights: fig1 to 3.8cm, fig2 to 3.4cm, fig3 to 3.2cm, fig4 to 3.4cm, fig8 to 0.64\linewidth
# Fix S2 tile text in section 4.4

# 2. Edit scripts/make_tables.py (Fix 3)
# Filter scaling_summary to only include p in config['process_counts']

# 3. Edit scripts/audit_numbers.py (Fix 11)
# Add algorithmic constants to known_config_numbers

# 4. Regenerate tables, macros, and audit
python scripts/make_tables.py
python scripts/audit_numbers.py

# 5. Recompile LaTeX report twice and verify page count
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cd ..

# 6. Verify page 1 and body page budget
pdftotext -f 1 -l 1 report/main.pdf -
# Confirm body starts on page 2 and finishes on page 7
```

---
*Report prepared for Bellatreche Mohamed Amine & Rezaiguia Soltane — Intensive Computing 2026.*

