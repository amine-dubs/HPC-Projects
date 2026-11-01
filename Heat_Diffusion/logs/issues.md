# Issues and Troubleshooting Log

This file records real issues encountered during development and benchmarking, their causes, and the applied fixes.

### Issue 1: `mpirun` not recognized in Windows PowerShell
- **Symptom:** `Get-Command mpirun` returned nothing. `where.exe mpiexec` also returned nothing initially.
- **Cause:** Microsoft MPI installs `mpiexec.exe` to `C:\Program Files\Microsoft MPI\Bin`. While present in Machine PATH, the active PowerShell session had not inherited the updated PATH, and the standard command name in Linux/Open MPI tutorials is `mpirun`. Furthermore, `.bat` wrappers in PowerShell can swallow child process output streams unless `.ps1` is defined.
- **Fix:** Created `mpirun.ps1` and `mpirun.cmd` wrappers invoking `mpiexec.exe @args` inside `Python312\Scripts` which is on PATH. Verified output capture works seamlessly.

### Issue 2: WSL2 has no installed Linux distribution
- **Symptom:** `wsl -l -v` returned `Wsl/WSL_E_DEFAULT_DISTRO_NOT_FOUND`.
- **Cause:** WSL2 kernel is present, but no Linux distribution (such as Ubuntu) is registered.
- **Fix:** Native Windows execution utilized using MS-MPI 10.1 and Python 3.12 with `mpi4py` 4.1.2 and MiKTeX `pdflatex`. Completely matches Phase 1 local scope.

### Issue 3: In Numba parallel loop (`prange`), scalar maximum reduction silently fails
- **Symptom:** In `@njit(parallel=True)`, computing `if d > diff: diff = d` across iterations of `prange` returned `0.0`.
- **Cause:** Numba's auto-parallelizer only synthesizes OpenMP reductions for standard in-place operators (like `+=`, `*=`). Conditional scalar updates inside `prange` are not recognized as reductions and suffer from thread-local scoping / race conditions.
- **Fix:** Used a thread-isolated array `row_diffs = np.zeros(N, dtype=np.float64)` where each loop iteration `i` writes `row_diffs[i] = max_row_diff`, followed by `np.max(row_diffs[1:-1])`. Verified against serial baseline with 100% exact match.

### Issue 4: Python subprocess.run on Windows failed to resolve `mpirun`
- **Symptom:** `subprocess.run(["mpirun", ...])` raised `FileNotFoundError: [WinError 2] The system cannot find the file specified`.
- **Cause:** On Windows, `mpirun` is a batch script wrapper (`mpirun.bat` / `mpirun.cmd`), which standard Windows API `CreateProcess` does not execute directly unless invoked through `cmd.exe` or resolved to the underlying binary `mpiexec.exe`.
- **Fix:** Created helper `get_mpirun_cmd()` that resolves `mpiexec.exe` from `MSMPI_BIN` / `C:\Program Files\Microsoft MPI\Bin` and passes `sys.executable` to guarantee consistent environment inheritance.

### Issue 5: Import error in non-blocking overlap runner
- **Symptom:** `ImportError: cannot import name 'run_mpi_overlap_solver' from 'heat_mpi'`.
- **Cause:** The overlap solver function lives in `src/heat_mpi_overlap.py`, but the runner script initially attempted to import it from `heat_mpi.py`.
- **Fix:** Updated the script name resolution in `scripts/run_mpi_grid.py` to dynamically target `heat_mpi_overlap.py`.

### Issue 6: Path resolution across relative project directories
- **Symptom:** Scripts failed when invoked from parent working directory with `FileNotFoundError: results/raw/pingpong_local.csv`.
- **Cause:** Hardcoded relative paths assumed the execution directory was `Heat_Diffusion/`.
- **Fix:** Refactored all helper scripts (`fit_cost_model.py`, `generate_plots.py`, `make_report_assets.py`, `animate.py`) to resolve paths using `PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))`.

### Issue 7: Unescaped underscores in LaTeX table generation
- **Symptom:** `pdflatex` failed with `Missing $ inserted` when compiling `table_correctness.tex`.
- **Cause:** String values containing raw underscores (`mpi_overlap`, `fixed_steps`) were written directly into LaTeX tabular rows without escaping `_` as `\_`.
- **Fix:** Added `.replace("_", "\\_")` escaping across all string columns in `scripts/make_report_assets.py`.

### Issue 8: Report main body page budget enforcement
- **Symptom:** Initial LaTeX compilation produced 13 pages with the main body spanning 9 pages, exceeding the 6-page main body constraint.
- **Cause:** Large 23-row tables and multiple secondary figures were placed in the main body.
- **Fix:** Selected 10 key representative tests for the main body correctness table, placed secondary figures (`seam_demo.png`, `convergence_curves.png`, `blocking_vs_overlap.png`, `cost_model_validation.png`) and full raw tables in `\appendix`, bringing the main body to exactly 6 pages with the appendix beginning on Page 7.

### Issue 9: Multiple Python environments on PATH silently broke the MPI runners
- **Symptom:** `scripts/run_mpi_grid.py --quick` failed on every configuration with `ModuleNotFoundError: No module named 'psutil'` in a shell where the first `python` on PATH is `fractal-hpc\3.13\Scripts\python.exe` (an unrelated environment with unpinned numpy 2.5.4 / numba 0.68.0 and no psutil). The same session's `tests/test_correctness.py` passed, hiding the problem.
- **Cause:** `heat_mpi.py`, `heat_mpi_overlap.py` and `heat_threads.py` imported `psutil` unconditionally inside their runner `main()` to flag oversubscription, while `test_correctness.py` calls the solver functions directly and therefore never touched that import. `run_mpi_grid.py` passes `sys.executable` to `mpiexec`, so whichever interpreter launched the runner was inherited by every rank.
- **Fix (two parts):** (1) added `get_cpu_counts()` to `src/heat_common.py` with an `os.cpu_count()` fallback when psutil is missing, and switched all three runner mains to it — verified identical checksums with and without psutil installed; (2) documented in `README.md` that scripts must run with the interpreter where `requirements.txt` is installed (the pinned env: numpy 2.4.2, numba 0.68.1, mpi4py 4.1.2, psutil 7.2.2) and how to check this in one command.

### Issue 10: Averaging instead of median aggregation, and stale numbers in the docs
- **Symptom:** `make_report_assets.py`, `generate_plots.py` and `fit_cost_model.py` aggregated repetitions with the mean, while instructions.md §3.3 requires the median; the report methodology sentence even contradicted the limitations section ("mean values are reported" vs "median aggregation was utilized"). Some short sub-second runs had large rep-to-rep spread (up to 57.2% on N=512, p=8), so the mean was fragile. Separately, `notes.md` §8 and `PROGRESS.md` still cited ping-pong numbers (α = 2.59 µs, β = 7997 MB/s) from an earlier measurement, contradicting `results/raw/pingpong_local.csv` (α = 1.92 µs, β = 7092.5 MB/s) and the generated macros.
- **Cause:** Aggregation statistic was chosen ad hoc during development; older numbers survived in prose files after the data was regenerated.
- **Fix:** Switched all three scripts to median aggregation and made the serial baseline the median of its repetitions (it previously used an arbitrary last-repetition value); added min/max-spread macros (`timing-spread-*`) from the raw CSV; rewrote every prose number in the report, notes and PROGRESS to keys produced by `make_report_assets.py`; replaced hand-typed result numbers in `main.tex` with `\R{...}` macros wherever possible.
