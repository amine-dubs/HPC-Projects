# Parallel 2D Heat Diffusion in Python: HPC Mini-Project (Phase 1: Local)

**Course:** Intensive Computing (5th year ING, Data Science, USTO-MB)  
**Authors:** Bellatreche Mohamed Amine, Rezaiguia Soltane  
**Supervisor:** Prof. Dairi Abdelkader  
**Academic Year:** 2026–2027  

---

## 1. Overview

This project implements and evaluates high-performance parallel solvers for the 2D steady-state heat diffusion equation ($\nabla^2 u = 0$) using the five-point Jacobi relaxation stencil on an $N \times N$ discrete grid. 

In **Phase 1 (Local Execution)**, all parallel paradigms run natively on a single Windows laptop host (AMD Ryzen 5 5500U, 6 physical cores / 12 logical threads):
1. **Version 0a (NumPy Reference):** Vectorized array-slice ground truth.
2. **Version 0b (Numba Serial):** JIT-compiled serial baseline (`@njit(cache=True)`).
3. **Version 1 (Numba Multithreading):** Shared-memory parallelism using OpenMP `prange` with GIL release.
4. **Version 2–4 (Distributed MPI):** 1D row domain decomposition with `MPI_Sendrecv` halo exchanges via Microsoft MPI (`mpi4py`).
5. **Version Extra (Non-Blocking MPI Overlap):** Asynchronous halo communication (`MPI_Isend`/`MPI_Irecv`) overlapped with interior row computations.

---

## 2. Prerequisites & Environment Setup

### Hardware & Operating System
- **OS:** Windows 10/11 (64-bit)
- **CPU:** Multicore x86_64 processor (tested on AMD Ryzen 5 5500U, 6 cores / 12 threads)
- **RAM:** Minimum 8 GB (16 GB recommended for $N=2048$)

### Software Requirements
- **Python 3.12+**
- **Microsoft MPI v10.1+** (Download from [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=100593) - install both `msmpisetup.exe` and `msmpisdk.msi`)
- **MiKTeX / TeX Live** (with `pdflatex` for compiling the PDF report)

> **Interpreter check (important on machines with several Python installs):** run
> `python -c "import psutil, numba, mpi4py; print(numba.__version__)"`
> If this fails, or prints an unpinned NumPy/Numba version, invoke every command below with the absolute path of the interpreter where `pip install -r requirements.txt` was executed (here: the Python 3.12 system install, not a newer side environment).

### Python Installation
```bash
# Clone repository and enter directory
cd Heat_Diffusion

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 3. Quick Start & Correctness Smoke Test

Run the full automated test suite verifying mathematical parity against the pure NumPy reference across all grid sizes and rank configurations ($max\,|u - u_{ref}| = 0.00e+00$):

```powershell
python tests/test_correctness.py
```

Run a quick 5-second smoke test of all parallel implementations:
```powershell
python scripts/run_mpi_grid.py --quick
python scripts/run_convergence.py --quick
```

---

## 4. Reproducing All Experiments

Each experiment can be run independently using the commands below:

| Experiment | Purpose & Configuration | Command | Approx Runtime | Output File |
|:---|:---|:---|:---:|:---|
| **E0: Environment** | Capture hardware and software metadata | `python scripts/collect_env.py` | < 1 s | `results/environment.json` |
| **E1: Correctness** | Verify numerical accuracy against NumPy | `python tests/test_correctness.py` | ~1-2 min (dominated by mpirun process startup) | `results/correctness.csv` |
| **E5: Ping-Pong** | Measure point-to-point latency & bandwidth | `mpirun -n 2 python src/pingpong.py` | ~10 s | `results/raw/pingpong_local.csv` |
| **E2: Threads** | Numba OpenMP scaling ($t \in \{1, 2, 4, 8, 12\}$) | `python scripts/run_threads.py` | ~45 s | `results/raw/scaling_local.csv` |
| **E3 & E8: MPI** | Blocking & non-blocking overlap scaling ($p \in \{1, 2, 4, 8\}$) | `python scripts/run_mpi_grid.py` | ~4 min | `results/raw/scaling_local.csv` |
| **E4: Convergence**| History tracking & $k$-sweep ($k \in \{1, 10, 50, 200\}$) | `python scripts/run_convergence.py` | ~15 s | `results/raw/ksweep.csv`, `convergence_N*.csv` |
| **E6: Cost Model** | Fit $\alpha$, $\beta$, $t_c$ and validate communication share | `python scripts/fit_cost_model.py` | < 1 s | `results/cost_model.csv` |
| **E9: Visualization**| Generate animated GIF, progression & seam demo | `mpirun -n 4 python src/animate.py --N 256 --steps 2000` | ~15 s | `results/animation/heat_diffusion.gif`, `figures/` |
| **E9: Seam Demo** | Illustrate boundary seam when halos are skipped | `mpirun -n 4 python src/animate.py --N 256 --steps 200 --seam-demo` | ~5 s | `figures/seam_demo.png` |
| **Asset Generation**| Build LaTeX tables, macros, and plots | `python scripts/generate_plots.py`<br>`python scripts/make_report_assets.py` | ~5 s | `report/generated/`, `figures/` |
| **Compile Report** | Build final PDF report with pdflatex | `cd report && pdflatex main.tex && pdflatex main.tex` | ~5 s | `report/main.pdf` -> `Heat_Diffusion_Report.pdf` |

---

## 5. Automated Pipeline Runners

We provide one-command runners for both PowerShell and Bash that automate every phase of the project:

```powershell
# Run quick smoke test
.\run_all.ps1 quick

# Run full correctness test suite
.\run_all.ps1 test

# Run scaling benchmarks (Threads E2, MPI E3, Overlap E8)
.\run_all.ps1 scaling

# Run convergence and k-sweep study (E4)
.\run_all.ps1 convergence

# Run ping-pong microbenchmark (E5)
.\run_all.ps1 pingpong

# Generate animations and static figures (E9)
.\run_all.ps1 animation

# Fit cost model, generate assets, and compile LaTeX report
.\run_all.ps1 report

# Run the complete end-to-end pipeline
.\run_all.ps1 all
```

*(On Linux / macOS / WSL, substitute `./run_all.sh <target>`.)*

---

## 6. Troubleshooting (Known Issues & Solutions)

All troubleshooting entries below are derived from real problems encountered during development and logged in `logs/issues.md`:

1. **`mpirun` not recognized in PowerShell:**
   - *Symptom:* `mpirun` returns a command-not-found error on Windows.
   - *Cause:* Microsoft MPI installs `mpiexec.exe` to `C:\Program Files\Microsoft MPI\Bin`.
   - *Solution:* Run via `mpiexec.exe` directly, or use the `mpirun.ps1` wrapper provided in the environment.
2. **Subprocess `CreateProcess` WinError 2 on `mpirun`:**
   - *Symptom:* `subprocess.run(["mpirun", ...])` fails with WinError 2.
   - *Cause:* Batch files (`mpirun.cmd`) cannot be directly executed by `CreateProcess` without `shell=True`.
   - *Solution:* Our runners implement `get_mpirun_cmd()` which resolves `mpiexec.exe` from `MSMPI_BIN` and passes `sys.executable`.
3. **Numba `prange` Reduction Silent Failure:**
   - *Symptom:* Computing conditional scalar maximums inside `@njit(parallel=True)` returns `0.0`.
   - *Cause:* Numba only synthesizes OpenMP reductions for in-place arithmetic operators (`+=`, `*=`); conditional maximum updates suffer thread-local scoping issues.
   - *Solution:* Implemented an array-buffered reduction `row_diffs[i] = max_row_diff` followed by `np.max(row_diffs[1:-1])`, guaranteeing 100% mathematical parity.
4. **Multi-environment interpreter / missing `psutil` (newest):**
   - *Symptom:* `ModuleNotFoundError: No module named 'psutil'` for every MPI configuration launched through `scripts/run_mpi_grid.py`, while `tests/test_correctness.py` passes.
   - *Cause:* A second Python environment (older/newer or without psutil) appears before the pinned one on PATH. The grid runner inherits that interpreter for all MPI ranks.
   - *Solution:* Run with the interpreter that has `requirements.txt` installed (see the interpreter check in section 2). The runner `main()`s now fall back to `get_cpu_counts()` when psutil is absent, but the pinned versions (numpy 2.4.2 / numba 0.68.1 / mpi4py 4.1.2) must still be used to reproduce the saved numbers.
5. **Thermal Throttling & System Noise:**
   - *Symptom:* Runtime variance on heavy 8-process runs and short sub-second runs.
   - *Cause:* Multicore laptop CPU dynamic frequency scaling under load.
   - *Solution:* All scaling benchmarks execute at least 3 repetitions and report the **median**; the min/max spread of each config is kept per repetition in the raw CSV (median spread 4.2%, worst 57.2% on the shortest oversubscribed runs).
6. **Report LaTeX Compilation Overflow:**
   - *Symptom:* LaTeX main body exceeding the 6-page ceiling.
   - *Solution:* Condensed main body tables to representative subsets and routed supplementary raw tables and secondary diagnostic plots to Appendix A.

---

## 7. Repository Directory Layout

```
Heat_Diffusion/
├── README.md               # User manual, commands, and troubleshooting (this file)
├── notes.md                # Presenter defense notes (16 sections, 30+ Q&As)
├── PROGRESS.md             # Project progress tracker
├── requirements.txt        # Pinned Python package versions
├── run_all.ps1             # PowerShell automated pipeline runner
├── run_all.sh              # Bash automated pipeline runner
├── Heat_Diffusion_Report.pdf # Final compiled 6-page LaTeX report
├── src/
│   ├── heat_common.py      # Shared initialization, boundaries, decomposition math
│   ├── heat_reference.py   # Version 0a: Pure NumPy reference
│   ├── heat_serial.py      # Version 0b: Numba JIT serial baseline
│   ├── heat_threads.py     # Version 1: Numba OpenMP prange multithreading
│   ├── heat_mpi.py         # Version 2-4: Distributed MPI with Sendrecv halos
│   ├── heat_mpi_overlap.py # Version Extra: Non-blocking MPI with Isend/Irecv overlap
│   ├── pingpong.py         # Point-to-point latency & bandwidth benchmark
│   └── animate.py          # Heat animation, field progression, and seam generator
├── tests/
│   └── test_correctness.py # Automated correctness test suite against NumPy
├── scripts/
│   ├── collect_env.py      # Hardware/software environment inspector
│   ├── run_threads.py      # Runner for Experiment E2 thread scaling
│   ├── run_mpi_grid.py     # Runner for Experiments E3 & E8 MPI grid scaling
│   ├── run_mpi_grid.sh     # Shell wrapper for MPI grid scaling
│   ├── run_convergence.py  # Runner for Experiment E4 convergence & k-sweep
│   ├── fit_cost_model.py   # Cost model regression and validation
│   ├── generate_plots.py   # Matplotlib chart generator (9 report figures)
│   └── make_report_assets.py # LaTeX table and macro number generator
├── results/
│   ├── environment.json    # Hardware and software environment details
│   ├── correctness.csv     # Correctness test results (66 test cases)
│   ├── cost_model.csv      # Analytical cost model validation table
│   ├── animation/
│   │   └── heat_diffusion.gif # 41-frame animated temperature evolution
│   └── raw/
│       ├── pingpong_local.csv # Ping-pong latency and bandwidth data
│       ├── scaling_local.csv  # Complete runtime measurements for serial, threads, MPI
│       ├── ksweep.csv         # Convergence check interval overhead data (k=1,10,50,200)
│       └── convergence_N*.csv # Convergence history curves (N=128, 256, 512)
├── figures/                # Publication-quality charts for report
└── report/
    ├── main.tex            # Main LaTeX document
    └── generated/          # Auto-generated LaTeX tables, numbers.tex, and SOURCES.md
```

---

## 8. Phase 2 Roadmap (Distributed Deployments)

In Phase 2, this codebase will be deployed across multi-node distributed environments without code modifications:
- **Version 2:** VirtualBox/VMware virtual machines with Bridged networking (`docs/vm_setup.md`).
- **Version 3:** Docker container clusters using Docker `macvlan` (`docker/Dockerfile`, `docker-compose.yml`).
- **Version 4:** Physical multi-node laptop cluster over a dedicated Gigabit Ethernet switch (`docs/lan_setup.md`, `docs/addressing_table.md`).

