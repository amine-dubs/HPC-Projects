# Parallel Fractal Zoom Movie in Python

Mini-project for the course **Intensive Computing** at USTO-MB (University of Science and Technology of Oran – Mohamed Boudiaf).  
Authors: **Bellatreche Mohamed Amine**, **Rezaiguia Soltane**  
Supervisor: **Prof. Dairi Abdelkader**  
Academic Year: **2026–2027, 5th year ING, Data Science**

---

## 1. Overview
This repository contains a high-performance Mandelbrot zoom movie engine written in Python. It investigates spatial and temporal load balancing across shared-memory threads (Numba + OpenMP) and distributed-memory message passing (MPI via `mpi4py`). Every parallel version is verified for bitwise identity against a serial baseline.

### Folder Layout
```
Fractal_Zoom/
├── README.md               # User guide and reproduction commands
├── notes.md                # Consultation defense guide and professor Q&A
├── instructions.md         # Course project specification
├── dev_log.md              # Real-time development diary and troubleshooting log
├── config.json             # Global zoom, geometry, and runtime parameters
├── requirements.txt        # Exact pinned Python package dependencies
├── docs/                   # Course proposal document
├── src/
│   ├── fractal_core.py     # Shared Numba @njit kernel and geometry helpers
│   ├── serial.py           # V0 serial baseline renderer
│   ├── threads.py          # V1 shared-memory parallel engine (Variants A and B)
│   ├── mpi_s1_rows.py      # MPI Strategy 1: Static row decomposition
│   ├── mpi_s2_tiles.py     # MPI Strategy 2: Master-worker dynamic tiles
│   ├── mpi_s3_frames.py    # MPI Strategy 3: Embarrassingly parallel frames
│   ├── make_movie.py       # Production movie and video assembly (E10)
│   └── pingpong.py         # Inter-process latency and bandwidth benchmark
├── tests/
│   └── run_correctness.py  # Validation test suite (orientation, coverage, bitwise hashes)
├── scripts/
│   ├── env_info.py         # System environment discovery
│   ├── mpi_launch.py       # Cross-platform MPI launcher wrapper
│   ├── warmup.py           # Kernel pre-compilation script
│   ├── compute_cost_profiles.py # E6 iteration cost analysis
│   ├── run_benchmarks.py   # Benchmark execution harness
│   ├── make_plots.py       # Publication-ready figure generator
│   ├── make_tables.py      # LaTeX table and macro generator
│   └── audit_numbers.py    # Rule 1 number-traceability audit for the report
├── results/                # Raw benchmark CSVs, logs, and reference arrays
├── figures/                # Rendered PNG/PDF figures
├── output/                 # Final zoom frames, zoom.gif, and zoom.mp4
└── report/
    ├── main.tex            # Full academic LaTeX report
    ├── main.pdf            # Compiled report document
    └── generated/          # Auto-generated tables and number macros
```

---

## 2. Requirements & Environment Setup

### System Prerequisites
- **Python:** 3.12 (or any Numba-supported version: 3.10–3.12).
- **MPI Runtime:** Microsoft MPI (MS-MPI v10+) on Windows, or Open MPI (`libopenmpi-dev`) on Linux/WSL2.
- **LaTeX Compiler:** MiKTeX (Windows) or TeX Live (Linux) providing `pdflatex`.
- **FFmpeg:** Handled automatically via `imageio-ffmpeg`.

### Windows Setup
```powershell
# 1. Install Microsoft MPI Runtime and SDK from Microsoft
# (Installs to C:\Program Files\Microsoft MPI\Bin)

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pre-warm Numba kernel
python scripts/warmup.py
```

### Linux / WSL2 Setup
```bash
# 1. Install system Open MPI and LaTeX
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev texlive-latex-base texlive-latex-extra

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies and warm up
pip install -r requirements.txt
python scripts/warmup.py
```

---

## 3. Quick Test (Approximately 2 Minutes)
To verify your installation, coordinate orientation, and bitwise correctness:
```powershell
python tests/run_correctness.py
```
This runs orientation checks, partition coverage tests, reference array comparisons, parallel identity checks, and live MPI subprocess executions (S1, S2, S3) that render full 120-frame workloads. All tests must report `[PASS]`.

---

## 4. How to Run Each Version

### V0: Serial Baseline
```powershell
# Render all frames serially (optionally save reference hashes)
python src/serial.py --save-reference --rep 1
```

### V1: Shared-Memory Threads (Numba + OpenMP)
```powershell
# Variant A: Parallel rows (p=4 threads)
python src/threads.py --variant A -p 4 --rep 1

# Variant B: Shuffled tiles (p=4 threads, tile size 20 px)
python src/threads.py --variant B -p 4 --tile 20 --rep 1
```

### MPI S1: Static Rows
```powershell
# Run across 4 ranks (Windows MS-MPI)
mpiexec -n 4 python src/mpi_s1_rows.py --rep 1

# Optional cluster hostfile syntax (Linux/Open MPI)
mpirun -n 4 --hostfile hosts.txt python src/mpi_s1_rows.py --rep 1
```

### MPI S2: Master-Worker Tiles
```powershell
# Run with 1 master + 3 workers (p=4) and 20px tiles
mpiexec -n 4 python src/mpi_s2_tiles.py --tile 20 --rep 1
```

### MPI S3: Whole Frames per Rank
```powershell
# Block assignment (contiguous frame slices)
mpiexec -n 4 python src/mpi_s3_frames.py --assignment block --rep 1

# Cyclic assignment (interleaved frames, recommended for load balance)
mpiexec -n 4 python src/mpi_s3_frames.py --assignment cyclic --rep 1
```

### MPI Ping-Pong Benchmark
```powershell
# Measure local shared-memory latency and bandwidth
mpiexec -n 2 python src/pingpong.py
```

---

## 5. Benchmarking & Full Reproduction

### Run Complete Benchmark Suite (E2–E5)
```powershell
python scripts/run_benchmarks.py
```
- Appends every result immediately to `results/` CSV files.
- Shuffles configurations per repetition to prevent thermal bias.
- Supports resuming: automatically skips completed configurations.
- Execution logs are saved to `results/logs/`.

### Generate Plots, Tables, and Cost Profiles
```powershell
# 1. Compute iteration cost profiles (E6)
python scripts/compute_cost_profiles.py

# 2. Generate all publication figures (Figures 1 to 8)
python scripts/make_plots.py

# 3. Generate LaTeX tables and number macros
python scripts/make_tables.py
```
The mapping from each PNG file in `figures/` to its Figure number and section in `report/main.pdf` is documented in `figures/README.md`.

### Generate Production Movie (E10)
```powershell
python src/make_movie.py -p 6 --tile 20
```
Outputs:
- Individual PNG frames: `output/frames/frame_000.png` $\dots$ `frame_119.png`
- Animated GIF: `output/zoom.gif` (24 fps target; 40 ms real frame delay)
- MP4 Video: `output/zoom.mp4` (H.264, 24 fps, exact $640 \times 360$ resolution)

### Compile the LaTeX Report
```powershell
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cd ..
```
Produces `report/main.pdf`.

---

## 6. Implementation Status Matrix

| Version | Target Architecture | Implementation Model | Status |
| :--- | :--- | :--- | :--- |
| **V0** | Local Host | Serial Python + Numba `@njit` | **EXECUTED** |
| **V1** | Local Host | Threads (Numba `prange` + OpenMP) | **EXECUTED** |
| **MPI Local** | Local Host | Distributed ranks (MS-MPI, S1, S2, S3) | **EXECUTED** |
| **V2** | Bridged VMs | MPI over virtual network | *DEFERRED* |
| **V3** | Docker Containers | MPI over container network / macvlan | *DEFERRED* |
| **V4** | Physical LAN | MPI over lab Ethernet switch | *DEFERRED* |

> **Note on Next Phase:** Virtual Machine (V2), Docker container (V3), and physical LAN (V4) deployments are intentionally deferred to the subsequent project phase. The existing codebase is fully prepared for multi-node execution via the `--hostfile` argument in `src/mpi_*.py` and `scripts/mpi_launch.py`.

---

## 7. Troubleshooting & Lessons Learned (from `dev_log.md`)
1. **`mpiexec` not recognized in Windows PowerShell:**  
   Microsoft MPI installer places binaries in `C:\Program Files\Microsoft MPI\Bin`, which may not be in user PATH. `scripts/mpi_launch.py` and `scripts/env_info.py` auto-detect and append this path dynamically.
2. **S2 Dynamic Work Queue Tag Collision:**  
   Master polling with `MPI.ANY_TAG` caused premature receipt of worker termination dictionaries (`tag=100`), causing a ValueError. Resolved by matching work returns explicitly with `tag=TAG_WORK`.
3. **Numba `set_num_threads` bounds:**  
   Setting `NUMBA_NUM_THREADS=1` prior to importing Numba restricts the maximum thread pool capacity. Thread limits must only be constrained for single-threaded MPI workers.
