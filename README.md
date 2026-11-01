# High-Performance Computing (HPC) Projects

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MPI](https://img.shields.io/badge/MPI-Microsoft%20MPI%20%2F%20mpi4py-green.svg)](https://mpi4py.readthedocs.io/)
[![Numba OpenMP](https://img.shields.io/badge/Numba-OpenMP%20Threads-orange.svg)](https://numba.pydata.org/)
[![Institution](https://img.shields.io/badge/USTO--MB-Intensive%20Computing-red.svg)](https://www.univ-usto.dz/)

This repository serves as the central hub for the laboratory mini-projects and benchmark suites developed for the **Intensive Computing (Calcul Intensif)** curriculum (5th Year Engineer, Data Science) at the **University of Science and Technology of Oran - Mohamed Boudiaf (USTO-MB)** during the 2026–2027 academic year.

- **Authors:** Bellatreche Mohamed Amine, Rezaiguia Soltane
- **Supervisor:** Prof. Abdelkader Dairi

---

## 📁 Repository Overview & Project Catalog

Each project is self-contained in its dedicated directory with its own source code, test suites, benchmark data, visualizations, and documentation:

```text
HPC-Projects/
├── Heat_Diffusion/                      # Project 1: Parallel 2D Heat Diffusion
│   ├── src/                             # Core solvers (NumPy, Numba, MPI, Non-blocking Overlap)
│   ├── tests/                           # Correctness test suite (66 configurations, bitwise parity)
│   ├── scripts/                         # Benchmark automation, Hockney cost model, plots
│   ├── results/                         # Empirical CSVs & 41-frame animated diffusion GIF
│   ├── figures/                         # 9 publication PNG figures + progression + seam demo
│   ├── report/                          # LaTeX report source, tables, and macros
│   ├── Heat_Diffusion_Report.pdf        # 10-page compiled PDF report (Publication-ready)
│   ├── README.md                        # Project documentation & execution guide
│   └── notes.md                         # Presenter defense notes (30 defense Q&As)
│
├── matrixmultiplication/                # Project 2: Parallel Matrix Multiplication
│   └── Projet-01-Matrix-Multiplication/
│       ├── src/                         # Matrix kernels (Serial, Threads, MPI)
│       ├── data/                        # Benchmark execution logs and data
│       ├── plots/                       # Performance scaling plots
│       ├── REPORT.docx                  # Full project report document
│       ├── REPORT.md                    # Markdown report summary
│       ├── notes.md                     # Presentation notes & defense Q&As
│       └── README.md                    # Setup and execution guide
│
└── Fractal_Zoom/                        # Project 3: Parallel Fractal Zoom (Mandelbrot)
    ├── fractal.py                       # Mandelbrot computation and rendering
    ├── fractal_kernels.py               # Serial and multithreaded prange Numba kernels
    ├── benchmark.py                     # Thread scaling benchmark harness
    ├── Fractal_Zoom_Mini_Project_Proposal.docx
    ├── learning-log.md                  # Milestone development log
    └── README.md                        # Detailed project guide & roadmap
```

---

## 🔬 Project 1: Parallel 2D Heat Diffusion

🔗 **Directory:** [`Heat_Diffusion/`](./Heat_Diffusion/)  
📄 **Report:** [`Heat_Diffusion_Report.pdf`](./Heat_Diffusion/Heat_Diffusion_Report.pdf)  
📖 **Defense Guide:** [`Heat_Diffusion/notes.md`](./Heat_Diffusion/notes.md)

Solves the 2D steady-state heat equation ($\nabla^2 u = 0$) using the classical Jacobi relaxation 5-point stencil on an $N \times N$ discrete grid with asymmetric Dirichlet boundary conditions ($100^\circ\text{C}$ top, $60^\circ\text{C}$ left, $0^\circ\text{C}$ right and bottom).

### Highlights
- **Paradigms Evaluated:**
  - **Version 0a:** Vectorized pure NumPy 2D array slicing reference.
  - **Version 0b:** Numba JIT-compiled serial loop ($10.6\times$ faster than NumPy).
  - **Version 1:** Shared-memory multithreading via Numba OpenMP `prange` ($4.00\times$ speedup at 4 physical cores).
  - **Version 2–4:** Distributed-memory MPI solver using 1D row decomposition and atomic, deadlock-free `MPI_Sendrecv` halo exchanges ($2.75\times$ peak speedup on local host).
  - **Version Extra:** Non-blocking MPI overlap hiding boundary communication behind interior computation (`MPI_Isend` / `MPI_Irecv`).
- **Microbenchmarking (E5):** Point-to-point ping-pong latency ($\alpha = 1.92\,\mu\text{s}$) and bandwidth ($\beta = 7.09\,\text{GB/s}$) across 21 message sizes (8 B to 8 MB).
- **Hockney Cost Model (E6):** Parametric analytical model ($\alpha = 5.21\,\mu\text{s}, \beta = 5.78\,\text{GB/s}, t_c = 1.23\,\text{ns/cell}$) predicting communication share vs grid size.
- **Convergence Optimization (E4):** Demonstrates that checking convergence every $k=50$ steps slashes `MPI_Allreduce` global barrier overhead from $60.9\%$ down to $3.8\%$.
- **Correctness (E1):** 66 test configurations passing with exact bitwise parity ($\max |u - u_{\text{ref}}| = 0.00\times 10^0$).

---

## ⚡ Project 2: Parallel Matrix Multiplication

🔗 **Directory:** [`matrixmultiplication/Projet-01-Matrix-Multiplication/`](./matrixmultiplication/Projet-01-Matrix-Multiplication/)  
📄 **Report:** [`REPORT.docx`](./matrixmultiplication/Projet-01-Matrix-Multiplication/REPORT.docx) / [`REPORT.md`](./matrixmultiplication/Projet-01-Matrix-Multiplication/REPORT.md)  
📖 **Guide:** [`README.md`](./matrixmultiplication/Projet-01-Matrix-Multiplication/README.md)

Empirical evaluation of dense matrix multiplication ($C = A \times B$) exploring cache hierarchy, algorithmic blocking, and parallel scaling:
- Serial NumPy / BLAS baselines.
- Multithreaded OpenMP implementations with loop tiling / cache blocking.
- Distributed memory MPI matrix distribution strategies.
- Profiling of cache misses, memory bandwidth, and core scaling efficiency.

---

## 🌀 Project 3: Parallel Fractal Zoom (Mandelbrot)

🔗 **Directory:** [`Fractal_Zoom/`](./Fractal_Zoom/)  
📖 **Guide:** [`Fractal_Zoom/README.md`](./Fractal_Zoom/README.md)

Explores shared-memory parallelism and load-balancing for dynamic escape-time Mandelbrot set rendering:
- Pure NumPy reference implementation vs. Numba JIT serial kernel.
- Shared-memory parallelization using Numba OpenMP `prange` across independent scanlines.
- Automated numerical self-checks ensuring exact array equivalence.
- Scalability benchmarking harness (`benchmark.py`) exploring thread count scaling and parallel efficiency.

---

## 🛠️ Global Prerequisites & Setup

### Environment Requirements
- **OS:** Windows 11 / Linux
- **Python:** 3.12+ (AMD64)
- **MPI:** Microsoft MPI (Windows) or Open MPI / MPICH (Linux) + `mpi4py`
- **Compiler:** Numba with OpenMP threading layer

### Installation
```bash
# Clone the complete repository
git clone https://github.com/amine-dubs/HPC-Projects.git
cd HPC-Projects

# Set up Heat Diffusion environment
cd Heat_Diffusion
pip install -r requirements.txt
```

### Running Quick Verification
```powershell
# From Heat_Diffusion directory:
.\run_all.ps1 quick
```

---

## 📜 Academic Integrity

All benchmark numbers, plots, and figures hosted in this repository reflect empirical measurements obtained directly on physical host hardware with strict mathematical traceability (Rule 1). No artificial or placeholder data is used.
