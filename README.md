# High-Performance Computing (HPC) Projects

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MPI](https://img.shields.io/badge/MPI-Microsoft%20MPI%20%2F%20mpi4py-green.svg)](https://mpi4py.readthedocs.io/)
[![Numba](https://img.shields.io/badge/Accelerated-Numba%20OpenMP-orange.svg)](https://numba.pydata.org/)
[![License: Academic](https://img.shields.io/badge/Course-Intensive%20Computing-red.svg)](https://www.univ-usto.dz/)

This repository contains the complete laboratory mini-projects and benchmark suites for the **Intensive Computing (Calcul Intensif)** course (5th Year Engineer, Data Science) at the **University of Science and Technology of Oran - Mohamed Boudiaf (USTO-MB)** for the Academic Year 2026–2027.

- **Authors:** Bellatreche Mohamed Amine, Rezaiguia Soltane
- **Supervisor:** Prof. Abdelkader Dairi

---

## 📁 Repository Structure & Projects

```text
HPC-Projects/
├── Heat_Diffusion/                   # Project: 2D Parallel Heat Diffusion
│   ├── src/                          # Computational solvers (NumPy, Numba, MPI, Overlap)
│   ├── tests/                        # Systematic correctness suite (66 configurations)
│   ├── scripts/                      # Automated benchmark runners, cost modeling, plots
│   ├── results/                      # Raw benchmark CSVs and animation GIF
│   ├── figures/                      # 9 publication PNG figures + progression + seam demo
│   ├── report/                       # LaTeX report source, tables, and macros
│   ├── Heat_Diffusion_Report.pdf     # 10-page compiled PDF report (Publication-ready)
│   ├── README.md                     # Complete project documentation & guide
│   └── notes.md                      # Presenter notes & 30 defense Q&As
│
├── matrixmultiplication/             # Project: Parallel Matrix Multiplication
│   └── Projet-01-Matrix-Multiplication/
│       ├── src/                      # Matrix kernels (Serial, Threads, MPI)
│       ├── data/                     # Benchmark execution logs and data
│       ├── plots/                    # Performance scaling plots
│       ├── REPORT.docx               # Full project report document
│       ├── REPORT.md                 # Markdown project summary
│       ├── notes.md                  # Presentation notes & defense Q&As
│       └── README.md                 # Setup and execution guide
│
├── fractal_kernels.py                # Project: Parallel Fractal Zoom (Mandelbrot)
├── fractal.py                        # Mandelbrot computation and rendering
├── benchmark.py                      # Thread scaling benchmark harness
├── Fractal_Zoom_Mini_Project_Proposal.docx
└── fractal_README.md                 # Detailed guide for Fractal Zoom
```

---

## 🔬 Project 1: Parallel 2D Heat Diffusion

Located in [`Heat_Diffusion/`](./Heat_Diffusion/).

Solves the 2D steady-state heat equation ($\nabla^2 u = 0$) using the classical Jacobi relaxation 5-point stencil on an $N \times N$ discrete grid with asymmetric Dirichlet boundary conditions ($100^\circ\text{C}$ top, $60^\circ\text{C}$ left, $0^\circ\text{C}$ right and bottom).

### Key Features
- **Architectures Evaluated:**
  - **Version 0a:** Pure NumPy 2D vectorized slicing reference.
  - **Version 0b:** Numba JIT-compiled serial loop ($10.6\times$ faster than NumPy).
  - **Version 1:** Shared-memory multithreading via Numba OpenMP `prange` ($4.00\times$ speedup at 4 physical cores).
  - **Version 2–4:** Distributed-memory MPI solver using 1D row decomposition and atomic, deadlock-free `MPI_Sendrecv` halo exchanges ($2.75\times$ peak speedup on local host).
  - **Version Extra:** Non-blocking MPI overlap hiding boundary communication behind interior computation (`MPI_Isend` / `MPI_Irecv`).
- **Ping-Pong Microbenchmark (E5):** Point-to-point latency ($\alpha = 1.92\,\mu\text{s}$) and peak bandwidth ($\beta = 7.09\,\text{GB/s}$) across 21 message sizes (8 B to 8 MB).
- **Hockney Cost Model (E6):** Parametric analytical model ($\alpha = 5.21\,\mu\text{s}, \beta = 5.78\,\text{GB/s}, t_c = 1.23\,\text{ns/cell}$) predicting communication share vs grid size.
- **Convergence & $k$-Sweep Study (E4):** Demonstrates that checking convergence every $k=50$ steps reduces `MPI_Allreduce` global barrier overhead from $60.9\%$ down to $3.8\%$.
- **Correctness Suite (E1):** 66 test configurations passing with exact bitwise parity ($\max |u - u_{\text{ref}}| = 0.00\times 10^0$) and identical stopping iterations.
- **Deliverables:**
  - Formal 10-page report: [`Heat_Diffusion/Heat_Diffusion_Report.pdf`](./Heat_Diffusion/Heat_Diffusion_Report.pdf)
  - Interactive Presenter & Defense Guide: [`Heat_Diffusion/notes.md`](./Heat_Diffusion/notes.md) (with 30 defense Q&As and 10-minute live demo script)
  - One-click pipeline runners: `.\run_all.ps1` and `./run_all.sh`

---

## ⚡ Project 2: Parallel Matrix Multiplication

Located in [`matrixmultiplication/Projet-01-Matrix-Multiplication/`](./matrixmultiplication/Projet-01-Matrix-Multiplication/).

Benchmarking dense matrix multiplication ($C = A \times B$) across CPU architectures and parallel paradigms:
- Serial NumPy / BLAS baselines.
- Shared-memory OpenMP / Numba multithreading.
- Distributed memory MPI matrix distribution.
- Memory hierarchy analysis and cache locality optimizations.
- Full experimental report: [`REPORT.docx`](./matrixmultiplication/Projet-01-Matrix-Multiplication/REPORT.docx) / [`REPORT.md`](./matrixmultiplication/Projet-01-Matrix-Multiplication/REPORT.md).

---

## 🌀 Project 3: Parallel Fractal Zoom (Mandelbrot)

Located in the root directory ([`fractal_README.md`](./fractal_README.md)).

Generates high-resolution Mandelbrot set fractals with escape-time iteration algorithms:
- Comparison between pure NumPy reference, Numba serial, and Numba multithreaded `prange` kernels.
- Automated correctness self-checks.
- Performance scaling and runtime benchmarking harness (`benchmark.py`).

---

## 🚀 Quick Start

### 1. Requirements & Setup
- Windows 11 / Linux
- Python 3.12+ with `numpy`, `numba`, `scipy`, `matplotlib`, and `imageio`
- Microsoft MPI (Windows) or Open MPI / MPICH (Linux)
- `mpi4py`

```bash
# Clone the repository
git clone https://github.com/amine-dubs/HPC-Projects.git
cd HPC-Projects

# Install dependencies for Heat Diffusion
cd Heat_Diffusion
pip install -r requirements.txt
```

### 2. Run Smoke Tests & Verification
```powershell
# From Heat_Diffusion directory:
.\run_all.ps1 quick
```

### 3. Generate Reports & Visualizations
```powershell
# Run cost model, generate all publication figures, and compile LaTeX report:
.\run_all.ps1 report
```

---

## 📜 Academic Integrity & Citation

This project is submitted as part of the **Intensive Computing** curriculum at **USTO-MB**. All benchmarks are empirical measurements obtained directly on host hardware with strict mathematical traceability (Rule 1).
