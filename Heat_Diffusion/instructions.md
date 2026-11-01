# instructions.md — Parallel 2D Heat Diffusion mini-project

You are a coding agent working inside the user's workspace. Your job is to build, run, measure and document the whole mini-project described below, then write the LaTeX report, the README and the presentation notes. The project proposal (`Heat_Diffusion_Mini_Project_Proposal.docx`) is the source of truth for the task. This file adds the working rules and the details you need.

The authors are two 5th-year Data Science engineering students. The report will be read by their professor, so it has to read like their work, be correct, and be honest.

---

## 0. Non-negotiable rules (read twice)

1. **Never invent or estimate numbers. Every figure in the report must come from saved output files.** This covers timings, speedups, efficiencies, iteration counts, latencies, bandwidths, error values, machine specs, everything. If a number is not in a file under `results/`, it does not go in the report or in `notes.md`.
2. **Run everything for real.** Do not describe an experiment you did not execute. Do not write "we expect" in place of a measurement. If something could not be run (no Docker, no second machine, no VM), say so plainly in the report and in `notes.md`.
3. **Honest limitations.** The report must contain a section on what did not work, what is uncertain, and what was not done (details in section 10.4). Do not hide problems to make results look cleaner.
4. **Compile check.** Run `pdflatex` on the report, read the log, fix every error and every undefined reference, and repeat until the PDF builds cleanly (details in section 10.6).
5. **Do not fake the LAN.** Versions 2, 3 and 4 on real VMs, real containers on a LAN and physical machines can only be measured by the students on their own hardware. You prepare the code, scripts and docs for it and you run whatever is possible on this machine. Never fill LAN or VM results with made-up or "simulated" numbers.
6. **Report only what exists.** The report and the notes must describe the code as it actually is, with the parameters actually used.
7. **No made-up references.** Cite only: the MPI Standard (MPI Forum), the Open MPI documentation, the mpi4py documentation, the Numba documentation, and *Using MPI* by Gropp, Lusk and Skjellum. Nothing else.

---

## 1. Working method

- Work in phases (section 11 gives the order). Finish and check one phase before starting the next.
- Keep three living files while you work:
  - `PROGRESS.md`: what is done, what is running, what is next. If the session is interrupted you must be able to resume from it.
  - `logs/issues.md`: every real problem you hit (error message, cause, fix). The "difficulties" paragraph of the report and part of `notes.md` use **only** these entries.
  - `notes.md`: add implementation decisions as you make them (decision, reason, file and function). You complete it at the end.
- Start every experiment with a `--quick` smoke run (small N, few steps, 1 repetition). Only launch the full run once the quick run works. Long runs go in the background with output redirected to `logs/`.
- Save raw results with timestamps. Never overwrite raw data, and never edit a result file by hand.
- If a tool is missing (`mpirun`, `pdflatex`, `docker`): try to install it with the OS package manager. If that needs a password or admin rights you do not have, stop and tell the user exactly what to install. Do not work around it by faking output. Skip the dependent parts and mark them as "not run" everywhere.
- Detect the OS first. If it is Windows, use WSL2 (Ubuntu) for MPI and Docker when available; if it is not available, tell the user.
- You may search the web for error messages.

---

## 2. The problem (fixed by the proposal)

| Item | Value |
|---|---|
| Grid | N x N, boundary cells included. Official sizes N = 512, 1024, 2048. Small sizes (64, 120, 128, 256) are for tests, convergence and animation. N must be divisible by the number of ranks p. |
| Boundaries | top row = 100, left column = 60, right column = 0, bottom row = 0. Fixed (Dirichlet). |
| Corner convention | Follow the starter code: columns override rows. Top-left = 60, top-right = 0, bottom-left = 60, bottom-right = 0. Implement it in **one** shared function used by every version, and write it down in the report. |
| Initial condition | zero interior. Optional hot spot held at 100 (flag `--hotspot`, off by default, used for one animation only). |
| Update | `u_new[i,j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])` |
| Stopping | max change < 1e-4 (convergence runs), or a fixed number of steps (scaling runs: 2000). |
| Precision | float64 everywhere. Do not use `fastmath` and do not reorder the arithmetic, so serial and parallel results can be compared tightly. |
| Correctness | the distributed result must satisfy `np.allclose` against the serial result after the same number of steps. Also save the max absolute difference. |

---

## 3. Versions to implement

| Version | What | Notes |
|---|---|---|
| 0a | `heat_reference.py`: pure NumPy serial solver | Simple and slow. Used only as the independent reference in tests. |
| 0b | `heat_serial.py`: Numba `@njit` serial solver | The timing baseline. Use `cache=True` and warm it up before timing. |
| 1 | `heat_threads.py`: Numba `prange` | Try `NUMBA_THREADING_LAYER=omp`. If it is not available, fall back to `tbb` or `workqueue` and say in the report which layer was really used (check `numba.threading_layer()` after the first parallel call and save it). Run each thread count in a fresh process. |
| 2, 3, 4 | `heat_mpi.py`: one MPI code (mpi4py) | The same file runs on VMs, in Docker and on physical machines. Only the launch changes. |
| Extra | `heat_mpi_overlap.py`: non-blocking `Isend`/`Irecv` that overlaps the halo exchange with the update of inner rows | Do it after everything else works. Compare against blocking `Sendrecv`. 2D block decomposition is NOT required; list it as not done in the report. |

### 3.1 MPI design (follow it)

- 1D row strips. Each rank owns `rows = N // p` rows plus one halo row above and one below, so the local array has shape `(rows + 2, N)`, C-contiguous. Assert `N % p == 0` with a clear error message.
- Each step: two `Sendrecv` calls (send first real row up and receive the bottom halo, send last real row down and receive the top halo). Neighbours of edge ranks are `MPI.PROC_NULL`. Send each edge row as one message.
- Then update the local interior. Global boundary rows (row 0 of rank 0, last row of the last rank) and boundary columns must stay fixed. The cleanest way is to set them once at initialisation in **both** arrays and never update them.
- Convergence: local max change on the **interior real rows only**, then `Allreduce(MAX)` every `k` steps (default `k = 50`, configurable). Never on every step by default.
- Output: on output steps only, rank 0 collects the full grid with `Gather` (or `Gatherv`) and writes frames. This is excluded from the compute timing.
- Command-line options: `--N`, `--steps`, `--tol`, `--check-every`, `--kernel {numpy,numba}`, `--out`, `--save-every`, `--reps`, `--target {local,docker,vm,lan}`, `--hotspot`, `--quick`.
- Kernel choice: implement both a NumPy-slicing kernel (as in the proposal's starter) and a single-threaded Numba kernel. Use the **Numba kernel** for the main scaling experiments so the comparison with versions 0 and 1 is fair. Run the NumPy kernel for at least N = 1024 as an extra comparison, and explain in the report why the results differ.
- For MPI runs set `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` so ranks do not also spawn threads.
- Use `--oversubscribe` (or the equivalent) only when p is larger than the cores available, and mark those runs `oversubscribed=1` in the CSV.

### 3.2 Known problems in the proposal's starter code (check each one, do not copy blindly)

1. `steps` is never defined.
2. Boundaries are applied only to `v` after the first update, so the first step runs with zero boundaries in `u`. Set boundaries in both arrays at the start. (The Numba kernel never writes boundary cells either.)
3. The convergence difference `np.abs(v - u).max()` covers the halo rows too. `v`'s halo rows are stale (only `u` is refreshed), so the value is inconsistent, and it makes the iteration count depend on p. Use the interior rows only.
4. When the loop exits through `break`, the newest field is in `v`. When it ends normally, the newest field is in `u` (after the swap). Keep a clear variable for "current field", otherwise the gathered result can be one step old.
5. Corner values depend on the order of the boundary assignments. Make sure serial and MPI use the same convention.
6. `Sendrecv(u[1], ...)` only works because rows of a C-contiguous array are contiguous. Keep the arrays C-contiguous.
7. If the hot spot is held fixed, it must be re-applied in the right local coordinates on the rank that owns it.

Write in `notes.md` which of these you actually hit and what you changed.

### 3.3 Timing method

- Warm up Numba (compile and run a few steps) before any timing.
- Time only the stepping loop. Use `MPI.Wtime()` (or `time.perf_counter()` for serial and threads). `Barrier` before starting the clock.
- Split MPI time into: `t_compute` (update), `t_halo` (Sendrecv), `t_reduce` (Allreduce), and `t_total`. Plotting and gathering time (`t_plot`) are measured separately and never counted in `t_total`.
- For each phase save the **max and the mean over ranks** (the slowest rank defines the run). Note in the notes that `Sendrecv` time also includes waiting for a slower neighbour.
- Each configuration: at least 3 repetitions (use 5 if cheap). Report the **median**; also store min and max for error bars.
- Speedup `S = T1 / Tp`, efficiency `E = S / p`, where `T1` is the same code with p = 1 (same kernel, same N). Also store the time of the Numba serial baseline as a reference.
- Add a `checksum` (sum of the final field, computed outside the timed region) to every run so you can confirm all runs of the same configuration produce the same answer.

---

## 4. Experiments

Create a CSV schema shared by all runs, in `results/raw/`:

`timestamp, target, version, kernel, N, p, steps, check_every, rep, t_total, t_compute_max, t_compute_mean, t_halo_max, t_halo_mean, t_reduce_max, t_reduce_mean, t_plot, iterations, final_diff, checksum, oversubscribed, threading_layer, notes`

| ID | Experiment | Parameters | Output |
|---|---|---|---|
| E0 | Environment capture | CPU model, physical and logical cores, RAM, OS, Python, numpy, numba, mpi4py, MPI library and version, Docker version, load average before the runs | `results/environment.json` |
| E1 | Correctness tests | serial Numba, threads and MPI against the NumPy reference. N in {64, 128, 120}; N = 120 allows p = 3, 5, 6, so also test non-power-of-two p. p in {1, 2, 4, 8} (plus 3, 5, 6 for N = 120). Fixed 500 steps, and one convergence-mode run where **the stop iteration must be identical** for serial and MPI. With the hot spot on and off. Save max abs difference and the `np.allclose` result. | `results/correctness.csv`, test log |
| E2 | Thread scaling (version 1) | N in {512, 1024, 2048}, threads in {1, 2, 4, 8} (cap at logical cores, and say so), 2000 steps, 3 or more reps | `results/raw/scaling_local.csv` |
| E3 | MPI scaling on this machine | N in {512, 1024, 2048}, p in {1, 2, 4, 8}, 2000 steps, 3 or more reps, Numba kernel; NumPy kernel for N = 1024 | same CSV |
| E4 | Convergence | N in {128, 256, 512} run to tol 1e-4 (cap the iterations and report the cap if hit). Save max change versus iteration (log every 10 iterations). Then a **k-sweep**: N = 256, p = 4, k in {1, 10, 50, 200}: total time, share of `t_reduce`, iteration at which it stopped. | `results/raw/convergence_*.csv`, `results/raw/ksweep.csv` |
| E5 | Ping-pong | `pingpong.py` with 2 ranks, message sizes from 8 bytes to at least 8 MB (powers of two), many repetitions. Latency (half round trip of the smallest messages) and bandwidth (largest messages). Run it in every environment that exists: local shared memory, Docker, VM, LAN. | `results/raw/pingpong_<target>.csv` |
| E6 | Cost model | See 4.1. | `results/cost_model.csv` |
| E7 | Docker | See section 5. Same experiments as E3 where possible, plus ping-pong. | `results/raw/scaling_docker.csv` |
| E8 | Blocking vs non-blocking | N = 1024, 2048, p = 2, 4, 8 | same CSV with `version=mpi_overlap` |
| E9 | Animation and plots | See section 6. | `figures/`, `results/animation/` |
| E10 | VM and LAN | Scripts and docs only, unless real VMs or machines are available here. See section 5. | scripts, `docs/` |

### 4.1 Cost model (derive it, fit it, compare it)

Per step, an interior rank computes about `N^2 / p` cells and exchanges two messages of `8N` bytes (N doubles). Model:

- `T_comp = (N^2 / p) * t_c`, with `t_c` fitted from the p = 1 compute time per cell.
- `T_halo = 2 * (alpha + 8N / beta)`, with `alpha` (latency) and `beta` (bandwidth) fitted from the ping-pong of the same environment.
- Predicted communication share = `T_halo / (T_comp + T_halo)`.

Note for the discussion: the ratio `T_halo / T_comp` equals `2p*alpha / (N^2 t_c) + 16p / (N beta t_c)`. The second (bandwidth) term grows like `p / N`, which is the proposal's rule of thumb; the first (latency) term shrinks like `p / N^2`. Plot predicted versus measured share against N and p, save the fitted `t_c`, `alpha`, `beta` and the resulting table, and say honestly where the model fits and where it does not. If you offer an explanation for a mismatch (cache effects, memory bandwidth, oversubscription, waiting for neighbours), label it as a hypothesis unless you measured it.

---

## 5. Docker, VMs and the physical LAN

**What you run here:**

- `docker/Dockerfile`: one image for every node (identical MPI everywhere), non-root user, `openssh-server`, Open MPI, Python, numpy, numba, mpi4py, matplotlib, imageio.
- `docker/docker-compose.yml`: 4 containers (`node1` to `node4`) on a user-defined bridge network with fixed IPs (for example `172.28.0.11`–`14`), a bind-mounted `results/docker/` output folder, a bind-mounted SSH key folder (a script generates the keys on the host; if bind-mounted key permissions do not work on this OS, bake a throwaway lab key into the image and say so), and the code available at the same path in every container.
- Hostfile with the container IPs and `slots`. Launch from `node1`: `mpirun -n P --hostfile hosts python src/heat_mpi.py ...`.
- A connectivity script: `ping` and `ssh hostname` for every node, then `mpirun hostname` across all nodes.
- If Docker is available, **run it**: connectivity check, E7 and the ping-pong. If Docker is not available, say so and do not report any Docker numbers.
- Known traps: pick the network interface with `--mca btl_tcp_if_include eth0`; run as non-root (or use `--allow-run-as-root` and note it); if you see shared-memory or `ptrace` errors inside containers, search for the Open MPI single-copy setting for your version.

**Important honesty point:** a bridge network on Docker Desktop is **not** the macvlan-on-a-physical-LAN setup the proposal asks for, and all containers share this machine's cores. State this clearly in the report. Also provide `docker/macvlan_setup.sh` and the macvlan shim steps as a **template** with the subnet, gateway and parent interface clearly marked "adapt to your lab". Do not run it unless the OS is Linux and you are really on the target LAN.

**What you only prepare (the students run it on their own hardware):**

- `docs/addressing_table.md`: a table for every node (name, type, IP, role) using the proposal's example subnet `192.168.10.0/24`: physical `.11-.12`, VMs `.21-.24`, Docker `.31-.34`. Mark the values "to confirm in the lab".
- `docs/vm_setup.md`: bridged-mode VMs (explain why NAT mode cannot form a cluster), static IPs, hostnames, passwordless SSH, installing the same MPI and Python environment, cloning VMs so versions match, hostfile, connectivity test. Include the proposal's warnings (wired Ethernet, switch ports accepting several MACs, VMs on one host share its cores).
- `docs/lan_setup.md` for the physical machines (same MPI version on every machine, same code path, SSH keys, firewall).
- `scripts/lan_check.sh <hostfile>`: ping, ssh and `mpirun hostname` for every node.
- `scripts/run_lan.sh <hostfile> <target>`: runs ping-pong and the E3 grid with the **same CSV schema** and `target=vm` or `target=lan`, writing into `results/raw/`. When the students copy their results back, `make_report_assets.py` must pick them up automatically.

In the report, the VM / physical LAN section must show those results **if the CSV files exist** and otherwise state clearly "not measured" (use LaTeX `\IfFileExists`). Never fill it with anything else.

---

## 6. Visualization

- Temperature animation: N = 256, p = 4 (and optionally p = 8), inferno colormap, dashed horizontal lines at the rank boundaries, a title with the step number, one frame every `k` steps (not every step). Rank 0 gathers on output steps only, writes frames with matplotlib (`Agg` backend, `savefig`) and assembles a GIF with `imageio` (MP4 only if ffmpeg is available). Time the plotting separately (`t_plot`). Keep the GIF reasonably small. Save into `results/animation/`.
- Static figures for the report: four frames of the field at different steps (with the rank boundaries), the final field.
- Charts (matplotlib, readable fonts, labelled axes with units, one figure per file, PDF or PNG): convergence curve (max change vs iteration, log scale), thread speedup and efficiency vs threads, MPI speedup and efficiency vs p (mark oversubscribed points), stacked bar of compute / halo / reduce per (N, p), predicted vs measured communication share, ping-pong latency and bandwidth curves, blocking vs overlap, deployment comparison (only for environments that were measured).
- Optional but cheap: a debug demo showing the visible **seam** at a rank boundary when the halo exchange is skipped (add a separate `--debug-skip-halo` flag, never on by default). It is a good figure for the discussion of correctness.

---

## 7. Repository layout

```
heat-diffusion/
├── README.md
├── notes.md
├── instructions.md          (copy of this file)
├── PROGRESS.md
├── requirements.txt         (pinned versions actually installed)
├── Makefile or run_all.sh   (whichever works on this OS; include `quick`, `test`, `scaling`, `convergence`, `pingpong`, `docker`, `animation`, `report`, `all`)
├── src/        heat_common.py  heat_reference.py  heat_serial.py  heat_threads.py
│               heat_mpi.py  heat_mpi_overlap.py  pingpong.py  animate.py
├── tests/      test_correctness.py (+ helper that runs the MPI cases through mpirun)
├── scripts/    collect_env.py  run_threads.py  run_mpi_grid.sh  make_report_assets.py
│               lan_check.sh  run_lan.sh
├── docker/     Dockerfile  docker-compose.yml  hosts  setup_ssh.sh  macvlan_setup.sh
├── docs/       addressing_table.md  vm_setup.md  lan_setup.md
├── results/    raw/  animation/  docker/  environment.json  correctness.csv  cost_model.csv
├── figures/
├── logs/       issues.md  run logs
└── report/     main.tex  generated/  (figures copied or referenced)  main.pdf
```

Shared logic (grid initialisation, boundary function, argument parsing, CSV writer, timing helpers) lives in `heat_common.py`, so no version redefines it.

Code quality matters (it is graded): short docstrings, clear names, comments that say **why**, no dead code, no copy-pasted blocks.

---

## 8. README.md (required content)

Write `README.md` for someone who has never seen the project. It must contain:

1. One-paragraph description and the list of versions.
2. Requirements and installation per OS (Linux, macOS, Windows through WSL2): Python venv, `pip install -r requirements.txt`, Open MPI or MPICH, TeX Live, Docker. Say how to check the install (`mpirun --version`, `python -c "import mpi4py"`, etc.).
3. **Exact commands**, copy-pasteable, for each of these, with what the output means and roughly how long it takes on this machine (from your own logs):
   - quick smoke test
   - correctness tests
   - serial and thread runs (with the `NUMBA_THREADING_LAYER` setting)
   - `mpirun` runs locally (including how to set the number of ranks and when to add `--oversubscribe`)
   - ping-pong
   - convergence study and k-sweep
   - animation
   - Docker: build, bring the cluster up, connectivity check, run, collect results, tear down
   - VM and physical LAN: hostfile, `lan_check.sh`, `run_lan.sh`, and how to bring the results back into `results/raw/`
   - generating tables and figures (`make_report_assets.py`) and compiling the report
4. Where each output file lands.
5. Troubleshooting (the real problems from `logs/issues.md`).
6. What was run in this environment and what was not.

---

## 9. notes.md (required content)

`notes.md` is the presenter's file. The student must be able to read it the evening before the consultation and defend the work. Write it in plain, clear language, in the first person plural ("we"), and ground every statement in the actual code and the actual results. Sections:

1. **The project in two minutes**: what the problem is, what Jacobi / the five-point stencil is, what a halo is, why this is a communication-heavy problem. No jargon without a one-line explanation.
2. **Proposal vs what we did**: a checklist of every objective and deliverable of the proposal with the status (done / partly / not done) and why.
3. **Environment**: machine and software from `environment.json`.
4. **File map**: what each file does and the one function to open first.
5. **Parameters used**: a table (N, p, steps, tolerance, check_every, repetitions, boundary values, kernel, threading layer, MPI flags, oversubscription) with the reason for each value.
6. **Implementation decisions**: a table with columns *decision / why / alternative we did not take / where in the code (file and function)*. Include at least: Numba instead of Python threads (GIL), `prange`, row strips instead of 2D blocks, `Sendrecv` + `PROC_NULL`, one message per row, `Allreduce` every k steps, interior-only convergence check, boundaries set once in both arrays, corner convention, shared init function, Numba kernel for the main runs, median of repetitions, warm-up, timing split, max over ranks, float64 and no fastmath, Docker bridge instead of macvlan on this machine.
7. **How the MPI loop works**, line by line, with a small diagram in words: what is sent, to whom, when, why it cannot deadlock, what the halo holds at each step.
8. **Results**: the real numbers, each with the file it came from. Then the interpretation: what scaled, what did not, where communication starts to dominate, what the cost model predicted and how far off it was.
9. **What did not work / uncertain**: same content as the report section, with more detail.
10. **Numbers cheat-sheet**: the 15 or so most important figures with their source file.
11. **Likely questions from the professor**: at least 30, in the format *Question → short answer (2–5 sentences) → "Show:" which figure, file or command*. Cover at least: why not Python threads; why Numba; what is a halo; why Sendrecv; what PROC_NULL does; deadlock with blocking Send/Recv; why one message per row; why Allreduce every k steps and what it costs; what if N is not divisible by p; how correctness was checked and what tolerance; why speedup is below p; Amdahl vs communication overhead; strong vs weak scaling; why efficiency drops; surface-to-volume and why 2D blocks would communicate less (`~ 4N/sqrt(p)` against `2N`); non-blocking overlap and when it helps; why median and how many repetitions; why warm-up; what oversubscription does to the results; why VMs / containers on one host cannot scale past its cores; NAT vs bridged VMs; macvlan vs bridge vs overlay; why Wi-Fi fails; what the ping-pong tells us; why the Jacobi method converges slowly and why N = 2048 uses a fixed step count; effect of k on the stopping iteration; what a seam at a rank boundary means. Answers must use **our** numbers and our code, not generic ones.
12. **Live demo script (10 minutes)**: exact commands in order, what the audience will see, timings, and a fallback if the LAN fails (use the recorded results and the animation).
13. **Weak points the professor may probe**, and honest ways to answer them. Also: "things we must not claim".
14. **After the LAN / VM runs**: a short checklist of what to copy back, which commands to run (`make_report_assets.py`, `pdflatex`), and which report paragraphs to review or update.
15. **Glossary** of the terms used.

---

## 10. The LaTeX report

### 10.1 Format

- `report/main.tex`, built with `pdflatex`. Use only standard packages (`inputenc`, `fontenc`, `geometry`, `graphicx`, `booktabs`, `amsmath`, `caption`, `hyperref`, `listings`, `xcolor`, `float`). Check availability before using any other package.
- Language: English. A4, 11 pt, normal margins.
- Length: title page + **at most 6 pages** of main body (the proposal asks for 4 to 6) + appendices. Put extra tables, setup details and secondary plots in the appendix.
- Every figure and table is referenced in the text and has a caption that names its source CSV.
- Show only short code listings (for example the halo exchange and the boundary handling), not whole files.

### 10.2 Title page (use exactly this content, centred)

```
USTO-MB
University of Science and Technology of Oran – Mohamed Boudiaf

Parallel 2D Heat Diffusion in Python:
Threads, Virtual Machines, Docker Containers and a Physical LAN

Course: Intensive Computing

by: Bellatreche Mohamed Amine, Rezaiguia Soltane

Supervisor: Prof. Dairi Abdelkader

Academic year: 2026–2027
5th year ING, Data Science
```

The title page contains nothing else (no abstract, no date line) and has no page number.

### 10.3 Structure

1. Introduction (short: problem, goal, what was built).
2. Problem and method (Jacobi, boundaries, corner convention, stopping rule, parameter table).
3. Implementation and design decisions (versions 0 to 4, the MPI design with a decomposition figure, halo exchange, convergence check, timing method). For each important decision say **what we chose and why**.
4. Experimental setup (hardware and software from `environment.json`, methodology: warm-up, repetitions, median, what ran where).
5. Results: correctness; thread scaling; MPI scaling on one machine; compute versus communication; cost model versus measurements; convergence and the k-sweep; blocking versus non-blocking; Docker; VMs and physical LAN (only if measured); animation (one figure with frames).
6. Discussion (what the numbers mean; the limit of scaling on shared cores; latency versus bandwidth; hypotheses clearly marked as hypotheses).
7. **Limitations, problems and uncertainties** (see 10.4).
8. Conclusion (short, no new results).
- Appendices: addressing table and setup summary; reproduction commands; extra plots.
- References: only the ones allowed in rule 7.

### 10.4 Limitations section (mandatory)

Write a real section, in prose. Include everything that is true from this list, and anything else that applies:

- what was not run (VMs, physical LAN, macvlan, anything missing in the environment);
- measurements done with oversubscribed ranks or threads, and what that does to them;
- timing noise (other programs running, thermal throttling, a laptop CPU), and how big the spread between repetitions was (from the CSV);
- Docker bridge on one host instead of macvlan on a LAN;
- the threading layer that was really available;
- places where the cost model did not match and any explanation that is only a hypothesis;
- Jacobi convergence being very slow for large N, so the large-N scaling runs use a fixed step count;
- features left out (2D decomposition, weak scaling, anything else);
- bugs found along the way and anything still uncertain (from `logs/issues.md`).

### 10.5 Writing style (important)

Write like two 5th-year engineering students who actually did the work and are writing it up honestly, not like a polished brochure.

- Use "we". Plain, direct sentences, mostly in paragraphs. Short bullet lists are fine in a few places, not everywhere.
- Say what was done, then why. State results with the number and then what it means in a sentence or two. Keep claims in proportion to the evidence ("this suggests", "we did not test whether") when something is not proven.
- Vary sentence and paragraph length. Do not give every section the same shape. Do not close every paragraph with a summarising sentence.
- Mention real difficulties and real fixes, using only entries from `logs/issues.md`.
- Avoid inflated or filler wording: *delve, leverage, robust, comprehensive, seamless, crucial, pivotal, tapestry, landscape, holistic, in today's world, it is worth noting, furthermore, moreover, in conclusion, plays a vital role*. Avoid heavy em-dash use, rhetorical questions, emojis, and triplets of adjectives.
- Technical terms are fine and expected (halo, stencil, Allreduce); explain them once, briefly.
- Tone example (about the style only, do not reuse the content): *"We time the halo exchange and the update separately. Each message is one row of N doubles, so 8N bytes, which is 8 KB for N = 1024. At that size we expect latency to matter more than bandwidth, and the ping-pong results are what we use to check that."*
- Do **not** add fake typos, fake mistakes or invented anecdotes. Everything stated must be true.

### 10.6 Numbers and the compile check

- `scripts/make_report_assets.py` reads `results/` and generates, into `report/generated/`: all tables (`.tex`), all figures, and `numbers.tex` containing every number cited in the running text as a keyed macro, for example:

  ```latex
  % generated/numbers.tex
  \expandafter\def\csname val-speedup-mpi-p4-N1024\endcsname{3.42}
  % main.tex
  \newcommand{\R}[1]{\csname val-#1\endcsname}
  ... we measured a speedup of \R{speedup-mpi-p4-N1024} on four ranks ...
  ```

  (The value above is a placeholder showing the syntax. The real values come from the CSV files.) A missing key makes LaTeX fail loudly, which is the point. Configuration values (N = 1024, k = 50, tolerance) can be typed directly because they are choices, not results.
- The script also writes `report/generated/SOURCES.md` listing which result file feeds which table, figure and number.
- Write the analysis text only after reading the generated tables and figures, so the words match the data.
- **Compile check:** run `pdflatex -interaction=nonstopmode main.tex` at least twice (three times if the references need it). Open the log. Fix every `Error`, every `Undefined control sequence`, every `undefined reference` or citation, and every `??` in the PDF. Fix overfull boxes that visibly run into the margin. Repeat until the log is clean. Then check the page count with `pdfinfo` (main body at most 6 pages plus title page and appendices), rasterize a few pages with `pdftoppm` and look at them if you can (figures present, tables not cut off, title page correct).
- Copy the final PDF to `Heat_Diffusion_Report.pdf` in the project root.

---

## 11. Order of work

1. **Setup**: detect the OS, install or check tools, create the layout, `requirements.txt`, `collect_env.py` (E0).
2. **Core code**: `heat_common.py`, `heat_reference.py`, `heat_serial.py`; first temperature image; E1 for serial.
3. **Threads**: `heat_threads.py`; E1 for threads; E2.
4. **MPI**: `heat_mpi.py`; E1 for MPI with all p, including N = 120; fix every correctness issue **before any timing**.
5. **Measurements**: E3, E4, E5, E6, E8. Run `--quick` first, then the full grids in the background.
6. **Docker**: image, compose, connectivity check, E7 (if Docker exists). Then write the VM and LAN docs and scripts.
7. **Visualization**: animation, static figures, charts, the seam demo.
8. **Assets**: `make_report_assets.py`.
9. **Writing**: README, notes.md, then the report (analysis written after the data).
10. **Compile and verify**: section 12.

---

## 12. Definition of done (verify each item, then print a short summary)

- [ ] All results files referenced above exist; raw data was never edited by hand.
- [ ] E1 passes for every version and every p tested. The stop iteration of the convergence-mode run is identical across serial and MPI.
- [ ] Every number in the report and in `notes.md` can be traced to a file in `results/` (check `SOURCES.md`). Search the `.tex` for numbers typed by hand and confirm each one is a configuration value or comes from a file.
- [ ] Nothing in the report claims a VM, LAN, macvlan or Docker measurement that was not actually made.
- [ ] The limitations section exists and is specific.
- [ ] The title page matches section 10.2 exactly.
- [ ] `pdflatex` finishes with no errors, no undefined references, no `??`; main body is at most 6 pages.
- [ ] `README.md` commands were tested from a fresh shell (quick test at least).
- [ ] `notes.md` has all 15 sections and at least 30 questions.
- [ ] No leftover `TODO`, placeholder text, "lorem ipsum" or dead code.
- [ ] Final message to the user: what was run, what was not run, where the key files are, and exactly what the students still have to do on their own hardware (VM and LAN runs, then regenerate the assets and recompile).
