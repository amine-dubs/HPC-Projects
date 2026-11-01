"""
scripts/generate_plots.py - Generates All Publication-Quality Figures for the Report (Experiment E9)

Figures generated:
1. figures/pingpong_perf.png: Ping-pong latency & bandwidth curves (log scale, protocol switch)
2. figures/thread_scaling.png: Thread speedup and parallel efficiency vs threads (marking physical core limit)
3. figures/mpi_scaling.png: MPI speedup and parallel efficiency vs ranks (marking oversubscription)
4. figures/mpi_time_breakdown.png: Stacked bar chart of Compute, Halo, and Reduce times per (N, p)
5. figures/convergence_curves.png: Convergence curves (log(Delta u) vs iteration) with tolerance threshold
6. figures/ksweep_tradeoff.png: Reduction overhead and runtime vs checking interval k
7. figures/cost_model_validation.png: Predicted vs measured communication share
8. figures/blocking_vs_overlap.png: Blocking vs non-blocking overlap comparison
9. figures/kernel_comparison.png: Numba JIT vs NumPy array operations comparison
"""

import os
import sys
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_pingpong(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(PROJECT_DIR, "results", "raw", "pingpong_local.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping pingpong plot (file missing: {csv_path})")
        return

    bytes_list, lat_us, bw_mbs = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            bytes_list.append(float(r["bytes"]))
            lat_us.append(float(r["one_way_sec"]) * 1e6)
            bw_mbs.append(float(r["bandwidth_MBs"]))

    bytes_arr = np.array(bytes_list)
    lat_arr = np.array(lat_us)
    bw_arr = np.array(bw_mbs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    # 1. Latency (log-log)
    ax1.plot(bytes_arr, lat_arr, "o-", color="#1f77b4", label="One-way Latency")
    ax1.axvline(x=4096, color="red", linestyle=":", label="Protocol Switch (4 KB)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Message Size (bytes)")
    ax1.set_ylabel("One-Way Latency (µs)")
    ax1.set_title("(a) Ping-Pong Latency vs Message Size")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # 2. Bandwidth (log-linear)
    ax2.plot(bytes_arr, bw_arr, "s-", color="#2ca02c", label="Measured Bandwidth")
    ax2.axvline(x=4096, color="red", linestyle=":", label="Protocol Switch (4 KB)")
    peak_bw = np.max(bw_arr)
    peak_idx = np.argmax(bw_arr)
    ax2.annotate(f"Peak: {peak_bw:.0f} MB/s\n({bytes_arr[peak_idx]/1024:.0f} KB)",
                 xy=(bytes_arr[peak_idx], peak_bw),
                 xytext=(bytes_arr[peak_idx] * 0.05, peak_bw * 0.8),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5))
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Message Size (bytes)")
    ax2.set_ylabel("Bandwidth (MB/s)")
    ax2.set_title("(b) Ping-Pong Bandwidth vs Message Size")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "pingpong_perf.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_thread_scaling(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(PROJECT_DIR, "results", "raw", "scaling_local.csv")
    if not os.path.exists(csv_path):
        return

    threads_data = {}  # N -> threads -> list of times
    serial_times = {}  # N -> list of serial times (one per repetition)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            v = r.get("version", "")
            k = r.get("kernel", "")
            N = int(r["N"])
            if ("serial" in v) and (k == "numba"):
                serial_times.setdefault(N, []).append(float(r["t_total"]))
            elif ("threads" in v):
                t = int(r.get("p") or r.get("threads") or 1)
                if N not in threads_data:
                    threads_data[N] = {}
                if t not in threads_data[N]:
                    threads_data[N][t] = []
                threads_data[N][t].append(float(r["t_total"]))

    if not threads_data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
    colors = {512: "#1f77b4", 1024: "#ff7f0e", 2048: "#2ca02c"}

    for N in sorted(threads_data.keys()):
        t_vals = sorted(threads_data[N].keys())
        med_times = [np.median(threads_data[N][t]) for t in t_vals]
        t_base = float(np.median(serial_times[N])) if N in serial_times else med_times[0]
        speedups = [t_base / m for m in med_times]
        efficiencies = [s / t for s, t in zip(speedups, t_vals)]

        ax1.plot(t_vals, speedups, "o-", color=colors.get(N, "black"), label=f"N={N}")
        ax2.plot(t_vals, efficiencies, "s-", color=colors.get(N, "black"), label=f"N={N}")

    all_t = sorted(list(next(iter(threads_data.values())).keys()))
    ax1.plot(all_t, all_t, "k--", alpha=0.5, label="Ideal Linear")
    ax1.axvline(x=6, color="gray", linestyle=":", label="6 Physical Cores")
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Speedup $S(t) = T_{serial} / T(t)$")
    ax1.set_title("(a) Multithreading Speedup (Numba OpenMP)")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Ideal (1.0)")
    ax2.axvline(x=6, color="gray", linestyle=":", label="6 Physical Cores")
    ax2.set_xlabel("Number of Threads")
    ax2.set_ylabel("Parallel Efficiency $E(t) = S(t) / t$")
    ax2.set_title("(b) Multithreading Parallel Efficiency")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True)
    ax2.legend(loc="lower left")

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "thread_scaling.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_mpi_scaling(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(PROJECT_DIR, "results", "raw", "scaling_local.csv")
    if not os.path.exists(csv_path):
        return

    mpi_data = {}  # N -> p -> list of dicts
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            v = r.get("version", "")
            k = r.get("kernel", "")
            if ("mpi" in v) and (v != "mpi_overlap") and (k == "numba"):
                N = int(r["N"])
                p = int(r.get("p") or r.get("size") or 1)
                if N not in mpi_data:
                    mpi_data[N] = {}
                if p not in mpi_data[N]:
                    mpi_data[N][p] = []
                mpi_data[N][p].append({
                    "t_total": float(r["t_total"]),
                    "t_comp": float(r.get("t_compute_mean") or r.get("t_compute_max") or 0.0),
                    "t_halo": float(r.get("t_halo_mean") or r.get("t_halo_max") or 0.0),
                    "t_red": float(r.get("t_reduce_mean") or r.get("t_reduce_max") or 0.0)
                })

    if not mpi_data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
    colors = {512: "#1f77b4", 1024: "#ff7f0e", 2048: "#2ca02c"}

    for N in sorted(mpi_data.keys()):
        p_vals = sorted(mpi_data[N].keys())
        med_tot = [np.median([x["t_total"] for x in mpi_data[N][p]]) for p in p_vals]
        t1 = med_tot[0]  # p=1 baseline
        speedups = [t1 / t for t in med_tot]
        efficiencies = [s / p for s, p in zip(speedups, p_vals)]

        ax1.plot(p_vals, speedups, "o-", color=colors.get(N, "black"), label=f"N={N}")
        ax2.plot(p_vals, efficiencies, "s-", color=colors.get(N, "black"), label=f"N={N}")

    all_p = sorted(list(next(iter(mpi_data.values())).keys()))
    ax1.plot(all_p, all_p, "k--", alpha=0.5, label="Ideal Linear")
    ax1.axvline(x=6, color="red", linestyle=":", label="Oversubscription Limit (6 Cores)")
    ax1.set_xlabel("MPI Process Count ($p$)")
    ax1.set_ylabel("Speedup $S(p) = T(1) / T(p)$")
    ax1.set_title("(a) MPI Strong Scaling Speedup")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Ideal (1.0)")
    ax2.axvline(x=6, color="red", linestyle=":", label="Oversubscription Limit (6 Cores)")
    ax2.set_xlabel("MPI Process Count ($p$)")
    ax2.set_ylabel("Parallel Efficiency $E(p) = S(p) / p$")
    ax2.set_title("(b) MPI Parallel Efficiency")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True)
    ax2.legend(loc="lower left")

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "mpi_scaling.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Stacked bar chart of breakdown
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5), dpi=200)
    labels = []
    comp_bars = []
    halo_bars = []
    red_bars = []

    for N in sorted(mpi_data.keys()):
        for p in sorted(mpi_data[N].keys()):
            labels.append(f"N={N}\np={p}")
            comp_bars.append(np.median([x["t_comp"] for x in mpi_data[N][p]]))
            halo_bars.append(np.median([x["t_halo"] for x in mpi_data[N][p]]))
            red_bars.append(np.median([x["t_red"] for x in mpi_data[N][p]]))

    x = np.arange(len(labels))
    width = 0.65
    ax_bar.bar(x, comp_bars, width, label="Compute Time", color="#1f77b4")
    ax_bar.bar(x, halo_bars, width, bottom=comp_bars, label="Halo Exchange", color="#ff7f0e")
    bottom_red = [c + h for c, h in zip(comp_bars, halo_bars)]
    ax_bar.bar(x, red_bars, width, bottom=bottom_red, label="Allreduce Overhead", color="#d62728")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=8)
    ax_bar.set_ylabel("Time (seconds)")
    ax_bar.set_title("MPI Runtime Breakdown: Compute vs Halo vs Reduction (2000 steps)")
    ax_bar.legend(loc="upper right")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    fig_bar.tight_layout()
    out_bar = os.path.join(FIGURES_DIR, "mpi_time_breakdown.png")
    fig_bar.savefig(out_bar)
    plt.close(fig_bar)
    print(f"Saved: {out_bar}")

def plot_convergence():
    pattern = os.path.join(PROJECT_DIR, "results", "raw", "convergence_N*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, fpath in enumerate(files):
        basename = os.path.basename(fpath)
        N_str = basename.replace("convergence_N", "").replace(".csv", "")
        iters, diffs = [], []
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                iters.append(int(r["iteration"]))
                diffs.append(float(r["max_change"]))

        ax.semilogy(iters, diffs, label=f"N={N_str}", color=colors[idx % len(colors)], linewidth=1.8)

    ax.axhline(y=1e-4, color="black", linestyle="--", label="Tolerance $\\epsilon = 10^{-4}$")
    ax.set_xlabel("Iteration ($k$)")
    ax.set_ylabel("Max Absolute Change $\\Delta u$ (log scale)")
    ax.set_title("Jacobi Iteration Convergence History ($\\|u^{(k+1)} - u^{(k)}\\|_{\\infty}$)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "convergence_curves.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_ksweep(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(PROJECT_DIR, "results", "raw", "ksweep.csv")
    if not os.path.exists(csv_path):
        return

    k_data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            k = int(r["k"])
            if k not in k_data:
                k_data[k] = []
            k_data[k].append({
                "t_total": float(r["t_total"]),
                "red_share": float(r["reduce_share_pct"]),
                "iters": int(r["iterations"])
            })

    if not k_data:
        return

    k_vals = sorted(k_data.keys())
    t_tot = [np.median([x["t_total"] for x in k_data[k]]) for k in k_vals]
    red_pct = [np.median([x["red_share"] for x in k_data[k]]) for k in k_vals]

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=200)

    color1 = "#1f77b4"
    ax1.plot(k_vals, t_tot, "o-", color=color1, linewidth=2, label="Runtime (s)")
    ax1.set_xlabel("Check Interval $k$ (steps between Allreduce)")
    ax1.set_ylabel("Total Runtime (seconds)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log", base=10)
    ax1.set_xticks(k_vals)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(k_vals, red_pct, "s--", color=color2, linewidth=2, label="Allreduce Share (%)")
    ax2.set_ylabel("Reduction Overhead Share (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Convergence Check Interval Tradeoff ($N=256, p=4$)")
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "ksweep_tradeoff.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_cost_model(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(PROJECT_DIR, "results", "cost_model.csv")
    if not os.path.exists(csv_path):
        return

    labels = []
    pred_comm = []
    meas_comm = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            labels.append(f"N={r['N']}\np={r['p']}")
            pred_comm.append(float(r["pred_comm_share_pct"]))
            meas_comm.append(float(r["meas_comm_share_pct"]))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.bar(x - width/2, pred_comm, width, label="Predicted Comm Share ($T_{halo} / (T_{comp} + T_{halo})$)", color="#1f77b4")
    ax.bar(x + width/2, meas_comm, width, label="Measured Comm Share", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Communication Share (%)")
    ax.set_title("Cost Model Validation: Predicted vs Measured Halo Communication Share")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "cost_model_validation.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_overlap(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(PROJECT_DIR, "results", "raw", "scaling_local.csv")
    if not os.path.exists(csv_path):
        return

    blocking_data = {}
    overlap_data = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            N = int(r["N"])
            p = int(r.get("p") or r.get("size") or 1)
            t = float(r["t_total"])
            v = r.get("version", "")
            k = r.get("kernel", "")
            if ("mpi" in v) and (v != "mpi_overlap") and (k == "numba"):
                key = (N, p)
                if key not in blocking_data:
                    blocking_data[key] = []
                blocking_data[key].append(t)
            elif v == "mpi_overlap":
                key = (N, p)
                if key not in overlap_data:
                    overlap_data[key] = []
                overlap_data[key].append(t)

    overlap_keys = sorted(overlap_data.keys())
    if not overlap_keys:
        return

    labels = [f"N={N}\np={p}" for (N, p) in overlap_keys]
    block_times = [np.median(blocking_data[k]) if k in blocking_data else 0 for k in overlap_keys]
    over_times = [np.median(overlap_data[k]) for k in overlap_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax.bar(x - width/2, block_times, width, label="Blocking MPI (Sendrecv)", color="#1f77b4")
    ax.bar(x + width/2, over_times, width, label="Non-Blocking MPI (Isend/Irecv Overlap)", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Experiment E8: Blocking vs Non-Blocking Communication Overlap (2000 steps)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "blocking_vs_overlap.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_kernel_comparison(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(PROJECT_DIR, "results", "raw", "scaling_local.csv")
    if not os.path.exists(csv_path):
        return

    numba_times = {}
    numpy_times = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if ("mpi" in r.get("version", "")) and (int(r["N"]) == 1024) and (r.get("version") != "mpi_overlap"):
                p = int(r.get("p") or r.get("size") or 1)
                t = float(r["t_total"])
                k = r.get("kernel")
                if k == "numba":
                    if p not in numba_times:
                        numba_times[p] = []
                    numba_times[p].append(t)
                elif k == "numpy":
                    if p not in numpy_times:
                        numpy_times[p] = []
                    numpy_times[p].append(t)

    if not numpy_times:
        return

    p_vals = sorted(numpy_times.keys())
    numba_med = [np.median(numba_times.get(p, [0])) for p in p_vals]
    numpy_med = [np.median(numpy_times[p]) for p in p_vals]

    x = np.arange(len(p_vals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax.bar(x - width/2, numpy_med, width, label="NumPy Vectorized Kernel", color="#d62728")
    ax.bar(x + width/2, numba_med, width, label="Numba JIT Kernel (@njit)", color="#1f77b4")

    for i in range(len(p_vals)):
        if numba_med[i] > 0:
            speedup = numpy_med[i] / numba_med[i]
            ax.text(x[i], max(numpy_med[i], numba_med[i]) + 0.8, f"{speedup:.1f}x",
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in p_vals])
    ax.set_ylabel("Runtime (seconds, 2000 steps)")
    ax.set_title("Experiment E3 Comparison: Numba JIT vs Pure NumPy Kernel (N=1024)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "kernel_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

def generate_all():
    print("Generating all figures...")
    plot_pingpong()
    plot_thread_scaling()
    plot_mpi_scaling()
    plot_convergence()
    plot_ksweep()
    plot_cost_model()
    plot_overlap()
    plot_kernel_comparison()
    print("All available figures generated.")

if __name__ == "__main__":
    generate_all()
