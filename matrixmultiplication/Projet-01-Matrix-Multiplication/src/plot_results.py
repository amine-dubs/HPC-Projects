"""
Plotting and Data Visualization Suite for Version 1 (Shared-Memory Threads)
High-Performance Computing (HPC) Project: Matrix Multiplication

Generates publication-quality plots:
1. Speedup curves: Version 1 (Threads) vs Ideal linear scaling
2. Parallel Efficiency curves: Version 1 across thread counts p in {1, 2, 4, 8}
3. Cache-Blocked (Tiling = 64x64) vs Naive Row-Parallel Performance

Outputs are saved in high resolution (300 DPI) into `plots/`.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# Styling configuration
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

COLORS = {
    "threads": "#2b5c8f",    # Deep blue
    "blocked": "#d95f02",    # Orange
    "ideal": "#7f7f7f"       # Grey
}


def load_csv(filepath):
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_speedup_threads(data, output_dir="plots"):
    """
    Plots Speedup S = T_1 / T_p for Version 1 (Threads) across matrix sizes n=1024, 2048, 4096.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
    sizes = [1024, 2048, 4096]
    markers = ["o", "s", "^"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    p_vals = [1, 2, 4, 8]
    # Ideal line
    ax.plot(p_vals, p_vals, linestyle="--", color=COLORS["ideal"], linewidth=1.8, label="Ideal Linear Speedup ($S=p$)")

    for idx, n in enumerate(sizes):
        subset = [r for r in data if r["n"] == str(n) and r["kernel"] == "naive_prange" and int(r["p"]) in p_vals]
        if not subset:
            continue
        p_list = [int(r["p"]) for r in subset]
        sp_list = [float(r["speedup"]) for r in subset]
        ax.plot(p_list, sp_list, marker=markers[idx], color=colors[idx], linewidth=2.2,
                markersize=7, label=f"Numba Threads (n={n})")

    ax.set_title("Shared-Memory Thread Speedup ($S = T_1 / T_p$)", fontweight="bold", pad=12)
    ax.set_xlabel("Number of Threads ($p$)")
    ax.set_ylabel("Speedup ($S$)")
    ax.set_xticks(p_vals)
    ax.set_yticks(range(1, 9))
    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(0.8, 8.5)
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True)

    out_png = os.path.join(output_dir, "speedup_threads.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out_png}")


def plot_efficiency_threads(data, output_dir="plots"):
    """
    Plots Parallel Efficiency E = S / p for Version 1 across thread counts.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
    sizes = [1024, 2048, 4096]
    markers = ["o", "s", "^"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    p_vals = [1, 2, 4, 8]
    ax.axhline(1.0, linestyle="--", color=COLORS["ideal"], linewidth=1.8, label="Ideal Efficiency ($E=1.0$)")

    for idx, n in enumerate(sizes):
        subset = [r for r in data if r["n"] == str(n) and r["kernel"] == "naive_prange" and int(r["p"]) in p_vals]
        if not subset:
            continue
        p_list = [int(r["p"]) for r in subset]
        eff_list = [float(r["efficiency"]) for r in subset]
        ax.plot(p_list, eff_list, marker=markers[idx], color=colors[idx], linewidth=2.2,
                markersize=7, label=f"Numba Threads (n={n})")

    ax.set_title("Shared-Memory Parallel Efficiency ($E = S / p$)", fontweight="bold", pad=12)
    ax.set_xlabel("Number of Threads ($p$)")
    ax.set_ylabel("Parallel Efficiency ($E$)")
    ax.set_xticks(p_vals)
    ax.set_ylim(0.15, 1.15)
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True)

    out_png = os.path.join(output_dir, "efficiency_threads.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out_png}")


def plot_cache_blocking_comparison(data, output_dir="plots"):
    """
    Compares Naive Row-Parallel vs Cache-Blocked (tiled) execution for n=2048.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    subset_naive = [r for r in data if r["n"] == "2048" and r["kernel"] == "naive_prange" and int(r["p"]) in [4, 8]]
    subset_blocked = [r for r in data if r["n"] == "2048" and r["kernel"] == "blocked_64" and int(r["p"]) in [4, 8]]

    if subset_naive and subset_blocked:
        threads = [int(r["p"]) for r in subset_naive]
        time_naive = [float(r["median_time_s"]) for r in subset_naive]
        time_blocked = [float(r["median_time_s"]) for r in subset_blocked]

        gflops_naive = [float(r["gflops"]) for r in subset_naive]
        gflops_blocked = [float(r["gflops"]) for r in subset_blocked]

        x = np.arange(len(threads))
        width = 0.35

        # Time comparison
        ax1.bar(x - width/2, time_naive, width, label="Naive Row Parallel", color=COLORS["threads"], edgecolor="black")
        ax1.bar(x + width/2, time_blocked, width, label="Cache-Blocked (Tile=64)", color=COLORS["blocked"], edgecolor="black")
        ax1.set_title("Execution Time Comparison ($n=2048$)", fontweight="bold")
        ax1.set_xlabel("Thread Count ($p$)")
        ax1.set_ylabel("Execution Time (seconds)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"p={t}" for t in threads])
        ax1.grid(True, axis="y")
        ax1.legend(frameon=True)

        # GFLOP/s comparison
        ax2.bar(x - width/2, gflops_naive, width, label="Naive Row Parallel", color=COLORS["threads"], edgecolor="black")
        ax2.bar(x + width/2, gflops_blocked, width, label="Cache-Blocked (Tile=64)", color=COLORS["blocked"], edgecolor="black")
        ax2.set_title("Computational Throughput ($n=2048$)", fontweight="bold")
        ax2.set_xlabel("Thread Count ($p$)")
        ax2.set_ylabel("Throughput (GFLOP/s)")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"p={t}" for t in threads])
        ax2.grid(True, axis="y")
        ax2.legend(frameon=True)

        fig.suptitle("Memory Hierarchy Optimization: Impact of Cache-Blocking / Tiling",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        out_png = os.path.join(output_dir, "cache_blocking_comparison.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[PLOT] Saved: {out_png}")


def generate_all_plots(data_dir="data", plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    serial_threads_csv = os.path.join(data_dir, "results_serial_threads.csv")

    if os.path.exists(serial_threads_csv):
        st_data = load_csv(serial_threads_csv)
        plot_speedup_threads(st_data, plots_dir)
        plot_efficiency_threads(st_data, plots_dir)
        plot_cache_blocking_comparison(st_data, plots_dir)

    print("\n[SUCCESS] Version 1 plots successfully generated in:", os.path.abspath(plots_dir))


if __name__ == "__main__":
    generate_all_plots()
