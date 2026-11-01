"""Generate publication-ready figures for the Fractal Zoom report.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


# Set consistent academic plotting style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Color palette for consistent strategy representation
COLORS = {
    "V0": "#333333",
    "threads_A": "#1f77b4",       # Blue
    "threads_B": "#2ca02c",       # Green
    "S1": "#d62728",              # Red
    "S2": "#9467bd",              # Purple
    "S3_block": "#ff7f0e",        # Orange
    "S3_cyclic": "#17becf",       # Cyan
}


def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def plot_fig1_scaling():
    """Figure 1: Speedup and Efficiency vs Process/Thread Count."""
    serial_file = Path("results/serial_times.csv")
    threads_file = Path("results/threads.csv")
    mpi_file = Path("results/mpi_scaling.csv")

    if not serial_file.is_file():
        print("Skipping Fig 1: serial_times.csv not found")
        return

    df_serial = pd.read_csv(serial_file)
    t_serial = df_serial["t_render"].median()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax_speedup, ax_eff = axes

    # Ideal linear speedup line
    p_max = 12
    p_ideal = np.array([1, 2, 4, 6, 8, 12])
    ax_speedup.plot(p_ideal, p_ideal, "k--", label="Ideal linear", alpha=0.6, zorder=1)
    ax_eff.axhline(1.0, color="k", linestyle="--", label="Ideal efficiency", alpha=0.6, zorder=1)

    # Plot Threads
    if threads_file.is_file():
        df_th = pd.read_csv(threads_file)
        # Variant A
        df_a = df_th[df_th["strategy"] == "threads_A"].groupby("p")["t_render"].median().reset_index()
        if not df_a.empty:
            s_a = t_serial / df_a["t_render"]
            e_a = s_a / df_a["p"]
            ax_speedup.plot(df_a["p"], s_a, "-o", color=COLORS["threads_A"], label="Threads V1-A (rows)")
            ax_eff.plot(df_a["p"], e_a, "-o", color=COLORS["threads_A"], label="Threads V1-A (rows)")

        # Variant B (tile=20)
        df_b = df_th[(df_th["strategy"] == "threads_B") & (df_th["tile"] == 20)].groupby("p")["t_render"].median().reset_index()
        if not df_b.empty:
            s_b = t_serial / df_b["t_render"]
            e_b = s_b / df_b["p"]
            ax_speedup.plot(df_b["p"], s_b, "-s", color=COLORS["threads_B"], label="Threads V1-B (tiles 20)")
            ax_eff.plot(df_b["p"], e_b, "-s", color=COLORS["threads_B"], label="Threads V1-B (tiles 20)")

    # Plot MPI
    if mpi_file.is_file():
        df_mpi = pd.read_csv(mpi_file)
        # S1
        df_s1 = df_mpi[df_mpi["strategy"] == "S1"].groupby("p")["t_render"].median().reset_index()
        if not df_s1.empty:
            s_s1 = t_serial / df_s1["t_render"]
            e_s1 = s_s1 / df_s1["p"]
            ax_speedup.plot(df_s1["p"], s_s1, "-^", color=COLORS["S1"], label="MPI S1 (rows)")
            ax_eff.plot(df_s1["p"], e_s1, "-^", color=COLORS["S1"], label="MPI S1 (rows)")

        # S2
        df_s2 = df_mpi[df_mpi["strategy"] == "S2"].groupby("p")["t_render"].median().reset_index()
        if not df_s2.empty:
            s_s2 = t_serial / df_s2["t_render"]
            # Efficiency relative to total processes p
            e_s2 = s_s2 / df_s2["p"]
            ax_speedup.plot(df_s2["p"], s_s2, "-D", color=COLORS["S2"], label="MPI S2 (tiles 20)")
            ax_eff.plot(df_s2["p"], e_s2, "-D", color=COLORS["S2"], label="MPI S2 (tiles 20)")

        # S3 block
        df_s3b = df_mpi[(df_mpi["strategy"] == "S3") & (df_mpi["assignment"] == "block")].groupby("p")["t_render"].median().reset_index()
        if not df_s3b.empty:
            s_s3b = t_serial / df_s3b["t_render"]
            e_s3b = s_s3b / df_s3b["p"]
            ax_speedup.plot(df_s3b["p"], s_s3b, "-v", color=COLORS["S3_block"], label="MPI S3 (block)")
            ax_eff.plot(df_s3b["p"], e_s3b, "-v", color=COLORS["S3_block"], label="MPI S3 (block)")

        # S3 cyclic
        df_s3c = df_mpi[(df_mpi["strategy"] == "S3") & (df_mpi["assignment"] == "cyclic")].groupby("p")["t_render"].median().reset_index()
        if not df_s3c.empty:
            s_s3c = t_serial / df_s3c["t_render"]
            e_s3c = s_s3c / df_s3c["p"]
            ax_speedup.plot(df_s3c["p"], s_s3c, "-p", color=COLORS["S3_cyclic"], label="MPI S3 (cyclic)")
            ax_eff.plot(df_s3c["p"], e_s3c, "-p", color=COLORS["S3_cyclic"], label="MPI S3 (cyclic)")

    # Formatting
    for ax in [ax_speedup, ax_eff]:
        ax.set_xlabel("Process / Thread Count $p$")
        ax.set_xticks([1, 2, 4, 6, 8, 12])
        ax.axvline(6, color="gray", linestyle=":", label="Physical cores (6)" if ax == ax_speedup else None)
        ax.grid(True)

    ax_speedup.set_ylabel("Speedup $S = T_{serial} / T_p$")
    ax_speedup.set_title("Speedup vs. Process/Thread Count")
    ax_speedup.legend(loc="upper left")

    ax_eff.set_ylabel("Parallel Efficiency $E = S / p$")
    ax_eff.set_title("Efficiency vs. Process/Thread Count")
    ax_eff.set_ylim(0, 1.15)
    ax_eff.legend(loc="lower left")

    plt.tight_layout()
    out_path = Path("figures/fig1_scaling.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_fig2_busy_time():
    """Figure 2: Per-worker busy time stacked (compute vs wait) at fixed p=4."""
    busy_file = Path("results/busy_time.csv")
    if not busy_file.is_file():
        print("Skipping Fig 2: busy_time.csv not found")
        return

    df = pd.read_csv(busy_file)
    # Target p=4, median over reps for rep=1 or median
    df_p4 = df[(df["p"] == 4) & (df["experiment"] == "E4")].copy()
    if df_p4.empty:
        # Fallback to any p=4
        df_p4 = df[df["p"] == 4].copy()
    if df_p4.empty:
        print("Skipping Fig 2: No p=4 data in busy_time.csv")
        return

    # Select representative rep
    rep_sel = df_p4["rep"].min()
    df_sel = df_p4[df_p4["rep"] == rep_sel]

    strategies = [
        ("S1", "static_rows", "MPI S1 (Rows)"),
        ("S2", "master_worker", "MPI S2 (Tiles 20)"),
        ("S3", "block", "MPI S3 (Block)"),
        ("S3", "cyclic", "MPI S3 (Cyclic)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.8), sharey=True)

    for ax, (strat, assign, title) in zip(axes, strategies):
        sub = df_sel[(df_sel["strategy"] == strat) & (df_sel["assignment"] == assign)].sort_values("rank")
        if sub.empty:
            ax.set_title(title)
            continue

        ranks = sub["rank"].values
        t_comp = sub["t_compute"].values
        t_wait = sub["t_wait"].values

        bars1 = ax.bar(ranks, t_comp, color="#2ca02c", alpha=0.85, label="Compute" if ax == axes[0] else "")
        bars2 = ax.bar(ranks, t_wait, bottom=t_comp, color="#d62728", alpha=0.75, label="Wait / Comm" if ax == axes[0] else "")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Rank")
        ax.set_xticks(ranks)
        ax.grid(True, axis="y")

    axes[0].set_ylabel("Time (seconds)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.98), ncol=2)
    plt.tight_layout()
    out_path = Path("figures/fig2_busy_time.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_fig3_tile_size():
    """Figure 3: Tile size study for S2 (time and messages vs tile size)."""
    tile_file = Path("results/tile_size.csv")
    busy_file = Path("results/busy_time.csv")
    if not tile_file.is_file():
        print("Skipping Fig 3: tile_size.csv not found")
        return

    df_tile = pd.read_csv(tile_file)
    df_med = df_tile.groupby("tile")["t_render"].median().reset_index()

    # Get master messages from busy_time.csv
    messages_by_tile = {}
    if busy_file.is_file():
        df_b = pd.read_csv(busy_file)
        sub_master = df_b[(df_b["experiment"] == "E5") & (df_b["rank"] == 0)]
        if not sub_master.empty:
            messages_by_tile = sub_master.groupby("tile")["messages"].median().to_dict()

    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    color_time = "#1f77b4"
    color_msgs = "#d62728"

    tiles = df_med["tile"].values
    times = df_med["t_render"].values

    ax1.plot(tiles, times, "-o", color=color_time, linewidth=2, label="Render time (s)")
    ax1.set_xlabel("Tile size (pixels)")
    ax1.set_ylabel("Rendering Time (s)", color=color_time)
    ax1.tick_params(axis="y", labelcolor=color_time)
    ax1.set_xticks([10, 20, 40, 80])
    ax1.grid(True)

    if messages_by_tile:
        ax2 = ax1.twinx()
        msgs = [messages_by_tile.get(t, 0) for t in tiles]
        ax2.plot(tiles, msgs, "--s", color=color_msgs, linewidth=2, label="Messages to master")
        ax2.set_ylabel("Messages exchanged", color=color_msgs)
        ax2.tick_params(axis="y", labelcolor=color_msgs)

    plt.title("S2 Master-Worker: Tile Size Trade-off ($p=4$)")
    plt.tight_layout()
    out_path = Path("figures/fig3_tile_size.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_fig4_cost_map():
    """Figure 4: Tile cost heatmaps (log scale) and S1 rowblock bars."""
    cost_npz = Path("results/cost_per_tile.npz")
    rowblock_csv = Path("results/cost_per_rowblock.csv")

    if not cost_npz.is_file() or not rowblock_csv.is_file():
        print("Skipping Fig 4: cost files not found")
        return

    npz = np.load(cost_npz)
    df_rows = pd.read_csv(rowblock_csv)

    frames = [0, 60, 119]
    frame_labels = ["Frame 0 (Start)", "Frame 60 (Mid)", "Frame 119 (Deep)"]

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), gridspec_kw={"height_ratios": [1.4, 1]})

    # Row 1: Tile heatmaps (20px tiles)
    for col_idx, (f, flabel) in enumerate(zip(frames, frame_labels)):
        ax = axes[0, col_idx]
        key = f"frame_{f}_tile_20"
        if key in npz:
            grid = npz[key]
            # Avoid log(0)
            grid_safe = np.maximum(grid, 1)
            im = ax.imshow(grid_safe, norm=LogNorm(), cmap="inferno", aspect="auto")
            ax.set_title(f"Tile Cost: {flabel}")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 2:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Iterations (log scale)")

    # Row 2: S1 row block cost distribution for p=4
    df_p4 = df_rows[df_rows["p"] == 4]
    for col_idx, (f, flabel) in enumerate(zip(frames, frame_labels)):
        ax = axes[1, col_idx]
        sub = df_p4[df_p4["frame"] == f].sort_values("rank")
        if not sub.empty:
            ranks = sub["rank"].values
            shares = sub["iterations_share"].values * 100
            bars = ax.bar(ranks, shares, color=COLORS["S1"], alpha=0.85)
            ax.axhline(25.0, color="k", linestyle="--", alpha=0.7, label="Ideal (25%)" if col_idx == 0 else "")
            ax.set_xlabel("Rank (Row Block)")
            ax.set_xticks(ranks)
            ax.set_ylim(0, max(50, np.max(shares) * 1.15))
            ax.grid(True, axis="y")
            if col_idx == 0:
                ax.set_ylabel("Share of Iterations (%)")
                ax.legend(loc="upper right")

    plt.tight_layout()
    out_path = Path("figures/fig4_cost_map.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_fig5_busy_vs_iterations():
    """Figure 5: Measured busy time share vs iteration share per rank."""
    busy_file = Path("results/busy_time.csv")
    if not busy_file.is_file():
        print("Skipping Fig 5: busy_time.csv not found")
        return

    df = pd.read_csv(busy_file)
    df_workers = df[df["role"] == "worker"].copy()
    if df_workers.empty:
        return

    # Compute group totals per run
    group_cols = ["experiment", "strategy", "assignment", "p", "rep"]
    totals = df_workers.groupby(group_cols)[["t_compute", "iterations_sum"]].transform("sum")
    df_workers["comp_share"] = df_workers["t_compute"] / totals["t_compute"]
    df_workers["iter_share"] = df_workers["iterations_sum"] / totals["iterations_sum"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 0.6], [0, 0.6], "k--", label="Ideal proportionality ($y=x$)", alpha=0.7)

    for strat, name, marker in [("S1", "S1 Rows", "^"), ("S2", "S2 Tiles", "D"), ("S3", "S3 Frames", "o")]:
        sub = df_workers[df_workers["strategy"] == strat]
        if not sub.empty:
            ax.scatter(sub["iter_share"], sub["comp_share"], label=name, alpha=0.65, marker=marker, s=35)

    ax.set_xlabel("Share of Iterations per Rank")
    ax.set_ylabel("Share of Compute Time per Rank")
    ax.set_title("Correlation: Iteration Count vs. Busy Time")
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.tight_layout()
    out_path = Path("figures/fig5_busy_vs_iterations.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_fig6_s3_block_vs_cyclic():
    """Figure 6: S3 block vs cyclic busy time comparison across ranks."""
    busy_file = Path("results/busy_time.csv")
    if not busy_file.is_file():
        return

    df = pd.read_csv(busy_file)
    sub = df[(df["strategy"] == "S3") & (df["p"] == 4)].copy()
    if sub.empty:
        return

    rep_sel = sub["rep"].min()
    sub_rep = sub[sub["rep"] == rep_sel]

    df_block = sub_rep[sub_rep["assignment"] == "block"].sort_values("rank")
    df_cyclic = sub_rep[sub_rep["assignment"] == "cyclic"].sort_values("rank")

    if df_block.empty or df_cyclic.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ranks = df_block["rank"].values
    width = 0.35

    ax.bar(ranks - width/2, df_block["t_compute"], width, label="S3 Block (t_compute)", color=COLORS["S3_block"], alpha=0.85)
    ax.bar(ranks + width/2, df_cyclic["t_compute"], width, label="S3 Cyclic (t_compute)", color=COLORS["S3_cyclic"], alpha=0.85)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Compute Time (seconds)")
    ax.set_title("S3 Frame Assignment Load Imbalance ($p=4$)")
    ax.set_xticks(ranks)
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    out_path = Path("figures/fig6_s3_block_vs_cyclic.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_fig7_pingpong():
    """Figure 7: Ping-pong latency and bandwidth vs message size."""
    pp_file = Path("results/pingpong.csv")
    if not pp_file.is_file():
        print("Skipping Fig 7: pingpong.csv not found")
        return

    df = pd.read_csv(pp_file)
    sizes = df["message_size_bytes"].values
    lat_us = df["latency_sec"].values * 1e6
    bw_mb = df["bandwidth_mbytes_per_sec"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # Latency
    ax1.loglog(sizes, lat_us, "-o", color="#1f77b4", linewidth=2)
    ax1.set_xlabel("Message size (bytes)")
    ax1.set_ylabel("One-way Latency ($\\mu$s)")
    ax1.set_title("MPI Latency vs. Message Size (Local Shared Memory)")
    ax1.grid(True, which="both")

    # Bandwidth
    ax2.semilogx(sizes, bw_mb, "-s", color="#2ca02c", linewidth=2)
    ax2.set_xlabel("Message size (bytes)")
    ax2.set_ylabel("Bandwidth (MB/s)")
    ax2.set_title("MPI Bandwidth vs. Message Size")
    ax2.grid(True, which="both")

    plt.tight_layout()
    out_path = Path("figures/fig7_pingpong.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Generated {out_path}")


def plot_fig8_filmstrip():
    """Figure 8: Four-frame filmstrip of the fractal zoom."""
    frames_dir = Path("output/frames")
    ref_dir = Path("results/reference")

    frames_to_show = [0, 40, 80, 119]
    frame_titles = ["Frame 0 (Overview)", "Frame 40 (Zoom 7.7x)", "Frame 80 (Zoom 60.5x)", "Frame 119 (Deep Filaments)"]

    images = []
    for f in frames_to_show:
        png_path = frames_dir / f"frame_{f:03d}.png"
        if png_path.is_file():
            img = plt.imread(png_path)
            images.append(img)
        else:
            # Fallback to generating from reference .npy if png not yet generated
            npy_path = ref_dir / f"frame_{f:03d}.npy"
            if npy_path.is_file():
                arr = np.load(npy_path)
                cmap = plt.get_cmap("magma")
                norm_val = (arr % 64) / 63.0
                rgb = (cmap(norm_val)[:, :, :3] * 255).astype(np.uint8)
                rgb[arr == arr.max()] = [0, 0, 0]
                images.append(rgb)

    if len(images) != 4:
        print("Skipping Fig 8: Not all 4 filmstrip frames are available")
        return

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.5))
    for ax, img, title in zip(axes, images, frame_titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9.5)
        ax.set_axis_off()

    plt.tight_layout()
    out_path = Path("figures/fig8_filmstrip.png")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Generated {out_path}")


def make_all_plots():
    print("Generating all project figures...")
    plot_fig1_scaling()
    plot_fig2_busy_time()
    plot_fig3_tile_size()
    plot_fig4_cost_map()
    plot_fig5_busy_vs_iterations()
    plot_fig6_s3_block_vs_cyclic()
    plot_fig7_pingpong()
    plot_fig8_filmstrip()
    print("Figure generation complete.")


if __name__ == "__main__":
    make_all_plots()

