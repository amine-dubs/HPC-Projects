"""
src/animate.py - MPI Heat Diffusion Visualizer & Animation Generator (Experiment E9)

Features:
- Runs 2D heat diffusion under MPI with domain decomposition (N=256, p=4, steps=2000, k=50).
- Rank 0 periodically gathers subdomains from all ranks.
- Draws frames using matplotlib ('inferno' colormap, dashed white lines at rank boundaries).
- Times plotting overhead separately (t_plot).
- Generates animated GIF saved to results/animation/heat_diffusion.gif.
- Generates 4-stage progression figure saved to figures/field_progression.png.
- Generates final temperature field figure saved to figures/field_final.png.
- Supports --seam-demo to visually demonstrate boundary seam discontinuities (saved to figures/seam_demo.png).
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from heat_common import apply_hotspot_local, TOP_TEMP, BOTTOM_TEMP, LEFT_TEMP, RIGHT_TEMP
from heat_mpi import jacobi_step_mpi_numba

def run_animation_mpi(N=256, steps=2000, k=50, skip_halo=False, hotspot=False,
                      out_gif=None, out_progression=None, out_final=None):
    if out_gif is None:
        out_gif = os.path.join(PROJECT_DIR, "results", "animation", "heat_diffusion.gif")
    if out_progression is None:
        out_progression = os.path.join(PROJECT_DIR, "figures", "field_progression.png")
    if out_final is None:
        out_final = os.path.join(PROJECT_DIR, "figures", "field_final.png")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rows = N // size
    u = np.zeros((rows + 2, N), dtype=np.float64, order="C")
    v = np.zeros((rows + 2, N), dtype=np.float64, order="C")

    # Set Dirichlet boundaries
    if rank == 0:
        u[1, :] = TOP_TEMP
        v[1, :] = TOP_TEMP
    if rank == size - 1:
        u[-2, :] = BOTTOM_TEMP
        v[-2, :] = BOTTOM_TEMP

    u[:, 0] = LEFT_TEMP
    v[:, 0] = LEFT_TEMP
    u[:, -1] = RIGHT_TEMP
    v[:, -1] = RIGHT_TEMP

    if hotspot:
        apply_hotspot_local(u, N, rank, rows)
        apply_hotspot_local(v, N, rank, rows)

    r_start = 2 if rank == 0 else 1
    r_end = rows if rank == size - 1 else rows + 1

    up = rank - 1 if rank > 0 else MPI.PROC_NULL
    down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    boundaries = [r * rows for r in range(1, size)]

    frames = []
    progression_milestones = [0, steps // 4, steps // 2, steps]
    progression_frames = {}
    t_plot_total = 0.0

    def record_frame(step_idx):
        nonlocal t_plot_total
        sendbuf = np.ascontiguousarray(u[1:rows+1, :])
        full_grid = np.empty((N, N), dtype=np.float64, order="C") if rank == 0 else None
        comm.Gather(sendbuf, full_grid, root=0)

        if rank == 0:
            t0 = time.perf_counter()
            fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
            im = ax.imshow(full_grid, cmap="inferno", vmin=0, vmax=100, origin="upper", aspect="equal")

            for b in boundaries:
                ax.axhline(y=b - 0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.85)

            for r in range(size):
                mid_y = r * rows + rows / 2.0
                ax.text(N - 5, mid_y, f"Rank {r}", color="white", fontsize=8, ha="right", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none"))

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Temperature (°C)", fontsize=10)
            ax.set_title(f"2D Heat Diffusion (Jacobi) - Step {step_idx}/{steps} (p={size})", fontsize=11, fontweight="bold")
            ax.set_xlabel("X (cells)")
            ax.set_ylabel("Y (cells)")
            fig.tight_layout()

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame_rgb = rgba[:, :, :3].copy()
            frames.append(frame_rgb)

            if step_idx in progression_milestones:
                progression_frames[step_idx] = full_grid.copy()

            plt.close(fig)
            t_plot_total += time.perf_counter() - t0

    # Record initial step 0
    record_frame(0)

    for step in range(1, steps + 1):
        if not skip_halo:
            comm.Sendrecv(sendbuf=u[1, :], dest=up, recvbuf=u[-1, :], source=down)
            comm.Sendrecv(sendbuf=u[-2, :], dest=down, recvbuf=u[0, :], source=up)

        jacobi_step_mpi_numba(u, v, r_start, r_end, N)
        u, v = v, u

        if step % k == 0 or step == steps:
            record_frame(step)

    if rank == 0:
        print(f"[Animate] Generated {len(frames)} frames. Total plotting time: {t_plot_total:.3f} s")

        # Save GIF
        os.makedirs(os.path.dirname(out_gif), exist_ok=True)
        try:
            import imageio
            imageio.mimsave(out_gif, frames, fps=12, loop=0)
            print(f"[Animate] Saved animation GIF to {out_gif}")
        except Exception as e:
            print(f"[Animate] Failed to save GIF: {e}")

        # Save 4-stage progression
        os.makedirs(os.path.dirname(out_progression), exist_ok=True)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
        sorted_milestones = sorted(progression_frames.keys())
        for idx, s in enumerate(sorted_milestones[:4]):
            ax = axes[idx]
            im = ax.imshow(progression_frames[s], cmap="inferno", vmin=0, vmax=100, origin="upper")
            for b in boundaries:
                ax.axhline(y=b - 0.5, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.set_title(f"Step {s}", fontsize=11, fontweight="bold")
            ax.set_xlabel("X (cells)")
            if idx == 0:
                ax.set_ylabel("Y (cells)")
            else:
                ax.set_yticks([])

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Temperature (°C)")
        fig.suptitle(f"2D Heat Diffusion Temporal Progression (N={N}, p={size})", fontsize=13, fontweight="bold", y=0.98)
        fig.savefig(out_progression, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Animate] Saved progression figure to {out_progression}")

        # Save final field
        os.makedirs(os.path.dirname(out_final), exist_ok=True)
        last_step = sorted_milestones[-1]
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
        im = ax.imshow(progression_frames[last_step], cmap="inferno", vmin=0, vmax=100, origin="upper")
        for b in boundaries:
            ax.axhline(y=b - 0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.85)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Temperature (°C)")
        ax.set_title(f"Final Steady-State Field (Step {last_step}, N={N}, p={size})", fontweight="bold")
        ax.set_xlabel("X (cells)")
        ax.set_ylabel("Y (cells)")
        fig.tight_layout()
        fig.savefig(out_final, dpi=150)
        plt.close(fig)
        print(f"[Animate] Saved final field figure to {out_final}")

def run_seam_comparison(N=256, steps=250, out_seam=None):
    if out_seam is None:
        out_seam = os.path.join(PROJECT_DIR, "figures", "seam_demo.png")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    rows = N // size

    boundaries = [r * rows for r in range(1, size)]

    def simulate(skip):
        u = np.zeros((rows + 2, N), dtype=np.float64, order="C")
        v = np.zeros((rows + 2, N), dtype=np.float64, order="C")
        if rank == 0:
            u[1, :] = TOP_TEMP
            v[1, :] = TOP_TEMP
        if rank == size - 1:
            u[-2, :] = BOTTOM_TEMP
            v[-2, :] = BOTTOM_TEMP
        u[:, 0] = LEFT_TEMP
        v[:, 0] = LEFT_TEMP
        u[:, -1] = RIGHT_TEMP
        v[:, -1] = RIGHT_TEMP

        r_start = 2 if rank == 0 else 1
        r_end = rows if rank == size - 1 else rows + 1
        up = rank - 1 if rank > 0 else MPI.PROC_NULL
        down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

        for _ in range(steps):
            if not skip:
                comm.Sendrecv(sendbuf=u[1, :], dest=up, recvbuf=u[-1, :], source=down)
                comm.Sendrecv(sendbuf=u[-2, :], dest=down, recvbuf=u[0, :], source=up)
            jacobi_step_mpi_numba(u, v, r_start, r_end, N)
            u, v = v, u

        sendbuf = np.ascontiguousarray(u[1:rows+1, :])
        gathered = np.empty((N, N), dtype=np.float64, order="C") if rank == 0 else None
        comm.Gather(sendbuf, gathered, root=0)
        return gathered

    field_normal = simulate(skip=False)
    field_skipped = simulate(skip=True)

    if rank == 0:
        os.makedirs(os.path.dirname(out_seam), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)

        im0 = axes[0].imshow(field_normal, cmap="inferno", vmin=0, vmax=100, origin="upper")
        for b in boundaries:
            axes[0].axhline(y=b - 0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.85)
        axes[0].set_title(f"Normal Execution (Halo Exchange Enabled)\nStep {steps}", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("X (cells)")
        axes[0].set_ylabel("Y (cells)")

        im1 = axes[1].imshow(field_skipped, cmap="inferno", vmin=0, vmax=100, origin="upper")
        for b in boundaries:
            axes[1].axhline(y=b - 0.5, color="cyan", linestyle="--", linewidth=1.5, alpha=0.95)
        axes[1].set_title(f"Skipped Halo Exchange (--skip-halo)\nVisible Seams / Boundary Discontinuities", fontsize=11, fontweight="bold", color="darkred")
        axes[1].set_xlabel("X (cells)")
        axes[1].set_yticks([])

        cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.035, pad=0.04)
        cbar.set_label("Temperature (°C)")
        fig.suptitle(f"Correctness Verification: Impact of MPI Halo Exchange (N={N}, p={size})", fontsize=12, fontweight="bold")
        fig.savefig(out_seam, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Animate] Saved seam demonstration figure to {out_seam}")

def main():
    parser = argparse.ArgumentParser(description="MPI Heat Diffusion Visualization & Animation")
    parser.add_argument("--N", type=int, default=256, help="Grid dimension N x N")
    parser.add_argument("--steps", type=int, default=2000, help="Total Jacobi steps")
    parser.add_argument("--k", type=int, default=50, help="Render frame every k steps")
    parser.add_argument("--skip-halo", action="store_true", help="Skip halo exchange for debug demo")
    parser.add_argument("--hotspot", action="store_true", help="Add circular hotspot")
    parser.add_argument("--seam-demo", action="store_true", help="Run side-by-side seam comparison demo")
    parser.add_argument("--out-gif", type=str, default=None)
    parser.add_argument("--out-progression", type=str, default=None)
    parser.add_argument("--out-final", type=str, default=None)
    parser.add_argument("--out-seam", type=str, default=None)

    args = parser.parse_args()

    if MPI is None:
        print("Error: mpi4py required for running animate.py", file=sys.stderr)
        sys.exit(1)

    if args.seam_demo:
        run_seam_comparison(N=args.N, steps=min(args.steps, 300), out_seam=args.out_seam)
    else:
        run_animation_mpi(
            N=args.N,
            steps=args.steps,
            k=args.k,
            skip_halo=args.skip_halo,
            hotspot=args.hotspot,
            out_gif=args.out_gif,
            out_progression=args.out_progression,
            out_final=args.out_final
        )

if __name__ == "__main__":
    main()
