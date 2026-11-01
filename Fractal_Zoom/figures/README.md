# Figures Manifest: Fractal Zoom Movie Report

All figures are automatically generated from raw benchmark CSV/JSON files by `scripts/make_plots.py` at 200 DPI using the `Agg` backend.

| Disk Filename | Report Element | Title / Description | Source File(s) |
| :--- | :--- | :--- | :--- |
| `fig1_scaling.png` | **Figure 1** (Body Sec 4.2) | Parallel speedup, efficiency, and FPS vs. process count $p$ (with min-max error bars and SMT demarcation) | `results/mpi_scaling.csv`, `results/threads.csv` |
| `fig2_busy_time.png` | **Figure 2** (Body Sec 4.3) | Stacked compute vs. wait time per rank at $p=4$ across S1, S2, S3 block, and S3 cyclic | `results/busy_time.csv` |
| `fig3_tile_size.png` | **Figure 3** (Body Sec 4.4) | S2 master-worker tile size sensitivity: execution time and messages exchanged ($p=4$) | `results/tile_size.csv`, `results/busy_time.csv` |
| `fig4_cost_map.png` | **Figure 4** (Body Sec 4.6) | Spatial iteration density heatmaps (log scale) and S1 row-block load distribution | `results/cost_per_tile.npz`, `results/cost_per_rowblock.csv` |
| `fig8_filmstrip.png` | **Figure 5** (Body Sec 4.8) | Four-frame zoom filmstrip into Seahorse Valley ($640 \times 360$) | `output/frames/` |
| `fig6_s3_block_vs_cyclic.png` | **Figure 6** (Appendix C) | Direct per-rank compute time comparison between S3 Block and Cyclic frame allocation ($p=4$) | `results/busy_time.csv` |
| `fig5_busy_vs_iterations.png` | **Figure 7** (Appendix D) | Correlation between rank iteration share and measured compute time share ($y=x$ validation) | `results/busy_time.csv` |
| `fig7_pingpong.png` | **Figure 8** (Appendix E) | Local MPI shared-memory Ping-Pong one-way latency and bandwidth curves | `results/pingpong.csv` |

*Note on Numbering:* Per Section 7 of `instructions.md`, primary figures (1, 2, 3, 4, and the filmstrip) are placed in the report body, while supplementary analyses (S3 block vs cyclic, iteration correlation, ping-pong) are placed in Appendices C, D, and E. LaTeX automatically numbers figures sequentially by order of appearance.
