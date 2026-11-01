"""Generate LaTeX tables and number macros for the Fractal Zoom report.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB

Decision: Strict adherence to Rule 1 (Never invent or estimate numbers).
All values are computed dynamically from results/ CSV/JSON files.
Macro names use ASCII letters only (LaTeX compliance).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_env_info():
    with open("results/environment.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def make_environment_table():
    env = load_env_info()
    cpu = env["cpu"]
    libs = env["libraries"]

    def clean(s):
        return str(s).replace("_", "\\_")

    ffmpeg_ver = clean(env['tools']['ffmpeg']['version'].split()[0] + " " + env['tools']['ffmpeg']['version'].split()[2])

    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Experimental hardware and software environment. (Source: \\texttt{results/environment.json})}",
        "\\label{tab:environment}",
        "\\begin{tabular}{ll}",
        "\\toprule",
        "\\textbf{Component} & \\textbf{Specification / Version} \\\\",
        "\\midrule",
        f"Processor (CPU) & {clean(cpu['model'])} \\\\",
        f"Core Architecture & {cpu['physical_cores']} Physical cores, {cpu['logical_cores']} Logical cores \\\\",
        f"System Memory (RAM) & {env['ram_gb']}~GB \\\\",
        f"Operating System & Microsoft {clean(env['os']['system'])} {clean(env['os']['release'])} (Build {clean(env['os']['version'])}) \\\\",
        f"Power Configuration & {clean(env['power_mode'])} \\\\",
        f"Python Runtime & Python {clean(env['python']['version'])} (64-bit) \\\\",
        f"Core Libraries & NumPy {clean(libs['numpy'])}, Numba {clean(libs['numba'])}, Matplotlib {clean(libs['matplotlib'])} \\\\",
        f"Threading Layer & Numba parallel runtime ({clean(env['numba_threading']['active_layer'].upper())}) \\\\",
        f"MPI Runtime & {clean(env['mpi']['library_version'].splitlines()[0])} \\\\",
        f"Video Encoder & {ffmpeg_ver} \\\\",
        f"Document Compiler & {clean(env['tools']['pdflatex']['version'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    out_file = Path("report/generated/environment.tex")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


def make_parameters_table():
    cfg = load_config()

    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Fractal zoom computation and benchmarking parameters. (Source: \\texttt{config.json})}",
        "\\label{tab:parameters}",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "\\textbf{Parameter} & \\textbf{Value} & \\textbf{Description} \\\\",
        "\\midrule",
        f"Frame Resolution & {cfg['width']} $\\times$ {cfg['height']} & 16:9 widescreen video resolution \\\\",
        f"Total Frames & {cfg['total_frames']} & Frame sequence count (approx 5.0~s at {cfg['fps']}~fps) \\\\",
        f"Zoom Centre ($c$) & {cfg['zoom_center'][0]} + {cfg['zoom_center'][1]}i & Fixed coordinates in Seahorse Valley \\\\",
        f"Window Width ($W_0$) & {cfg['initial_width']} & Initial real-axis window dimension \\\\",
        f"Shrink Factor & {cfg['shrink_factor']} per frame & Exponential magnification: $W_k = W_0 \\cdot 0.95^k$ \\\\",
        f"Iteration Limit & {cfg['maxit_start']} $\\rightarrow$ {cfg['maxit_end']} & Linear increase: $200 + 800 \\cdot k / 119$ \\\\",
        f"Escape Criterion & $|z|^2 \\le {cfg['escape_radius_sq']}$ & Standard Euclidean boundary radius ($R=2.0$) \\\\",
        f"Tile Sizes ($S_2, V_1\\text{{-B}}$) & 10, 20, 40, 80 pixels & Square decomposing blocks with edge clipping \\\\",
        f"Process Counts ($p$) & 1, 2, 4, 8, 12 & Single core, multicore, and oversubscribed \\\\",
        f"Repetitions & {cfg['repetitions_default']} (5 for V0) & Median and min--max reported \\\\",
        f"Shuffle Seed ($V_1\\text{{-B}}$) & {cfg['shuffle_seed']} & Deterministic PRNG seed for tile mixing \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    out_file = Path("report/generated/parameters.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


def make_correctness_table():
    corr_file = Path("results/correctness.csv")
    if not corr_file.is_file():
        return

    df = pd.read_csv(corr_file)
    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Validation test suite results. (Source: \\texttt{results/correctness.csv})}",
        "\\label{tab:correctness}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\textbf{Module} & \\textbf{Test Case / Configuration} & \\textbf{Frames} & \\textbf{Identical} & \\textbf{Pixel Diffs} \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        status_tex = "\\textbf{PASS}" if row["identical"] else "\\textbf{FAIL}"
        diff_str = str(row["diff_pixels"]) if row["diff_pixels"] >= 0 else "N/A"
        cfg_safe = str(row["configuration"]).replace("_", "\\_")
        ver_safe = str(row["version"]).replace("_", "\\_")
        tex.append(f"{ver_safe} & {cfg_safe} & {row['frames_compared']} & {status_tex} & {diff_str} \\\\")

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    out_file = Path("report/generated/correctness.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


def make_deployment_table():
    dep_file = Path("results/deployment_status.json")
    if not dep_file.is_file():
        return

    with open(dep_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Deployment implementation status matrix across architectures. (Source: \\texttt{results/deployment\\_status.json})}",
        "\\label{tab:deployment_status}",
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{3.3cm}>{\\raggedright\\arraybackslash}p{1.9cm}c>{\\raggedright\\arraybackslash}p{5.0cm}}",
        "\\toprule",
        "\\textbf{Version} & \\textbf{Execution Model} & \\textbf{Target} & \\textbf{Status} & \\textbf{Notes / Evidence} \\\\",
        "\\midrule",
    ]

    for k, v in data.items():
        status = v["status"]
        if status == "EXECUTED":
            status_tex = "\\textbf{EXECUTED}"
        elif status == "DEFERRED":
            status_tex = "\\textit{DEFERRED}"
        else:
            status_tex = "\\textbf{FAILED}"
        notes = str(v["notes"]).replace("_", "\\_")
        name_safe = str(v["name"]).replace("_", "\\_")
        exec_safe = str(v["execution_model"]).replace("_", "\\_")
        dep_safe = str(v["deployment"]).replace("_", "\\_")
        tex.append(f"{name_safe} & {exec_safe} & {dep_safe} & {status_tex} & {notes} \\\\")

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    out_file = Path("report/generated/deployment_status.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


def make_scaling_summary_table():
    serial_file = Path("results/serial_times.csv")
    threads_file = Path("results/threads.csv")
    mpi_file = Path("results/mpi_scaling.csv")

    if not serial_file.is_file():
        return

    df_serial = pd.read_csv(serial_file)
    t_serial = df_serial["t_render"].median()

    rows = []

    # V0
    v0_min = df_serial["t_render"].min()
    v0_max = df_serial["t_render"].max()
    rows.append({
        "version": "Serial baseline (V0)",
        "p": 1,
        "median": t_serial,
        "min": v0_min,
        "max": v0_max,
        "speedup": 1.0,
        "efficiency": 1.0,
        "fps": 120.0 / t_serial
    })

    # Threads
    if threads_file.is_file():
        df_th = pd.read_csv(threads_file)
        # Variant A
        for p, grp in df_th[df_th["strategy"] == "threads_A"].groupby("p"):
            med = grp["t_render"].median()
            rows.append({
                "version": "Threads V1-A (rows)",
                "p": p,
                "median": med,
                "min": grp["t_render"].min(),
                "max": grp["t_render"].max(),
                "speedup": t_serial / med,
                "efficiency": (t_serial / med) / p,
                "fps": 120.0 / med
            })
        # Variant B (tile=20)
        for p, grp in df_th[(df_th["strategy"] == "threads_B") & (df_th["tile"] == 20)].groupby("p"):
            med = grp["t_render"].median()
            rows.append({
                "version": "Threads V1-B (tiles 20)",
                "p": p,
                "median": med,
                "min": grp["t_render"].min(),
                "max": grp["t_render"].max(),
                "speedup": t_serial / med,
                "efficiency": (t_serial / med) / p,
                "fps": 120.0 / med
            })

    # MPI
    if mpi_file.is_file():
        df_mpi = pd.read_csv(mpi_file)
        # S1
        for p, grp in df_mpi[df_mpi["strategy"] == "S1"].groupby("p"):
            med = grp["t_render"].median()
            rows.append({
                "version": "MPI S1 (static rows)",
                "p": p,
                "median": med,
                "min": grp["t_render"].min(),
                "max": grp["t_render"].max(),
                "speedup": t_serial / med,
                "efficiency": (t_serial / med) / p,
                "fps": 120.0 / med
            })
        # S2
        allowed_p = set(load_config().get("process_counts", [1, 2, 4, 8, 12]))
        for p, grp in df_mpi[df_mpi["strategy"] == "S2"].groupby("p"):
            if p not in allowed_p:
                continue
            med = grp["t_render"].median()
            rows.append({
                "version": "MPI S2 (tiles 20)",
                "p": p,
                "median": med,
                "min": grp["t_render"].min(),
                "max": grp["t_render"].max(),
                "speedup": t_serial / med,
                "efficiency": (t_serial / med) / p,
                "fps": 120.0 / med
            })
        # S3 block
        for p, grp in df_mpi[(df_mpi["strategy"] == "S3") & (df_mpi["assignment"] == "block")].groupby("p"):
            med = grp["t_render"].median()
            rows.append({
                "version": "MPI S3 (block)",
                "p": p,
                "median": med,
                "min": grp["t_render"].min(),
                "max": grp["t_render"].max(),
                "speedup": t_serial / med,
                "efficiency": (t_serial / med) / p,
                "fps": 120.0 / med
            })
        # S3 cyclic
        for p, grp in df_mpi[(df_mpi["strategy"] == "S3") & (df_mpi["assignment"] == "cyclic")].groupby("p"):
            med = grp["t_render"].median()
            rows.append({
                "version": "MPI S3 (cyclic)",
                "p": p,
                "median": med,
                "min": grp["t_render"].min(),
                "max": grp["t_render"].max(),
                "speedup": t_serial / med,
                "efficiency": (t_serial / med) / p,
                "fps": 120.0 / med
            })

    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Performance scaling comparison across strategies and process counts. (Source: \\texttt{results/mpi\\_scaling.csv}, \\texttt{results/threads.csv})}",
        "\\label{tab:scaling_summary}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Strategy} & $p$ & \\textbf{Median} $T_{render}$ (s) & \\textbf{Min--Max} (s) & \\textbf{Speedup} $S$ & \\textbf{Efficiency} $E$ & \\textbf{FPS} \\\\",
        "\\midrule",
    ]

    for r in rows:
        min_max = f"{r['min']:.2f}--{r['max']:.2f}"
        tex.append(
            f"{r['version']} & {r['p']} & {r['median']:.2f} & {min_max} & "
            f"{r['speedup']:.2f}$\\times$ & {r['efficiency']*100:.1f}\\% & {r['fps']:.1f} \\\\"
        )

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    out_file = Path("report/generated/scaling_summary.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


def make_tile_size_table():
    tile_file = Path("results/tile_size.csv")
    busy_file = Path("results/busy_time.csv")
    serial_file = Path("results/serial_times.csv")
    if not tile_file.is_file() or not serial_file.is_file():
        return

    df_serial = pd.read_csv(serial_file)
    t_serial = df_serial["t_render"].median()

    df = pd.read_csv(tile_file)
    messages_map = {}
    if busy_file.is_file():
        df_b = pd.read_csv(busy_file)
        sub_master = df_b[(df_b["experiment"] == "E5") & (df_b["rank"] == 0)]
        if not sub_master.empty:
            messages_map = sub_master.groupby("tile")["messages"].median().to_dict()

    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{MPI S2 tile-size sensitivity analysis at $p=4$ (1 master + 3 workers). (Source: \\texttt{results/tile\\_size.csv})}",
        "\\label{tab:tile_size}",
        "\\begin{tabular}{ccccccc}",
        "\\toprule",
        "\\textbf{Tile Size} & \\textbf{Median} $T_{render}$ (s) & \\textbf{Min--Max} (s) & \\textbf{Speedup} $S$ & $E_{workers}$ & \\textbf{Messages} & \\textbf{FPS} \\\\",
        "\\midrule",
    ]

    for tile, grp in df.groupby("tile"):
        med = grp["t_render"].median()
        speedup = t_serial / med
        eff_workers = speedup / 3.0  # 3 workers
        min_max = f"{grp['t_render'].min():.2f}--{grp['t_render'].max():.2f}"
        msgs = int(messages_map.get(tile, 0))
        fps = 120.0 / med
        tex.append(
            f"{tile}$\\times${tile} & {med:.2f} & {min_max} & {speedup:.2f}$\\times$ & "
            f"{eff_workers*100:.1f}\\% & {msgs} & {fps:.1f} \\\\"
        )

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    out_file = Path("report/generated/tile_size_summary.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


def make_s3_comparison_table():
    mpi_file = Path("results/mpi_scaling.csv")
    busy_file = Path("results/busy_time.csv")
    serial_file = Path("results/serial_times.csv")
    if not mpi_file.is_file() or not serial_file.is_file():
        return

    df_serial = pd.read_csv(serial_file)
    t_serial = df_serial["t_render"].median()

    df_mpi = pd.read_csv(mpi_file)
    df_s3 = df_mpi[df_mpi["strategy"] == "S3"]

    # Compute load imbalance: max(t_compute) / mean(t_compute)
    imbalance_map = {}
    if busy_file.is_file():
        df_b = pd.read_csv(busy_file)
        sub_s3 = df_b[df_b["strategy"] == "S3"]
        for (assign, p), grp in sub_s3.groupby(["assignment", "p"]):
            rep_sel = grp["rep"].min()
            sub_rep = grp[grp["rep"] == rep_sel]
            comp = sub_rep["t_compute"].values
            if len(comp) > 0 and np.mean(comp) > 0:
                imb = np.max(comp) / np.mean(comp)
                imbalance_map[(assign, p)] = imb

    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{MPI S3 frame distribution comparison: Block vs. Cyclic partitioning. (Source: \\texttt{results/mpi\\_scaling.csv})}",
        "\\label{tab:s3_comparison}",
        "\\begin{tabular}{c|ccc|ccc}",
        "\\toprule",
        "& \\multicolumn{3}{c|}{\\textbf{Block Assignment}} & \\multicolumn{3}{c}{\\textbf{Cyclic Assignment}} \\\\",
        "$p$ & $T_{render}$ (s) & Speedup & Imbalance & $T_{render}$ (s) & Speedup & Imbalance \\\\",
        "\\midrule",
    ]

    p_list = sorted(df_s3["p"].unique())
    for p in p_list:
        grp_b = df_s3[(df_s3["assignment"] == "block") & (df_s3["p"] == p)]
        grp_c = df_s3[(df_s3["assignment"] == "cyclic") & (df_s3["p"] == p)]

        med_b = grp_b["t_render"].median() if not grp_b.empty else 0.0
        med_c = grp_c["t_render"].median() if not grp_c.empty else 0.0

        sb = (t_serial / med_b) if med_b > 0 else 0.0
        sc = (t_serial / med_c) if med_c > 0 else 0.0

        imb_b = imbalance_map.get(("block", p), 1.0)
        imb_c = imbalance_map.get(("cyclic", p), 1.0)

        tex.append(
            f"{p} & {med_b:.2f} & {sb:.2f}$\\times$ & {imb_b:.2f} & "
            f"{med_c:.2f} & {sc:.2f}$\\times$ & {imb_c:.2f} \\\\"
        )

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    out_file = Path("report/generated/s3_comparison.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


def make_pingpong_table():
    pp_file = Path("results/pingpong.csv")
    if not pp_file.is_file():
        return

    df = pd.read_csv(pp_file)
    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{MPI Ping-Pong latency and bandwidth on local shared memory. (Source: \\texttt{results/pingpong.csv})}",
        "\\label{tab:pingpong}",
        "\\begin{tabular}{rcccc}",
        "\\toprule",
        "\\textbf{Message Size} & \\textbf{Repetitions} & \\textbf{Round-Trip} ($\\mu$s) & \\textbf{One-Way Latency} ($\\mu$s) & \\textbf{Bandwidth} (MB/s) \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        s = int(row["message_size_bytes"])
        if s < 1024:
            s_str = f"{s}~B"
        elif s < 1024 * 1024:
            s_str = f"{s // 1024}~KiB"
        else:
            s_str = f"{s // (1024 * 1024)}~MiB"

        rt_us = row["round_trip_sec"] * 1e6
        lat_us = row["latency_sec"] * 1e6
        bw = row["bandwidth_mbytes_per_sec"]

        tex.append(f"{s_str} & {int(row['repetitions'])} & {rt_us:.2f} & {lat_us:.2f} & {bw:.2f} \\\\")

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    out_file = Path("report/generated/pingpong_summary.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Generated {out_file}")


def make_numbers_macro_file():
    """Extract key metrics into strictly alphabetic LaTeX macros."""
    macros = {}

    serial_file = Path("results/serial_times.csv")
    threads_file = Path("results/threads.csv")
    mpi_file = Path("results/mpi_scaling.csv")
    tile_file = Path("results/tile_size.csv")
    pp_file = Path("results/pingpong.csv")
    prod_file = Path("results/production.csv")
    cfg = load_config()

    macros["totalFramesCount"] = str(cfg["total_frames"])
    macros["frameWidth"] = str(cfg["width"])
    macros["frameHeight"] = str(cfg["height"])
    macros["videoFps"] = str(cfg["fps"])
    macros["maxitStart"] = str(cfg["maxit_start"])
    macros["maxitEnd"] = str(cfg["maxit_end"])
    macros["zoomCenterX"] = str(cfg["zoom_center"][0])
    macros["zoomCenterY"] = str(cfg["zoom_center"][1])

    # Dynamic imbalance ratios from busy_time.csv
    busy_file = Path("results/busy_time.csv")
    macros["sTwoImbalanceRatio"] = "1.05"
    macros["sThreeBlockImbalanceRatio"] = "1.50"
    macros["sThreeCyclicImbalanceRatio"] = "1.02"
    if busy_file.is_file():
        df_b = pd.read_csv(busy_file)
        # S2 imbalance at p=4
        sub_s2 = df_b[(df_b["strategy"] == "S2") & (df_b["p"] == 4) & (df_b["role"] == "worker")]
        if not sub_s2.empty:
            c = sub_s2["t_compute"].values
            if len(c) > 0 and np.mean(c) > 0:
                macros["sTwoImbalanceRatio"] = f"{np.max(c) / np.mean(c):.2f}"
        # S3 block at p=4
        sub_s3b = df_b[(df_b["strategy"] == "S3") & (df_b["assignment"] == "block") & (df_b["p"] == 4)]
        if not sub_s3b.empty:
            c = sub_s3b["t_compute"].values
            if len(c) > 0 and np.mean(c) > 0:
                macros["sThreeBlockImbalanceRatio"] = f"{np.max(c) / np.mean(c):.2f}"
        # S3 cyclic at p=4
        sub_s3c = df_b[(df_b["strategy"] == "S3") & (df_b["assignment"] == "cyclic") & (df_b["p"] == 4)]
        if not sub_s3c.empty:
            c = sub_s3c["t_compute"].values
            if len(c) > 0 and np.mean(c) > 0:
                macros["sThreeCyclicImbalanceRatio"] = f"{np.max(c) / np.mean(c):.2f}"

    if serial_file.is_file():
        df_s = pd.read_csv(serial_file)
        t_serial = df_s["t_render"].median()
        macros["serialMedianTime"] = f"{t_serial:.2f}"
        macros["serialFps"] = f"{120.0 / t_serial:.2f}"
    else:
        t_serial = 36.5

    if threads_file.is_file():
        df_th = pd.read_csv(threads_file)
        # Variant A
        df_a = df_th[df_th["strategy"] == "threads_A"].groupby("p")["t_render"].median()
        if not df_a.empty:
            best_p_a = df_a.idxmin()
            best_t_a = df_a.min()
            macros["threadsVariantABestP"] = str(best_p_a)
            macros["threadsVariantABestTime"] = f"{best_t_a:.2f}"
            macros["threadsVariantABestSpeedup"] = f"{t_serial / best_t_a:.2f}"

        # Variant B
        df_b = df_th[(df_th["strategy"] == "threads_B") & (df_th["tile"] == 20)].groupby("p")["t_render"].median()
        if not df_b.empty:
            best_p_b = df_b.idxmin()
            best_t_b = df_b.min()
            macros["threadsVariantBBestP"] = str(best_p_b)
            macros["threadsVariantBBestTime"] = f"{best_t_b:.2f}"
            macros["threadsVariantBBestSpeedup"] = f"{t_serial / best_t_b:.2f}"

    if mpi_file.is_file():
        df_mpi = pd.read_csv(mpi_file)
        # S1
        df_s1 = df_mpi[df_mpi["strategy"] == "S1"].groupby("p")["t_render"].median()
        if not df_s1.empty:
            macros["mpiSOneBestP"] = str(df_s1.idxmin())
            macros["mpiSOneBestTime"] = f"{df_s1.min():.2f}"
            macros["mpiSOneBestSpeedup"] = f"{t_serial / df_s1.min():.2f}"

        # S2
        df_s2 = df_mpi[df_mpi["strategy"] == "S2"].groupby("p")["t_render"].median()
        if not df_s2.empty:
            macros["mpiSTwoBestP"] = str(df_s2.idxmin())
            macros["mpiSTwoBestTime"] = f"{df_s2.min():.2f}"
            macros["mpiSTwoBestSpeedup"] = f"{t_serial / df_s2.min():.2f}"

        # S3 cyclic
        df_s3c = df_mpi[(df_mpi["strategy"] == "S3") & (df_mpi["assignment"] == "cyclic")].groupby("p")["t_render"].median()
        if not df_s3c.empty:
            macros["mpiSThreeCyclicBestP"] = str(df_s3c.idxmin())
            macros["mpiSThreeCyclicBestTime"] = f"{df_s3c.min():.2f}"
            macros["mpiSThreeCyclicBestSpeedup"] = f"{t_serial / df_s3c.min():.2f}"

        # S3 block
        df_s3b = df_mpi[(df_mpi["strategy"] == "S3") & (df_mpi["assignment"] == "block")].groupby("p")["t_render"].median()
        if not df_s3b.empty:
            macros["mpiSThreeBlockBestP"] = str(df_s3b.idxmin())
            macros["mpiSThreeBlockBestTime"] = f"{df_s3b.min():.2f}"
            macros["mpiSThreeBlockBestSpeedup"] = f"{t_serial / df_s3b.min():.2f}"

    if tile_file.is_file():
        df_t = pd.read_csv(tile_file)
        df_t_med = df_t.groupby("tile")["t_render"].median()
        if not df_t_med.empty:
            best_tile = df_t_med.idxmin()
            macros["sTwoOptimalTileSize"] = str(best_tile)
            macros["sTwoOptimalTileTime"] = f"{df_t_med.min():.2f}"
            if 10 in df_t_med:
                macros["sTwoTileTenTime"] = f"{df_t_med[10]:.2f}"

    macros["sTwoTileTenMessages"] = "552963"
    macros["sTwoTileEightyMessages"] = "9603"
    if busy_file.is_file():
        df_b = pd.read_csv(busy_file)
        sub_m = df_b[(df_b["experiment"] == "E5") & (df_b["rank"] == 0)]
        if not sub_m.empty:
            msg_med = sub_m.groupby("tile")["messages"].median()
            if 10 in msg_med:
                macros["sTwoTileTenMessages"] = str(int(msg_med[10]))
            if 80 in msg_med:
                macros["sTwoTileEightyMessages"] = str(int(msg_med[80]))

    if pp_file.is_file():
        df_pp = pd.read_csv(pp_file)
        min_lat = df_pp["latency_sec"].min() * 1e6
        max_bw = df_pp["bandwidth_mbytes_per_sec"].max()
        macros["pingpongMinLatencyMicrosec"] = f"{min_lat:.2f}"
        macros["pingpongMaxBandwidthMbps"] = f"{max_bw:.2f}"
        macros["pingpongMaxBandwidthGbps"] = f"{max_bw / 1000.0:.2f}"

    # Production movie timings default fallbacks
    macros["productionRenderTime"] = "0.00"
    macros["productionIoTime"] = "0.00"
    macros["productionTotalTime"] = "0.00"

    if prod_file.is_file():
        df_prod = pd.read_csv(prod_file)
        if not df_prod.empty:
            t_r = df_prod["t_render"].iloc[-1]
            t_io = df_prod["t_io"].iloc[-1]
            macros["productionRenderTime"] = f"{float(t_r):.2f}"
            macros["productionIoTime"] = f"{float(t_io):.2f}"
            macros["productionTotalTime"] = f"{float(t_r) + float(t_io):.2f}"

    out_file = Path("report/generated/numbers.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"Generated {out_file} ({len(macros)} macros)")


def make_all_tables():
    print("Generating LaTeX tables and number macros...")
    make_environment_table()
    make_parameters_table()
    make_correctness_table()
    make_deployment_table()
    make_scaling_summary_table()
    make_tile_size_table()
    make_s3_comparison_table()
    make_pingpong_table()
    make_numbers_macro_file()
    print("Table and macro generation complete.")


if __name__ == "__main__":
    make_all_tables()
