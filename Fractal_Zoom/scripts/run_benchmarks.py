"""Benchmark harness for local execution of experiments E2 to E5.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Add src and scripts to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from fractal_core import load_config
from mpi_launch import build_mpi_cmd


def count_existing_reps(csv_file, match_dict):
    """Count how many times a configuration matching match_dict appears in csv_file."""
    if not Path(csv_file).is_file():
        return 0
    count = 0
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matches = True
            for k, v in match_dict.items():
                if str(row.get(k, "")) != str(v):
                    matches = False
                    break
            if matches:
                count += 1
    return count


def run_command_logged(cmd, log_prefix, env=None):
    """Run a subprocess command, stream output to console, and save to log file."""
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_prefix}_{timestamp}.log"

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    # Ensure MS-MPI bin in PATH if present
    mpi_default = r"C:\Program Files\Microsoft MPI\Bin"
    if os.path.isdir(mpi_default) and mpi_default not in run_env.get("PATH", ""):
        run_env["PATH"] = mpi_default + os.pathsep + run_env.get("PATH", "")

    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=run_env
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"WARNING: Command failed with exit code {proc.returncode}. See {log_file}", file=sys.stderr)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Run complete benchmark suite E2-E5")
    parser.add_argument("--dry-run", action="store_true", help="Print schedule without running")
    parser.add_argument("--quick", action="store_true", help="Run 1 repetition for quick test")
    parser.add_argument("--clean", action="store_true", help="Remove existing benchmark CSVs before running")
    args = parser.parse_args()

    config = load_config()
    p_values = config["process_counts"]  # [1, 2, 4, 8, 12]
    tile_sizes = config["tile_sizes"]    # [10, 20, 40, 80]
    n_reps_default = 1 if args.quick else config.get("repetitions_default", 3)
    n_reps_serial = 1 if args.quick else config.get("repetitions_serial", 5)

    py_exe = sys.executable

    # Define tasks to run
    # Format: (experiment_id, target_csv, match_criteria, command_builder)
    plan = []

    # E2: Serial baseline V0 (5 reps)
    for rep in range(1, n_reps_serial + 1):
        plan.append({
            "exp": "E2",
            "name": f"Serial V0 (rep {rep})",
            "csv": "results/serial_times.csv",
            "match": {"version": "V0", "rep": rep},
            "cmd": [py_exe, "src/serial.py", "--rep", str(rep), "--out", "results/serial_times.csv"],
            "log": f"serial_v0_rep{rep}",
            "env": {}
        })

    # For parallel benchmarks, loop over repetitions outermost, shuffled inside
    for rep in range(1, n_reps_default + 1):
        rep_tasks = []

        # E3: Threads V1-A (rows)
        for p in p_values:
            rep_tasks.append({
                "exp": "E3",
                "name": f"Threads V1-A rows p={p} (rep {rep})",
                "csv": "results/threads.csv",
                "match": {"version": "V1", "strategy": "threads_A", "p": p, "rep": rep},
                "cmd": [py_exe, "src/threads.py", "--variant", "A", "-p", str(p), "--rep", str(rep)],
                "log": f"threads_v1A_p{p}_rep{rep}",
                "env": {"NUMBA_THREADING_LAYER": "omp"}
            })

        # E3: Threads V1-B (tiles 20 and 40 per Section 5.3)
        for p in p_values:
            for t_sz in [20, 40]:
                rep_tasks.append({
                    "exp": "E3",
                    "name": f"Threads V1-B tiles p={p} tile={t_sz} (rep {rep})",
                    "csv": "results/threads.csv",
                    "match": {"version": "V1", "strategy": "threads_B", "p": p, "tile": t_sz, "rep": rep},
                    "cmd": [py_exe, "src/threads.py", "--variant", "B", "-p", str(p), "--tile", str(t_sz), "--rep", str(rep)],
                    "log": f"threads_v1B_p{p}_tile{t_sz}_rep{rep}",
                    "env": {"NUMBA_THREADING_LAYER": "omp"}
                })

        # E4: MPI S1 (static rows)
        for p in p_values:
            mpi_cmd = build_mpi_cmd(p, "src/mpi_s1_rows.py", ["--rep", str(rep)])
            rep_tasks.append({
                "exp": "E4",
                "name": f"MPI S1 rows p={p} (rep {rep})",
                "csv": "results/mpi_scaling.csv",
                "match": {"version": "MPI", "strategy": "S1", "p": p, "rep": rep},
                "cmd": mpi_cmd,
                "log": f"mpi_s1_p{p}_rep{rep}",
                "env": {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            })

        # E4: MPI S2 (master-worker tiles, tile=20)
        for p in p_values:
            if p >= 2:
                mpi_cmd = build_mpi_cmd(p, "src/mpi_s2_tiles.py", ["--tile", "20", "--experiment", "E4", "--rep", str(rep)])
                rep_tasks.append({
                    "exp": "E4",
                    "name": f"MPI S2 tiles p={p} tile=20 (rep {rep})",
                    "csv": "results/mpi_scaling.csv",
                    "match": {"version": "MPI", "strategy": "S2", "p": p, "tile": 20, "rep": rep},
                    "cmd": mpi_cmd,
                    "log": f"mpi_s2_p{p}_tile20_rep{rep}",
                    "env": {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
                })

        # E4: MPI S3 (block frames)
        for p in p_values:
            mpi_cmd = build_mpi_cmd(p, "src/mpi_s3_frames.py", ["--assignment", "block", "--rep", str(rep)])
            rep_tasks.append({
                "exp": "E4",
                "name": f"MPI S3 block p={p} (rep {rep})",
                "csv": "results/mpi_scaling.csv",
                "match": {"version": "MPI", "strategy": "S3", "assignment": "block", "p": p, "rep": rep},
                "cmd": mpi_cmd,
                "log": f"mpi_s3_block_p{p}_rep{rep}",
                "env": {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            })

        # E4: MPI S3 (cyclic frames)
        for p in p_values:
            mpi_cmd = build_mpi_cmd(p, "src/mpi_s3_frames.py", ["--assignment", "cyclic", "--rep", str(rep)])
            rep_tasks.append({
                "exp": "E4",
                "name": f"MPI S3 cyclic p={p} (rep {rep})",
                "csv": "results/mpi_scaling.csv",
                "match": {"version": "MPI", "strategy": "S3", "assignment": "cyclic", "p": p, "rep": rep},
                "cmd": mpi_cmd,
                "log": f"mpi_s3_cyclic_p{p}_rep{rep}",
                "env": {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            })

        # E5: Tile size study for S2 at fixed p=4 (1 master + 3 workers)
        # Tile sizes: 10, 20, 40, 80
        for t_sz in tile_sizes:
            mpi_cmd = build_mpi_cmd(4, "src/mpi_s2_tiles.py", ["--tile", str(t_sz), "--experiment", "E5", "--rep", str(rep)])
            rep_tasks.append({
                "exp": "E5",
                "name": f"Tile study S2 p=4 tile={t_sz} (rep {rep})",
                "csv": "results/tile_size.csv",
                "match": {"version": "MPI", "strategy": "S2", "p": 4, "tile": t_sz, "rep": rep},
                "cmd": mpi_cmd,
                "log": f"tile_study_s2_p4_tile{t_sz}_rep{rep}",
                "env": {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            })

        # Shuffle tasks within this repetition with fixed seed
        random.Random(42 + rep).shuffle(rep_tasks)
        plan.extend(rep_tasks)

    # Filter out tasks that already exist in target CSVs unless --clean is passed
    if args.clean:
        for f in ["results/serial_times.csv", "results/threads.csv", "results/mpi_scaling.csv", "results/tile_size.csv", "results/busy_time.csv"]:
            if Path(f).is_file():
                Path(f).unlink()
        print("Cleaned existing CSV files.")

    pending_tasks = []
    for t in plan:
        exists = count_existing_reps(t["csv"], t["match"])
        if exists == 0:
            pending_tasks.append(t)

    print("=" * 60)
    print(f"BENCHMARK PLAN: Total scheduled tasks: {len(plan)} | Pending to execute: {len(pending_tasks)}")
    est_seconds = len(pending_tasks) * 14.0
    print(f"Estimated runtime: {est_seconds / 60.0:.1f} minutes ({est_seconds:.0f} seconds)")
    print("=" * 60)

    if args.dry_run:
        for i, t in enumerate(pending_tasks, 1):
            print(f"{i:>3}. [{t['exp']}] {t['name']}")
        return

    # Execute pending tasks
    t_start = time.perf_counter()
    for idx, task in enumerate(pending_tasks, 1):
        print(f"\n[{idx}/{len(pending_tasks)}] Starting: {task['name']}")
        ret = run_command_logged(task["cmd"], task["log"], task["env"])
        if ret != 0:
            print(f"ERROR: Task {task['name']} exited with code {ret}", file=sys.stderr)

    t_end = time.perf_counter()
    print("\n" + "=" * 60)
    print(f"BENCHMARKS FINISHED in {(t_end - t_start) / 60.0:.2f} minutes.")
    print("=" * 60)


if __name__ == "__main__":
    main()

