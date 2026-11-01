"""
tests/test_correctness.py - Experiment E1 Correctness Verification Suite

Compares serial Numba, threads, and MPI against pure NumPy reference.
Tests:
- N in {64, 120, 128} with power-of-two and non-power-of-two p (e.g. 3, 5, 6)
- Hotspot on and off
- Fixed 500 steps
- Convergence-mode verification where stop iteration MUST match identically
Outputs: results/correctness.csv and logs/correctness_test.log
"""

import os
import sys
import csv
import shutil
import subprocess
import tempfile
import numpy as np

# Ensure src is importable
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from heat_reference import solve_numpy
from heat_serial import solve_numba_serial
from heat_threads import solve_numba_threads
from heat_common import compute_checksum

def get_mpirun_cmd():
    msmpi = os.path.join(os.environ.get("MSMPI_BIN", r"C:\Program Files\Microsoft MPI\Bin"), "mpiexec.exe")
    if os.path.exists(msmpi):
        return msmpi
    which = shutil.which("mpirun") or shutil.which("mpiexec")
    return which or "mpirun"

def run_mpi_subprocess(script_name, p, N, steps, tol, check_every, hotspot, kernel="numba"):
    """
    Run MPI solver in a separate mpirun process, gather grid, and save to a temporary .npy file.
    """
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_npy = tmp.name

    func_name = "run_mpi_solver" if script_name == "heat_mpi" else "run_mpi_overlap_solver"
    solve_call = (
        f"res = run_mpi_solver(comm, {N}, {steps}, {tol}, {check_every}, '{kernel}', {hotspot})"
        if script_name == "heat_mpi"
        else f"res = run_mpi_overlap_solver(comm, {N}, {steps}, {tol}, {check_every}, {hotspot})"
    )

    runner_code = f"""
import sys
sys.path.insert(0, r"{SRC_DIR}")
from mpi4py import MPI
import numpy as np
from {script_name} import {func_name}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

{solve_call}

if rank == 0:
    np.save(r"{tmp_npy}", res["full_grid"])
    print(f"ITERS:{{res['iterations']}}")
    print(f"DIFF:{{res['final_diff']}}")
"""

    mpi_cmd = get_mpirun_cmd()
    cmd = [mpi_cmd, "-n", str(p), sys.executable, "-u", "-c", runner_code]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MPI failed with code {proc.returncode}:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")

    iters = steps
    diff = 0.0
    for line in proc.stdout.splitlines():
        if line.startswith("ITERS:"):
            iters = int(line.split(":")[1])
        elif line.startswith("DIFF:"):
            diff = float(line.split(":")[1])

    grid = np.load(tmp_npy)
    try:
        os.remove(tmp_npy)
    except OSError:
        pass

    return grid, iters, diff

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run reduced quick smoke test")
    args = parser.parse_args()

    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if args.quick:
        out_csv = os.path.join(PROJECT_DIR, "results", "quick_correctness.csv")
    else:
        out_csv = os.path.join(PROJECT_DIR, "results", "correctness.csv")
    log_file = os.path.join(PROJECT_DIR, "logs", "correctness_test.log")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    csv_fields = ["version", "kernel", "N", "p", "steps", "mode", "hotspot", "max_abs_diff", "allclose_pass", "iterations"]
    csv_rows = []

    with open(log_file, "w", encoding="utf-8") as log:
        def log_print(msg):
            print(msg)
            log.write(msg + "\n")
            log.flush()

        log_print("=========================================================")
        log_print("EXPERIMENT E1: SYSTEMATIC CORRECTNESS VERIFICATION SUITE")
        log_print("=========================================================\n")

        # -----------------------------------------------------------
        # PART 1: Fixed Step Tests (500 steps)
        # -----------------------------------------------------------
        fixed_steps = 100 if args.quick else 500

        if args.quick:
            test_configs = [
                (64, [1, 2, 4], False),
                (64, [1, 2], True),
            ]
        else:
            test_configs = [
                # (N, p_list, hotspot)
                (64, [1, 2, 4, 8], False),
                (64, [1, 2, 4], True),
                (120, [1, 2, 3, 4, 5, 6, 8], False), # non-power-of-two ranks
                (128, [1, 2, 4, 8], False),
            ]

        all_passed = True

        for N, p_list, hotspot in test_configs:
            log_print(f"\n--- Testing N={N}, steps={fixed_steps}, hotspot={hotspot} ---")
            # 1. Compute ground truth with pure NumPy reference
            log_print(f"  Computing ground truth NumPy reference (N={N})...")
            ref_grid, _, _, _ = solve_numpy(N, steps=fixed_steps, tol=0.0, check_every=0, hotspot=hotspot)

            # 2. Test Numba serial
            log_print("  Testing Version 0b (Numba serial)...")
            ser_grid, _, _, _ = solve_numba_serial(N, steps=fixed_steps, tol=0.0, check_every=0, hotspot=hotspot)
            diff_ser = float(np.max(np.abs(ref_grid - ser_grid)))
            pass_ser = np.allclose(ref_grid, ser_grid, atol=1e-12, rtol=1e-10)
            log_print(f"    Serial vs Ref: max_diff={diff_ser:.2e}, allclose={pass_ser}")
            csv_rows.append({
                "version": "serial_numba", "kernel": "numba", "N": N, "p": 1,
                "steps": fixed_steps, "mode": "fixed_steps", "hotspot": hotspot,
                "max_abs_diff": diff_ser, "allclose_pass": pass_ser, "iterations": fixed_steps
            })
            if not pass_ser:
                all_passed = False

            # 3. Test Numba threads for each p
            for p in p_list:
                log_print(f"  Testing Version 1 (Numba prange, threads={p})...")
                th_grid, _, _, _, _ = solve_numba_threads(N, steps=fixed_steps, tol=0.0, check_every=0, hotspot=hotspot)
                diff_th = float(np.max(np.abs(ref_grid - th_grid)))
                pass_th = np.allclose(ref_grid, th_grid, atol=1e-12, rtol=1e-10)
                log_print(f"    Threads (p={p}) vs Ref: max_diff={diff_th:.2e}, allclose={pass_th}")
                csv_rows.append({
                    "version": "threads_numba", "kernel": "numba", "N": N, "p": p,
                    "steps": fixed_steps, "mode": "fixed_steps", "hotspot": hotspot,
                    "max_abs_diff": diff_th, "allclose_pass": pass_th, "iterations": fixed_steps
                })
                if not pass_th:
                    all_passed = False

            # 4. Test MPI blocking Sendrecv for each p
            for p in p_list:
                log_print(f"  Testing Version 2-4 (MPI Sendrecv, p={p})...")
                mpi_grid, iters, _ = run_mpi_subprocess("heat_mpi", p, N, fixed_steps, 0.0, 0, hotspot)
                diff_mpi = float(np.max(np.abs(ref_grid - mpi_grid)))
                pass_mpi = np.allclose(ref_grid, mpi_grid, atol=1e-12, rtol=1e-10)
                log_print(f"    MPI (p={p}) vs Ref: max_diff={diff_mpi:.2e}, allclose={pass_mpi}")
                csv_rows.append({
                    "version": "mpi", "kernel": "numba", "N": N, "p": p,
                    "steps": fixed_steps, "mode": "fixed_steps", "hotspot": hotspot,
                    "max_abs_diff": diff_mpi, "allclose_pass": pass_mpi, "iterations": iters
                })
                if not pass_mpi:
                    all_passed = False

            # 5. Test MPI non-blocking overlap for each p
            for p in p_list:
                log_print(f"  Testing Version Extra (MPI overlap, p={p})...")
                ov_grid, iters, _ = run_mpi_subprocess("heat_mpi_overlap", p, N, fixed_steps, 0.0, 0, hotspot)
                diff_ov = float(np.max(np.abs(ref_grid - ov_grid)))
                pass_ov = np.allclose(ref_grid, ov_grid, atol=1e-12, rtol=1e-10)
                log_print(f"    MPI Overlap (p={p}) vs Ref: max_diff={diff_ov:.2e}, allclose={pass_ov}")
                csv_rows.append({
                    "version": "mpi_overlap", "kernel": "numba", "N": N, "p": p,
                    "steps": fixed_steps, "mode": "fixed_steps", "hotspot": hotspot,
                    "max_abs_diff": diff_ov, "allclose_pass": pass_ov, "iterations": iters
                })
                if not pass_ov:
                    all_passed = False

        # -----------------------------------------------------------
        # PART 2: Convergence Mode Test (Stop iteration must match)
        # -----------------------------------------------------------
        log_print("\n--- Testing Convergence Mode (Stop Iteration Identity) ---")
        conv_N = 64
        conv_tol = 1e-4
        conv_check_every = 20
        max_steps = 5000

        # Reference convergence run
        ref_cgrid, _, ref_iters, ref_diff = solve_numpy(
            conv_N, steps=max_steps, tol=conv_tol, check_every=conv_check_every, hotspot=False
        )
        log_print(f"  NumPy Reference stopped at iteration: {ref_iters} with diff: {ref_diff:.4e}")

        # Serial Numba convergence run
        ser_cgrid, _, ser_iters, ser_diff = solve_numba_serial(
            conv_N, steps=max_steps, tol=conv_tol, check_every=conv_check_every, hotspot=False
        )
        pass_ser_iter = (ser_iters == ref_iters)
        diff_ser_c = float(np.max(np.abs(ref_cgrid - ser_cgrid)))
        log_print(f"  Serial Numba stopped at: {ser_iters} (matches Ref: {pass_ser_iter}), max_diff={diff_ser_c:.2e}")
        csv_rows.append({
            "version": "serial_numba", "kernel": "numba", "N": conv_N, "p": 1,
            "steps": max_steps, "mode": "convergence", "hotspot": False,
            "max_abs_diff": diff_ser_c, "allclose_pass": pass_ser_iter, "iterations": ser_iters
        })
        if not pass_ser_iter:
            all_passed = False

        # Threads Numba convergence runs for p in {1, 2, 4}
        th_p_list = [1, 2] if args.quick else [1, 2, 4]
        for p in th_p_list:
            th_cgrid, _, th_iters, th_diff, _ = solve_numba_threads(
                conv_N, steps=max_steps, tol=conv_tol, check_every=conv_check_every, hotspot=False
            )
            pass_th_iter = (th_iters == ref_iters)
            diff_th_c = float(np.max(np.abs(ref_cgrid - th_cgrid)))
            log_print(f"  Threads p={p} stopped at: {th_iters} (matches Ref: {pass_th_iter}), max_diff={diff_th_c:.2e}")
            csv_rows.append({
                "version": "threads_numba", "kernel": "numba", "N": conv_N, "p": p,
                "steps": max_steps, "mode": "convergence", "hotspot": False,
                "max_abs_diff": diff_th_c, "allclose_pass": pass_th_iter, "iterations": th_iters
            })
            if not pass_th_iter:
                all_passed = False

        # MPI convergence runs for p in {1, 2, 4, 8}
        mpi_p_list = [1, 2] if args.quick else [1, 2, 4, 8]
        for p in mpi_p_list:
            mpi_cgrid, mpi_iters, mpi_diff = run_mpi_subprocess(
                "heat_mpi", p, conv_N, max_steps, conv_tol, conv_check_every, False
            )
            pass_mpi_iter = (mpi_iters == ref_iters)
            diff_mpi_c = float(np.max(np.abs(ref_cgrid - mpi_cgrid)))
            log_print(f"  MPI p={p} stopped at: {mpi_iters} (matches Ref: {pass_mpi_iter}), max_diff={diff_mpi_c:.2e}")
            csv_rows.append({
                "version": "mpi", "kernel": "numba", "N": conv_N, "p": p,
                "steps": max_steps, "mode": "convergence", "hotspot": False,
                "max_abs_diff": diff_mpi_c, "allclose_pass": pass_mpi_iter, "iterations": mpi_iters
            })
            if not pass_mpi_iter:
                all_passed = False

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(csv_rows)

        log_print(f"\nCorrectness test results written to {out_csv}")
        if all_passed:
            log_print(">>> ALL CORRECTNESS AND CONVERGENCE TESTS PASSED! <<<")
        else:
            log_print(">>> SOME TESTS FAILED. CHECK LOGS. <<<")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
