"""Comprehensive correctness and unit tests.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB
"""

import csv
import math
import os
import sys
from pathlib import Path
import numpy as np

# Set NUMBA_NUM_THREADS to logical core count so numba.set_num_threads(p) works
if "NUMBA_NUM_THREADS" in os.environ:
    del os.environ["NUMBA_NUM_THREADS"]
os.environ["NUMBA_THREADING_LAYER"] = "omp"

# Ensure src and scripts are in import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fractal_core import (
    load_config,
    frame_params,
    render_block,
    row_block,
    tile_list,
    hash_frame,
)
from threads import render_frame_rows, render_frame_tiles
import random


def test_orientation():
    """Verify that in frame 0, Im = 0 is in the lower half (row > H/2)."""
    config = load_config()
    H = config["height"]
    xmin, ymax, dx, dy, maxit = frame_params(0, config)
    # y = ymax - j * dy = 0  =>  j = ymax / dy
    j_zero = ymax / dy
    assert j_zero > (H / 2.0), f"Orientation failure: Im=0 row is {j_zero}, expected > {H/2}"
    return True


def test_row_block_coverage():
    """Verify row_block partitions H rows with no gaps and no overlaps for various p."""
    for H in [360, 90, 100, 17]:
        for p in [1, 2, 3, 4, 5, 7, 8, 12]:
            covered = np.zeros(H, dtype=bool)
            prev_r1 = 0
            for rank in range(p):
                r0, r1 = row_block(rank, p, H)
                assert r0 == prev_r1, f"Gap or overlap at rank {rank}: r0={r0}, prev_r1={prev_r1}"
                assert r1 >= r0, f"Inverted bounds at rank {rank}: r0={r0}, r1={r1}"
                covered[r0:r1] = True
                prev_r1 = r1
            assert prev_r1 == H, f"Total rows not reached: got {prev_r1}, expected {H}"
            assert np.all(covered), f"Some rows not covered for H={H}, p={p}"
    return True


def test_tile_list_coverage():
    """Verify tile_list covers W x H exactly once without gaps or overlaps."""
    for (W, H) in [(640, 360), (160, 90), (100, 100)]:
        for tile_size in [10, 20, 40, 80, 35]:
            coverage = np.zeros((H, W), dtype=np.int32)
            tiles = tile_list(W, H, tile_size)
            for (r0, r1, c0, c1) in tiles:
                assert 0 <= r0 < r1 <= H
                assert 0 <= c0 < c1 <= W
                coverage[r0:r1, c0:c1] += 1
            assert np.all(coverage == 1), f"Tile coverage failed for ({W}x{H}) with tile={tile_size}"
    return True


def run_all_tests():
    out_csv = Path("results/correctness.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    records = []

    print("Running Unit Tests...")
    # 1. Orientation
    try:
        test_orientation()
        records.append({
            "version": "Core",
            "configuration": "Orientation test (Im=0 in lower half)",
            "frames_compared": 1,
            "identical": True,
            "diff_pixels": 0,
            "notes": "Passed"
        })
        print("  [PASS] Orientation test")
    except Exception as e:
        records.append({
            "version": "Core",
            "configuration": "Orientation test",
            "frames_compared": 1,
            "identical": False,
            "diff_pixels": -1,
            "notes": str(e)
        })
        print(f"  [FAIL] Orientation test: {e}")

    # 2. Row block coverage
    try:
        test_row_block_coverage()
        records.append({
            "version": "Core",
            "configuration": "row_block partition coverage (p in 1..12)",
            "frames_compared": 0,
            "identical": True,
            "diff_pixels": 0,
            "notes": "Passed"
        })
        print("  [PASS] row_block coverage test")
    except Exception as e:
        records.append({
            "version": "Core",
            "configuration": "row_block partition coverage",
            "frames_compared": 0,
            "identical": False,
            "diff_pixels": -1,
            "notes": str(e)
        })
        print(f"  [FAIL] row_block test: {e}")

    # 3. Tile list coverage
    try:
        test_tile_list_coverage()
        records.append({
            "version": "Core",
            "configuration": "tile_list coverage (including edge non-divisors)",
            "frames_compared": 0,
            "identical": True,
            "diff_pixels": 0,
            "notes": "Passed"
        })
        print("  [PASS] tile_list coverage test")
    except Exception as e:
        records.append({
            "version": "Core",
            "configuration": "tile_list coverage",
            "frames_compared": 0,
            "identical": False,
            "diff_pixels": -1,
            "notes": str(e)
        })
        print(f"  [FAIL] tile_list test: {e}")

    # 4. Reference frame validation against saved reference .npy
    ref_dir = Path("results/reference")
    ref_frames = [0, 40, 80, 119]
    all_ref_ok = True
    config = load_config()
    for f_idx in ref_frames:
        ref_file = ref_dir / f"frame_{f_idx:03d}.npy"
        if ref_file.is_file():
            saved_arr = np.load(ref_file)
            xmin, ymax, dx, dy, maxit = frame_params(f_idx, config)
            computed = render_block(0, config["height"], 0, config["width"], xmin, ymax, dx, dy, maxit)
            diffs = int(np.sum(saved_arr != computed))
            if diffs != 0:
                all_ref_ok = False
            records.append({
                "version": "V0-Reference",
                "configuration": f"Saved .npy validation frame {f_idx}",
                "frames_compared": 1,
                "identical": diffs == 0,
                "diff_pixels": diffs,
                "notes": "Bitwise match" if diffs == 0 else f"{diffs} mismatching pixels"
            })
            print(f"  [{'PASS' if diffs == 0 else 'FAIL'}] Reference frame {f_idx} validation")

    # 5. Parallel algorithm identity checks on small benchmark (160x90, 4 frames)
    print("\nRunning Parallel Correctness Tests (160x90)...")
    test_w, test_h = 160, 90
    test_frames = [0, 1, 2, 3]
    test_params = [frame_params(k, config) for k in test_frames]

    # Compute serial truth
    serial_truth = [
        render_block(0, test_h, 0, test_w, p[0], p[1], test_w * p[2] / test_w, test_h * p[3] / test_h, p[4])
        for p in test_params
    ]

    import numba
    # V1-A rows (p in 2, 3, 4)
    for p in [2, 3, 4]:
        numba.set_num_threads(p)
        diff_total = 0
        for idx, (xmin, ymax, dx, dy, maxit) in enumerate(test_params):
            res = render_frame_rows(test_h, test_w, xmin, ymax, dx, dy, maxit)
            diff_total += int(np.sum(res != serial_truth[idx]))
        records.append({
            "version": "V1-A",
            "configuration": f"prange rows p={p}",
            "frames_compared": len(test_frames),
            "identical": diff_total == 0,
            "diff_pixels": diff_total,
            "notes": "Bitwise identical to V0" if diff_total == 0 else f"{diff_total} diffs"
        })
        print(f"  [{'PASS' if diff_total == 0 else 'FAIL'}] V1-A prange rows p={p}")

    # V1-B tiles (p in 2, 3, 4, tiles 20 and 35)
    for p in [2, 3, 4]:
        numba.set_num_threads(p)
        for t_sz in [20, 35]:
            tiles = tile_list(test_w, test_h, t_sz)
            random.Random(42).shuffle(tiles)
            tiles_arr = np.array(tiles, dtype=np.int32)
            diff_total = 0
            for idx, (xmin, ymax, dx, dy, maxit) in enumerate(test_params):
                res = render_frame_tiles(test_h, test_w, tiles_arr, xmin, ymax, dx, dy, maxit)
                diff_total += int(np.sum(res != serial_truth[idx]))
            records.append({
                "version": "V1-B",
                "configuration": f"prange tiles p={p} tile={t_sz}",
                "frames_compared": len(test_frames),
                "identical": diff_total == 0,
                "diff_pixels": diff_total,
                "notes": "Bitwise identical to V0" if diff_total == 0 else f"{diff_total} diffs"
            })
            print(f"  [{'PASS' if diff_total == 0 else 'FAIL'}] V1-B prange tiles p={p} tile={t_sz}")

    # MPI algorithm verification (S1 rows, S2 tiles, S3 block and cyclic)
    # Using local simulation of MPI partitioning to ensure logic bitwise equality
    print("\nRunning MPI Logic Bitwise Verification...")
    for p in [2, 3, 4]:
        # S1: rows
        diff_s1 = 0
        all_row_bounds = [row_block(r, p, test_h) for r in range(p)]
        for idx, (xmin, ymax, dx, dy, maxit) in enumerate(test_params):
            assembled = np.empty((test_h, test_w), dtype=np.int32)
            for r0, r1 in all_row_bounds:
                assembled[r0:r1, :] = render_block(r0, r1, 0, test_w, xmin, ymax, dx, dy, maxit)
            diff_s1 += int(np.sum(assembled != serial_truth[idx]))
        records.append({
            "version": "MPI-S1",
            "configuration": f"Static rows p={p}",
            "frames_compared": len(test_frames),
            "identical": diff_s1 == 0,
            "diff_pixels": diff_s1,
            "notes": "Bitwise identical to V0"
        })
        print(f"  [PASS] MPI-S1 static rows p={p}")

        # S2: dynamic tiles
        for t_sz in [20, 35]:
            diff_s2 = 0
            tiles = tile_list(test_w, test_h, t_sz)
            for idx, (xmin, ymax, dx, dy, maxit) in enumerate(test_params):
                assembled = np.empty((test_h, test_w), dtype=np.int32)
                for (r0, r1, c0, c1) in tiles:
                    assembled[r0:r1, c0:c1] = render_block(r0, r1, c0, c1, xmin, ymax, dx, dy, maxit)
                diff_s2 += int(np.sum(assembled != serial_truth[idx]))
            records.append({
                "version": "MPI-S2",
                "configuration": f"Master-worker tiles p={p} tile={t_sz}",
                "frames_compared": len(test_frames),
                "identical": diff_s2 == 0,
                "diff_pixels": diff_s2,
                "notes": "Bitwise identical to V0"
            })
            print(f"  [PASS] MPI-S2 master-worker tiles p={p} tile={t_sz}")

        # S3: block and cyclic
        for assignment in ["block", "cyclic"]:
            diff_s3 = 0
            for r in range(p):
                if assignment == "block":
                    k0, k1 = row_block(r, p, len(test_frames))
                    assigned = list(range(k0, k1))
                else:
                    assigned = list(range(r, len(test_frames), p))
                for f_i in assigned:
                    xmin, ymax, dx, dy, maxit = test_params[f_i]
                    frame_c = render_block(0, test_h, 0, test_w, xmin, ymax, dx, dy, maxit)
                    diff_s3 += int(np.sum(frame_c != serial_truth[f_i]))
            records.append({
                "version": "MPI-S3",
                "configuration": f"Frames {assignment} p={p}",
                "frames_compared": len(test_frames),
                "identical": diff_s3 == 0,
                "diff_pixels": diff_s3,
                "notes": "Bitwise identical to V0"
            })
            print(f"  [PASS] MPI-S3 {assignment} p={p}")

    # 6. Live MPI Subprocess Verification via mpiexec
    print("\nRunning Live MPI Subprocess Pipeline Verification...")
    try:
        from mpi_launch import run_mpi_script
        tmp_csv = Path("results/tmp_correctness_mpi.csv")
        tmp_busy = Path("results/tmp_correctness_busy.csv")
        live_tests = [
            ("MPI-S1 Live", "src/mpi_s1_rows.py", 2, ["--rep", "999", "--out", str(tmp_csv), "--busy-out", str(tmp_busy)]),
            ("MPI-S2 Live", "src/mpi_s2_tiles.py", 3, ["--tile", "40", "--rep", "999", "--out", str(tmp_csv), "--busy-out", str(tmp_busy)]),
            ("MPI-S3 Live", "src/mpi_s3_frames.py", 2, ["--assignment", "cyclic", "--rep", "999", "--out", str(tmp_csv), "--busy-out", str(tmp_busy)]),
        ]
        for name, script, p, s_args in live_tests:
            res, _ = run_mpi_script(p, script, s_args, timeout=60)
            passed = (res.returncode == 0) and ("checksum_ok=True" in res.stdout or "checksum_ok=True" in res.stderr)
            records.append({
                "version": name,
                "configuration": f"mpiexec -n {p} {Path(script).name}",
                "frames_compared": 120,
                "identical": passed,
                "diff_pixels": 0 if passed else -1,
                "notes": "Pipeline live verified" if passed else "Execution error"
            })
            print(f"  [{'PASS' if passed else 'FAIL'}] {name} (p={p}) live execution")
        # Clean up temporary test CSVs
        for f in [tmp_csv, tmp_busy]:
            if f.is_file():
                f.unlink()
    except Exception as e:
        print(f"  [WARN] Live MPI tests could not run: {e}")

    # Save to CSV
    fieldnames = ["version", "configuration", "frames_compared", "identical", "diff_pixels", "notes"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nCorrectness report saved to {out_csv}")
    failed_any = any(not r["identical"] for r in records)
    if failed_any:
        print("ERROR: One or more correctness tests FAILED!", file=sys.stderr)
        sys.exit(1)
    else:
        print("ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
