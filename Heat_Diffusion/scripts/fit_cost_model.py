"""
scripts/fit_cost_model.py - HPC Cost Model Derivation & Validation (Experiment E6)

Formulation (from Section 4.1):
- Ping-pong model: t_one_way(S) = alpha + S / beta
  Fitted using linear regression on results/raw/pingpong_local.csv.
- Compute model: T_comp_step = (N^2 / p) * t_c
  t_c is fitted from p=1 compute time per cell:
  t_c = mean( T_comp(N, p=1) / (steps * N^2) )
- Halo exchange per step (for 1D decomposition):
  Interior ranks exchange two messages of 8N bytes (N doubles):
  T_halo_step = 2 * (alpha + 8N / beta)
- Predicted Communication Share:
  comm_share_pred = T_halo_step / (T_comp_step + T_halo_step)
- Validation:
  Compare comm_share_pred against comm_share_meas from results/raw/scaling_local.csv
  for N in {512, 1024, 2048} and p in {2, 4, 8}.
Outputs:
- results/cost_model.csv
- Prints fitted parameters: alpha (us), beta (GB/s or MB/s), t_c (ns/cell).
"""

import os
import sys
import csv
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def fit_pingpong(pingpong_csv=None):
    if pingpong_csv is None:
        pingpong_csv = os.path.join(PROJECT_DIR, "results", "raw", "pingpong_local.csv")
    if not os.path.exists(pingpong_csv):
        raise FileNotFoundError(f"Pingpong file not found: {pingpong_csv}")

    sizes = []
    times = []
    with open(pingpong_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            b = float(r["bytes"])
            t = float(r["one_way_sec"])
            # Focus regression on relevant range for halos (4 KB to 64 KB) or entire range
            sizes.append(b)
            times.append(t)

    sizes = np.array(sizes)
    times = np.array(times)

    # We fit over the message sizes relevant to our halos:
    # N=512 -> 4096 B, N=1024 -> 8192 B, N=2048 -> 16384 B
    # Let's use the range 1024 <= size <= 65536 for high-accuracy halo regime
    mask = (sizes >= 1024) & (sizes <= 65536)
    if np.sum(mask) >= 3:
        fit_s = sizes[mask]
        fit_t = times[mask]
    else:
        fit_s = sizes
        fit_t = times

    # Linear regression: time = alpha + (1/beta) * bytes
    p = np.polyfit(fit_s, fit_t, 1)
    slope = p[0]       # 1 / beta (sec / byte)
    intercept = p[1]   # alpha (sec)

    alpha_sec = max(intercept, 1e-7)
    beta_bytes_sec = 1.0 / slope if slope > 0 else 5e9

    return alpha_sec, beta_bytes_sec

def fit_tc(scaling_csv=None):
    if scaling_csv is None:
        scaling_csv = os.path.join(PROJECT_DIR, "results", "raw", "scaling_local.csv")
    if not os.path.exists(scaling_csv):
        raise FileNotFoundError(f"Scaling file not found: {scaling_csv}")

    tc_estimates = []
    with open(scaling_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            v = r.get("version", "")
            k = r.get("kernel", "")
            p_val = int(r.get("p") or r.get("size") or 1)
            if ("mpi" in v) and (k == "numba") and (p_val == 1):
                N = int(r["N"])
                steps = int(r["steps"])
                t_comp = float(r.get("t_compute_mean") or r.get("t_compute_max") or r["t_total"])
                per_cell_sec = t_comp / (steps * (N ** 2))
                tc_estimates.append((N, per_cell_sec))

    if not tc_estimates:
        # Fallback to serial
        with open(scaling_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                v = r.get("version", "")
                k = r.get("kernel", "")
                if ("serial" in v) and (k == "numba"):
                    N = int(r["N"])
                    steps = int(r["steps"])
                    t_comp = float(r["t_total"])
                    per_cell_sec = t_comp / (steps * (N ** 2))
                    tc_estimates.append((N, per_cell_sec))

    med_tc = float(np.median([tc[1] for tc in tc_estimates])) if tc_estimates else 1.5e-9
    return med_tc, tc_estimates

def evaluate_cost_model(out_csv=None,
                        scaling_csv=None,
                        pingpong_csv=None):
    if out_csv is None:
        out_csv = os.path.join(PROJECT_DIR, "results", "cost_model.csv")
    if scaling_csv is None:
        scaling_csv = os.path.join(PROJECT_DIR, "results", "raw", "scaling_local.csv")
    if pingpong_csv is None:
        pingpong_csv = os.path.join(PROJECT_DIR, "results", "raw", "pingpong_local.csv")
    alpha_sec, beta_bytes_sec = fit_pingpong(pingpong_csv)
    tc_sec, tc_list = fit_tc(scaling_csv)

    print("=========================================================")
    print("EXPERIMENT E6: COST MODEL FITTING")
    print("=========================================================")
    print(f"Latency (alpha):        {alpha_sec * 1e6:.3f} us")
    print(f"Bandwidth (beta):       {beta_bytes_sec / (1024**2):.2f} MB/s ({beta_bytes_sec / (1024**3):.2f} GB/s)")
    print(f"Compute cell time (tc): {tc_sec * 1e9:.3f} ns/cell")
    for N_val, tc_val in tc_list:
        print(f"  - N={N_val}: {tc_val * 1e9:.3f} ns/cell")
    print("---------------------------------------------------------")

    # Read measured scaling data
    measured_data = {}
    with open(scaling_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if ("mpi" in r.get("version", "")) and (r.get("kernel") == "numba") and (r.get("version") != "mpi_overlap"):
                N = int(r["N"])
                p = int(r.get("p") or r.get("size") or 1)
                key = (N, p)
                t_comp = float(r.get("t_compute_mean") or r.get("t_compute_max") or 0.0)
                t_halo = float(r.get("t_halo_mean") or r.get("t_halo_max") or 0.0)
                t_tot = float(r["t_total"])
                steps = int(r["steps"])

                if key not in measured_data:
                    measured_data[key] = []
                measured_data[key].append({
                    "steps": steps,
                    "t_comp": t_comp,
                    "t_halo": t_halo,
                    "t_tot": t_tot
                })

    rows = []
    print(f"{'N':>5} {'p':>3} {'T_comp_pred(s)':>15} {'T_halo_pred(s)':>15} {'Pred Comm%':>12} {'Meas Comm%':>12} {'Abs Diff%':>10}")
    print("-" * 75)

    for (N, p), rep_list in sorted(measured_data.items()):
        if p == 1:
            continue
        # Median across reps (slowest-rep extremes excluded, per instructions.md 3.3)
        steps = rep_list[0]["steps"]
        meas_comp = np.median([x["t_comp"] for x in rep_list])
        meas_halo = np.median([x["t_halo"] for x in rep_list])
        meas_tot = np.median([x["t_tot"] for x in rep_list])

        meas_comm_share = (meas_halo / (meas_comp + meas_halo) * 100.0) if (meas_comp + meas_halo) > 0 else 0.0

        # Model predictions per step
        t_comp_step_pred = ((N * N) / p) * tc_sec
        msg_bytes = 8 * N
        t_halo_step_pred = 2.0 * (alpha_sec + msg_bytes / beta_bytes_sec)

        # Scale to total steps
        T_comp_pred_tot = t_comp_step_pred * steps
        T_halo_pred_tot = t_halo_step_pred * steps
        T_tot_pred = T_comp_pred_tot + T_halo_pred_tot
        pred_comm_share = (T_halo_pred_tot / T_tot_pred * 100.0)

        abs_diff = abs(pred_comm_share - meas_comm_share)

        print(f"{N:5d} {p:3d} {T_comp_pred_tot:15.4f} {T_halo_pred_tot:15.4f} {pred_comm_share:11.2f}% {meas_comm_share:11.2f}% {abs_diff:9.2f}%")

        rows.append({
            "N": N,
            "p": p,
            "steps": steps,
            "alpha_us": alpha_sec * 1e6,
            "beta_MBs": beta_bytes_sec / (1024**2),
            "tc_ns": tc_sec * 1e9,
            "T_comp_pred_sec": T_comp_pred_tot,
            "T_halo_pred_sec": T_halo_pred_tot,
            "T_tot_pred_sec": T_tot_pred,
            "pred_comm_share_pct": pred_comm_share,
            "meas_comp_sec": meas_comp,
            "meas_halo_sec": meas_halo,
            "meas_tot_sec": meas_tot,
            "meas_comm_share_pct": meas_comm_share,
            "abs_error_pct": abs_diff
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "N", "p", "steps", "alpha_us", "beta_MBs", "tc_ns",
            "T_comp_pred_sec", "T_halo_pred_sec", "T_tot_pred_sec", "pred_comm_share_pct",
            "meas_comp_sec", "meas_halo_sec", "meas_tot_sec", "meas_comm_share_pct", "abs_error_pct"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCost model table saved to {out_csv}")
    return alpha_sec, beta_bytes_sec, tc_sec

if __name__ == "__main__":
    evaluate_cost_model()
