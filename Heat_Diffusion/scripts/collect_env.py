#!/usr/bin/env python3
"""
scripts/collect_env.py - Environment capture (Experiment E0)
Collects CPU details, core counts, RAM, OS version, package versions, and MPI information.
Saves to results/environment.json.
"""

import json
import os
import platform
import sys
import psutil

def get_cpu_model():
    # Try Windows CIM / registry first
    if platform.system() == "Windows":
        try:
            import subprocess
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True
            ).strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor()

def collect_environment(output_path="results/environment.json"):
    env = {}
    env["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine()
    }
    
    cpu_model = get_cpu_model()
    env["cpu"] = {
        "model": cpu_model,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
    }
    
    mem = psutil.virtual_memory()
    env["ram"] = {
        "total_gb": round(mem.total / (1024 ** 3), 2),
        "available_gb": round(mem.available / (1024 ** 3), 2),
        "percent_used": mem.percent
    }
    
    # Load average or CPU percent before runs
    try:
        load_avg = os.getloadavg()
    except AttributeError:
        # Windows does not have getloadavg
        load_avg = [psutil.cpu_percent(interval=0.5)]
    env["load_before_runs"] = load_avg

    # Python & packages
    import numpy as np
    import numba
    import mpi4py
    from mpi4py import MPI
    import matplotlib
    import imageio

    # Check Numba threading layer
    from numba import njit, prange
    @njit(parallel=True)
    def _warmup_prange():
        s = 0
        for i in prange(4): s += i
        return s
    _warmup_prange()
    threading_layer = numba.threading_layer()

    mpi_vendor = MPI.get_vendor()
    mpi_lib_version = MPI.Get_library_version()

    env["software"] = {
        "python": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "numba_threading_layer": threading_layer,
        "mpi4py": mpi4py.__version__,
        "mpi_version": MPI.Get_version(),
        "mpi_vendor": mpi_vendor,
        "mpi_library": mpi_lib_version,
        "matplotlib": matplotlib.__version__,
        "imageio": imageio.__version__,
        "docker": "Skipped (Phase 1 Local Only scope)"
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    print(f"Environment captured successfully to {output_path}")
    print(f"CPU: {cpu_model} ({env['cpu']['physical_cores']} physical, {env['cpu']['logical_cores']} logical cores)")
    print(f"RAM: {env['ram']['total_gb']} GB")
    print(f"MPI: {mpi_lib_version}")
    print(f"Numba threading layer: {threading_layer}")
    return env

if __name__ == "__main__":
    out = "results/environment.json"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    collect_environment(out)

