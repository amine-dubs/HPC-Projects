"""Collect and save system and software environment metadata.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Add MS-MPI to PATH if needed on Windows
mpi_default_bin = r"C:\Program Files\Microsoft MPI\Bin"
if os.path.isdir(mpi_default_bin) and mpi_default_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = mpi_default_bin + os.pathsep + os.environ.get("PATH", "")


def get_cpu_info():
    model = platform.processor()
    physical_cores = None
    logical_cores = os.cpu_count()
    if platform.system() == "Windows":
        try:
            cmd = "Get-CimInstance Win32_Processor | Select-Object -Property Name, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json"
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True)
            data = json.loads(out)
            if isinstance(data, list):
                data = data[0]
            model = data.get("Name", model).strip()
            physical_cores = data.get("NumberOfCores", None)
            logical_cores = data.get("NumberOfLogicalProcessors", logical_cores)
        except Exception:
            pass
    return model, physical_cores, logical_cores


def get_ram_gb():
    if platform.system() == "Windows":
        try:
            cmd = "Get-CimInstance Win32_OperatingSystem | Select-Object -Property TotalVisibleMemorySize | ConvertTo-Json"
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True)
            data = json.loads(out)
            total_kb = data.get("TotalVisibleMemorySize", 0)
            return round(total_kb / (1024 * 1024), 2)
        except Exception:
            pass
    return None


def get_power_status():
    if platform.system() == "Windows":
        try:
            cmd = "Get-CimInstance Win32_Battery | Select-Object -Property BatteryStatus, EstimatedChargeRemaining | ConvertTo-Json"
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True)
            data = json.loads(out)
            status_code = data.get("BatteryStatus", 0)
            charge = data.get("EstimatedChargeRemaining", 0)
            status_str = "AC Plugged In" if status_code == 2 else f"Battery ({charge}%)"
            return status_str
        except Exception:
            pass
    return "Unknown"


def get_command_version(cmd, arg="--version"):
    exe = shutil.which(cmd)
    if not exe and platform.system() == "Windows" and cmd == "mpiexec":
        if os.path.isfile(os.path.join(mpi_default_bin, "mpiexec.exe")):
            exe = os.path.join(mpi_default_bin, "mpiexec.exe")
    if not exe:
        return {"installed": False, "version": None}
    try:
        res = subprocess.run([exe, arg], capture_output=True, text=True, timeout=5)
        text = res.stdout if res.stdout else res.stderr
        first_line = text.strip().splitlines()[0] if text.strip() else "Installed"
        return {"installed": True, "path": exe, "version": first_line}
    except Exception as e:
        return {"installed": True, "path": exe, "version": f"Error querying: {e}"}


def main():
    import numpy as np
    import numba
    import matplotlib
    import pandas as pd
    import imageio

    try:
        from mpi4py import MPI
        mpi_lib = MPI.Get_library_version()
    except Exception as e:
        mpi_lib = f"Error: {e}"

    # Check Numba threading layer
    try:
        @numba.njit(parallel=True)
        def _test_parallel(a):
            for i in numba.prange(len(a)):
                a[i] += 1
        _dummy = np.zeros(4)
        _test_parallel(_dummy)
        threading_layer = numba.threading_layer()
    except Exception as e:
        threading_layer = f"Unavailable: {e}"

    cpu_model, phys_cores, log_cores = get_cpu_info()
    ram_gb = get_ram_gb()
    power_mode = get_power_status()

    # ffmpeg check
    ffmpeg_info = {"installed": False, "version": None}
    try:
        import imageio_ffmpeg
        ff_exe = imageio_ffmpeg.get_ffmpeg_exe()
        res = subprocess.run([ff_exe, "-version"], capture_output=True, text=True, timeout=5)
        ffmpeg_info = {"installed": True, "path": ff_exe, "version": res.stdout.splitlines()[0]}
    except Exception:
        ffmpeg_info = get_command_version("ffmpeg", "-version")

    env_data = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine()
        },
        "cpu": {
            "model": cpu_model,
            "physical_cores": phys_cores,
            "logical_cores": log_cores
        },
        "ram_gb": ram_gb,
        "power_mode": power_mode,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable
        },
        "libraries": {
            "numpy": np.__version__,
            "numba": numba.__version__,
            "mpi4py": getattr(sys.modules.get("mpi4py"), "__version__", "unknown"),
            "matplotlib": matplotlib.__version__,
            "pandas": pd.__version__,
            "imageio": imageio.__version__
        },
        "mpi": {
            "library_version": mpi_lib,
            "launcher": shutil.which("mpiexec") or os.path.join(mpi_default_bin, "mpiexec.exe")
        },
        "numba_threading": {
            "active_layer": threading_layer
        },
        "tools": {
            "pdflatex": get_command_version("pdflatex", "--version"),
            "ffmpeg": ffmpeg_info,
            "docker": get_command_version("docker", "--version")
        }
    }

    out_file = Path("results/environment.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(env_data, f, indent=2)

    print(f"Environment info saved to {out_file}")
    print(f"CPU: {cpu_model} ({phys_cores} physical / {log_cores} logical cores)")
    print(f"RAM: {ram_gb} GB")
    print(f"Threading layer: {threading_layer}")
    print(f"MPI: {mpi_lib.splitlines()[0] if mpi_lib else 'None'}")


if __name__ == "__main__":
    main()

