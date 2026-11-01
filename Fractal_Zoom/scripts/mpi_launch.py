"""MPI Launch helper for MS-MPI and Open MPI.

Authors: Bellatreche Mohamed Amine, Rezaiguia Soltane
Course: Intensive Computing, USTO-MB
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def get_mpi_launcher():
    """Locate the MPI launcher executable."""
    # On Windows, check standard Microsoft MPI installation paths first
    if platform.system() == "Windows":
        for default_dir in [
            r"C:\Program Files\Microsoft MPI\Bin",
            r"C:\Program Files (x86)\Microsoft MPI\Bin",
        ]:
            exe = os.path.join(default_dir, "mpiexec.exe")
            if os.path.isfile(exe):
                return exe

    # Check PATH
    for candidate in ["mpiexec", "mpirun"]:
        path = shutil.which(candidate)
        if path:
            return path

    raise RuntimeError("No MPI launcher (mpiexec or mpirun) found.")


def build_mpi_cmd(p, script_path, script_args=None, hostfile=None, python_exe=None):
    """Build the command-line list to launch an MPI script with p processes."""
    launcher = get_mpi_launcher()
    if python_exe is None:
        python_exe = sys.executable

    cmd = [launcher, "-n", str(p)]

    # Distinguish MS-MPI vs Open MPI / MPICH
    is_ms_mpi = "microsoft mpi" in launcher.lower() or platform.system() == "Windows"

    if hostfile and os.path.isfile(hostfile):
        if is_ms_mpi:
            cmd.extend(["-machinefile", str(hostfile)])
        else:
            cmd.extend(["--hostfile", str(hostfile)])

    # Check for Open MPI oversubscription
    if not is_ms_mpi:
        cmd.append("--oversubscribe")

    # Target python script
    cmd.append(python_exe)
    cmd.append(str(script_path))

    if script_args:
        cmd.extend([str(a) for a in script_args])

    return cmd


def run_mpi_script(p, script_path, script_args=None, hostfile=None, timeout=300, env=None):
    """Execute an MPI script with single-threaded Numba/OMP constraints."""
    cmd = build_mpi_cmd(p, script_path, script_args, hostfile)

    run_env = os.environ.copy()
    run_env["NUMBA_NUM_THREADS"] = "1"
    run_env["OMP_NUM_THREADS"] = "1"
    if env:
        run_env.update(env)

    # Ensure launcher dir is in PATH for any subprocesses
    launcher = get_mpi_launcher()
    launcher_dir = os.path.dirname(launcher)
    if launcher_dir not in run_env.get("PATH", ""):
        run_env["PATH"] = launcher_dir + os.pathsep + run_env.get("PATH", "")

    cmd_str = subprocess.list2cmdline(cmd)
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=run_env
    )
    return res, cmd_str


if __name__ == "__main__":
    launcher = get_mpi_launcher()
    print(f"MPI Launcher: {launcher}")
    test_cmd = build_mpi_cmd(2, "-c", ["from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"])
    print("Test command:", subprocess.list2cmdline(test_cmd))
