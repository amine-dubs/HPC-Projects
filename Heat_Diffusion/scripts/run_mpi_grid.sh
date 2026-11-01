#!/usr/bin/env bash
# scripts/run_mpi_grid.sh - Shell wrapper for MPI scaling and overlap grid (E3 & E8)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
python scripts/run_mpi_grid.py "$@"

