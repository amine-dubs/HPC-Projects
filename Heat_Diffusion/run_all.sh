#!/usr/bin/env bash
# Automated execution runner for Parallel 2D Heat Diffusion (Phase 1: Local)
set -e
TARGET="${1:-quick}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================================="
echo " Parallel 2D Heat Diffusion - Runner Target: ${TARGET}"
echo "=========================================================="

case "${TARGET}" in
    quick)
        echo "[1/4] Running Quick Correctness Check..."
        python tests/test_correctness.py --quick
        echo "[2/4] Running Quick Ping-Pong Benchmark..."
        mpirun -n 2 python src/pingpong.py --quick
        echo "[3/4] Running Quick MPI Scaling Grid..."
        python scripts/run_mpi_grid.py --quick
        echo "[4/4] Running Quick Convergence & k-Sweep..."
        python scripts/run_convergence.py --quick
        echo ">>> Quick smoke tests completed successfully! <<<"
        ;;
    test)
        python tests/test_correctness.py
        ;;
    scaling)
        python scripts/run_threads.py
        python scripts/run_mpi_grid.py
        ;;
    convergence)
        python scripts/run_convergence.py
        ;;
    pingpong)
        mpirun -n 2 python src/pingpong.py --target local --out results/raw/pingpong_local.csv
        ;;
    docker)
        echo "[Phase 2 Notice] Docker deployment is deferred to Phase 2 (Local host only in Phase 1)."
        ;;
    animation)
        python src/animate.py
        ;;
    report)
        python scripts/fit_cost_model.py
        python scripts/generate_plots.py
        python scripts/make_report_assets.py
        cd report && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex && cd ..
        cp report/main.pdf Heat_Diffusion_Report.pdf
        echo ">>> PDF generated cleanly at Heat_Diffusion_Report.pdf <<<"
        ;;
    all)
        "$0" test
        "$0" pingpong
        "$0" scaling
        "$0" convergence
        "$0" animation
        "$0" report
        echo ">>> Entire project pipeline completed successfully! <<<"
        ;;
    *)
        echo "Unknown target: ${TARGET}. Valid targets: quick, test, scaling, convergence, pingpong, docker, animation, report, all"
        exit 1
        ;;
esac
