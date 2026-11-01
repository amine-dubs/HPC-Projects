<#
.SYNOPSIS
    Automated execution runner for Parallel 2D Heat Diffusion (Phase 1: Local)
.DESCRIPTION
    Supported targets: quick, test, scaling, convergence, pingpong, docker, animation, report, all
#>
param (
    [Parameter(Position=0)]
    [ValidateSet("quick", "test", "scaling", "convergence", "pingpong", "docker", "animation", "report", "all")]
    [string]$Target = "quick"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Parallel 2D Heat Diffusion - Runner Target: $Target" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

switch ($Target) {
    "quick" {
        Write-Host "`n[1/4] Running Quick Correctness Check..." -ForegroundColor Green
        python tests/test_correctness.py --quick
        Write-Host "`n[2/4] Running Quick Ping-Pong Benchmark..." -ForegroundColor Green
        mpirun -n 2 python src/pingpong.py --quick
        Write-Host "`n[3/4] Running Quick MPI Scaling Grid..." -ForegroundColor Green
        python scripts/run_mpi_grid.py --quick
        Write-Host "`n[4/4] Running Quick Convergence & k-Sweep..." -ForegroundColor Green
        python scripts/run_convergence.py --quick
        Write-Host "`n>>> Quick smoke tests completed successfully! <<<" -ForegroundColor Green
    }
    "test" {
        Write-Host "`nRunning Full Correctness Suite (E1)..." -ForegroundColor Green
        python tests/test_correctness.py
    }
    "scaling" {
        Write-Host "`nRunning Thread Scaling Grid (E2)..." -ForegroundColor Green
        python scripts/run_threads.py
        Write-Host "`nRunning MPI Strong Scaling & Overlap Grid (E3 & E8)..." -ForegroundColor Green
        python scripts/run_mpi_grid.py
    }
    "convergence" {
        Write-Host "`nRunning Full Convergence & k-Sweep Study (E4)..." -ForegroundColor Green
        python scripts/run_convergence.py
    }
    "pingpong" {
        Write-Host "`nRunning Full Ping-Pong Benchmark (E5)..." -ForegroundColor Green
        mpirun -n 2 python src/pingpong.py --target local --out results/raw/pingpong_local.csv
    }
    "docker" {
        Write-Host "`n[Phase 2 Notice] Docker container cluster deployment is deferred to Phase 2." -ForegroundColor Yellow
        Write-Host "Local host execution with MS-MPI is active in Phase 1." -ForegroundColor Yellow
    }
    "animation" {
        Write-Host "`nGenerating Visualizations and Animated GIF (E9)..." -ForegroundColor Green
        python src/animate.py
    }
    "report" {
        Write-Host "`n[1/4] Fitting Analytical Cost Model (E6)..." -ForegroundColor Green
        python scripts/fit_cost_model.py
        Write-Host "`n[2/4] Generating Publication Figures..." -ForegroundColor Green
        python scripts/generate_plots.py
        Write-Host "`n[3/4] Compiling Numbers and LaTeX Tables..." -ForegroundColor Green
        python scripts/make_report_assets.py
        Write-Host "`n[4/4] Building LaTeX PDF Report with pdflatex..." -ForegroundColor Green
        Push-Location report
        pdflatex -interaction=nonstopmode main.tex
        pdflatex -interaction=nonstopmode main.tex
        Pop-Location
        Copy-Item -Path "report/main.pdf" -Destination "Heat_Diffusion_Report.pdf" -Force
        Write-Host "`n>>> PDF generated cleanly at Heat_Diffusion_Report.pdf <<<" -ForegroundColor Green
    }
    "all" {
        Write-Host "`nExecuting Complete Workflow..." -ForegroundColor Green
        & $PSCommandPath test
        & $PSCommandPath pingpong
        & $PSCommandPath scaling
        & $PSCommandPath convergence
        & $PSCommandPath animation
        & $PSCommandPath report
        Write-Host "`n>>> Entire project pipeline completed successfully! <<<" -ForegroundColor Green
    }
}
