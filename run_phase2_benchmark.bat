@echo off
REM Phase 2 Benchmark Script

echo ============================================
echo Phase 2 Optimization Benchmark
echo ============================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check command line arguments
if "%1"=="" (
    echo Usage: run_phase2_benchmark.bat ^<input_pdf^>
    echo.
    echo Example: run_phase2_benchmark.bat "data\raw_pdfs\sample.pdf"
    echo.
    pause
    exit /b 1
)

set INPUT_PDF=%~1

echo Input PDF: %INPUT_PDF%
echo.
echo This will benchmark:
echo   1. Standard OCR
echo   2. Multi-scale OCR
echo   3. Confidence Retry
echo   4. Smart Orchestrator (Full Phase 2)
echo.
echo ============================================
echo.

REM Run benchmark
python test_phase2_optimizations.py "%INPUT_PDF%"

echo.
pause
