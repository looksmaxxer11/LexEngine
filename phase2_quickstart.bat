@echo off
REM Phase 2 Quick Start - Run OCR Pipeline with Phase 2 Optimizations

echo ============================================
echo Phase 2 OCR Pipeline - Quick Start
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
    echo Usage: phase2_quickstart.bat ^<input_pdf^> [output_json]
    echo.
    echo Example: phase2_quickstart.bat "C:\Users\looksmaxxer11\Desktop\needs scanning\2022\document.pdf"
    echo.
    pause
    exit /b 1
)

REM Set input and output paths
set INPUT_PDF=%~1
set OUTPUT_JSON=%~2

REM If no output specified, use default
if "%OUTPUT_JSON%"=="" (
    set OUTPUT_JSON=output\result.json
)

echo Input PDF: %INPUT_PDF%
echo Output JSON: %OUTPUT_JSON%
echo.
echo Phase 2 Features Enabled:
echo   - Multi-scale OCR (1.0x, 1.5x, 2.0x)
echo   - Confidence-based retry (up to 3 attempts)
echo   - Adaptive preprocessing strategies
echo.
echo ============================================
echo.

REM Run pipeline with Phase 2
python -m src.orchestrator --input "%INPUT_PDF%" --output "%OUTPUT_JSON%" --phase2

echo.
echo ============================================
echo Phase 2 Processing Complete!
echo ============================================
echo.

pause
