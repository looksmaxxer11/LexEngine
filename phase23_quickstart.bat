@echo off
REM Phase 2 + Phase 3 Quick Start

echo ============================================
echo Phase 2 + 3 OCR Pipeline - Full Power
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
    echo Usage: phase23_quickstart.bat ^<input_pdf^> [output_json]
    echo.
    echo Example: phase23_quickstart.bat "document.pdf"
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
echo Phase 2 Features:
echo   - Multi-scale OCR (1.0x, 1.5x, 2.0x)
echo   - Confidence-based retry (up to 3 attempts)
echo   - Adaptive preprocessing
echo.
echo Phase 3 Features:
echo   - Advanced layout analysis
echo   - Multi-column detection
echo   - Reading order optimization  
echo   - Post-OCR text correction
echo   - Language-aware error fixing
echo.
echo ============================================
echo.

REM Run pipeline with Phase 2 + 3
python -m src.orchestrator --input "%INPUT_PDF%" --output "%OUTPUT_JSON%" --phase2 --phase3

echo.
echo ============================================
echo Phase 2 + 3 Processing Complete!
echo ============================================
echo.

pause
