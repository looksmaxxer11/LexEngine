@echo off
setlocal
REM Fast launcher for Tesseract OCR pipeline server

REM Activate Python venv if present
if exist venv\Scripts\activate.bat (
  call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)

REM Set data root
if not defined DATA_ROOT set DATA_ROOT=data

echo.
echo ============================================
echo   Tesseract OCR Pipeline Server
echo ============================================
echo   Server will start on: http://127.0.0.1:8000
echo   Professional UI: http://127.0.0.1:8000
echo   Uses system Tesseract (no heavy model preload)
echo   Ensure Tesseract-OCR is installed on Windows
echo ============================================
echo.

python -m uvicorn server:app --host 127.0.0.1 --port 8000 --log-level info
endlocal
