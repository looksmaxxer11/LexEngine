@echo off
REM Quick setup script for Vision AI with Groq (FREE!)

echo ========================================
echo Vision AI Setup - Groq (FREE!)
echo ========================================
echo.

REM Check if groq is installed
python -c "import groq" 2>nul
if errorlevel 1 (
    echo [STEP 1] Installing Groq package...
    call venv\Scripts\python.exe -m pip install groq
    echo Done!
) else (
    echo [STEP 1] Groq package already installed
)

echo.
echo [STEP 2] Get your FREE Groq API key:
echo.
echo   1. Visit: https://console.groq.com
echo   2. Sign up FREE (no credit card!)
echo   3. Click "API Keys"
echo   4. Create new key (starts with gsk_...)
echo.
echo [STEP 3] Set the API key:
echo.
echo   PowerShell:
echo   $env:GROQ_API_KEY="gsk-your-key-here"
echo.
echo   CMD:
echo   set GROQ_API_KEY=gsk-your-key-here
echo.
echo ========================================
echo Ready to use! The web UI will automatically
echo detect and use Groq for vision OCR.
echo.
echo Start: start_server.bat
echo Then: http://127.0.0.1:8000
echo ========================================
pause
