@echo off
REM Batch Process All PDFs in Input Folder
REM Automatically processes all PDF files in the input directory

setlocal enabledelayedexpansion

echo.
echo ========================================================
echo   DeepSeek Document Intelligence - Batch Processor
echo ========================================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
    echo.
)

REM Check if input directory exists
if not exist "input" (
    echo Error: input directory not found!
    echo Creating input directory...
    mkdir input
    echo.
    echo Please place PDF files in the input folder and run again.
    pause
    exit /b 1
)

REM Count PDF files
set /a count=0
for %%f in (input\*.pdf) do set /a count+=1

if %count%==0 (
    echo No PDF files found in input directory!
    echo.
    echo Please place PDF files in the input folder.
    echo.
    pause
    exit /b 1
)

echo Found %count% PDF file(s) in input directory
echo.
echo Processing all PDFs...
echo.

REM Run batch processing
python cli.py batch --input-dir input --output-dir output

REM Check result
if errorlevel 1 (
    echo.
    echo ========================================================
    echo   Some files may have failed processing
    echo ========================================================
    echo.
    echo Check logs\deepseek_pipeline.log for details
) else (
    echo.
    echo ========================================================
    echo   SUCCESS! All documents processed
    echo ========================================================
    echo.
    echo Output files saved to: output\
)

echo.
echo Opening output folder...
start "" "output"

echo.
pause
