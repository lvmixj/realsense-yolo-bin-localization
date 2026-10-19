@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python environment is missing. Run setup_windows.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Usage: test_image_windows.bat C:\path\to\image.png
    pause
    exit /b 2
)

".venv\Scripts\python.exe" "detect_image.py" "%~1"
if errorlevel 1 (
    echo.
    echo Image test failed.
    pause
    exit /b 1
)

echo.
echo Image test completed.
pause
