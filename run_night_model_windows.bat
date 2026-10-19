@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python environment is missing. Run setup_windows.bat first.
    pause
    exit /b 1
)

if not exist "models\night_bin_yolo11s_v1.pt" (
    echo Night model is missing: models\night_bin_yolo11s_v1.pt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "d435i_yolo11_3d_detection.py" --model "models\night_bin_yolo11s_v1.pt" --print_result %*
if errorlevel 1 (
    echo.
    echo Detection exited with an error.
    pause
    exit /b 1
)
