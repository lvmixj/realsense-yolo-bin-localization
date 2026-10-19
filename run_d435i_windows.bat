@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python environment is missing. Run setup_windows.bat first.
    pause
    exit /b 1
)

rem Skip warm-up for faster camera startup; the first inference can be slower.
rem Conditional exit prevents cmd from prompting after Ctrl+C.
".venv\Scripts\python.exe" "d435i_yolo11_3d_detection.py" --print_result --no_warmup %* && exit /b 0 || exit /b 1
