@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
if errorlevel 1 (
    echo.
    echo Setup failed. Review the error message above.
    pause
    exit /b 1
)

echo.
echo Setup completed successfully.
pause
