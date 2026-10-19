param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return @("py", "-3.11")
        }

        & py -3.10 -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return @("py", "-3.10")
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }

    throw "Python was not found. Install 64-bit Python 3.11, then run this script again."
}

$PythonCommand = Find-Python
$PythonExe = $PythonCommand[0]
$PythonArgs = @()
if ($PythonCommand.Count -gt 1) {
    $PythonArgs = $PythonCommand[1..($PythonCommand.Count - 1)]
}

Write-Host "[1/4] Creating Python virtual environment..."
& $PythonExe @PythonArgs -m venv .venv

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment creation failed: $VenvPython"
}

Write-Host "[2/4] Updating pip..."
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "[3/4] Installing PyTorch from $TorchIndexUrl ..."
& $VenvPython -m pip install --index-url $TorchIndexUrl torch torchvision

Write-Host "[4/4] Installing YOLO, OpenCV and RealSense packages..."
& $VenvPython -m pip install -r requirements-windows.txt

Write-Host ""
Write-Host "Running installation verification..."
& $VenvPython verify_windows_setup.py

Write-Host ""
Write-Host "Setup completed. Run run_d435i_windows.bat to start detection."
