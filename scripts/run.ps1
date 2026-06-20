param(
    [switch]$Setup,
    [switch]$CpuOnly,
    [switch]$Dev,
    [switch]$Check
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"

Set-Location $Root

if ($Setup -or -not (Test-Path $PythonExe)) {
    $SetupArgs = @()
    if ($CpuOnly) {
        $SetupArgs += "-CpuOnly"
    }
    if ($Dev) {
        $SetupArgs += "-Dev"
    }
    & (Join-Path $PSScriptRoot "setup.ps1") @SetupArgs
}

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment is missing. Run .\scripts\setup.ps1 first."
}

if ($Check) {
    & $PythonExe -c "from PySide6.QtWidgets import QApplication; from gui.main_window import MainWindow; print('Runtime check OK')"
    exit $LASTEXITCODE
}

& $PythonExe app.py
exit $LASTEXITCODE
