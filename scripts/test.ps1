param(
    [switch]$Setup
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"

Set-Location $Root

if ($Setup -or -not (Test-Path $PythonExe)) {
    & (Join-Path $PSScriptRoot "setup.ps1") -CpuOnly -Dev
}

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment is missing. Run .\scripts\setup.ps1 -Dev first."
}

Write-Host "Compiling Python files..."
& $PythonExe -m compileall -q app.py audio config core gui speech storage tests
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Running tests..."
& $PythonExe -m pytest -q
exit $LASTEXITCODE
