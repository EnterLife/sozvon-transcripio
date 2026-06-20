param(
    [switch]$CpuOnly,
    [switch]$Dev,
    [switch]$Recreate,
    [switch]$UseSystemProxy,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Venv = Join-Path $Root ".venv"
$PythonExe = Join-Path $Venv "Scripts\python.exe"

Set-Location $Root

if (-not $UseSystemProxy) {
    $env:NO_PROXY = "*"
    $env:no_proxy = "*"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

if ($Recreate -and (Test-Path $Venv)) {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -LiteralPath $Venv -Recurse -Force
}

if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating virtual environment in .venv..."
    Invoke-Checked -Command $Python -Arguments @("-m", "venv", $Venv)
}

Write-Host "Upgrading pip..."
Invoke-Checked -Command $PythonExe -Arguments @("-m", "pip", "install", "--upgrade", "pip")

$Extras = @()
if (-not $CpuOnly) {
    $Extras += "gpu"
}
if ($Dev) {
    $Extras += "dev"
}

if ($Extras.Count -gt 0) {
    $InstallTarget = ".[" + ($Extras -join ",") + "]"
} else {
    $InstallTarget = "."
}

Write-Host "Installing project dependencies: $InstallTarget"
Invoke-Checked -Command $PythonExe -Arguments @("-m", "pip", "install", "-e", $InstallTarget)

Write-Host ""
Write-Host "Environment is ready."
Write-Host "Run the app with: .\scripts\run.ps1"
