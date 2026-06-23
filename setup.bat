@echo off
setlocal

set "SOZVON_SETUP_FROM_BAT=1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
set "SETUP_EXIT=%ERRORLEVEL%"

echo.
if "%SETUP_EXIT%"=="0" (
    echo Setup completed successfully.
) else (
    echo Setup failed with exit code %SETUP_EXIT%.
)
echo Press any key to close this window.
pause >nul

exit /b %SETUP_EXIT%
