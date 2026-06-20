@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\test.ps1" %*
exit /b %ERRORLEVEL%

