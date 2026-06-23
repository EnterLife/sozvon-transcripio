# Realtime Call Transcriber

Desktop Windows app for local real-time call transcription.

## Status

Implemented MVP foundation:

- PySide6 desktop UI.
- Microphone capture.
- WASAPI loopback capture for system audio.
- Source-based speaker separation: `Я` / `Собеседник`.
- Hardware detection and automatic Whisper model selection.
- Local `faster-whisper` transcription.
- Transcript autosave to TXT and JSON.
- Settings dialog with audio, recognition, test mode, and storage options.

## Quick Start

Double-click `setup.bat`, then `run.bat`, or run:

```bat
setup.bat
run.bat
```

When started by double-click, `setup.bat` keeps the console window open until you
press a key, so setup errors remain visible.

Check the runtime without opening the GUI:

```bat
run.bat -Check
```

PowerShell entry points are also available:

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1
```

CPU-only setup:

```powershell
.\scripts\setup.ps1 -CpuOnly
.\scripts\run.ps1
```

From `cmd.exe`:

```bat
setup.bat -CpuOnly
run.bat
```

By default setup bypasses the system proxy for pip because Windows may expose SOCKS
proxy settings that pip cannot use without extra packages. To use the system proxy:

```bat
setup.bat -UseSystemProxy
```

Development setup with tests:

```powershell
.\scripts\setup.ps1 -Dev
.\scripts\test.ps1
```

From `cmd.exe`:

```bat
setup.bat -Dev
test.bat
```

## Manual Run With GPU

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[gpu]"
python app.py
```

## Manual Run CPU-Only

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python app.py
```

Application data is stored in `%APPDATA%\RealtimeCallTranscriber`.

## Test Mode

Open `Settings` and enable `Use test transcript engine` to test the GUI, audio routing,
session controls, and autosave without downloading or loading a Whisper model.

## Checks

```powershell
python -m compileall -q app.py audio config core gui speech storage tests
python -m ruff check .
python -m pytest -q
```

If `pytest` is not installed:

```powershell
pip install -e ".[dev]"
```

## Roadmap

1. Validate real WASAPI loopback behavior on Windows audio devices.
2. Add live partial transcription aggregation instead of one final result per chunk.
3. Add audio level meters and device test buttons.
4. Add packaged Windows build.
5. Add optional diarization with `pyannote.audio`.
