# Realtime Call Transcriber

Desktop Windows app for local real-time call transcription.

## Status

Implemented MVP foundation:

- PySide6 desktop UI.
- Microphone capture.
- WASAPI loopback capture for system audio through `soundcard`.
- Source-based speaker separation: `Я` / `Собеседник`.
- Hardware detection and automatic Whisper model selection.
- Local `faster-whisper` transcription.
- Buffered recognition windows to reduce phrase cuts during real-time capture.
- Transcript autosave to TXT and JSON.
- Export to TXT, Markdown, and JSON.
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

## Model Downloads

The app can download public Whisper models from Hugging Face without a token. Set
`HF_TOKEN` only if you need higher Hugging Face rate limits or faster downloads.
You can enter the token in `Settings` under `Hugging Face token`; it is used for
model downloads.

## GPU Acceleration

Open `Settings` and choose `CUDA GPU` under `Device` to force NVIDIA GPU mode, or keep
`Auto` to use CUDA when an NVIDIA GPU is detected and fall back to CPU if CUDA cannot
be loaded. `Compute type` can stay on `Auto`: the app uses `float16` on CUDA and `int8`
on CPU.

GPU mode needs NVIDIA CUDA runtime DLLs used by CTranslate2. Normal setup installs the
GPU extras unless you run setup with `-CpuOnly`. If the DLLs are still missing, keep
`Auto-install missing CUDA runtime` enabled in `Settings`; the app will try to install
the required NVIDIA runtime wheels before loading the Whisper model.

`Recognition window seconds` controls how much audio is sent to Whisper at a time.
Lower values reduce delay; higher values usually reduce cut-off phrases.

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
