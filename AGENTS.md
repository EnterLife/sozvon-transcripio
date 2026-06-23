# AGENTS.md

## Project

Sozvon Transcripio is a local-first Windows desktop application for real-time call
transcription. The app captures microphone audio and WASAPI loopback system audio,
routes audio chunks through a local transcription worker, runs `faster-whisper`, and
autosaves transcripts as TXT and JSON.

The main goal is to keep the product simple, local-only by default, practical for
desktop calls, and easy to extend without leaking GUI, audio, model, or storage details
across module boundaries.

## Tech Stack

- Language: Python 3.12+
- UI: PySide6
- Audio capture: `sounddevice`
- Speech to text: `faster-whisper`
- Hardware detection: `psutil`, optional GPU packages
- Packaging: `pyproject.toml` with editable install
- Tests: `pytest`
- Linting: `ruff`
- Local environment: `.venv`

## Repository Structure

- `app.py` - PySide6 application entry point.
- `gui/main_window.py` - main window, controls, model loading, transcription workflow.
- `gui/settings_dialog.py` - settings UI.
- `audio/` - audio device discovery, routing, microphone capture, and WASAPI loopback capture.
- `speech/` - hardware detection, model selection/loading, transcription engine, and workers.
- `config/settings.py` - settings dataclasses, loading, saving, and validation.
- `storage/transcript_store.py` - transcript records, autosave, TXT/JSON output.
- `core/` - shared runtime, paths, and filesystem helpers.
- `tests/` - focused behavior tests.
- `scripts/setup.ps1` - creates `.venv` and installs dependencies.
- `scripts/run.ps1` - runs setup when needed and launches/checks the app.
- `scripts/test.ps1` - test/check entry point.
- `README.md` - setup, launch, and usage notes.

## Coding Rules

- Keep edits focused on the requested behavior.
- Preserve user changes already present in the working tree.
- Do not refactor unrelated code while fixing a local issue.
- Do not commit secrets, tokens, local model weights, Hugging Face caches, audio recordings,
  transcripts, generated output, or runtime artifacts.
- Avoid hardcoded credentials, absolute local paths, tokens, server URLs, device indexes, or
  machine-specific model paths.
- Keep project dependencies in `pyproject.toml`.
- Prefer running commands through `.venv\Scripts\python.exe` on Windows.
- Do not change generated/runtime folders unless the task requires it:
  - `.venv/`
  - `.pytest_cache/`
  - `.ruff_cache/`
  - `__pycache__/`
  - `realtime_call_transcriber.egg-info/`
- Do not leave unused imports, helpers, parameters, or dead code after changing files.
- Use explicit names that describe audio, transcription, model, or storage behavior over generic
  helper names.
- Keep comments short and useful; avoid comments for obvious assignments.

## Code Quality Rules

- Before changing behavior, identify the expected user-visible outcome and the most likely failure
  case.
- For bug fixes, add or update a regression test that would fail before the fix when practical.
- For new behavior, add focused tests for:
  - the successful path;
  - at least one failure, edge, or invalid-input path.
- Do not add tests that only verify mocks, implementation details, or duplicated source logic.
- Prefer tests that assert product behavior: settings normalization, transcript output, audio
  routing, model selection, worker errors, or storage results.
- If a change is documentation-only, formatting-only, or a trivial internal cleanup, tests are not
  required unless behavior could change.
- Keep ordinary tests small and local; they must not require real models, real audio devices,
  network access, Hugging Face tokens, or large files.

## Architecture Rules

- Keep GUI code in `gui/`; keep audio capture/routing in `audio/`; keep model and transcription
  logic in `speech/`; keep persistence in `storage/`.
- Keep provider-specific model code behind adapters such as `WhisperEngine` and `ModelManager`.
- Keep raw audio, transcripts, settings, and model execution local by default.
- Do not introduce cloud transcription, cloud diarization, telemetry, or cloud AI calls unless
  explicitly requested.
- If future cloud integrations are added, make them opt-in and isolated behind provider interfaces.
- Do not put low-level audio capture logic directly in the GUI.
- Do not put GUI behavior into transcription or storage modules.
- Keep settings validation centralized in `config/settings.py`.
- Keep atomic file writes centralized in `core/atomic_write.py`.
- Add focused tests when changing model selection, settings normalization, transcript persistence,
  audio routing, or worker behavior.

## Local Model Rules

- `faster-whisper` may accept a known model name or a local CTranslate2 model path.
- Do not commit downloaded models or Hugging Face caches.
- Do not hardcode local model paths in source files.
- Handle missing model dependencies, failed downloads, invalid proxy settings, and unsupported
  hardware with clear user-facing errors.
- Keep model selection deterministic from `HardwareInfo` and user settings.
- For fully offline usage, prefer documenting local model paths and cache setup instead of adding
  hidden network requirements.

## Audio Rules

- Treat microphone and WASAPI loopback capture as independent sources.
- Preserve source-based speaker separation: `Я` for microphone and `Собеседник` for system audio.
- Keep audio routing in `AudioRouter`; avoid coupling captures directly to transcription engines.
- Do not require real audio devices in ordinary automated tests.
- Surface audio-device failures with concise user-facing errors and useful diagnostics.
- Be careful with sample rate and chunk duration changes; they affect latency, CPU load, and model
  behavior.

## UI Rules

- Keep the PySide6 UI practical and task-first: choose devices/settings, start, stop, inspect,
  export, and manage sessions.
- Do not add landing-page or marketing sections.
- Keep controls simple and understandable for local desktop use.
- Do not expose implementation jargon unless it helps the user configure a local model or audio
  device.
- For long-running model loading or transcription work, keep status messages clear.
- Do not block the UI thread with model loading, audio capture, or transcription work.

## Test Design Rules

- Tests should verify product behavior rather than implementation trivia.
- For behavior changes, include at least one happy-path test and one negative or edge-case test when
  applicable.
- Prefer focused tests for:
  - settings loading, validation, and saving;
  - transcript TXT/JSON storage behavior;
  - model selection by detected hardware;
  - audio routing and source labels;
  - proxy and model-loading error handling;
  - worker behavior with fake or mock engines.
- Do not require large models, real microphone/loopback devices, network access, or local media
  files in ordinary unit tests.
- If model-backed or audio-device integration tests are added later, keep them opt-in and document
  required local setup.
- Update docs when setup commands, model behavior, runtime requirements, export formats, or user
  workflows change.

## Review Rules

- For non-trivial code changes, perform a final review from the perspective of a fresh reader.
- Review only the user request, changed files, and relevant tests.
- Check for:
  - missing negative tests;
  - behavior that is only partially tested;
  - violations of local-only model/audio/storage rules;
  - leaked GUI/audio/model/storage concerns across module boundaries;
  - unused imports, dead code, or speculative abstractions;
  - unclear user-facing errors.
- If using another AI pass or a new context window for review, provide it with the request, diff,
  and test results, but not the full prior reasoning.
- Treat AI review as advisory; verify any suggested issue against the code before applying changes.

## Completion Criteria

A code change is complete only when:

- the requested behavior is implemented;
- relevant positive and negative tests are added or intentionally skipped with a clear reason;
- focused checks have been run when feasible;
- changed files are reviewed before the final response;
- remaining risks are mentioned, especially untested model-backed transcription or real audio-device
  flows.

## Useful Commands

Create or update the local environment:

```powershell
.\scripts\setup.ps1
```

Create a CPU-only environment:

```powershell
.\scripts\setup.ps1 -CpuOnly
```

Create a development environment:

```powershell
.\scripts\setup.ps1 -Dev
```

Run the app:

```powershell
.\scripts\run.ps1
```

Check runtime imports without opening the GUI:

```powershell
.\scripts\run.ps1 -Check
```

Run focused checks:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py audio config core gui speech storage tests
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Run the project test script:

```powershell
.\scripts\test.ps1
```

Check key imports from the virtual environment:

```powershell
.\.venv\Scripts\python.exe -c "import PySide6, sounddevice, faster_whisper; print('ok')"
```

## Commit Message Suggestions

- After each completed work chunk, include a suggested commit message in the final response.
- Use a lowercase prefix and a short lowercase summary.
- Keep the message in one line without a period at the end.
- Choose the prefix by intent:
  - `add:` for new user-visible features, adapters, tests, or app flows.
  - `fix:` for bug fixes, broken behavior, failing checks, or bad errors.
  - `upd:` for updates to existing features, docs, configs, or expected behavior.
  - `refactor:` for internal restructuring without behavior changes.
  - `docs:` for documentation-only changes.
  - `test:` for test-only maintenance that does not add new coverage.
  - `chore:` for tooling, cleanup, dependency, or repository maintenance.

Examples:

- `add: realtime transcription app scaffold`
- `fix: model download proxy handling`
- `upd: virtual environment setup`
- `docs: document local model setup`

## Before Finishing Work

- Review changed files.
- Run the most focused relevant checks when feasible.
- For Python code changes, prefer:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py audio config core gui speech storage tests
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

- For docs-only changes, a syntax or test run is usually not required.
- If a required tool is unavailable, say so explicitly in the final response and mention the
  remaining risk.
- Mention any remaining risk, especially model-backed transcription or real audio-device flows that
  were not executed.
