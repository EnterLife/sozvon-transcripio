from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "RealtimeCallTranscriber"


@dataclass(frozen=True)
class AppPaths:
    root_dir: Path
    config_dir: Path
    models_dir: Path
    logs_dir: Path
    transcripts_dir: Path
    settings_file: Path


def app_data_root() -> Path:
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / APP_NAME
    return Path.home() / ".realtime-call-transcriber"


def ensure_app_dirs() -> AppPaths:
    root = app_data_root()
    paths = AppPaths(
        root_dir=root,
        config_dir=root / "config",
        models_dir=root / "models",
        logs_dir=root / "logs",
        transcripts_dir=root / "transcripts",
        settings_file=root / "config" / "settings.json",
    )
    for directory in (
        paths.root_dir,
        paths.config_dir,
        paths.models_dir,
        paths.logs_dir,
        paths.transcripts_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths

