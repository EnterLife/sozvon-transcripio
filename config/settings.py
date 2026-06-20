from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any


@dataclass
class AudioSettings:
    microphone_device: int | None = None
    loopback_device: int | None = None
    sample_rate: int = 16_000
    chunk_seconds: float = 1.0


@dataclass
class RecognitionSettings:
    language: str = "ru"
    model_size: str | None = None
    auto_select_model: bool = True
    compute_type: str = "auto"
    dry_run: bool = False


@dataclass
class StorageSettings:
    autosave_seconds: int = 30
    transcript_dir: str | None = None


@dataclass
class AppSettings:
    audio: AudioSettings = field(default_factory=AudioSettings)
    recognition: RecognitionSettings = field(default_factory=RecognitionSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)


def _merge_dataclass(cls: type, data: dict[str, Any]):
    valid = {field_name for field_name in cls.__dataclass_fields__}
    return cls(**{key: value for key, value in data.items() if key in valid})


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def load_settings(path: Path) -> AppSettings:
    if not path.exists():
        return AppSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError):
        return AppSettings()
    if not isinstance(data, dict):
        return AppSettings()
    return AppSettings(
        audio=_merge_dataclass(AudioSettings, _section(data, "audio")),
        recognition=_merge_dataclass(RecognitionSettings, _section(data, "recognition")),
        storage=_merge_dataclass(StorageSettings, _section(data, "storage")),
    )


def save_settings(path: Path, settings: AppSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
