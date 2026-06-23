from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from core.atomic_write import write_text_atomic


SAMPLE_RATES = {8000, 16000, 22050, 44100, 48000}
MODEL_SIZES = {"tiny", "base", "small", "medium", "large-v3"}
LANGUAGES = {"", "ru", "en"}
DEVICE_MODES = {"auto", "cpu", "cuda"}
COMPUTE_TYPES = {"auto", "int8", "int8_float16", "float16", "float32"}


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
    device: str = "auto"
    compute_type: str = "auto"
    auto_install_cuda_runtime: bool = True
    hf_token: str | None = None
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


def _optional_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return min(max(value, minimum), maximum)


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return min(max(float(value), minimum), maximum)


def _choice(value: Any, default: str, allowed: set[str]) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _optional_choice(value: Any, default: str | None, allowed: set[str]) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) and value in allowed else default


def _optional_non_empty_string(value: Any, default: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return default
    stripped = value.strip()
    return stripped or None


def _bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


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
    return _normalize_settings(
        AppSettings(
            audio=_merge_dataclass(AudioSettings, _section(data, "audio")),
            recognition=_merge_dataclass(RecognitionSettings, _section(data, "recognition")),
            storage=_merge_dataclass(StorageSettings, _section(data, "storage")),
        )
    )


def save_settings(path: Path, settings: AppSettings) -> None:
    normalized = _normalize_settings(settings)
    write_text_atomic(path, json.dumps(asdict(normalized), ensure_ascii=False, indent=2))


def _normalize_settings(settings: AppSettings) -> AppSettings:
    audio_defaults = AudioSettings()
    recognition_defaults = RecognitionSettings()
    storage_defaults = StorageSettings()

    sample_rate = _bounded_int(
        settings.audio.sample_rate,
        audio_defaults.sample_rate,
        min(SAMPLE_RATES),
        max(SAMPLE_RATES),
    )
    if sample_rate not in SAMPLE_RATES:
        sample_rate = audio_defaults.sample_rate

    return AppSettings(
        audio=AudioSettings(
            microphone_device=_optional_int(
                settings.audio.microphone_device,
                audio_defaults.microphone_device,
            ),
            loopback_device=_optional_int(settings.audio.loopback_device, audio_defaults.loopback_device),
            sample_rate=sample_rate,
            chunk_seconds=_bounded_float(
                settings.audio.chunk_seconds,
                audio_defaults.chunk_seconds,
                0.5,
                2.0,
            ),
        ),
        recognition=RecognitionSettings(
            language=_choice(
                settings.recognition.language,
                recognition_defaults.language,
                LANGUAGES,
            ),
            model_size=_optional_choice(
                settings.recognition.model_size,
                recognition_defaults.model_size,
                MODEL_SIZES,
            ),
            auto_select_model=_bool(
                settings.recognition.auto_select_model,
                recognition_defaults.auto_select_model,
            ),
            device=_choice(
                settings.recognition.device,
                recognition_defaults.device,
                DEVICE_MODES,
            ),
            compute_type=_choice(
                settings.recognition.compute_type,
                recognition_defaults.compute_type,
                COMPUTE_TYPES,
            ),
            auto_install_cuda_runtime=_bool(
                settings.recognition.auto_install_cuda_runtime,
                recognition_defaults.auto_install_cuda_runtime,
            ),
            hf_token=_optional_non_empty_string(
                settings.recognition.hf_token,
                recognition_defaults.hf_token,
            ),
            dry_run=_bool(settings.recognition.dry_run, recognition_defaults.dry_run),
        ),
        storage=StorageSettings(
            autosave_seconds=_bounded_int(
                settings.storage.autosave_seconds,
                storage_defaults.autosave_seconds,
                5,
                3600,
            ),
            transcript_dir=_optional_non_empty_string(
                settings.storage.transcript_dir,
                storage_defaults.transcript_dir,
            ),
        ),
    )
