from __future__ import annotations

from pathlib import Path


def resolve_transcript_dir(configured_dir: str | None, default_dir: Path) -> Path:
    if configured_dir:
        return Path(configured_dir).expanduser()
    return default_dir


def same_transcript_dir(left: Path, right: Path) -> bool:
    return _stable_path(left) == _stable_path(right)


def _stable_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)
