from __future__ import annotations

from audio.types import AudioSource


def speaker_for_source(source: AudioSource) -> str:
    if source is AudioSource.USER_MIC:
        return "Я"
    return "Собеседник"

