from __future__ import annotations

from audio.types import AudioSource


def parse_capture_status(message: str) -> tuple[AudioSource | None, str]:
    source_text, separator, state_text = message.partition(":")
    if not separator:
        return None, "idle"
    try:
        source = AudioSource(source_text.strip())
    except ValueError:
        return None, state_text.strip() or "idle"
    state = state_text.strip() or "idle"
    return source, state
