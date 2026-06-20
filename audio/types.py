from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AudioSource(StrEnum):
    USER_MIC = "USER_MIC"
    REMOTE_AUDIO = "REMOTE_AUDIO"


@dataclass(frozen=True)
class AudioChunk:
    source: AudioSource
    timestamp: float
    sample_rate: int
    pcm: bytes

