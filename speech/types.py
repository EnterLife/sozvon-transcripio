from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class TranscriptEvent:
    speaker: str
    timestamp: float
    text: str
    source: str
    is_final: bool = True
    words: tuple[WordTimestamp, ...] = ()
