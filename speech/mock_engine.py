from __future__ import annotations

import itertools

import numpy as np

from audio.types import AudioChunk
from speech.diarization import speaker_for_source
from speech.types import TranscriptEvent


class MockTranscriptionEngine:
    def __init__(self) -> None:
        self._counter = itertools.count(1)

    def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptEvent | None:
        audio = np.frombuffer(chunk.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0 or float(np.sqrt(np.mean(audio * audio))) < 0.008:
            return None
        number = next(self._counter)
        return TranscriptEvent(
            speaker=speaker_for_source(chunk.source),
            timestamp=chunk.timestamp,
            text=f"Тестовая реплика #{number} ({chunk.source})",
            source=str(chunk.source),
        )
