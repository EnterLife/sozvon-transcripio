from __future__ import annotations

import io
import logging
import wave

import numpy as np

from audio.types import AudioChunk
from speech.diarization import speaker_for_source
from speech.types import TranscriptEvent

logger = logging.getLogger(__name__)


class WhisperEngine:
    def __init__(self, model, language: str = "ru") -> None:
        self.model = model
        self.language = language

    def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptEvent | None:
        audio = np.frombuffer(chunk.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self._is_silence(audio):
            return None

        try:
            segments, _info = self.model.transcribe(
                self._wav_bytes(chunk.pcm, chunk.sample_rate),
                language=self.language or None,
                vad_filter=True,
                beam_size=1,
                condition_on_previous_text=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception:
            logger.exception("Transcription failed for %s", chunk.source)
            return None

        if not text:
            return None
        return TranscriptEvent(
            speaker=speaker_for_source(chunk.source),
            timestamp=chunk.timestamp,
            text=text,
            source=str(chunk.source),
        )

    def _is_silence(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return True
        return float(np.sqrt(np.mean(audio * audio))) < 0.008

    def _wav_bytes(self, pcm: bytes, sample_rate: int) -> io.BytesIO:
        data = io.BytesIO()
        with wave.open(data, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        data.seek(0)
        return data
