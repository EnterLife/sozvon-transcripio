from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass

import numpy as np

from audio.types import AudioChunk
from speech.diarization import speaker_for_source
from speech.types import TranscriptEvent, WordTimestamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhisperQuality:
    beam_size: int
    vad_filter: bool
    condition_on_previous_text: bool


QUALITY_PRESETS = {
    "fast": WhisperQuality(beam_size=1, vad_filter=True, condition_on_previous_text=False),
    "balanced": WhisperQuality(beam_size=3, vad_filter=True, condition_on_previous_text=False),
    "accurate": WhisperQuality(beam_size=5, vad_filter=True, condition_on_previous_text=True),
}


class WhisperEngine:
    def __init__(
        self,
        model,
        language: str = "ru",
        quality_mode: str = "balanced",
        glossary_terms: str | None = None,
        word_timestamps: bool = False,
    ) -> None:
        self.model = model
        self.language = language
        self.quality = QUALITY_PRESETS.get(quality_mode, QUALITY_PRESETS["balanced"])
        self.glossary_terms = glossary_terms
        self.word_timestamps = word_timestamps

    def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptEvent | None:
        audio = np.frombuffer(chunk.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self._is_silence(audio):
            return None

        try:
            segments, _info = self.model.transcribe(
                self._wav_bytes(chunk.pcm, chunk.sample_rate),
                language=self.language or None,
                vad_filter=self.quality.vad_filter,
                beam_size=self.quality.beam_size,
                condition_on_previous_text=self.quality.condition_on_previous_text,
                initial_prompt=self._initial_prompt(),
                word_timestamps=self.word_timestamps,
            )
            segment_list = list(segments)
            text = " ".join(segment.text.strip() for segment in segment_list).strip()
            words = self._word_timestamps(segment_list, chunk.timestamp)
        except Exception as exc:
            logger.exception("Transcription failed for %s", chunk.source)
            raise RuntimeError(f"Transcription failed for {chunk.source}: {exc}") from exc

        if not text:
            return None
        return TranscriptEvent(
            speaker=speaker_for_source(chunk.source),
            timestamp=chunk.timestamp,
            text=text,
            source=str(chunk.source),
            words=words,
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

    def _initial_prompt(self) -> str | None:
        if not self.glossary_terms:
            return None
        terms = [term.strip() for term in self.glossary_terms.replace(";", "\n").splitlines()]
        terms = [term for term in terms if term]
        if not terms:
            return None
        return "Use these domain terms and names exactly when spoken: " + ", ".join(terms)

    def _word_timestamps(self, segments, chunk_timestamp: float) -> tuple[WordTimestamp, ...]:
        if not self.word_timestamps:
            return ()
        words: list[WordTimestamp] = []
        for segment in segments:
            for word in getattr(segment, "words", None) or ():
                text = str(getattr(word, "word", "")).strip()
                if not text:
                    continue
                start = float(getattr(word, "start", 0.0) or 0.0)
                end = float(getattr(word, "end", start) or start)
                words.append(
                    WordTimestamp(
                        word=text,
                        start_seconds=chunk_timestamp + start,
                        end_seconds=chunk_timestamp + end,
                    )
                )
        return tuple(words)
