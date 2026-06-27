from types import SimpleNamespace

import numpy as np

from audio.types import AudioChunk, AudioSource
from speech.whisper_engine import WhisperEngine


def audio_chunk() -> AudioChunk:
    samples = np.array([0.25, -0.25, 0.2, -0.2], dtype=np.float32)
    pcm = (samples * 32767).astype(np.int16).tobytes()
    return AudioChunk(
        source=AudioSource.USER_MIC,
        timestamp=100.0,
        sample_rate=16_000,
        pcm=pcm,
    )


class FakeModel:
    def __init__(self) -> None:
        self.kwargs = {}

    def transcribe(self, _audio, **kwargs):
        self.kwargs = kwargs
        words = [
            SimpleNamespace(word=" Sozvon", start=0.1, end=0.3),
            SimpleNamespace(word=" test", start=0.35, end=0.7),
        ]
        return iter([SimpleNamespace(text="Sozvon test", words=words)]), None


def test_whisper_engine_uses_quality_glossary_and_word_timestamps() -> None:
    model = FakeModel()
    engine = WhisperEngine(
        model,
        language="en",
        quality_mode="accurate",
        glossary_terms="Sozvon\nCTranslate2",
        word_timestamps=True,
    )

    event = engine.transcribe_chunk(audio_chunk())

    assert event is not None
    assert event.text == "Sozvon test"
    assert event.words[0].word == "Sozvon"
    assert event.words[0].start_seconds == 100.1
    assert event.words[1].end_seconds == 100.7
    assert model.kwargs["language"] == "en"
    assert model.kwargs["beam_size"] == 5
    assert model.kwargs["condition_on_previous_text"] is True
    assert model.kwargs["word_timestamps"] is True
    assert "Sozvon" in model.kwargs["initial_prompt"]


def test_whisper_engine_fast_mode_disables_context_conditioning() -> None:
    model = FakeModel()
    engine = WhisperEngine(model, quality_mode="fast")

    event = engine.transcribe_chunk(audio_chunk())

    assert event is not None
    assert model.kwargs["beam_size"] == 1
    assert model.kwargs["condition_on_previous_text"] is False
    assert model.kwargs["initial_prompt"] is None
    assert event.words == ()
