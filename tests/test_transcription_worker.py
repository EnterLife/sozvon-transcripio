import time

from audio.audio_router import AudioRouter
from audio.types import AudioChunk, AudioSource
from speech.transcription_worker import TranscriptionWorker
from speech.types import TranscriptEvent


def chunk(timestamp: float, frames: int = 2, sample_rate: int = 4) -> AudioChunk:
    return AudioChunk(
        source=AudioSource.USER_MIC,
        timestamp=timestamp,
        sample_rate=sample_rate,
        pcm=b"\x01\x00" * frames,
    )


class RecordingEngine:
    def __init__(self) -> None:
        self.chunks: list[AudioChunk] = []

    def transcribe_chunk(self, audio_chunk: AudioChunk) -> TranscriptEvent:
        self.chunks.append(audio_chunk)
        return TranscriptEvent(
            speaker="Я",
            timestamp=audio_chunk.timestamp,
            text=f"{len(audio_chunk.pcm)} bytes",
            source=str(audio_chunk.source),
        )


class FailingEngine:
    def transcribe_chunk(self, _audio_chunk: AudioChunk) -> None:
        raise RuntimeError("engine crashed")


def test_transcription_worker_buffers_short_chunks_before_transcribing() -> None:
    router = AudioRouter()
    engine = RecordingEngine()
    events = []
    errors = []
    worker = TranscriptionWorker(
        router,
        engine,
        events.append,
        errors.append,
        window_seconds=1.0,
    )

    worker.start()
    router.push(chunk(1.0))
    time.sleep(0.02)
    assert events == []

    router.push(chunk(1.5))
    worker.stop()

    assert errors == []
    assert [event.text for event in events] == ["8 bytes"]
    assert len(engine.chunks) == 1
    assert engine.chunks[0].timestamp == 1.0


def test_transcription_worker_flushes_remaining_audio_on_stop() -> None:
    router = AudioRouter()
    engine = RecordingEngine()
    events = []
    worker = TranscriptionWorker(
        router,
        engine,
        events.append,
        lambda _message: None,
        window_seconds=10.0,
    )

    worker.start()
    router.push(chunk(1.0))
    worker.stop()

    assert [event.text for event in events] == ["4 bytes"]


def test_transcription_worker_reports_engine_errors() -> None:
    router = AudioRouter()
    errors = []
    worker = TranscriptionWorker(
        router,
        FailingEngine(),
        lambda _event: None,
        errors.append,
        window_seconds=0.0,
    )

    worker.start()
    router.push(chunk(1.0))
    worker.stop()

    assert errors == ["engine crashed"]
