from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from audio.audio_router import AudioRouter
from audio.types import AudioChunk
from speech.audio_window import AudioWindowBuffer
from speech.types import TranscriptEvent
from speech.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)


class TranscriptionWorker:
    def __init__(
        self,
        router: AudioRouter,
        engine: WhisperEngine,
        on_event: Callable[[TranscriptEvent], None],
        on_error: Callable[[str], None],
        window_seconds: float = 3.0,
    ) -> None:
        self.router = router
        self.engine = engine
        self.on_event = on_event
        self.on_error = on_error
        self.window_seconds = window_seconds
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="transcription-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.router.stop()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        buffer = AudioWindowBuffer(self.window_seconds)
        for chunk in self.router.chunks():
            for ready_chunk in buffer.add(chunk):
                self._transcribe(ready_chunk)
        for ready_chunk in buffer.flush():
            self._transcribe(ready_chunk)

    def _transcribe(self, chunk: AudioChunk) -> None:
        try:
            event = self.engine.transcribe_chunk(chunk)
        except Exception as exc:
            logger.exception("Unhandled transcription error")
            self.on_error(str(exc))
            return
        if event:
            self.on_event(event)
