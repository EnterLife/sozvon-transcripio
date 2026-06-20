from __future__ import annotations

from collections.abc import Callable

from audio.capture_base import SoundDeviceCapture
from audio.types import AudioChunk, AudioSource


class MicrophoneCapture(SoundDeviceCapture):
    def __init__(
        self,
        device_index: int | None,
        sample_rate: int,
        chunk_seconds: float,
        on_chunk: Callable[[AudioChunk], None],
        on_error: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(
            AudioSource.USER_MIC,
            device_index,
            sample_rate,
            chunk_seconds,
            on_chunk,
            on_error,
            on_status,
            loopback=False,
        )
