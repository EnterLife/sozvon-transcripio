from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from audio.types import AudioChunk, AudioSource

logger = logging.getLogger(__name__)


class SoundDeviceCapture:
    def __init__(
        self,
        source: AudioSource,
        device_index: int | None,
        sample_rate: int,
        chunk_seconds: float,
        on_chunk: Callable[[AudioChunk], None],
        on_error: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        loopback: bool = False,
    ) -> None:
        self.source = source
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.on_chunk = on_chunk
        self.on_error = on_error
        self.on_status = on_status
        self.loopback = loopback
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"{self.source}-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            self._fail(f"sounddevice is not installed: {exc}")
            return

        blocksize = max(1, int(self.sample_rate * self.chunk_seconds))
        extra_settings = None
        channels = 1

        if self.loopback:
            try:
                extra_settings = sd.WasapiSettings(loopback=True)
                device_info = sd.query_devices(self.device_index)
                channels = max(1, min(2, int(device_info["max_output_channels"])))
            except Exception as exc:
                self._fail(f"WASAPI loopback is unavailable: {exc}")
                return
        elif self.device_index is not None:
            try:
                device_info = sd.query_devices(self.device_index)
                channels = max(1, min(1, int(device_info["max_input_channels"])))
            except Exception as exc:
                self._fail(f"Microphone device is unavailable: {exc}")
                return

        logger.info(
            "Starting capture source=%s device=%s loopback=%s",
            self.source,
            self.device_index,
            self.loopback,
        )
        if self.on_status:
            self.on_status(f"{self.source}: started")

        def callback(indata, frames, callback_time, status) -> None:
            if status:
                logger.warning("Audio status for %s: %s", self.source, status)
            mono = np.asarray(indata, dtype=np.float32)
            if mono.ndim > 1:
                mono = mono.mean(axis=1)
            mono = np.clip(mono, -1.0, 1.0)
            pcm = (mono * 32767).astype(np.int16).tobytes()
            self.on_chunk(
                AudioChunk(
                    source=self.source,
                    timestamp=float(getattr(callback_time, "inputBufferAdcTime", time.time())),
                    sample_rate=self.sample_rate,
                    pcm=pcm,
                )
            )

        try:
            with sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=channels,
                dtype="float32",
                blocksize=blocksize,
                callback=callback,
                extra_settings=extra_settings,
            ):
                while not self._stop.is_set():
                    time.sleep(0.1)
        except Exception:
            logger.exception("Capture failed for %s", self.source)
            self._fail(f"Capture failed for {self.source}. Check the selected audio device.")
        finally:
            if self.on_status:
                self.on_status(f"{self.source}: stopped")

    def _fail(self, message: str) -> None:
        logger.error(message)
        if self.on_error:
            self.on_error(message)
