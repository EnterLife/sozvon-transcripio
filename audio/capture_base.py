from __future__ import annotations

import logging
import inspect
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
        stream_device_index = self.device_index

        if self.loopback:
            if not _sounddevice_wasapi_supports_loopback(sd):
                self._run_soundcard_loopback(sd, blocksize)
                return
            try:
                stream_device_index = _resolve_sounddevice_loopback_index(sd, self.device_index)
                if stream_device_index is None:
                    self._fail("No WASAPI output device found for loopback capture.")
                    return
                extra_settings = sd.WasapiSettings(loopback=True)
                device_info = sd.query_devices(stream_device_index)
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
            stream_device_index,
            self.loopback,
        )
        if self.on_status:
            self.on_status(f"{self.source}: started")

        def callback(indata, frames, callback_time, status) -> None:
            if status:
                logger.warning("Audio status for %s: %s", self.source, status)
            self._push_samples(
                indata,
                float(getattr(callback_time, "inputBufferAdcTime", time.time())),
            )

        failed = False
        try:
            with sd.InputStream(
                device=stream_device_index,
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
            failed = True
            logger.exception("Capture failed for %s", self.source)
            self._fail(f"Capture failed for {self.source}. Check the selected audio device.")
        finally:
            if self.on_status and not failed:
                self.on_status(f"{self.source}: stopped")

    def _run_soundcard_loopback(self, sd, blocksize: int) -> None:
        try:
            import soundcard as sc
        except Exception as exc:
            self._fail(f"soundcard is required for WASAPI loopback with this sounddevice version: {exc}")
            return

        try:
            output_name = _sounddevice_output_name(sd, self.device_index)
            microphone = _select_soundcard_loopback_microphone(sc, output_name)
            channels = max(1, min(2, int(getattr(microphone, "channels", 2))))
        except Exception as exc:
            self._fail(f"WASAPI loopback is unavailable: {exc}")
            return

        logger.info(
            "Starting soundcard loopback source=%s device=%s loopback=%s",
            self.source,
            getattr(microphone, "name", self.device_index),
            True,
        )
        if self.on_status:
            self.on_status(f"{self.source}: started")

        failed = False
        try:
            with microphone.recorder(
                samplerate=self.sample_rate,
                channels=channels,
                blocksize=blocksize,
            ) as recorder:
                while not self._stop.is_set():
                    data = recorder.record(numframes=blocksize)
                    self._push_samples(data, time.time())
        except Exception:
            failed = True
            logger.exception("Loopback capture failed for %s", self.source)
            self._fail(f"Loopback capture failed for {self.source}. Check the selected output device.")
        finally:
            if self.on_status and not failed:
                self.on_status(f"{self.source}: stopped")

    def _push_samples(self, samples, timestamp: float) -> None:
        mono = np.asarray(samples, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        mono = np.clip(mono, -1.0, 1.0)
        pcm = (mono * 32767).astype(np.int16).tobytes()
        self.on_chunk(
            AudioChunk(
                source=self.source,
                timestamp=timestamp,
                sample_rate=self.sample_rate,
                pcm=pcm,
            )
        )

    def _fail(self, message: str) -> None:
        logger.error(message)
        if self.on_status:
            self.on_status(f"{self.source}: failed")
        if self.on_error:
            self.on_error(message)


def _sounddevice_wasapi_supports_loopback(sd) -> bool:
    try:
        return "loopback" in inspect.signature(sd.WasapiSettings).parameters
    except Exception:
        return False


def _sounddevice_output_name(sd, device_index: int | None) -> str | None:
    if device_index is None:
        return None
    device_info = sd.query_devices(device_index)
    return str(device_info.get("name", "")).strip() or None


def _resolve_sounddevice_loopback_index(sd, device_index: int | None) -> int | None:
    if device_index is not None:
        return device_index
    from audio.devices import default_loopback_index

    return default_loopback_index(sd)


def _select_soundcard_loopback_microphone(sc, output_name: str | None):
    candidates = [mic for mic in sc.all_microphones(include_loopback=True) if _is_loopback(mic)]
    if not candidates:
        raise RuntimeError("No loopback recording device found.")

    names_to_try = []
    if output_name:
        names_to_try.append(output_name)
    else:
        try:
            speaker = sc.default_speaker()
            names_to_try.extend([getattr(speaker, "id", ""), getattr(speaker, "name", "")])
        except Exception:
            pass

    for name in names_to_try:
        if not name:
            continue
        try:
            microphone = sc.get_microphone(id=name, include_loopback=True)
        except Exception:
            continue
        if _is_loopback(microphone):
            return microphone

    if output_name:
        normalized_output = _normalize_device_name(output_name)
        for microphone in candidates:
            normalized_mic = _normalize_device_name(getattr(microphone, "name", ""))
            if normalized_output in normalized_mic or normalized_mic in normalized_output:
                return microphone

    return candidates[0]


def _is_loopback(microphone) -> bool:
    return bool(getattr(microphone, "isloopback", False))


def _normalize_device_name(name: str) -> str:
    return " ".join(name.lower().split())
