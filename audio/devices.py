from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int


@dataclass(frozen=True)
class AudioDiagnostics:
    sounddevice_available: bool
    microphone_available: bool
    loopback_available: bool
    messages: list[str]


def list_audio_devices() -> list[AudioDevice]:
    try:
        import sounddevice as sd
    except Exception as exc:
        logger.warning("sounddevice is unavailable: %s", exc)
        return []

    hostapis = sd.query_hostapis()
    devices = []
    for index, raw in enumerate(sd.query_devices()):
        hostapi_name = hostapis[raw["hostapi"]]["name"]
        devices.append(
            AudioDevice(
                index=index,
                name=str(raw["name"]),
                hostapi=str(hostapi_name),
                max_input_channels=int(raw["max_input_channels"]),
                max_output_channels=int(raw["max_output_channels"]),
            )
        )
    return devices


def diagnose_audio(microphone_device: int | None, loopback_device: int | None) -> AudioDiagnostics:
    try:
        import sounddevice  # noqa: F401
    except Exception as exc:
        return AudioDiagnostics(
            sounddevice_available=False,
            microphone_available=False,
            loopback_available=False,
            messages=[f"sounddevice is unavailable: {exc}"],
        )

    devices = list_audio_devices()
    messages: list[str] = []
    mic = _find_device(devices, microphone_device)
    loopback = _find_device(devices, loopback_device)

    microphone_available = mic is None or mic.max_input_channels > 0
    loopback_available = (
        loopback is None
        or ("WASAPI" in loopback.hostapi.upper() and loopback.max_output_channels > 0)
    )

    if not devices:
        messages.append("No audio devices detected.")
    if mic and mic.max_input_channels <= 0:
        messages.append(f"Selected microphone has no input channels: {mic.name}")
    if loopback and "WASAPI" not in loopback.hostapi.upper():
        messages.append(f"Loopback works best with a WASAPI output device: {loopback.name}")
    if loopback and loopback.max_output_channels <= 0:
        messages.append(f"Selected loopback device has no output channels: {loopback.name}")
    if loopback_device is None and default_loopback_index() is None:
        messages.append("No WASAPI output device found for loopback capture.")

    return AudioDiagnostics(
        sounddevice_available=True,
        microphone_available=microphone_available,
        loopback_available=loopback_available,
        messages=messages,
    )


def default_microphone_index() -> int | None:
    try:
        import sounddevice as sd

        index = sd.default.device[0]
        return int(index) if index is not None and index >= 0 else None
    except Exception as exc:
        logger.warning("Could not resolve default microphone: %s", exc)
        return None


def default_loopback_index() -> int | None:
    devices = list_audio_devices()
    wasapi_outputs = [
        device
        for device in devices
        if "WASAPI" in device.hostapi.upper() and device.max_output_channels > 0
    ]
    if wasapi_outputs:
        return wasapi_outputs[0].index
    return None


def _find_device(devices: list[AudioDevice], index: int | None) -> AudioDevice | None:
    if index is None:
        return None
    for device in devices:
        if device.index == index:
            return device
    return None
