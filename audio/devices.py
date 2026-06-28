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

    try:
        return _list_audio_devices(sd)
    except Exception as exc:
        logger.warning("Could not list audio devices: %s", exc)
        return []


def list_microphone_devices(devices: list[AudioDevice] | None = None) -> list[AudioDevice]:
    devices = devices if devices is not None else list_audio_devices()
    inputs = [device for device in devices if device.max_input_channels > 0]
    wasapi_inputs = [device for device in inputs if _is_wasapi(device)]
    return wasapi_inputs or inputs


def list_loopback_devices(devices: list[AudioDevice] | None = None) -> list[AudioDevice]:
    devices = devices if devices is not None else list_audio_devices()
    return [device for device in devices if _is_wasapi(device) and device.max_output_channels > 0]


def resolve_microphone_index(device_index: int | None) -> int | None:
    if device_index is None:
        return None
    devices = list_audio_devices()
    microphones = list_microphone_devices(devices)
    selected = _find_device(devices, device_index)
    preferred = _find_device(microphones, device_index)
    if preferred is not None:
        return preferred.index
    if selected is not None:
        matched = _find_matching_device_by_name(microphones, selected.name)
        if matched is not None:
            return matched.index
    return default_microphone_index()


def resolve_loopback_index(device_index: int | None) -> int | None:
    if device_index is None:
        return None
    devices = list_audio_devices()
    loopback_devices = list_loopback_devices(devices)
    selected = _find_device(devices, device_index)
    preferred = _find_device(loopback_devices, device_index)
    if preferred is not None:
        return preferred.index
    if selected is not None:
        matched = _find_matching_device_by_name(loopback_devices, selected.name)
        if matched is not None:
            return matched.index
    return default_loopback_index()


def _list_audio_devices(sd) -> list[AudioDevice]:
    hostapis = sd.query_hostapis()
    devices: list[AudioDevice] = []
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
    mic_index = default_microphone_index() if microphone_device is None else microphone_device
    loopback_index = default_loopback_index() if loopback_device is None else loopback_device
    mic = _find_device(devices, mic_index)
    loopback = _find_device(devices, loopback_index)

    microphone_available = mic is not None and mic.max_input_channels > 0
    loopback_available = (
        loopback is not None and _is_wasapi(loopback) and loopback.max_output_channels > 0
    )

    if not devices:
        messages.append("No audio devices detected.")
    if microphone_device is not None and mic is None:
        messages.append("Selected microphone was not found.")
    elif mic is None:
        messages.append("No microphone input device found.")
    elif mic.max_input_channels <= 0:
        messages.append(f"Selected microphone has no input channels: {mic.name}")
    if loopback_device is not None and loopback is None:
        messages.append("Selected output device was not found.")
    elif loopback is None:
        messages.append("No WASAPI output device found for loopback capture.")
    elif not _is_wasapi(loopback):
        messages.append(f"Loopback requires a WASAPI output device: {loopback.name}")
    elif loopback.max_output_channels <= 0:
        messages.append(f"Selected loopback device has no output channels: {loopback.name}")
    return AudioDiagnostics(
        sounddevice_available=True,
        microphone_available=microphone_available,
        loopback_available=loopback_available,
        messages=messages,
    )


def default_microphone_index(sd=None) -> int | None:
    if sd is None:
        try:
            import sounddevice as sd
        except Exception as exc:
            logger.warning("Could not resolve default microphone: %s", exc)
            return None
    try:
        return _default_audio_device_index(sd, input_device=True)
    except Exception as exc:
        logger.warning("Could not resolve default microphone: %s", exc)
        return None


def default_loopback_index(sd=None) -> int | None:
    if sd is None:
        try:
            import sounddevice as sd
        except Exception as exc:
            logger.warning("Could not resolve default loopback output: %s", exc)
            return None
    try:
        return _default_audio_device_index(sd, input_device=False)
    except Exception as exc:
        logger.warning("Could not resolve default loopback output: %s", exc)
        return None


def _find_device(devices: list[AudioDevice], index: int | None) -> AudioDevice | None:
    if index is None:
        return None
    for device in devices:
        if device.index == index:
            return device
    return None


def _default_audio_device_index(sd, input_device: bool) -> int | None:
    devices = _list_audio_devices(sd)
    preferred = list_microphone_devices(devices) if input_device else list_loopback_devices(devices)
    if not preferred:
        return None

    hostapi_default = _hostapi_default_device_index(sd, devices, input_device)
    hostapi_device = _find_device(preferred, hostapi_default)
    if hostapi_device is not None:
        return hostapi_device.index

    default_index = _sounddevice_default_index(sd, input_device)
    default_device = _find_device(devices, default_index)
    if default_device is not None:
        preferred_default = _find_device(preferred, default_device.index)
        if preferred_default is not None:
            return preferred_default.index
        matched = _find_matching_device_by_name(preferred, default_device.name)
        if matched is not None:
            return matched.index

    return preferred[0].index


def _hostapi_default_device_index(sd, devices: list[AudioDevice], input_device: bool) -> int | None:
    hostapis = sd.query_hostapis()
    key = "default_input_device" if input_device else "default_output_device"
    for hostapi in hostapis:
        if "WASAPI" not in str(hostapi.get("name", "")).upper():
            continue
        index = _valid_device_index(hostapi.get(key))
        device = _find_device(devices, index)
        if device is None:
            continue
        if input_device and device.max_input_channels > 0:
            return device.index
        if not input_device and device.max_output_channels > 0:
            return device.index
    return None


def _sounddevice_default_index(sd, input_device: bool) -> int | None:
    try:
        index = sd.default.device[0 if input_device else 1]
    except Exception:
        return None
    return _valid_device_index(index)


def _valid_device_index(value) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def _find_matching_device_by_name(
    devices: list[AudioDevice],
    device_name: str,
) -> AudioDevice | None:
    normalized_name = _normalize_device_name(device_name)
    if not normalized_name:
        return None
    for device in devices:
        if _normalize_device_name(device.name) == normalized_name:
            return device
    for device in devices:
        candidate = _normalize_device_name(device.name)
        if normalized_name in candidate or candidate in normalized_name:
            return device
    return None


def _is_wasapi(device: AudioDevice) -> bool:
    return "WASAPI" in device.hostapi.upper()


def _normalize_device_name(name: str) -> str:
    return " ".join(name.lower().split())
