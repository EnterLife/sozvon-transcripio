import sys
from types import SimpleNamespace

from audio.devices import (
    AudioDevice,
    default_loopback_index,
    default_microphone_index,
    diagnose_audio,
    list_audio_devices,
    list_loopback_devices,
    list_microphone_devices,
    resolve_loopback_index,
)


def test_default_loopback_prefers_wasapi_default_output() -> None:
    sd = fake_sounddevice(
        hostapis=[
            {"name": "MME", "default_input_device": -1, "default_output_device": 0},
            {"name": "Windows WASAPI", "default_input_device": 1, "default_output_device": 2},
        ],
        devices=[
            raw_output_device("Speakers", hostapi=0),
            raw_input_device("Microphone", hostapi=1),
            raw_output_device("Headphones", hostapi=1),
        ],
        default_device=[1, 0],
    )

    assert default_loopback_index(sd) == 2


def test_default_loopback_matches_global_default_output_name_to_wasapi() -> None:
    sd = fake_sounddevice(
        hostapis=[
            {"name": "MME", "default_input_device": -1, "default_output_device": 0},
            {"name": "Windows WASAPI", "default_input_device": -1, "default_output_device": -1},
        ],
        devices=[
            raw_output_device("Speakers (Realtek Audio)", hostapi=0),
            raw_output_device("Speakers (Realtek Audio)", hostapi=1),
            raw_output_device("Headphones", hostapi=1),
        ],
        default_device=[-1, 0],
    )

    assert default_loopback_index(sd) == 1


def test_default_microphone_prefers_wasapi_default_input() -> None:
    sd = fake_sounddevice(
        hostapis=[
            {"name": "MME", "default_input_device": 0, "default_output_device": -1},
            {"name": "Windows WASAPI", "default_input_device": 1, "default_output_device": -1},
        ],
        devices=[
            raw_input_device("Microphone", hostapi=0),
            raw_input_device("Microphone", hostapi=1),
        ],
        default_device=[0, -1],
    )

    assert default_microphone_index(sd) == 1


def test_list_loopback_devices_filters_to_wasapi_outputs() -> None:
    devices = [
        audio_output_device(0, "Speakers", "MME"),
        audio_input_device(1, "Microphone", "Windows WASAPI"),
        audio_output_device(2, "Headphones", "Windows WASAPI"),
    ]

    assert [device.name for device in list_loopback_devices(devices)] == ["Headphones"]


def test_list_microphone_devices_prefers_wasapi_inputs() -> None:
    devices = [
        audio_input_device(0, "Microphone", "MME"),
        audio_input_device(1, "Microphone", "Windows WASAPI"),
        audio_input_device(2, "Webcam Mic", "Windows WASAPI"),
    ]

    assert [device.index for device in list_microphone_devices(devices)] == [1, 2]


def test_resolve_loopback_migrates_matching_non_wasapi_output(monkeypatch) -> None:
    sd = fake_sounddevice(
        hostapis=[
            {"name": "MME", "default_input_device": -1, "default_output_device": 0},
            {"name": "Windows WASAPI", "default_input_device": -1, "default_output_device": 2},
        ],
        devices=[
            raw_output_device("Headphones", hostapi=0),
            raw_output_device("Headphones", hostapi=1),
            raw_output_device("Speakers", hostapi=1),
        ],
        default_device=[-1, 2],
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd)

    assert resolve_loopback_index(0) == 1


def test_diagnose_audio_rejects_non_wasapi_loopback(monkeypatch) -> None:
    sd = fake_sounddevice(
        hostapis=[
            {"name": "MME", "default_input_device": 1, "default_output_device": 0},
            {"name": "Windows WASAPI", "default_input_device": 1, "default_output_device": 2},
        ],
        devices=[
            raw_output_device("Speakers", hostapi=0),
            raw_input_device("Microphone", hostapi=1),
            raw_output_device("Speakers", hostapi=1),
        ],
        default_device=[1, 0],
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd)

    diagnostics = diagnose_audio(microphone_device=1, loopback_device=0)

    assert diagnostics.loopback_available is False
    assert diagnostics.messages == ["Loopback requires a WASAPI output device: Speakers"]


def test_list_audio_devices_returns_empty_when_hostapi_query_fails(monkeypatch) -> None:
    sd = SimpleNamespace(
        query_hostapis=lambda: (_ for _ in ()).throw(RuntimeError("portaudio offline")),
        query_devices=lambda index=None: [],
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd)

    assert list_audio_devices() == []


def test_diagnose_audio_reports_no_devices_when_device_query_fails(monkeypatch) -> None:
    sd = SimpleNamespace(
        query_hostapis=lambda: [{"name": "Windows WASAPI"}],
        query_devices=lambda index=None: (_ for _ in ()).throw(RuntimeError("device scan failed")),
        default=SimpleNamespace(device=[-1, -1]),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd)

    diagnostics = diagnose_audio(microphone_device=None, loopback_device=None)

    assert diagnostics.sounddevice_available is True
    assert diagnostics.microphone_available is False
    assert diagnostics.loopback_available is False
    assert diagnostics.messages == [
        "No audio devices detected.",
        "No microphone input device found.",
        "No WASAPI output device found for loopback capture.",
    ]


def fake_sounddevice(hostapis, devices, default_device):
    return SimpleNamespace(
        query_hostapis=lambda: hostapis,
        query_devices=lambda index=None: devices if index is None else devices[index],
        default=SimpleNamespace(device=default_device),
    )


def raw_input_device(name: str, hostapi: int) -> dict:
    return raw_device(name, hostapi, inputs=1, outputs=0)


def raw_output_device(name: str, hostapi: int) -> dict:
    return raw_device(name, hostapi, inputs=0, outputs=2)


def raw_device(name: str, hostapi: int, inputs: int, outputs: int) -> dict:
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


def audio_input_device(index: int, name: str, hostapi: str) -> AudioDevice:
    return audio_device(index, name, hostapi, inputs=1, outputs=0)


def audio_output_device(index: int, name: str, hostapi: str) -> AudioDevice:
    return audio_device(index, name, hostapi, inputs=0, outputs=2)


def audio_device(
    index: int,
    name: str,
    hostapi: str,
    inputs: int,
    outputs: int,
) -> AudioDevice:
    return AudioDevice(
        index=index,
        name=name,
        hostapi=hostapi,
        max_input_channels=inputs,
        max_output_channels=outputs,
    )
