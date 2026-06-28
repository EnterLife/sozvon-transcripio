from types import SimpleNamespace
import sys

import numpy as np

from audio.capture_base import (
    SoundDeviceCapture,
    _resolve_sounddevice_loopback_index,
    _select_soundcard_loopback_microphone,
    _sounddevice_output_name,
    _sounddevice_wasapi_supports_loopback,
)
from audio.types import AudioSource


def test_sounddevice_wasapi_loopback_support_detection() -> None:
    class NewWasapiSettings:
        def __init__(self, exclusive=False) -> None:
            pass

    class OldWasapiSettings:
        def __init__(self, loopback=False) -> None:
            pass

    assert _sounddevice_wasapi_supports_loopback(SimpleNamespace(WasapiSettings=OldWasapiSettings))
    assert not _sounddevice_wasapi_supports_loopback(
        SimpleNamespace(WasapiSettings=NewWasapiSettings)
    )


def test_sounddevice_output_name_uses_selected_device() -> None:
    sd = SimpleNamespace(query_devices=lambda index: {"name": f"Speakers {index}"})

    assert _sounddevice_output_name(sd, 8) == "Speakers 8"


def test_sounddevice_loopback_resolves_default_wasapi_output() -> None:
    devices = [
        {
            "name": "Speakers",
            "hostapi": 0,
            "max_input_channels": 0,
            "max_output_channels": 2,
        },
        {
            "name": "Speakers",
            "hostapi": 1,
            "max_input_channels": 0,
            "max_output_channels": 2,
        },
    ]
    sd = SimpleNamespace(
        query_hostapis=lambda: [
            {"name": "MME", "default_input_device": -1, "default_output_device": 0},
            {"name": "Windows WASAPI", "default_input_device": -1, "default_output_device": 1},
        ],
        query_devices=lambda index=None: devices if index is None else devices[index],
        default=SimpleNamespace(device=[-1, 0]),
    )

    assert _resolve_sounddevice_loopback_index(sd, None) == 1


def test_select_soundcard_loopback_by_output_name() -> None:
    selected = SimpleNamespace(name="Speakers (Realtek Audio)", isloopback=True)
    other = SimpleNamespace(name="Monitor", isloopback=True)

    sc = SimpleNamespace(
        all_microphones=lambda include_loopback: [other, selected],
        get_microphone=lambda id, include_loopback: selected,
    )

    assert _select_soundcard_loopback_microphone(sc, "Speakers (Realtek Audio)") is selected


def test_select_soundcard_loopback_falls_back_to_name_match() -> None:
    selected = SimpleNamespace(name="Speakers (Realtek Audio)", isloopback=True)
    other = SimpleNamespace(name="Monitor", isloopback=True)

    def get_microphone(_id, include_loopback):
        raise RuntimeError("not found")

    sc = SimpleNamespace(
        all_microphones=lambda include_loopback: [other, selected],
        get_microphone=get_microphone,
    )

    assert _select_soundcard_loopback_microphone(sc, "Speakers") is selected


def test_push_samples_converts_stereo_float_to_mono_pcm() -> None:
    chunks = []
    capture = SoundDeviceCapture(
        AudioSource.REMOTE_AUDIO,
        device_index=None,
        sample_rate=16_000,
        chunk_seconds=1.0,
        on_chunk=chunks.append,
    )
    samples = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)

    capture._push_samples(samples, timestamp=123.0)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source is AudioSource.REMOTE_AUDIO
    assert chunk.timestamp == 123.0
    assert chunk.sample_rate == 16_000
    assert np.frombuffer(chunk.pcm, dtype=np.int16).tolist() == [0, 16383]


def test_soundcard_loopback_failure_does_not_emit_stopped_after_failed(monkeypatch) -> None:
    statuses = []
    errors = []
    microphone = SimpleNamespace(
        name="Speakers",
        channels=2,
        isloopback=True,
        recorder=lambda **_kwargs: FailingRecorder(),
    )
    soundcard = SimpleNamespace(
        all_microphones=lambda include_loopback: [microphone],
        get_microphone=lambda id, include_loopback: microphone,
    )
    sd = SimpleNamespace(query_devices=lambda _index: {"name": "Speakers"})
    monkeypatch.setitem(sys.modules, "soundcard", soundcard)
    capture = SoundDeviceCapture(
        AudioSource.REMOTE_AUDIO,
        device_index=8,
        sample_rate=16_000,
        chunk_seconds=1.0,
        on_chunk=lambda _chunk: None,
        on_error=errors.append,
        on_status=statuses.append,
    )

    capture._run_soundcard_loopback(sd, blocksize=1600)

    assert statuses == ["REMOTE_AUDIO: started", "REMOTE_AUDIO: failed"]
    assert errors == ["Loopback capture failed for REMOTE_AUDIO. Check the selected output device."]


class FailingRecorder:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def record(self, numframes):
        raise RuntimeError("device lost")
