from types import SimpleNamespace

from audio.capture_base import (
    _select_soundcard_loopback_microphone,
    _sounddevice_output_name,
    _sounddevice_wasapi_supports_loopback,
)


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
