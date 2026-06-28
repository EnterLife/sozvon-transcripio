from audio.devices import AudioDiagnostics
from audio.types import AudioSource
from gui.main_window import capture_sources_for_diagnostics
from gui.status import parse_capture_status


def test_parse_capture_status_for_known_source() -> None:
    assert parse_capture_status("USER_MIC: started") == (AudioSource.USER_MIC, "started")
    assert parse_capture_status("REMOTE_AUDIO: failed") == (AudioSource.REMOTE_AUDIO, "failed")


def test_parse_capture_status_handles_unknown_message() -> None:
    assert parse_capture_status("Ready") == (None, "idle")
    assert parse_capture_status("OTHER: started") == (None, "started")


def test_capture_sources_include_available_microphone_and_loopback() -> None:
    diagnostics = AudioDiagnostics(
        sounddevice_available=True,
        microphone_available=True,
        loopback_available=True,
        messages=[],
    )

    assert capture_sources_for_diagnostics(diagnostics) == [
        AudioSource.USER_MIC,
        AudioSource.REMOTE_AUDIO,
    ]


def test_capture_sources_allow_loopback_without_microphone() -> None:
    diagnostics = AudioDiagnostics(
        sounddevice_available=True,
        microphone_available=False,
        loopback_available=True,
        messages=["No microphone input device found."],
    )

    assert capture_sources_for_diagnostics(diagnostics) == [AudioSource.REMOTE_AUDIO]


def test_capture_sources_are_empty_when_no_audio_source_is_available() -> None:
    diagnostics = AudioDiagnostics(
        sounddevice_available=True,
        microphone_available=False,
        loopback_available=False,
        messages=[],
    )

    assert capture_sources_for_diagnostics(diagnostics) == []


def test_capture_sources_are_empty_without_sounddevice() -> None:
    diagnostics = AudioDiagnostics(
        sounddevice_available=False,
        microphone_available=True,
        loopback_available=True,
        messages=[],
    )

    assert capture_sources_for_diagnostics(diagnostics) == []
