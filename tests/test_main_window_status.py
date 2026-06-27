from audio.types import AudioSource
from gui.status import parse_capture_status


def test_parse_capture_status_for_known_source() -> None:
    assert parse_capture_status("USER_MIC: started") == (AudioSource.USER_MIC, "started")
    assert parse_capture_status("REMOTE_AUDIO: failed") == (AudioSource.REMOTE_AUDIO, "failed")


def test_parse_capture_status_handles_unknown_message() -> None:
    assert parse_capture_status("Ready") == (None, "idle")
    assert parse_capture_status("OTHER: started") == (None, "started")
