import shutil

from speech.types import TranscriptEvent
from storage.transcript_store import TranscriptStore


def test_transcript_store_saves_txt_and_json(tmp_path) -> None:
    store = TranscriptStore(tmp_path)
    store.add(
        TranscriptEvent(
            speaker="Я",
            timestamp=1_700_000_000,
            text="Привет",
            source="USER_MIC",
        )
    )

    store.save()

    assert store.txt_path.exists()
    assert store.json_path.exists()
    assert "Привет" in store.txt_path.read_text(encoding="utf-8")
    assert "USER_MIC" in store.json_path.read_text(encoding="utf-8")


def test_transcript_store_clear(tmp_path) -> None:
    store = TranscriptStore(tmp_path)
    store.add(TranscriptEvent(speaker="Я", timestamp=1_700_000_000, text="Привет", source="USER_MIC"))
    store.clear()
    store.save()

    assert store.records == []
    assert store.json_path.read_text(encoding="utf-8") == "[]"


def test_transcript_store_uses_unique_session_paths(tmp_path) -> None:
    first = TranscriptStore(tmp_path)
    second = TranscriptStore(tmp_path)

    assert first.txt_path != second.txt_path
    assert first.json_path != second.json_path


def test_transcript_store_recreates_directory_on_save(tmp_path) -> None:
    store = TranscriptStore(tmp_path)
    shutil.rmtree(tmp_path)

    store.save()

    assert store.txt_path.exists()
    assert store.json_path.exists()


def test_transcript_store_does_not_leave_temp_files(tmp_path) -> None:
    store = TranscriptStore(tmp_path)

    store.save()

    assert list(tmp_path.glob("*.tmp")) == []
