from storage.session_paths import resolve_transcript_dir, same_transcript_dir


def test_resolve_transcript_dir_uses_default_when_unset(tmp_path) -> None:
    assert resolve_transcript_dir(None, tmp_path) == tmp_path


def test_resolve_transcript_dir_uses_configured_path(tmp_path) -> None:
    selected = tmp_path / "custom"

    assert resolve_transcript_dir(str(selected), tmp_path) == selected


def test_same_transcript_dir_treats_equivalent_paths_as_same(tmp_path) -> None:
    assert same_transcript_dir(tmp_path, tmp_path / ".")
