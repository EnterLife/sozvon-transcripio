from config.settings import AppSettings, load_settings, save_settings


def test_settings_roundtrip(tmp_path) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.audio.microphone_device = 3
    settings.audio.chunk_seconds = 0.75
    settings.recognition.language = "en"
    settings.recognition.dry_run = True
    settings.storage.autosave_seconds = 15

    save_settings(path, settings)
    loaded = load_settings(path)

    assert loaded.audio.microphone_device == 3
    assert loaded.audio.chunk_seconds == 0.75
    assert loaded.recognition.language == "en"
    assert loaded.recognition.dry_run is True
    assert loaded.storage.autosave_seconds == 15


def test_missing_settings_returns_defaults(tmp_path) -> None:
    settings = load_settings(tmp_path / "missing.json")

    assert settings.audio.sample_rate == 16_000
    assert settings.recognition.language == "ru"
    assert settings.storage.autosave_seconds == 30


def test_corrupt_settings_returns_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{broken", encoding="utf-8")

    settings = load_settings(path)

    assert settings.audio.sample_rate == 16_000
    assert settings.recognition.language == "ru"
    assert settings.storage.autosave_seconds == 30


def test_invalid_settings_sections_are_ignored(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        '{"audio": null, "recognition": ["ru"], "storage": {"autosave_seconds": 10}}',
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.audio.sample_rate == 16_000
    assert settings.recognition.language == "ru"
    assert settings.storage.autosave_seconds == 10
