from config.settings import AppSettings, load_settings, save_settings


def test_settings_roundtrip(tmp_path) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.audio.microphone_device = 3
    settings.audio.chunk_seconds = 0.75
    settings.recognition.language = "en"
    settings.recognition.model_size = "small"
    settings.recognition.local_model_path = "C:/models/faster-whisper-small"
    settings.recognition.quality_mode = "accurate"
    settings.recognition.glossary_terms = "Sozvon\nCTranslate2"
    settings.recognition.word_timestamps = True
    settings.recognition.auto_select_model = False
    settings.recognition.device = "cuda"
    settings.recognition.compute_type = "float16"
    settings.recognition.transcription_window_seconds = 4.5
    settings.recognition.auto_install_cuda_runtime = False
    settings.recognition.hf_token = "hf_test_token"
    settings.recognition.offline_mode = True
    settings.recognition.dry_run = True
    settings.storage.autosave_seconds = 15

    save_settings(path, settings)
    loaded = load_settings(path)

    assert loaded.audio.microphone_device == 3
    assert loaded.audio.chunk_seconds == 0.75
    assert loaded.recognition.language == "en"
    assert loaded.recognition.model_size == "small"
    assert loaded.recognition.local_model_path == "C:/models/faster-whisper-small"
    assert loaded.recognition.quality_mode == "accurate"
    assert loaded.recognition.glossary_terms == "Sozvon\nCTranslate2"
    assert loaded.recognition.word_timestamps is True
    assert loaded.recognition.auto_select_model is False
    assert loaded.recognition.device == "cuda"
    assert loaded.recognition.compute_type == "float16"
    assert loaded.recognition.transcription_window_seconds == 4.5
    assert loaded.recognition.auto_install_cuda_runtime is False
    assert loaded.recognition.hf_token == "hf_test_token"
    assert loaded.recognition.offline_mode is True
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


def test_invalid_setting_values_fall_back_or_clamp(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        """
        {
          "audio": {
            "microphone_device": "default",
            "loopback_device": true,
            "sample_rate": 12345,
            "chunk_seconds": 9
          },
          "recognition": {
            "language": "de",
            "model_size": "huge",
            "local_model_path": "   ",
            "quality_mode": "maximum",
            "glossary_terms": "   ",
            "word_timestamps": "yes",
            "auto_select_model": "yes",
            "device": "quantum",
            "compute_type": "surprise",
            "transcription_window_seconds": 99,
            "auto_install_cuda_runtime": "please",
            "hf_token": 123,
            "offline_mode": "yes",
            "dry_run": 1
          },
          "storage": {
            "autosave_seconds": 1,
            "transcript_dir": "   "
          }
        }
        """,
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.audio.microphone_device is None
    assert settings.audio.loopback_device is None
    assert settings.audio.sample_rate == 16_000
    assert settings.audio.chunk_seconds == 2.0
    assert settings.recognition.language == "ru"
    assert settings.recognition.model_size is None
    assert settings.recognition.local_model_path is None
    assert settings.recognition.quality_mode == "balanced"
    assert settings.recognition.glossary_terms is None
    assert settings.recognition.word_timestamps is False
    assert settings.recognition.auto_select_model is True
    assert settings.recognition.device == "auto"
    assert settings.recognition.compute_type == "auto"
    assert settings.recognition.transcription_window_seconds == 8.0
    assert settings.recognition.auto_install_cuda_runtime is True
    assert settings.recognition.hf_token is None
    assert settings.recognition.offline_mode is False
    assert settings.recognition.dry_run is False
    assert settings.storage.autosave_seconds == 5
    assert settings.storage.transcript_dir is None
