from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a headless PySide GUI smoke check.")
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    with tempfile.TemporaryDirectory(prefix="sozvon-gui-smoke-") as temp_dir:
        os.environ["APPDATA"] = temp_dir

        from PySide6.QtWidgets import QApplication

        from config.settings import AppSettings, save_settings
        from core.paths import ensure_app_dirs
        from gui.main_window import MainWindow

        paths = ensure_app_dirs()
        settings = AppSettings()
        settings.recognition.dry_run = args.local_model_path is None
        settings.recognition.local_model_path = args.local_model_path
        settings.recognition.offline_mode = args.local_model_path is not None
        settings.recognition.device = args.device
        settings.storage.transcript_dir = str(Path(temp_dir) / "transcripts")
        save_settings(paths.settings_file, settings)

        app = QApplication([])
        window = MainWindow(paths)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if settings.recognition.dry_run:
                break
            if window.model is not None or "failed" in window.model_label.text().lower():
                break
            time.sleep(0.05)

        checks = [
            window.start_button.isEnabled(),
            not window.stop_button.isEnabled(),
            Path(settings.storage.transcript_dir).exists(),
        ]
        if settings.recognition.dry_run:
            checks.append("test mode" in window.model_label.text().lower())
        else:
            checks.extend(
                [
                    window.model is not None,
                    "failed" not in window.model_label.text().lower(),
                ]
            )
        window.close()
        app.processEvents()
        return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
