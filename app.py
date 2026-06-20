from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from core.paths import ensure_app_dirs
from core.runtime import configure_logging
from gui.main_window import MainWindow


def main() -> int:
    paths = ensure_app_dirs()
    configure_logging(paths.logs_dir)
    logging.info("Starting Realtime Call Transcriber")

    app = QApplication(sys.argv)
    app.setApplicationName("Realtime Call Transcriber")
    app.setOrganizationName("RealtimeCallTranscriber")

    window = MainWindow(paths)
    window.resize(1040, 720)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

