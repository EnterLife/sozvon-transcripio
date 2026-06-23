from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class HuggingFaceUnauthenticatedWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "huggingface_hub.utils._http":
            return True
        return "You are sending unauthenticated requests to the HF Hub" not in record.getMessage()


def configure_logging(logs_dir: Path) -> None:
    log_file = logs_dir / "app.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    hf_auth_filter = HuggingFaceUnauthenticatedWarningFilter()

    file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(hf_auth_filter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(hf_auth_filter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
