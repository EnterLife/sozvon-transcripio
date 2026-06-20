from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from speech.hardware_detector import HardwareDetector, HardwareInfo, choose_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSelection:
    model_size: str
    device: str
    compute_type: str
    hardware: HardwareInfo


class ModelManager:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.detector = HardwareDetector()

    def select(self, configured_model: str | None = None, auto_select: bool = True) -> ModelSelection:
        hardware = self.detector.detect()
        model_size = choose_model(hardware) if auto_select or not configured_model else configured_model
        device = "cuda" if hardware.cuda_available else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        selection = ModelSelection(model_size, device, compute_type, hardware)
        logger.info("Selected model: %s", selection)
        return selection

    def ensure_model(self, selection: ModelSelection, progress: callable | None = None):
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install dependencies with `pip install -e .[gpu]`."
            ) from exc

        if progress:
            progress(f"Loading model {selection.model_size}...")

        return WhisperModel(
            selection.model_size,
            device=selection.device,
            compute_type=selection.compute_type,
            download_root=str(self.models_dir),
        )

