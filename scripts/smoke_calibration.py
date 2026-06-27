from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speech.model_manager import ModelManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real faster-whisper calibration smoke check.")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-auto-install-cuda-runtime", action="store_true")
    parser.add_argument("--audio-seconds", type=float, default=1.0)
    parser.add_argument("--max-rtf", type=float, default=1.5)
    args = parser.parse_args()

    models_dir = args.models_dir or Path.cwd() / ".smoke-models"
    model = args.model
    if model is None and args.local_model_path is None:
        model = "tiny"
    manager = ModelManager(models_dir)
    report = manager.calibrate(
        configured_model=model,
        auto_select=False,
        device_mode=args.device,
        configured_compute_type=args.compute_type,
        local_model_path=args.local_model_path,
        offline_mode=args.offline,
        auto_install_cuda_runtime=not args.no_auto_install_cuda_runtime,
        audio_seconds=args.audio_seconds,
        max_realtime_factor=args.max_rtf,
        progress=print,
    )
    print(json.dumps(_json_safe(asdict(report)), ensure_ascii=False, indent=2))
    if report.selected is None:
        return 1
    return 0 if report.selected.error is None and report.selected.passed else 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
