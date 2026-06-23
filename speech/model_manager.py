from __future__ import annotations

import logging
import os
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from speech.hardware_detector import HardwareDetector, HardwareInfo, choose_model

logger = logging.getLogger(__name__)

_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_NO_PROXY_ENV_VARS = ("NO_PROXY", "no_proxy")
_SUPPORTED_PROXY_SCHEMES = {"http", "https", "socks5", "socks5h"}


def _env_identity(name: str) -> str:
    return name.upper() if os.name == "nt" else name


def _unique_env_names(names: tuple[str, ...]) -> Iterator[str]:
    seen = set()
    for name in names:
        identity = _env_identity(name)
        if identity in seen:
            continue
        seen.add(identity)
        yield name


@dataclass(frozen=True)
class ModelSelection:
    model_size: str
    device: str
    compute_type: str
    hardware: HardwareInfo


def _no_proxy_bypasses_all() -> bool:
    return any(os.environ.get(name, "").strip() == "*" for name in _unique_env_names(_NO_PROXY_ENV_VARS))


def _unsupported_proxy_env() -> dict[str, str]:
    if _no_proxy_bypasses_all():
        return {}

    unsupported = {}
    for name in _unique_env_names(_PROXY_ENV_VARS):
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        scheme = urlparse(value).scheme.lower()
        if scheme not in _SUPPORTED_PROXY_SCHEMES:
            unsupported[name] = value
    for name, value in urllib.request.getproxies().items():
        if name == "no":
            continue
        scheme = urlparse(value).scheme.lower()
        if scheme not in _SUPPORTED_PROXY_SCHEMES:
            unsupported[f"system:{name}"] = value
    return unsupported


@contextmanager
def _model_download_env(progress: callable | None = None) -> Iterator[None]:
    unsupported = _unsupported_proxy_env()
    if not unsupported:
        yield
        return

    logger.warning("Bypassing unsupported proxy settings for model download: %s", unsupported)
    if progress:
        progress("Unsupported system proxy ignored for model download; using direct connection")

    no_proxy_names = tuple(_unique_env_names(_NO_PROXY_ENV_VARS))
    previous_no_proxy = {name: os.environ.get(name) for name in no_proxy_names}
    try:
        for name in no_proxy_names:
            os.environ[name] = "*"
        yield
    finally:
        for name, value in previous_no_proxy.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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

        try:
            with _model_download_env(progress):
                return WhisperModel(
                    selection.model_size,
                    device=selection.device,
                    compute_type=selection.compute_type,
                    download_root=str(self.models_dir),
                )
        except ValueError as exc:
            if "Unknown scheme for proxy URL" in str(exc):
                raise RuntimeError(
                    "Model download failed because the system proxy uses an unsupported scheme. "
                    "Use http://, https://, socks5://, or socks5h:// for proxy environment "
                    "variables, or clear them before starting the app."
                ) from exc
            raise
