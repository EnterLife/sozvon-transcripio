from __future__ import annotations

import logging
import os
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from core.cuda_runtime import configure_cuda_dll_paths, install_cuda_runtime_packages
from speech.hardware_detector import HardwareDetector, HardwareInfo, choose_model

logger = logging.getLogger(__name__)

_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_NO_PROXY_ENV_VARS = ("NO_PROXY", "no_proxy")
_HF_TOKEN_ENV_VAR = "HF_TOKEN"
_SUPPORTED_PROXY_SCHEMES = {"http", "https", "socks5", "socks5h"}
_CUDA_ERROR_MARKERS = ("cuda", "cublas", "cudnn", "nvidia")


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
def _model_download_env(
    progress: callable | None = None,
    hf_token: str | None = None,
) -> Iterator[None]:
    unsupported = _unsupported_proxy_env()
    token = hf_token.strip() if hf_token else None
    if not unsupported and not token:
        yield
        return

    if unsupported:
        logger.warning("Bypassing unsupported proxy settings for model download: %s", unsupported)
    if unsupported and progress:
        progress("Unsupported system proxy ignored for model download; using direct connection")

    no_proxy_names = tuple(_unique_env_names(_NO_PROXY_ENV_VARS))
    previous_no_proxy = {name: os.environ.get(name) for name in no_proxy_names}
    previous_hf_token = os.environ.get(_HF_TOKEN_ENV_VAR)
    try:
        if unsupported:
            for name in no_proxy_names:
                os.environ[name] = "*"
        if token:
            os.environ[_HF_TOKEN_ENV_VAR] = token
        yield
    finally:
        if token:
            if previous_hf_token is None:
                os.environ.pop(_HF_TOKEN_ENV_VAR, None)
            else:
                os.environ[_HF_TOKEN_ENV_VAR] = previous_hf_token
        if unsupported:
            for name, value in previous_no_proxy.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class ModelManager:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.detector = HardwareDetector()

    def select(
        self,
        configured_model: str | None = None,
        auto_select: bool = True,
        device_mode: str = "auto",
        configured_compute_type: str = "auto",
    ) -> ModelSelection:
        hardware = self.detector.detect()
        model_size = choose_model(hardware) if auto_select or not configured_model else configured_model
        device = _select_device(device_mode, hardware)
        compute_type = _select_compute_type(configured_compute_type, device)
        selection = ModelSelection(model_size, device, compute_type, hardware)
        logger.info("Selected model: %s", selection)
        return selection

    def ensure_model(
        self,
        selection: ModelSelection,
        progress: callable | None = None,
        hf_token: str | None = None,
        auto_install_cuda_runtime: bool = True,
        allow_cpu_fallback: bool = True,
    ):
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install dependencies with `pip install -e .[gpu]`."
            ) from exc

        if progress:
            progress(f"Loading model {selection.model_size}...")

        try:
            return self._create_model(
                WhisperModel,
                selection,
                progress,
                hf_token,
                auto_install_cuda_runtime,
            )
        except Exception as exc:
            if _is_proxy_scheme_error(exc):
                raise _proxy_scheme_runtime_error() from exc
            if not _should_fallback_to_cpu(exc, selection.device, allow_cpu_fallback):
                raise
            logger.warning("CUDA model loading failed; falling back to CPU int8", exc_info=True)
            if progress:
                progress("CUDA unavailable; falling back to CPU int8")
            cpu_selection = ModelSelection(
                model_size=selection.model_size,
                device="cpu",
                compute_type="int8",
                hardware=selection.hardware,
            )
            try:
                return self._create_model(
                    WhisperModel,
                    cpu_selection,
                    progress,
                    hf_token,
                    auto_install_cuda_runtime=False,
                )
            except Exception as fallback_exc:
                if _is_proxy_scheme_error(fallback_exc):
                    raise _proxy_scheme_runtime_error() from fallback_exc
                raise

    def _create_model(
        self,
        model_class,
        selection: ModelSelection,
        progress: callable | None,
        hf_token: str | None,
        auto_install_cuda_runtime: bool,
    ):
        if selection.device == "cuda":
            _prepare_cuda_runtime(auto_install_cuda_runtime, progress)
        with _model_download_env(progress, hf_token):
            return model_class(
                selection.model_size,
                device=selection.device,
                compute_type=selection.compute_type,
                download_root=str(self.models_dir),
            )


def _select_device(device_mode: str, hardware: HardwareInfo) -> str:
    if device_mode == "cpu":
        return "cpu"
    if device_mode == "cuda":
        return "cuda"
    return "cuda" if hardware.cuda_available else "cpu"


def _select_compute_type(configured_compute_type: str, device: str) -> str:
    if configured_compute_type != "auto":
        return configured_compute_type
    return "float16" if device == "cuda" else "int8"


def _prepare_cuda_runtime(auto_install: bool, progress: callable | None = None) -> None:
    cuda_status = configure_cuda_dll_paths()
    if not cuda_status.is_ready and auto_install:
        if progress:
            progress("Installing missing CUDA runtime packages...")
        completed = install_cuda_runtime_packages()
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Could not install CUDA runtime packages: {details}")
        cuda_status = configure_cuda_dll_paths()
    if not cuda_status.is_ready:
        missing = ", ".join(cuda_status.missing_dlls)
        raise RuntimeError(f"Missing CUDA runtime DLLs: {missing}")


def _should_fallback_to_cpu(exc: Exception, device: str, allow_cpu_fallback: bool) -> bool:
    if not allow_cpu_fallback or device != "cuda":
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _CUDA_ERROR_MARKERS)


def _is_proxy_scheme_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "Unknown scheme for proxy URL" in str(exc)


def _proxy_scheme_runtime_error() -> RuntimeError:
    return RuntimeError(
        "Model download failed because the system proxy uses an unsupported scheme. "
        "Use http://, https://, socks5://, or socks5h:// for proxy environment "
        "variables, or clear them before starting the app."
    )
