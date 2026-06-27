from __future__ import annotations

import logging
import os
import time
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
MODEL_QUALITY_ORDER = ("tiny", "base", "small", "medium", "large-v3")


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
    local_model_path: Path | None = None

    @property
    def model_identifier(self) -> str:
        if self.local_model_path is not None:
            return str(self.local_model_path)
        return self.model_size

    @property
    def display_name(self) -> str:
        if self.local_model_path is not None:
            return self.local_model_path.name
        return self.model_size

    @property
    def is_local_model(self) -> bool:
        return self.local_model_path is not None


@dataclass(frozen=True)
class CalibrationResult:
    model_size: str
    device: str
    compute_type: str
    audio_seconds: float
    elapsed_seconds: float
    realtime_factor: float
    passed: bool
    error: str | None = None


@dataclass(frozen=True)
class CalibrationReport:
    results: tuple[CalibrationResult, ...]
    selected: CalibrationResult | None


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
        local_model_path: str | None = None,
    ) -> ModelSelection:
        hardware = self.detector.detect()
        resolved_local_model_path = _resolve_local_model_path(local_model_path)
        model_size = choose_model(hardware) if auto_select or not configured_model else configured_model
        device = _select_device(device_mode, hardware)
        compute_type = _select_compute_type(configured_compute_type, device)
        selection = ModelSelection(
            model_size,
            device,
            compute_type,
            hardware,
            resolved_local_model_path,
        )
        logger.info("Selected model: %s", selection)
        return selection

    def ensure_model(
        self,
        selection: ModelSelection,
        progress: callable | None = None,
        hf_token: str | None = None,
        auto_install_cuda_runtime: bool = True,
        allow_cpu_fallback: bool = True,
        offline_mode: bool = False,
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
                offline_mode,
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
                    offline_mode=offline_mode,
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
        offline_mode: bool,
    ):
        if selection.device == "cuda":
            _prepare_cuda_runtime(auto_install_cuda_runtime, progress)
        is_local_model = bool(getattr(selection, "is_local_model", False))
        model_identifier = getattr(selection, "model_identifier", selection.model_size)
        model_kwargs = {
            "device": selection.device,
            "compute_type": selection.compute_type,
            "download_root": str(self.models_dir),
        }
        if offline_mode:
            if not _supports_local_files_only(model_class):
                if not is_local_model:
                    raise RuntimeError(
                        "Offline mode needs a local CTranslate2 model folder with this "
                        "faster-whisper version."
                    )
            else:
                model_kwargs["local_files_only"] = True
        with _model_download_env(progress, hf_token):
            return model_class(
                model_identifier,
                **model_kwargs,
            )

    def calibration_candidates(self, hardware: HardwareInfo) -> list[str]:
        max_model = choose_model(hardware)
        max_index = MODEL_QUALITY_ORDER.index(max_model)
        return list(MODEL_QUALITY_ORDER[: max_index + 1])

    def calibrate(
        self,
        configured_model: str | None = None,
        auto_select: bool = True,
        device_mode: str = "auto",
        configured_compute_type: str = "auto",
        local_model_path: str | None = None,
        progress: callable | None = None,
        hf_token: str | None = None,
        auto_install_cuda_runtime: bool = True,
        offline_mode: bool = False,
        audio_seconds: float = 3.0,
        max_realtime_factor: float = 0.8,
    ) -> CalibrationReport:
        hardware = self.detector.detect()
        resolved_local_model_path = _resolve_local_model_path(local_model_path)
        candidates = [configured_model or resolved_local_model_path.name] if resolved_local_model_path else (
            self.calibration_candidates(hardware)
            if auto_select or not configured_model
            else [configured_model]
        )
        results = []
        for model_size in candidates:
            device = _select_device(device_mode, hardware)
            selection = ModelSelection(
                model_size=model_size,
                device=device,
                compute_type=_select_compute_type(configured_compute_type, device),
                hardware=hardware,
                local_model_path=resolved_local_model_path,
            )
            if progress:
                progress(f"Calibrating {selection.display_name} on {selection.device}")
            try:
                model = self.ensure_model(
                    selection,
                    progress,
                    hf_token,
                    auto_install_cuda_runtime,
                    allow_cpu_fallback=device_mode == "auto",
                    offline_mode=offline_mode,
                )
                elapsed_seconds = _benchmark_model(model, audio_seconds)
                realtime_factor = elapsed_seconds / audio_seconds
                results.append(
                    CalibrationResult(
                        model_size=selection.display_name,
                        device=selection.device,
                        compute_type=selection.compute_type,
                        audio_seconds=audio_seconds,
                        elapsed_seconds=elapsed_seconds,
                        realtime_factor=realtime_factor,
                        passed=realtime_factor <= max_realtime_factor,
                    )
                )
            except Exception as exc:
                logger.exception("Calibration failed for %s", model_size)
                results.append(
                    CalibrationResult(
                        model_size=model_size,
                        device=selection.device,
                        compute_type=selection.compute_type,
                        audio_seconds=audio_seconds,
                        elapsed_seconds=0.0,
                        realtime_factor=float("inf"),
                        passed=False,
                        error=str(exc),
                    )
                )
        return CalibrationReport(tuple(results), _best_calibration_result(results))


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


def _resolve_local_model_path(local_model_path: str | None) -> Path | None:
    if not local_model_path:
        return None
    path = Path(local_model_path).expanduser().resolve(strict=False)
    if not path.is_dir():
        raise RuntimeError(f"Local model folder does not exist: {path}")
    if not (path / "model.bin").is_file():
        raise RuntimeError(f"Local model folder is missing model.bin: {path}")
    return path


def _supports_local_files_only(model_class) -> bool:
    try:
        import inspect

        return "local_files_only" in inspect.signature(model_class.__init__).parameters
    except Exception:
        return False


def _benchmark_model(model, audio_seconds: float) -> float:
    import io
    import math
    import wave

    import numpy as np

    sample_rate = 16_000
    seconds = max(1.0, audio_seconds)
    frames = int(sample_rate * seconds)
    samples = np.fromiter(
        (0.05 * math.sin(2 * math.pi * 220 * index / sample_rate) for index in range(frames)),
        dtype=np.float32,
        count=frames,
    )
    pcm = (samples * 32767).astype(np.int16).tobytes()
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    data.seek(0)

    started = time.perf_counter()
    segments, _info = model.transcribe(
        data,
        language=None,
        vad_filter=False,
        beam_size=1,
        condition_on_previous_text=False,
    )
    list(segments)
    return time.perf_counter() - started


def _best_calibration_result(results: list[CalibrationResult]) -> CalibrationResult | None:
    passed = [result for result in results if result.passed and result.error is None]
    if passed:
        return passed[-1]
    usable = [result for result in results if result.error is None]
    return min(usable, key=lambda result: result.realtime_factor, default=None)


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
