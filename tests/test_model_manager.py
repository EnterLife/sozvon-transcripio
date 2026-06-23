import os
import sys
import types
from pathlib import Path

from speech.hardware_detector import GpuInfo, HardwareInfo
from speech.model_manager import ModelManager, _model_download_env, _unsupported_proxy_env


def hardware(cuda: bool = False) -> HardwareInfo:
    gpus = []
    if cuda:
        gpus.append(
            GpuInfo(
                name="NVIDIA RTX",
                vram_gb=8,
                is_nvidia=True,
                cuda_available=True,
                directml_available=False,
            )
        )
    return HardwareInfo(
        cpu_cores=8,
        cpu_frequency_mhz=None,
        ram_gb=16,
        gpus=gpus,
        os_name="Windows 11",
    )


def test_unsupported_socks4_proxy_is_detected(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks4://127.0.0.1:10808")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})

    assert _unsupported_proxy_env() == {"ALL_PROXY": "socks4://127.0.0.1:10808"}


def test_supported_socks5_proxy_is_allowed(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:10808")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})

    assert _unsupported_proxy_env() == {}


def test_unsupported_windows_system_proxy_is_detected(monkeypatch) -> None:
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(
        "urllib.request.getproxies",
        lambda: {"http": "socks4://127.0.0.1:10808", "https": "socks4://127.0.0.1:10808"},
    )

    assert _unsupported_proxy_env() == {
        "system:http": "socks4://127.0.0.1:10808",
        "system:https": "socks4://127.0.0.1:10808",
    }


def test_model_download_env_temporarily_bypasses_unsupported_proxy(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks4://127.0.0.1:10808")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})

    with _model_download_env():
        assert os.environ["NO_PROXY"] == "*"
        assert os.environ["no_proxy"] == "*"

    assert os.environ["NO_PROXY"] == "localhost"
    if os.name == "nt":
        assert os.environ["no_proxy"] == "localhost"
    else:
        assert "no_proxy" not in os.environ


def test_model_download_env_temporarily_sets_hf_token(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})

    with _model_download_env(hf_token=" hf_test_token "):
        assert os.environ["HF_TOKEN"] == "hf_test_token"

    assert "HF_TOKEN" not in os.environ


def test_model_download_env_restores_existing_hf_token(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "existing_token")
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})

    with _model_download_env(hf_token="new_token"):
        assert os.environ["HF_TOKEN"] == "new_token"

    assert os.environ["HF_TOKEN"] == "existing_token"


def test_no_proxy_star_already_bypasses_proxy_validation(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks4://127.0.0.1:10808")
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setattr("urllib.request.getproxies", lambda: {"http": "socks4://127.0.0.1:10808"})

    assert _unsupported_proxy_env() == {}


def test_select_auto_uses_cuda_for_nvidia_gpu(monkeypatch, tmp_path: Path) -> None:
    manager = ModelManager(tmp_path)
    monkeypatch.setattr(manager.detector, "detect", lambda: hardware(cuda=True))

    selection = manager.select(device_mode="auto", configured_compute_type="auto")

    assert selection.device == "cuda"
    assert selection.compute_type == "float16"


def test_select_cpu_override_uses_cpu(monkeypatch, tmp_path: Path) -> None:
    manager = ModelManager(tmp_path)
    monkeypatch.setattr(manager.detector, "detect", lambda: hardware(cuda=True))

    selection = manager.select(device_mode="cpu", configured_compute_type="auto")

    assert selection.device == "cpu"
    assert selection.compute_type == "int8"


def test_select_custom_compute_type_is_preserved(monkeypatch, tmp_path: Path) -> None:
    manager = ModelManager(tmp_path)
    monkeypatch.setattr(manager.detector, "detect", lambda: hardware(cuda=True))

    selection = manager.select(device_mode="cuda", configured_compute_type="float32")

    assert selection.device == "cuda"
    assert selection.compute_type == "float32"


def test_cuda_load_error_falls_back_to_cpu(monkeypatch, tmp_path: Path) -> None:
    created: list[tuple[str, str]] = []

    class FakeWhisperModel:
        def __init__(self, _model_size, *, device, compute_type, download_root) -> None:
            created.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found")
            self.download_root = download_root

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    monkeypatch.setattr("speech.model_manager.configure_cuda_dll_paths", _ready_cuda)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})

    manager = ModelManager(tmp_path)
    model = manager.ensure_model(
        types.SimpleNamespace(
            model_size="small",
            device="cuda",
            compute_type="float16",
            hardware=hardware(cuda=True),
        ),
        allow_cpu_fallback=True,
    )

    assert isinstance(model, FakeWhisperModel)
    assert created == [("cuda", "float16"), ("cpu", "int8")]


def test_cuda_runtime_auto_install(monkeypatch, tmp_path: Path) -> None:
    configure_calls = 0
    install_calls = 0

    def fake_configure_cuda_dll_paths():
        nonlocal configure_calls
        configure_calls += 1
        if configure_calls == 1:
            return types.SimpleNamespace(is_ready=False, missing_dlls=("cublas64_12.dll",))
        return types.SimpleNamespace(is_ready=True, missing_dlls=())

    def fake_install_cuda_runtime_packages():
        nonlocal install_calls
        install_calls += 1
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    monkeypatch.setattr("speech.model_manager.configure_cuda_dll_paths", fake_configure_cuda_dll_paths)
    monkeypatch.setattr("speech.model_manager.install_cuda_runtime_packages", fake_install_cuda_runtime_packages)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})

    manager = ModelManager(tmp_path)
    manager.ensure_model(
        types.SimpleNamespace(
            model_size="small",
            device="cuda",
            compute_type="float16",
            hardware=hardware(cuda=True),
        ),
        auto_install_cuda_runtime=True,
        allow_cpu_fallback=False,
    )

    assert configure_calls == 2
    assert install_calls == 1


def _ready_cuda():
    return types.SimpleNamespace(is_ready=True, missing_dlls=())
