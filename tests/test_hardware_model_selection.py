import types

from speech.hardware_detector import GpuInfo, HardwareDetector, HardwareInfo, choose_model


def hardware(cpu_cores: int, ram_gb: float, vram_gb: float = 0) -> HardwareInfo:
    gpus = []
    if vram_gb:
        gpus.append(
            GpuInfo(
                name="NVIDIA RTX",
                vram_gb=vram_gb,
                is_nvidia=True,
                cuda_available=True,
                directml_available=False,
            )
        )
    return HardwareInfo(
        cpu_cores=cpu_cores,
        cpu_frequency_mhz=None,
        ram_gb=ram_gb,
        gpus=gpus,
        os_name="Windows 11",
    )


def test_choose_tiny_for_weak_cpu() -> None:
    assert choose_model(hardware(cpu_cores=4, ram_gb=16)) == "tiny"


def test_choose_small_for_middle_cpu() -> None:
    assert choose_model(hardware(cpu_cores=8, ram_gb=16)) == "small"


def test_choose_medium_for_10gb_nvidia() -> None:
    assert choose_model(hardware(cpu_cores=8, ram_gb=32, vram_gb=10)) == "medium"


def test_choose_large_for_16gb_nvidia() -> None:
    assert choose_model(hardware(cpu_cores=8, ram_gb=32, vram_gb=16)) == "large-v3"


def test_nvidia_smi_detection_finds_gpu(monkeypatch) -> None:
    detector = HardwareDetector()

    monkeypatch.setattr(detector, "_directml_available", lambda: False)

    def fake_run(*_args, **_kwargs):
        return types.SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4070, 12282\n",
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    gpus = detector._detect_nvidia_with_nvidia_smi()

    assert gpus == [
        GpuInfo(
            name="NVIDIA GeForce RTX 4070",
            vram_gb=12282 / 1024,
            is_nvidia=True,
            cuda_available=True,
            directml_available=False,
        )
    ]


def test_windows_cim_detection_finds_nvidia_gpu(monkeypatch) -> None:
    detector = HardwareDetector()

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(detector, "_directml_available", lambda: False)

    def fake_run(*_args, **_kwargs):
        return types.SimpleNamespace(
            returncode=0,
            stdout='{"Name":"NVIDIA GeForce RTX 3060","AdapterRAM":6442450944}',
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    gpus = detector._detect_nvidia_with_windows_cim()

    assert gpus == [
        GpuInfo(
            name="NVIDIA GeForce RTX 3060",
            vram_gb=6,
            is_nvidia=True,
            cuda_available=True,
            directml_available=False,
        )
    ]
