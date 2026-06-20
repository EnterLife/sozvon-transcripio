from speech.hardware_detector import GpuInfo, HardwareInfo, choose_model


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

