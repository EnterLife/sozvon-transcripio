from __future__ import annotations

import logging
import platform
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_gb: float
    is_nvidia: bool
    cuda_available: bool
    directml_available: bool


@dataclass(frozen=True)
class HardwareInfo:
    cpu_cores: int
    cpu_frequency_mhz: float | None
    ram_gb: float
    gpus: list[GpuInfo]
    os_name: str

    @property
    def best_nvidia_vram_gb(self) -> float:
        nvidia = [gpu.vram_gb for gpu in self.gpus if gpu.is_nvidia]
        return max(nvidia, default=0.0)

    @property
    def cuda_available(self) -> bool:
        return any(gpu.cuda_available for gpu in self.gpus)


class HardwareDetector:
    def detect(self) -> HardwareInfo:
        try:
            import psutil
        except Exception as exc:
            raise RuntimeError(
                "psutil is not installed. Install dependencies with `pip install -e .`."
            ) from exc

        cpu_freq = psutil.cpu_freq()
        info = HardwareInfo(
            cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1,
            cpu_frequency_mhz=cpu_freq.current if cpu_freq else None,
            ram_gb=psutil.virtual_memory().total / (1024**3),
            gpus=self._detect_gpus(),
            os_name=f"{platform.system()} {platform.release()}",
        )
        logger.info("Detected hardware: %s", info)
        return info

    def _detect_gpus(self) -> list[GpuInfo]:
        gpus: list[GpuInfo] = []
        cuda_available = self._torch_cuda_available()

        try:
            import GPUtil

            for gpu in GPUtil.getGPUs():
                name = gpu.name or "Unknown GPU"
                gpus.append(
                    GpuInfo(
                        name=name,
                        vram_gb=float(gpu.memoryTotal) / 1024,
                        is_nvidia="nvidia" in name.lower(),
                        cuda_available=cuda_available and "nvidia" in name.lower(),
                        directml_available=self._directml_available(),
                    )
                )
        except Exception as exc:
            logger.info("GPUtil detection unavailable: %s", exc)

        if not gpus:
            gpus.extend(self._detect_nvidia_with_nvml(cuda_available))
        return gpus

    def _detect_nvidia_with_nvml(self, cuda_available: bool) -> list[GpuInfo]:
        try:
            import pynvml

            pynvml.nvmlInit()
            result = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                result.append(
                    GpuInfo(
                        name=name,
                        vram_gb=mem.total / (1024**3),
                        is_nvidia=True,
                        cuda_available=cuda_available,
                        directml_available=self._directml_available(),
                    )
                )
            pynvml.nvmlShutdown()
            return result
        except Exception as exc:
            logger.info("NVML detection unavailable: %s", exc)
            return []

    def _torch_cuda_available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _directml_available(self) -> bool:
        try:
            import torch_directml

            torch_directml.device()
            return True
        except Exception:
            return False


def choose_model(info: HardwareInfo) -> str:
    vram = info.best_nvidia_vram_gb
    if vram >= 16:
        return "large-v3"
    if vram >= 10:
        return "medium"
    if vram >= 6:
        return "small"
    if info.cpu_cores < 6 or info.ram_gb < 8:
        return "tiny"
    if info.cpu_cores >= 6 and info.ram_gb >= 16:
        return "small"
    return "base"
