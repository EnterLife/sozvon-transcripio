from __future__ import annotations

from dataclasses import dataclass, field

from audio.types import AudioChunk, AudioSource

BYTES_PER_SAMPLE = 2


@dataclass
class _BufferedAudio:
    source: AudioSource
    timestamp: float
    sample_rate: int
    pcm_parts: list[bytes] = field(default_factory=list)
    frame_count: int = 0

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frame_count / self.sample_rate

    def append(self, chunk: AudioChunk) -> None:
        self.pcm_parts.append(chunk.pcm)
        self.frame_count += len(chunk.pcm) // BYTES_PER_SAMPLE

    def to_chunk(self) -> AudioChunk:
        return AudioChunk(
            source=self.source,
            timestamp=self.timestamp,
            sample_rate=self.sample_rate,
            pcm=b"".join(self.pcm_parts),
        )


class AudioWindowBuffer:
    def __init__(self, target_seconds: float = 3.0) -> None:
        self.target_seconds = max(0.0, target_seconds)
        self._buffers: dict[str, _BufferedAudio] = {}

    def add(self, chunk: AudioChunk) -> list[AudioChunk]:
        if self.target_seconds <= 0:
            return [chunk]

        key = str(chunk.source)
        ready: list[AudioChunk] = []
        buffered = self._buffers.get(key)
        if buffered and buffered.sample_rate != chunk.sample_rate:
            ready.append(buffered.to_chunk())
            buffered = None

        if buffered is None:
            buffered = _BufferedAudio(
                source=chunk.source,
                timestamp=chunk.timestamp,
                sample_rate=chunk.sample_rate,
            )
            self._buffers[key] = buffered

        buffered.append(chunk)
        if buffered.duration_seconds >= self.target_seconds:
            ready.append(buffered.to_chunk())
            del self._buffers[key]
        return ready

    def flush(self) -> list[AudioChunk]:
        chunks = [buffered.to_chunk() for buffered in self._buffers.values()]
        self._buffers.clear()
        return sorted(chunks, key=lambda chunk: chunk.timestamp)
