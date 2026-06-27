from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass

from audio.types import AudioChunk


@dataclass(frozen=True)
class AudioRouterStats:
    pushed_chunks: int
    dropped_chunks: int
    queued_chunks: int
    max_chunks: int


class AudioRouter:
    def __init__(
        self,
        max_chunks: int = 64,
        on_backpressure: Callable[[AudioRouterStats], None] | None = None,
    ) -> None:
        self._max_chunks = max(1, max_chunks)
        self._queue: queue.Queue[AudioChunk | None] = queue.Queue(maxsize=self._max_chunks)
        self._on_backpressure = on_backpressure
        self._stats_lock = threading.Lock()
        self._pushed_chunks = 0
        self._dropped_chunks = 0

    def push(self, chunk: AudioChunk) -> None:
        with self._stats_lock:
            self._pushed_chunks += 1
        self._put_drop_oldest(chunk)

    def stop(self) -> None:
        self._put_drop_oldest(None)

    def stats(self) -> AudioRouterStats:
        with self._stats_lock:
            return AudioRouterStats(
                pushed_chunks=self._pushed_chunks,
                dropped_chunks=self._dropped_chunks,
                queued_chunks=self._queue.qsize(),
                max_chunks=self._max_chunks,
            )

    def _put_drop_oldest(self, item: AudioChunk | None) -> None:
        dropped = False
        while True:
            try:
                self._queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    removed = self._queue.get_nowait()
                except queue.Empty:
                    continue
                if removed is not None:
                    dropped = True

        if not dropped:
            return

        with self._stats_lock:
            self._dropped_chunks += 1
        if self._on_backpressure:
            self._on_backpressure(self.stats())

    def chunks(self) -> Iterator[AudioChunk]:
        while True:
            item = self._queue.get()
            if item is None:
                return
            yield item
