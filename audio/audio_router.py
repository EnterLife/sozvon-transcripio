from __future__ import annotations

import queue
from collections.abc import Iterator

from audio.types import AudioChunk


class AudioRouter:
    def __init__(self, max_chunks: int = 64) -> None:
        self._queue: queue.Queue[AudioChunk | None] = queue.Queue(maxsize=max_chunks)

    def push(self, chunk: AudioChunk) -> None:
        self._put_drop_oldest(chunk)

    def stop(self) -> None:
        self._put_drop_oldest(None)

    def _put_drop_oldest(self, item: AudioChunk | None) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def chunks(self) -> Iterator[AudioChunk]:
        while True:
            item = self._queue.get()
            if item is None:
                return
            yield item
