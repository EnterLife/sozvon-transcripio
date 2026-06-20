from audio.audio_router import AudioRouter
from audio.types import AudioChunk


def chunk(sequence: int) -> AudioChunk:
    return AudioChunk(
        source="USER_MIC",
        pcm=sequence.to_bytes(2, byteorder="little", signed=True),
        sample_rate=16_000,
        timestamp=float(sequence),
    )


def test_audio_router_keeps_newest_chunk_under_backpressure() -> None:
    router = AudioRouter(max_chunks=2)
    router.push(chunk(1))
    router.push(chunk(2))
    router.push(chunk(3))
    router.stop()

    chunks = list(router.chunks())

    assert [item.timestamp for item in chunks] == [3.0]


def test_audio_router_stop_does_not_block_when_full() -> None:
    router = AudioRouter(max_chunks=1)
    router.push(chunk(1))

    router.stop()

    assert list(router.chunks()) == []
