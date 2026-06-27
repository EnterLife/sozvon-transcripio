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
    assert router.stats().dropped_chunks == 2
    assert router.stats().pushed_chunks == 3


def test_audio_router_stop_does_not_block_when_full() -> None:
    router = AudioRouter(max_chunks=1)
    router.push(chunk(1))

    router.stop()

    assert list(router.chunks()) == []


def test_audio_router_reports_backpressure_stats() -> None:
    reports = []
    router = AudioRouter(max_chunks=1, on_backpressure=reports.append)
    router.push(chunk(1))
    router.push(chunk(2))

    assert len(reports) == 1
    assert reports[0].dropped_chunks == 1
    assert reports[0].pushed_chunks == 2
    assert reports[0].queued_chunks == 1


def test_audio_router_uses_at_least_one_queue_slot() -> None:
    router = AudioRouter(max_chunks=0)

    router.push(chunk(1))
    router.stop()

    assert [item.timestamp for item in router.chunks()] == []
    assert router.stats().max_chunks == 1
