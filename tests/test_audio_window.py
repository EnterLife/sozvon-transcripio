from audio.types import AudioChunk, AudioSource
from speech.audio_window import AudioWindowBuffer


def chunk(source: AudioSource, timestamp: float, frames: int, sample_rate: int = 4) -> AudioChunk:
    return AudioChunk(
        source=source,
        timestamp=timestamp,
        sample_rate=sample_rate,
        pcm=b"\x01\x00" * frames,
    )


def test_audio_window_buffers_until_target_duration() -> None:
    buffer = AudioWindowBuffer(target_seconds=1.0)

    assert buffer.add(chunk(AudioSource.USER_MIC, 10.0, 2)) == []
    ready = buffer.add(chunk(AudioSource.USER_MIC, 10.5, 2))

    assert len(ready) == 1
    assert ready[0].source is AudioSource.USER_MIC
    assert ready[0].timestamp == 10.0
    assert ready[0].sample_rate == 4
    assert ready[0].pcm == b"\x01\x00" * 4


def test_audio_window_keeps_sources_separate() -> None:
    buffer = AudioWindowBuffer(target_seconds=1.0)

    assert buffer.add(chunk(AudioSource.USER_MIC, 1.0, 2)) == []
    assert buffer.add(chunk(AudioSource.REMOTE_AUDIO, 2.0, 4))[0].source is AudioSource.REMOTE_AUDIO

    flushed = buffer.flush()

    assert [item.source for item in flushed] == [AudioSource.USER_MIC]


def test_audio_window_flushes_remaining_chunks_in_timestamp_order() -> None:
    buffer = AudioWindowBuffer(target_seconds=10.0)
    buffer.add(chunk(AudioSource.REMOTE_AUDIO, 2.0, 1))
    buffer.add(chunk(AudioSource.USER_MIC, 1.0, 1))

    flushed = buffer.flush()

    assert [item.timestamp for item in flushed] == [1.0, 2.0]


def test_audio_window_flushes_before_sample_rate_change() -> None:
    buffer = AudioWindowBuffer(target_seconds=10.0)
    buffer.add(chunk(AudioSource.USER_MIC, 1.0, 2, sample_rate=4))

    ready = buffer.add(chunk(AudioSource.USER_MIC, 2.0, 2, sample_rate=8))

    assert len(ready) == 1
    assert ready[0].sample_rate == 4
    assert buffer.flush()[0].sample_rate == 8
