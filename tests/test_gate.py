import numpy as np

from peek.voice.gate import NoiseGate, chunk_db

CHUNK = 1600
rng = np.random.default_rng(0)


def noise(db: float) -> bytes:
    amp = 32768 * 10 ** (db / 20)
    return np.clip(rng.normal(0, amp, CHUNK), -32768, 32767).astype(np.int16).tobytes()


def is_silent(chunk: bytes) -> bool:
    return not any(chunk)


def test_chunk_db_levels():
    assert chunk_db(np.zeros(CHUNK, np.int16)) == -70
    assert abs(chunk_db(np.frombuffer(noise(-30), np.int16)) - -30) < 1


def test_background_noise_is_silenced():
    gate = NoiseGate()
    out = [gate.process(noise(-45)) for _ in range(100)]
    assert all(is_silent(c) for c in out)


def test_speech_over_noise_passes_with_preroll():
    gate = NoiseGate(preroll_chunks=3, hold_chunks=8)
    for _ in range(60):
        gate.process(noise(-45))
    lead = [noise(-45) for _ in range(3)]
    speech = [noise(-20) for _ in range(10)]
    sent = [gate.process(c) for c in lead + speech]
    # Output lags by 3 chunks; the 3 quiet chunks right before speech come through.
    assert sent[3:6] == lead
    assert sent[6:] == speech[:7]


def test_gate_holds_between_words_then_closes():
    gate = NoiseGate(preroll_chunks=0, hold_chunks=4)
    for _ in range(60):
        gate.process(noise(-45))
    gate.process(noise(-20))
    gaps = [gate.process(noise(-45)) for _ in range(6)]
    assert not any(is_silent(c) for c in gaps[:3])
    assert all(is_silent(c) for c in gaps[4:])


def test_floor_adapts_to_louder_room():
    gate = NoiseGate()
    for _ in range(200):
        gate.process(noise(-30))  # a fan turns on
    assert not gate.is_open
    assert abs(gate.floor_db - -30) < 2


def test_reset_drops_buffered_audio():
    gate = NoiseGate(preroll_chunks=3)
    for _ in range(5):
        gate.process(noise(-20))
    gate.reset()
    assert not gate.is_open
    assert is_silent(gate.process(noise(-45)))
