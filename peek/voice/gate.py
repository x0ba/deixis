"""A noise gate that only lets through speech clearly louder than the room.

It learns the background level from the quiet moments of the last few
seconds, then opens only for chunks well above it. That keeps fans, typing
and far-off voices out of the transcript, and stops them from starting
turns. It holds open briefly between words, and replays a short pre-roll
when it opens so the first syllable isn't clipped.
"""

from collections import deque

import numpy as np

FLOOR_DB = -70.0  # quietest background we assume, so digital silence can't open the gate
MIN_SPEECH_DB = -50.0  # never open for anything quieter than this


def chunk_db(samples: np.ndarray) -> float:
    """RMS level of int16 samples in dBFS."""
    if samples.size == 0:
        return FLOOR_DB
    rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768) ** 2)))
    return max(20 * np.log10(rms) if rms > 0 else FLOOR_DB, FLOOR_DB)


class NoiseGate:
    """Delays audio by ``preroll_chunks`` so it can let through the moments just
    before speech began. Emits exactly one chunk per chunk in, silenced while
    closed, so the stream stays continuous and in real time."""

    def __init__(
        self,
        margin_db: float = 12.0,
        window_chunks: int = 150,
        percentile: float = 20.0,
        hold_chunks: int = 8,
        preroll_chunks: int = 3,
    ):
        self.margin_db = margin_db
        self.percentile = percentile
        self.hold_chunks = hold_chunks
        self.preroll_chunks = preroll_chunks
        self._history: deque[float] = deque(maxlen=window_chunks)
        self._delay: deque[list] = deque()  # [chunk, passes]
        self._hold = 0
        self.floor_db = FLOOR_DB

    @property
    def is_open(self) -> bool:
        return self._hold > 0

    def reset(self) -> None:
        """Drop buffered audio, e.g. while the mic is muted. Keeps the learned floor."""
        self._delay.clear()
        self._hold = 0

    def process(self, data: bytes) -> bytes:
        """Feed one chunk of int16 audio; get back the chunk to send."""
        level = chunk_db(np.frombuffer(data, dtype=np.int16))
        self._history.append(level)
        self.floor_db = max(float(np.percentile(self._history, self.percentile)), FLOOR_DB)

        if level >= max(self.floor_db + self.margin_db, MIN_SPEECH_DB):
            self._hold = self.hold_chunks
        elif self._hold > 0:
            self._hold -= 1

        self._delay.append([data, False])
        if self.is_open:
            for item in self._delay:
                item[1] = True
        if len(self._delay) <= self.preroll_chunks:
            return bytes(len(data))
        chunk, passes = self._delay.popleft()
        return chunk if passes else bytes(len(chunk))
