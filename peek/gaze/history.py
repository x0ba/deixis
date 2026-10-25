"""A time-stamped buffer of gaze points and the fixations within a window.

All timestamps are `time.monotonic()` seconds. Points are global screen
coordinates with a top-left origin (the Quartz / Accessibility convention).
"""

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

FIXATION_RADIUS_PX = 90.0
MIN_FIXATION_S = 0.12


@dataclass(frozen=True)
class Fixation:
    x: float
    y: float
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class GazeHistory:
    def __init__(self, seconds: float = 30.0):
        self._samples: deque[tuple[float, float, float]] = deque()
        self._seconds = seconds
        self._lock = threading.Lock()

    def add(self, t: float, x: float, y: float) -> None:
        with self._lock:
            self._samples.append((t, x, y))
            while self._samples and self._samples[0][0] < t - self._seconds:
                self._samples.popleft()

    def latest(self) -> tuple[float, float, float] | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def window(self, start: float, end: float) -> np.ndarray:
        with self._lock:
            rows = [s for s in self._samples if start <= s[0] <= end]
        return np.array(rows, dtype=float).reshape(-1, 3)

    def fixations(self, start: float, end: float) -> list[Fixation]:
        return fixations(self.window(start, end))


def fixations(samples: np.ndarray) -> list[Fixation]:
    """Group consecutive samples that stay within a radius of their running centroid."""
    result: list[Fixation] = []
    if len(samples) == 0:
        return result
    group = [samples[0]]
    for row in samples[1:]:
        centroid = np.mean([g[1:] for g in group], axis=0)
        if np.hypot(*(row[1:] - centroid)) <= FIXATION_RADIUS_PX:
            group.append(row)
            continue
        _close(group, result)
        group = [row]
    _close(group, result)
    return _merge_neighbours(result)


def _close(group: list[np.ndarray], out: list[Fixation]) -> None:
    start, end = group[0][0], group[-1][0]
    if end - start < MIN_FIXATION_S:
        return
    xy = np.median([g[1:] for g in group], axis=0)
    out.append(Fixation(float(xy[0]), float(xy[1]), float(start), float(end)))


def _merge_neighbours(items: list[Fixation]) -> list[Fixation]:
    merged: list[Fixation] = []
    for f in items:
        if merged and np.hypot(f.x - merged[-1].x, f.y - merged[-1].y) <= FIXATION_RADIUS_PX:
            prev = merged[-1]
            w0, w1 = prev.duration, f.duration
            total = (w0 + w1) or 1.0
            merged[-1] = Fixation(
                (prev.x * w0 + f.x * w1) / total,
                (prev.y * w0 + f.y * w1) / total,
                prev.start,
                f.end,
            )
        else:
            merged.append(f)
    return merged


def primary(items: list[Fixation]) -> Fixation | None:
    """The longest fixation, which is usually what the speaker meant by "this"."""
    return max(items, key=lambda f: f.duration, default=None)
