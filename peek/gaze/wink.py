"""Spots a deliberate wink in the per-frame eyelid blendshapes.

A wink is one eye shut while the other stays open, held longer than a blink
and released. Blinks close both eyes, so any frame with both shut throws the
attempt away. Natural blinks are often lopsided for a frame or two, which the
minimum hold filters out.
"""

CLOSED = 0.5  # eyeBlink score for a shut eye
OPEN = 0.35  # the other eye must stay below this
GAP = 0.3  # and be at least this much more open
MIN_S = 0.18
MAX_S = 1.5
COOLDOWN_S = 0.6


class WinkDetector:
    def __init__(self):
        self._start: float | None = None
        self._side: str | None = None
        self._spoiled = False
        self._last = -COOLDOWN_S

    def reset(self) -> None:
        self._start = None
        self._side = None
        self._spoiled = False

    def update(self, t: float, left: float, right: float) -> float | None:
        """Feed one frame's eyeBlinkLeft/Right scores. Returns the time the wink
        began once a wink is released, otherwise None."""
        side = _wink_side(left, right)
        both = left > CLOSED and right > CLOSED
        if self._start is not None:
            if both:
                self._spoiled = True
                return None
            if side == self._side:
                if t - self._start > MAX_S:
                    self._spoiled = True
                return None
            start, spoiled = self._start, self._spoiled
            self.reset()
            if spoiled or t - start < MIN_S or start - self._last < COOLDOWN_S:
                return None
            self._last = t
            return start
        if side is not None and not both:
            self._start, self._side, self._spoiled = t, side, False
        return None


def _wink_side(left: float, right: float) -> str | None:
    if left > CLOSED and right < OPEN and left - right >= GAP:
        return "left"
    if right > CLOSED and left < OPEN and right - left >= GAP:
        return "right"
    return None
