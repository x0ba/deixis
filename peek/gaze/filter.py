"""One Euro filter (Casiez et al. 2012) for a 2D point stream."""

import math


class OneEuro:
    def __init__(self, min_cutoff: float = 0.6, beta: float = 0.004, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: tuple[float, float] | None = None
        self._dx = (0.0, 0.0)
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = None
        self._t = None
        self._dx = (0.0, 0.0)

    def __call__(self, t: float, x: float, y: float) -> tuple[float, float]:
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t = (x, y), t
            return x, y
        dt = t - self._t
        self._t = t
        a_d = self._alpha(self.d_cutoff, dt)
        raw_dx = ((x - self._x[0]) / dt, (y - self._x[1]) / dt)
        self._dx = (
            a_d * raw_dx[0] + (1 - a_d) * self._dx[0],
            a_d * raw_dx[1] + (1 - a_d) * self._dx[1],
        )
        speed = math.hypot(*self._dx)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self._x = (a * x + (1 - a) * self._x[0], a * y + (1 - a) * self._x[1])
        return self._x
